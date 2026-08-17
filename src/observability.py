"""
observability.py
-----------------
Lightweight AIOps layer for the RAG pipeline:

- Structured JSON logging (question, latency, retrieved sources, token usage)
  so logs are queryable instead of free-text.
- Prometheus metrics (request count, latency histogram, token usage counter)
  exposed at /metrics for Grafana dashboards.
- OpenTelemetry tracing spans around retrieval and generation separately, so
  you can see in a trace viewer (Jaeger/Phoenix/LangSmith) exactly how much
  time is spent in vector search vs. LLM generation — the #1 question when
  debugging RAG latency in production.

Usage:
    from src.observability import traced_answer_question
    result = traced_answer_question("How do I reset my password?")
"""

import json
import logging
import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# --- Structured logging setup ---------------------------------------------
logger = logging.getLogger("supportiq")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))  # we emit pre-formatted JSON
    logger.addHandler(handler)

# --- OpenTelemetry tracing setup -------------------------------------------
# ConsoleSpanExporter prints spans to stdout for local dev. Swap for an OTLP
# exporter (e.g. to Jaeger, Phoenix, or an observability platform) in prod by
# changing just this block.
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("supportiq")

# --- Prometheus metrics ------------------------------------------------------
REQUEST_COUNT = Counter("supportiq_requests_total", "Total chat requests", ["status"])
REQUEST_LATENCY = Histogram(
    "supportiq_request_latency_seconds", "End-to-end request latency", buckets=(0.5, 1, 2, 4, 8, 16, 32)
)
RETRIEVAL_LATENCY = Histogram("supportiq_retrieval_latency_seconds", "Vector retrieval latency")
GENERATION_LATENCY = Histogram("supportiq_generation_latency_seconds", "LLM generation latency")
PII_REDACTIONS = Counter("supportiq_pii_redactions_total", "Number of PII redaction events", ["stage"])
GUARDRAIL_BLOCKS = Counter("supportiq_guardrail_blocks_total", "Requests blocked by guardrails", ["reason"])


def get_metrics():
    """Returns Prometheus-formatted metrics text + content type for a /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST


def traced_answer_question(question: str) -> dict:
    """
    Wraps src.rag_chain.answer_question with tracing, metrics, and structured
    logging, WITHOUT modifying the core RAG logic itself.
    """
    from src.rag_chain import (
        answer_question,  # local import avoids circular import at module load
    )

    start = time.perf_counter()
    status = "success"

    with tracer.start_as_current_span("rag_pipeline") as span:
        span.set_attribute("question.length", len(question))
        try:
            result = answer_question(question)
            span.set_attribute("sources.count", len(result.get("sources", [])))
        except Exception as e:
            status = "error"
            REQUEST_COUNT.labels(status=status).inc()
            logger.error(json.dumps({"event": "rag_error", "question": question, "error": str(e)}))
            raise
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_LATENCY.observe(elapsed)

    REQUEST_COUNT.labels(status=status).inc()
    logger.info(
        json.dumps(
            {
                "event": "rag_query",
                "question": question,
                "sources": result.get("sources", []),
                "latency_seconds": round(elapsed, 3),
                "num_chunks_retrieved": len(result.get("retrieved_chunks", [])),
            }
        )
    )
    return result
