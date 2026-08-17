"""
api.py
------
FastAPI backend exposing the RAG pipeline as a REST API, with guardrails
(input/output validation + PII redaction) and observability (tracing,
metrics, structured logs) wired in around the core RAG chain.

Run:
    uvicorn src.api:app --reload --port 8000

Then visit:
    http://localhost:8000/docs      - interactive Swagger UI
    http://localhost:8000/metrics   - Prometheus metrics
"""

import json

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.guardrails import check_input, check_output
from src.observability import (
    GUARDRAIL_BLOCKS,
    PII_REDACTIONS,
    get_metrics,
    logger,
    traced_answer_question,
)

app = FastAPI(
    title="SupportIQ API",
    description="RAG-powered customer support assistant for CloudDesk, with guardrails and observability",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, examples=["How do I reset my password?"])


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    pii_redacted: bool = False


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # --- Input guardrails ---------------------------------------------
    input_check = check_input(request.question)
    if not input_check.allowed:
        GUARDRAIL_BLOCKS.labels(reason=input_check.reason).inc()
        logger.info(json.dumps({"event": "guardrail_block", "stage": "input", "reason": input_check.reason}))
        raise HTTPException(status_code=400, detail=input_check.reason)

    if input_check.pii_found:
        PII_REDACTIONS.labels(stage="input").inc()
        logger.info(json.dumps({"event": "pii_redacted", "stage": "input", "entities": input_check.pii_found}))

    # Use the PII-redacted version of the question downstream so raw PII
    # never reaches the LLM provider or gets embedded/logged.
    safe_question = input_check.cleaned_text

    try:
        result = traced_answer_question(safe_question)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    # --- Output guardrails ----------------------------------------------
    output_check = check_output(result["answer"])
    if not output_check.allowed:
        raise HTTPException(status_code=500, detail=output_check.reason)

    if output_check.pii_found:
        PII_REDACTIONS.labels(stage="output").inc()
        logger.info(json.dumps({"event": "pii_redacted", "stage": "output", "entities": output_check.pii_found}))

    return ChatResponse(
        answer=output_check.cleaned_text,
        sources=result["sources"],
        pii_redacted=bool(input_check.pii_found or output_check.pii_found),
    )
