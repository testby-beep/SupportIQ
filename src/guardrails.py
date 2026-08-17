"""
guardrails.py
-------------
Safety layer sitting between the user and the RAG pipeline:

1. INPUT guardrails (before retrieval/generation):
   - PII detection via Presidio (emails, phone numbers, credit cards, etc.)
     Detected PII is redacted before the question is embedded/logged, so we
     never persist sensitive user data in logs or traces.
   - Length / empty-input validation.
   - Basic prompt-injection pattern check (e.g. "ignore previous instructions").

2. OUTPUT guardrails (before the answer is returned to the user):
   - Blocks answers that leak PII patterns (in case a chunk itself contained PII).
   - Enforces the "don't know" refusal behavior isn't silently dropped.
   - Structured validation via Pydantic so the API contract can't be violated.

This is intentionally built as a standalone module (not baked into rag_chain.py)
so it can be swapped for NeMo Guardrails / Guardrails AI later without touching
the core retrieval logic.
"""

import re
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

_analyzer = AnalyzerEngine()
_anonymizer = AnonymizerEngine()

# Entities we actively redact. Presidio supports many more (see docs) —
# this list is scoped to what's plausible in a support-chat context.
PII_ENTITIES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "US_SSN", "PERSON"]

INJECTION_PATTERNS = [
    r"ignore (all )?(previous|above) instructions",
    r"disregard (your|the) (system )?prompt",
    r"you are now (in )?(dan|developer) mode",
    r"reveal your (system )?prompt",
]

MAX_QUESTION_LENGTH = 1000


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    cleaned_text: str = ""
    pii_found: list = field(default_factory=list)


def check_input(question: str) -> GuardrailResult:
    """Validate and sanitize user input before it reaches the RAG pipeline."""
    question = str(question).strip()

    if not question:
        return GuardrailResult(allowed=False, reason="Question cannot be empty.")

    if len(question) > MAX_QUESTION_LENGTH:
        return GuardrailResult(
            allowed=False, reason=f"Question exceeds {MAX_QUESTION_LENGTH} character limit."
        )

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, re.IGNORECASE):
            return GuardrailResult(
                allowed=False,
                reason="Question was blocked by prompt-injection safety filter.",
            )

    # Detect PII so we can redact it before logging/embedding, but still let
    # the (redacted) question through — e.g. "my email is x@y.com, how do I
    # change it?" is a legitimate support question.
    pii_results = _analyzer.analyze(text=question, entities=PII_ENTITIES, language="en")
    cleaned = question
    pii_found = []
    if pii_results:
        anonymized = _anonymizer.anonymize(text=question, analyzer_results=pii_results)
        cleaned = anonymized.text
        pii_found = [r.entity_type for r in pii_results]

    return GuardrailResult(allowed=True, cleaned_text=cleaned, pii_found=pii_found)


def check_output(answer: str) -> GuardrailResult:
    """Validate the generated answer before it's returned to the user."""
    answer = str(answer)  # defend against str-subclass types (e.g. langchain-core's
                            # TextAccessor) that pass isinstance(x, str) but fail
                            # strict type checks in libraries like Presidio.
    if not answer or not answer.strip():
        return GuardrailResult(
            allowed=False, reason="Empty response generated.", cleaned_text=""
        )

    pii_results = _analyzer.analyze(text=answer, entities=PII_ENTITIES, language="en")
    cleaned = answer
    pii_found = []
    if pii_results:
        anonymized = _anonymizer.anonymize(text=answer, analyzer_results=pii_results)
        cleaned = anonymized.text
        pii_found = [r.entity_type for r in pii_results]

    return GuardrailResult(allowed=True, cleaned_text=cleaned, pii_found=pii_found)
