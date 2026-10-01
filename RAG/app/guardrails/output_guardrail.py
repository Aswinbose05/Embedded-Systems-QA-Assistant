import re
from dataclasses import dataclass
from typing import List, Optional
from app.core.logging import get_logger

logger = get_logger("guardrails.output")

UNAVAILABLE_PHRASE = "I couldn't find the answer in the uploaded documents."


@dataclass
class OutputGuardrailResult:
    is_safe: bool
    sanitized_text: str
    is_grounded: bool
    confidence: str = "grounded"
    category: Optional[str] = None
    reason: Optional[str] = None


class OutputGuardrail:
    """
    Validates LLM outputs before delivering them to users.
    - Masks/blocks leaked API keys, tokens, and internal stack traces.
    - Validates context grounding: confirms the model did not invent data when context is absent.
    - Returns standardized safe responses.
    """

    LEAK_PATTERNS = [
        re.compile(r"gsk_[a-zA-Z0-9_-]{20,}"),
        re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
        re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
        re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE),
        re.compile(r"File \".*?\.py\", line \d+", re.IGNORECASE),
    ]

    UNAVAILABLE_INDICATORS = [
        "couldn't find the answer in the uploaded documents",
        "could not find the answer in the uploaded documents",
        "not mentioned in the provided documents",
        "not found in the uploaded documents",
        "provided context does not contain",
        "does not provide information",
        "not present in the context",
    ]

    def validate(
        self,
        output_text: str,
        context_chunks: Optional[List[str]] = None,
        is_rag_query: bool = True,
    ) -> OutputGuardrailResult:
        if not output_text:
            return OutputGuardrailResult(
                is_safe=False,
                sanitized_text=UNAVAILABLE_PHRASE,
                is_grounded=False,
                confidence="unsupported",
                category="empty_response",
                reason="LLM returned an empty response.",
            )

        sanitized = output_text

        # 1. Leak Detection
        for pattern in self.LEAK_PATTERNS:
            if pattern.search(sanitized):
                logger.warning("Sensitive credential or traceback detected in output. Redacting.")
                sanitized = pattern.sub("[REDACTED_SECURITY_DATA]", sanitized)

        # 2. Grounding check for RAG queries
        lower_output = sanitized.lower()
        has_unavailable_phrase = any(phrase in lower_output for phrase in self.UNAVAILABLE_INDICATORS)

        if is_rag_query:
            # If no context was retrieved at all
            if not context_chunks or len(context_chunks) == 0:
                logger.info("No context was retrieved; enforcing grounded unavailability phrase.")
                return OutputGuardrailResult(
                    is_safe=True,
                    sanitized_text=UNAVAILABLE_PHRASE,
                    is_grounded=True,
                    confidence="no_context",
                    reason="No document context was retrieved.",
                )

            # If model acknowledged the answer is not in documents
            if has_unavailable_phrase:
                return OutputGuardrailResult(
                    is_safe=True,
                    sanitized_text=sanitized,
                    is_grounded=True,
                    confidence="not_in_context",
                    reason="Model correctly indicated information is not in the uploaded documents.",
                )

        return OutputGuardrailResult(
            is_safe=True,
            sanitized_text=sanitized,
            is_grounded=True,
            confidence="grounded",
        )


output_guardrail = OutputGuardrail()
