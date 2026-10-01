import re
from dataclasses import dataclass
from typing import Optional
from app.core.logging import get_logger

logger = get_logger("guardrails.input")


@dataclass
class InputGuardrailResult:
    is_safe: bool
    sanitized_text: str
    category: Optional[str] = None
    reason: Optional[str] = None


class InputGuardrail:
    """
    Validates and sanitizes incoming user queries before routing or RAG processing.
    Detects prompt injection, malicious commands, and toxic instructions.
    """

    # Prompt injection patterns
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?(system\s+prompt|initial\s+prompt|instructions)", re.IGNORECASE),
        re.compile(r"what\s+is\s+your\s+system\s+prompt", re.IGNORECASE),
        re.compile(r"print\s+(your\s+)?(system\s+prompt|instructions)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
        re.compile(r"act\s+as\s+DAN\b", re.IGNORECASE),
        re.compile(r"bypass\s+all\s+(filters|rules|safety)", re.IGNORECASE),
        re.compile(r"override\s+system\s+(prompt|rules|instructions)", re.IGNORECASE),
        re.compile(r"show\s+me\s+the\s+hidden\s+prompt", re.IGNORECASE),
        re.compile(r"repeat\s+everything\s+above", re.IGNORECASE),
    ]

    # Malicious commands / code execution patterns
    MALICIOUS_PATTERNS = [
        re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
        re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
        re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
        re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
        re.compile(r"cat\s+/etc/(passwd|shadow)", re.IGNORECASE),
        re.compile(r"powershell\s+-enc\b", re.IGNORECASE),
        re.compile(r"/bin/(ba)?sh", re.IGNORECASE),
        re.compile(r"curl\s+.*?\|\s*sh", re.IGNORECASE),
    ]

    def validate(self, text: str) -> InputGuardrailResult:
        """
        Runs comprehensive input safety checks.
        """
        if not text or not text.strip():
            return InputGuardrailResult(
                is_safe=False,
                sanitized_text="",
                category="empty_input",
                reason="Query cannot be empty.",
            )

        cleaned = text.strip()

        # Check prompt injection
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(cleaned):
                logger.warning("Prompt injection detected matching pattern: %s", pattern.pattern)
                return InputGuardrailResult(
                    is_safe=False,
                    sanitized_text=cleaned,
                    category="prompt_injection",
                    reason="Request contains unauthorized system override or injection instructions.",
                )

        # Check malicious commands
        for pattern in self.MALICIOUS_PATTERNS:
            if pattern.search(cleaned):
                logger.warning("Malicious pattern detected matching: %s", pattern.pattern)
                return InputGuardrailResult(
                    is_safe=False,
                    sanitized_text=cleaned,
                    category="malicious_instruction",
                    reason="Request contains unsafe system commands or exploit patterns.",
                )

        return InputGuardrailResult(
            is_safe=True,
            sanitized_text=cleaned,
        )


input_guardrail = InputGuardrail()
