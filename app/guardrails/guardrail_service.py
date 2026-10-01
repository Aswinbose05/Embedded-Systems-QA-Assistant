from typing import List, Optional
from app.guardrails.input_guardrail import input_guardrail, InputGuardrailResult
from app.guardrails.output_guardrail import output_guardrail, OutputGuardrailResult


class GuardrailService:
    """
    Unified AI Guardrails service orchestrating both input and output defenses.
    """

    def __init__(self):
        self.input_guardrail = input_guardrail
        self.output_guardrail = output_guardrail

    def check_input(self, text: str) -> InputGuardrailResult:
        """Validates incoming prompt/query."""
        return self.input_guardrail.validate(text)

    def check_output(
        self,
        output_text: str,
        context_chunks: Optional[List[str]] = None,
        is_rag_query: bool = True,
    ) -> OutputGuardrailResult:
        """Validates generated LLM response."""
        return self.output_guardrail.validate(
            output_text=output_text,
            context_chunks=context_chunks,
            is_rag_query=is_rag_query,
        )


guardrail_service = GuardrailService()
