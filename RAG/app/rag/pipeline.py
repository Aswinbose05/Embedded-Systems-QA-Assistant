from typing import List, Tuple
from langchain_core.documents import Document
from app.gateway.llm_gateway import llm_gateway, LLMResponse
from app.guardrails.guardrail_service import guardrail_service, OutputGuardrailResult
from app.rag.retriever import rag_retriever
from app.core.logging import get_logger

logger = get_logger("rag.pipeline")

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


RAG_SYSTEM_PROMPT = (
    "You are an intelligent document question-answering assistant specializing in Embedded Systems. "
    "Answer only from the supplied document context."
)


def format_rag_user_prompt(context: str, question: str) -> str:
    """Constructs prompt matching the original application rules."""
    return f"""You are an intelligent document question-answering assistant.

Your job is to answer the user's question using the provided document context.

IMPORTANT RULES:
1. Read the entire provided context carefully.
2. Answer using information from the context.
3. If the answer is clearly present, explain it naturally.
4. You may combine information from multiple retrieved chunks.
5. Do not say "I don't know" if the answer can reasonably be found in the context.
6. Do not use outside knowledge.
7. If the answer genuinely does not exist in the context, say:
   "I couldn't find the answer in the uploaded documents."
8. Give a clear and useful answer.
9. Use simple language.
10. If appropriate, use bullet points or examples.

DOCUMENT CONTEXT:
=================

{context}

=================

USER QUESTION:

{question}

ANSWER:
"""


class RAGPipeline:
    """
    Coordinates RAG prompt synthesis, LLM Gateway invocation, and Output Guardrails.
    """

    @traceable(name="rag_pipeline_execution", run_type="chain")
    def run(
        self,
        question: str,
        retrieved_documents: List[Document],
    ) -> Tuple[str, LLMResponse, OutputGuardrailResult]:
        """
        Executes generation through the LLM Gateway and validates output via Guardrails.
        """
        context_str = rag_retriever.build_context(retrieved_documents)
        context_chunks = [d.page_content for d in retrieved_documents]

        prompt = format_rag_user_prompt(context_str, question)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Call LLM Gateway
        llm_response = llm_gateway.generate(messages=messages, temperature=0.0)

        # Apply Output Guardrail
        guardrail_result = guardrail_service.check_output(
            output_text=llm_response.content,
            context_chunks=context_chunks,
            is_rag_query=True,
        )

        final_answer = guardrail_result.sanitized_text
        return final_answer, llm_response, guardrail_result


rag_pipeline = RAGPipeline()
