import time
from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse, SourceDocument
from app.guardrails.guardrail_service import guardrail_service
from app.router.intent_router import intent_router, QueryIntent
from app.rag.retriever import rag_retriever
from app.rag.pipeline import rag_pipeline
from app.gateway.llm_gateway import llm_gateway

logger = get_logger("service.rag")

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class RAGService:
    """
    High-level orchestrator executing the full query pipeline:
    User Query -> Input Guardrail -> Intent Router -> RAG Pipeline / Direct Response -> Output Guardrail -> Response.
    """

    @traceable(name="rag_service_query", run_type="chain")
    def handle_query(self, request: QueryRequest) -> QueryResponse:
        start_time = time.perf_counter()
        query = request.query.strip()
        top_k = request.top_k or settings.TOP_K

        # =========================================================
        # 1. INPUT GUARDRAILS
        # =========================================================
        input_check = guardrail_service.check_input(query)
        if not input_check.is_safe:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.warning("Query rejected by input guardrail (%s): %s", input_check.category, query)
            return QueryResponse(
                query=query,
                intent=QueryIntent.UNSAFE.value,
                answer=(
                    "I cannot fulfill this request as it violates safety guidelines or "
                    "contains unauthorized system override instructions."
                ),
                retrieved_sources=[],
                model_used="guardrails",
                is_fallback=False,
                confidence="unsafe_blocked",
                execution_time_ms=round(duration_ms, 2),
            )

        sanitized_query = input_check.sanitized_text

        # =========================================================
        # 2. CONVERSATIONAL INTENT ROUTER (BEFORE VECTOR SEARCH)
        # =========================================================
        intent, natural_reply = intent_router.route(sanitized_query)

        if intent != QueryIntent.RAG_QUERY and natural_reply is not None:
            # Generate a dynamic, contextual, friendly response adapted to the user's greeting/query
            try:
                conv_messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly, intelligent, and concise Embedded Systems QA Assistant. "
                            "Respond warmly, naturally, and politely (1 to 2 sentences) to the user's greeting, appreciation, or casual message. "
                            "Directly acknowledge what they said (e.g. if they say 'good morning', greet them with good morning; if they say 'thank you', say you're welcome) "
                            "and invite them to ask about their embedded systems documents. "
                            "Do not repeat robotic boilerplate text. Do not invent document facts."
                        ),
                    },
                    {"role": "user", "content": sanitized_query},
                ]
                conv_res = llm_gateway.generate(messages=conv_messages, temperature=0.7, max_tokens=100)
                dynamic_answer = conv_res.content.strip()
                conv_model = conv_res.model_used
            except Exception as exc:
                logger.warning("Dynamic greeting generation failed: %s. Using default reply.", str(exc))
                dynamic_answer = natural_reply
                conv_model = "intent_router"

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Conversational query routed directly without vector search (intent: %s)", intent.value)
            return QueryResponse(
                query=sanitized_query,
                intent=intent.value,
                answer=dynamic_answer,
                retrieved_sources=[],
                model_used=conv_model,
                is_fallback=False,
                confidence="conversational",
                execution_time_ms=round(duration_ms, 2),
            )

        # =========================================================
        # 3. RAG RETRIEVAL & VECTOR SEARCH
        # =========================================================
        selected_doc = getattr(request, "selected_document", None)
        session_id = getattr(request, "session_id", None)
        raw_docs, source_docs = rag_retriever.retrieve(
            sanitized_query, top_k=top_k, selected_document=selected_doc, session_id=session_id
        )

        # Handle empty knowledge base / zero retrieved chunks
        if not raw_docs:
            duration_ms = (time.perf_counter() - start_time) * 1000
            if session_id:
                from app.rag.vectorstore import vector_store_manager
                session_docs = vector_store_manager.get_indexed_documents(session_id=session_id)
                if not session_docs:
                    return QueryResponse(
                        query=sanitized_query,
                        intent=QueryIntent.RAG_QUERY.value,
                        answer="You haven't uploaded any documents to your session yet! Please upload a PDF, DOCX, TXT, or CSV file in the sidebar to begin asking questions.",
                        retrieved_sources=[],
                        model_used="system",
                        is_fallback=False,
                        confidence="no_documents_uploaded",
                        execution_time_ms=round(duration_ms, 2),
                    )

            return QueryResponse(
                query=sanitized_query,
                intent=QueryIntent.RAG_QUERY.value,
                answer="I couldn't find the answer in the uploaded documents.",
                retrieved_sources=[],
                model_used=settings.PRIMARY_MODEL,
                is_fallback=False,
                confidence="no_context",
                execution_time_ms=round(duration_ms, 2),
            )

        # =========================================================
        # 4. RAG GENERATION & OUTPUT GUARDRAILS
        # =========================================================
        final_answer, llm_resp, guard_result = rag_pipeline.run(
            question=sanitized_query,
            retrieved_documents=raw_docs,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        return QueryResponse(
            query=sanitized_query,
            intent=QueryIntent.RAG_QUERY.value,
            answer=final_answer,
            retrieved_sources=source_docs,
            model_used=llm_resp.model_used,
            is_fallback=llm_resp.is_fallback,
            confidence=guard_result.confidence,
            execution_time_ms=round(duration_ms, 2),
        )


rag_service = RAGService()
