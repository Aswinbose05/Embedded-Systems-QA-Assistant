from typing import List, Optional, Tuple
from langchain_core.documents import Document
from app.models.schemas import SourceDocument
from app.rag.vectorstore import vector_store_manager
from app.core.logging import get_logger

logger = get_logger("rag.retriever")

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class RAGRetriever:
    """
    Retrieves context chunks from ChromaDB and formats them for prompting and user sources.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    @traceable(name="rag_retrieval", run_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        selected_document: Optional[str] = None,
    ) -> Tuple[List[Document], List[SourceDocument]]:
        """
        Executes vector search and returns both LangChain Documents and formatted SourceDocuments.
        If selected_document is specified, results are filtered to that document only.
        """
        k = top_k or self.top_k
        doc_filter = None
        if selected_document and selected_document.strip() and selected_document.strip().lower() != "all":
            doc_filter = {"source": selected_document.strip()}
            logger.info("Restricting RAG retrieval to document: '%s'", selected_document)

        results = vector_store_manager.similarity_search_with_score(query, k=k, filter=doc_filter)

        raw_docs: List[Document] = []
        source_docs: List[SourceDocument] = []
        seen_keys = set()

        for doc, score in results:
            raw_docs.append(doc)

            source_name = doc.metadata.get("source", "Unknown")
            page_val = doc.metadata.get("page_number")
            if page_val is None:
                raw_page = doc.metadata.get("page")
                page_val = int(raw_page) + 1 if raw_page is not None else "N/A"

            # De-duplicate identical source+page combos in final display sources
            key = (source_name, page_val)
            snippet = doc.page_content[:200].replace("\n", " ").strip()
            if len(doc.page_content) > 200:
                snippet += "..."

            source_docs.append(
                SourceDocument(
                    document_name=source_name,
                    page_number=page_val,
                    relevance_score=round(float(score), 4) if score is not None else None,
                    snippet=snippet,
                )
            )

        logger.info("Retrieved %d documents for query: %s", len(raw_docs), query)
        return raw_docs, source_docs

    @staticmethod
    def build_context(documents: List[Document]) -> str:
        """
        Builds the formatted context block matching the original RAG prompt specification.
        """
        context_parts = []
        for index, document in enumerate(documents):
            source = document.metadata.get("source", "Unknown")
            page_val = document.metadata.get("page_number")
            if page_val is None:
                raw_page = document.metadata.get("page")
                page_val = int(raw_page) + 1 if raw_page is not None else "N/A"

            context_parts.append(
                f"DOCUMENT {index + 1}\n\n"
                f"Source: {source}\n\n"
                f"Page: {page_val}\n\n"
                f"Content:\n{document.page_content}"
            )
        return "\n\n".join(context_parts)


rag_retriever = RAGRetriever()
