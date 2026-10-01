from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from chromadb.utils import embedding_functions
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import RetrieverError

logger = get_logger("rag.vectorstore")


class LowMemoryEmbeddings(Embeddings):
    """
    Lightweight ONNX-powered all-MiniLM-L6-v2 embeddings.
    Consumes ~40MB RAM (vs 550MB for PyTorch), preventing Out-Of-Memory (OOM)
    errors on memory-constrained servers like Render Free Tier (512MB limit).
    Produces identical 384-dimensional vectors fully compatible with existing collections.
    """
    def __init__(self):
        self._ef = embedding_functions.ONNXMiniLM_L6_V2()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]


class VectorStoreManager:
    """
    Singleton manager for ChromaDB vector store and embeddings.
    Preserves existing persistence format, collection name, and embeddings.
    """

    def __init__(self):
        self._embeddings: Optional[Embeddings] = None
        self._vectordb: Optional[Chroma] = None

    def get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            try:
                logger.info("Initializing Low-Memory ONNX embeddings (all-MiniLM-L6-v2)...")
                self._embeddings = LowMemoryEmbeddings()
                logger.info("Low-Memory ONNX embeddings active (~40MB RAM footprint).")
            except Exception as exc:
                logger.warning("ONNX initialization failed: %s. Falling back to HuggingFaceEmbeddings.", exc)
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL
                )
        return self._embeddings

    def get_vector_store(self) -> Chroma:
        if self._vectordb is None:
            logger.info(
                "Connecting to ChromaDB collection '%s' at '%s'",
                settings.COLLECTION_NAME,
                settings.CHROMA_DIRECTORY,
            )
            embeddings = self.get_embeddings()
            self._vectordb = Chroma(
                collection_name=settings.COLLECTION_NAME,
                persist_directory=settings.CHROMA_DIRECTORY,
                embedding_function=embeddings,
            )
        return self._vectordb

    def add_documents(self, chunks: List[Document], ids: List[str]) -> int:
        """Adds chunks to ChromaDB."""
        if not chunks:
            return 0
        try:
            store = self.get_vector_store()
            store.add_documents(documents=chunks, ids=ids)
            logger.info("Added %d chunks to ChromaDB", len(chunks))
            return len(chunks)
        except Exception as exc:
            logger.error("Failed to add documents to ChromaDB: %s", str(exc))
            raise RetrieverError(f"ChromaDB insert failed: {str(exc)}") from exc

    def similarity_search_with_score(
        self, query: str, k: int = 5, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Performs similarity search with relevance scores and optional metadata filter."""
        try:
            store = self.get_vector_store()
            if filter:
                return store.similarity_search_with_score(query, k=k, filter=filter)
            return store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error("Vector search failed for query '%s' (filter=%s): %s", query, filter, str(exc))
            raise RetrieverError(f"Vector search failed: {str(exc)}") from exc

    def get_total_chunks(self) -> int:
        """Returns total number of chunks currently indexed."""
        try:
            store = self.get_vector_store()
            data = store.get()
            return len(data.get("ids", []))
        except Exception as exc:
            logger.warning("Could not read chunk count from ChromaDB: %s", str(exc))
            return 0

    def get_indexed_documents(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans all documents in the vector store and aggregates unique document metadata.
        Returns mapping of document_id -> {document_name, chunk_count, pages, ...}
        """
        try:
            store = self.get_vector_store()
            data = store.get()
            docs: Dict[str, Dict[str, Any]] = {}
            for meta in data.get("metadatas", []):
                if not meta:
                    continue
                doc_id = meta.get("document_id", "unknown")
                source = meta.get("source", "Unknown")
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_id": doc_id,
                        "document_name": source,
                        "chunk_count": 0,
                    }
                docs[doc_id]["chunk_count"] += 1
            return docs
        except Exception as exc:
            logger.error("Failed to inspect indexed documents: %s", str(exc))
            return {}


vector_store_manager = VectorStoreManager()
