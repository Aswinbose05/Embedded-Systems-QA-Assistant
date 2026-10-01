from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("rag.splitter")


def split_documents(
    documents: List[Document],
    document_id: str,
    session_id: Optional[str] = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits documents into overlapping chunks and enriches metadata with chunk IDs and session ID.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        chunk.metadata["document_id"] = document_id
        chunk.metadata["chunk_id"] = idx
        if session_id:
            chunk.metadata["session_id"] = session_id

    logger.info("Split %d documents into %d chunks (chunk_size=%d, overlap=%d, session=%s)", len(documents), len(chunks), chunk_size, chunk_overlap, session_id)
    return chunks
