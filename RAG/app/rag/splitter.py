from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("rag.splitter")


def split_documents(
    documents: List[Document],
    document_id: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits documents into overlapping chunks and enriches metadata with chunk IDs.
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

    logger.info("Split %d documents into %d chunks (chunk_size=%d, overlap=%d)", len(documents), len(chunks), chunk_size, chunk_overlap)
    return chunks
