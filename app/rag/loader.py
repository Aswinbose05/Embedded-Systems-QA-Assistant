import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)
from app.core.logging import get_logger
from app.core.exceptions import DocumentProcessingError

logger = get_logger("rag.loader")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


def load_file_content(file_path: str, original_filename: str) -> List[Document]:
    """
    Loads document content from PDF, DOCX, TXT, or CSV files into LangChain Documents.
    Preserves and enriches metadata (source filename and page numbers).
    """
    if not os.path.exists(file_path):
        raise DocumentProcessingError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file format '{ext}'. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext == ".csv":
            loader = CSVLoader(file_path)
        else:
            raise DocumentProcessingError(f"Unsupported file format: {ext}")

        documents = loader.load()

        # Normalize and enrich metadata
        for doc in documents:
            doc.metadata["source"] = original_filename
            # Normalize page numbers to 1-indexed integers when available
            if "page" in doc.metadata:
                try:
                    doc.metadata["page_number"] = int(doc.metadata["page"]) + 1
                except (ValueError, TypeError):
                    doc.metadata["page_number"] = doc.metadata.get("page", "N/A")
            else:
                doc.metadata["page_number"] = "N/A"

        logger.info("Successfully loaded %d pages/records from %s", len(documents), original_filename)
        return documents

    except Exception as exc:
        logger.error("Failed to load file '%s': %s", original_filename, str(exc))
        raise DocumentProcessingError(f"Error loading {original_filename}: {str(exc)}") from exc
