import os
import shutil
from typing import Dict, List, Tuple
from fastapi import UploadFile
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import DocumentProcessingError
from app.utils.hash_utils import get_bytes_hash, get_file_hash
from app.rag.loader import load_file_content
from app.rag.splitter import split_documents
from app.rag.vectorstore import vector_store_manager
from app.models.schemas import DocumentInfo, DocumentListResponse

logger = get_logger("service.document")

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class DocumentService:
    """
    Manages document upload, storage, indexing, and inventory.
    """

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploaded_file(self, upload_file: UploadFile) -> Tuple[str, str, int]:
        """
        Saves incoming multipart file to upload directory.
        Returns: (file_path, file_hash, file_size)
        """
        filename = os.path.basename(upload_file.filename or "unknown_file")
        destination_path = os.path.join(self.upload_dir, filename)

        try:
            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            file_size = os.path.getsize(destination_path)
            file_hash = get_file_hash(destination_path)
            logger.info("Saved upload: %s (size: %d bytes, hash: %s)", filename, file_size, file_hash)
            return destination_path, file_hash, file_size

        except Exception as exc:
            logger.error("Failed to save uploaded file %s: %s", filename, str(exc))
            raise DocumentProcessingError(f"Could not save file {filename}: {str(exc)}") from exc

    @traceable(name="document_processing_pipeline", run_type="parser")
    def process_file(self, file_path: str, original_filename: str) -> Tuple[str, int]:
        """
        Parses, splits, and indexes a file into ChromaDB.
        Avoids redundant re-indexing if hash already exists in Chroma.
        Returns: (document_id, chunk_count)
        """
        document_id = get_file_hash(file_path)

        # Check if document already indexed
        indexed_docs = vector_store_manager.get_indexed_documents()
        if document_id in indexed_docs:
            existing_count = indexed_docs[document_id]["chunk_count"]
            logger.info("Document %s (%s) already indexed (%d chunks). Skipping duplicate index.", original_filename, document_id, existing_count)
            return document_id, existing_count

        # 1. Load document
        documents = load_file_content(file_path, original_filename)
        if not documents:
            logger.warning("No text content could be extracted from %s", original_filename)
            return document_id, 0

        # 2. Split document
        chunks = split_documents(documents, document_id=document_id)

        # 3. Create unique deterministic IDs matching existing pattern
        ids = [f"{document_id}_{i}" for i in range(len(chunks))]

        # 4. Add to vector store
        vector_store_manager.add_documents(chunks, ids)
        logger.info("Successfully indexed %d chunks for document: %s", len(chunks), original_filename)
        return document_id, len(chunks)

    def process_all_uploaded_files(self) -> Tuple[List[str], int]:
        """
        Scans uploaded_documents directory and processes any files not yet indexed.
        """
        processed_files = []
        total_chunks = 0

        if not os.path.exists(self.upload_dir):
            return processed_files, total_chunks

        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in {".pdf", ".docx", ".txt", ".csv"}:
                    try:
                        doc_id, chunks_added = self.process_file(file_path, filename)
                        processed_files.append(filename)
                        total_chunks += chunks_added
                    except Exception as exc:
                        logger.error("Error processing %s during batch process: %s", filename, str(exc))

        return processed_files, total_chunks

    def list_documents(self) -> DocumentListResponse:
        """
        Returns all indexed documents and summary metrics.
        """
        indexed_map = vector_store_manager.get_indexed_documents()
        total_chunks = vector_store_manager.get_total_chunks()

        docs_list: List[DocumentInfo] = []
        for doc_id, info in indexed_map.items():
            source_path = os.path.join(self.upload_dir, info["document_name"])
            docs_list.append(
                DocumentInfo(
                    document_name=info["document_name"],
                    document_id=doc_id,
                    chunk_count=info["chunk_count"],
                    source_path=source_path if os.path.exists(source_path) else None,
                )
            )

        return DocumentListResponse(
            total_documents=len(docs_list),
            total_chunks=total_chunks,
            documents=docs_list,
        )


document_service = DocumentService()
