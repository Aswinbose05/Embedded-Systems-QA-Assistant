import os
import shutil
from typing import Dict, List, Optional, Tuple
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

    def save_uploaded_file(self, upload_file: UploadFile, session_id: Optional[str] = None) -> Tuple[str, str, int]:
        """
        Saves incoming multipart file to upload directory (scoped by session_id if provided).
        Returns: (file_path, file_hash, file_size)
        """
        filename = os.path.basename(upload_file.filename or "unknown_file")
        target_dir = os.path.join(self.upload_dir, session_id) if session_id else self.upload_dir
        os.makedirs(target_dir, exist_ok=True)
        destination_path = os.path.join(target_dir, filename)

        try:
            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            file_size = os.path.getsize(destination_path)
            file_hash = get_file_hash(destination_path)
            logger.info("Saved upload: %s (size: %d bytes, hash: %s, session: %s)", filename, file_size, file_hash, session_id)
            return destination_path, file_hash, file_size

        except Exception as exc:
            logger.error("Failed to save uploaded file %s: %s", filename, str(exc))
            raise DocumentProcessingError(f"Could not save file {filename}: {str(exc)}") from exc

    @traceable(name="document_processing_pipeline", run_type="parser")
    def process_file(self, file_path: str, original_filename: str, session_id: Optional[str] = None) -> Tuple[str, int]:
        """
        Parses, splits, and indexes a file into ChromaDB, tagging chunks with session_id.
        Avoids redundant re-indexing if hash already exists in Chroma for this session.
        Returns: (document_id, chunk_count)
        """
        document_id = get_file_hash(file_path)

        # Check if document already indexed for this session
        indexed_docs = vector_store_manager.get_indexed_documents(session_id=session_id)
        if document_id in indexed_docs:
            existing_count = indexed_docs[document_id]["chunk_count"]
            logger.info("Document %s (%s) already indexed (%d chunks) for session %s. Skipping duplicate index.", original_filename, document_id, existing_count, session_id)
            return document_id, existing_count

        # 1. Load document
        documents = load_file_content(file_path, original_filename)
        if not documents:
            logger.warning("No text content could be extracted from %s", original_filename)
            return document_id, 0

        # 2. Split document with session_id
        chunks = split_documents(documents, document_id=document_id, session_id=session_id)

        # 3. Create unique deterministic IDs
        prefix = f"{session_id}_" if session_id else ""
        ids = [f"{prefix}{document_id}_{i}" for i in range(len(chunks))]

        # 4. Add to vector store
        vector_store_manager.add_documents(chunks, ids)
        logger.info("Successfully indexed %d chunks for document: %s (session: %s)", len(chunks), original_filename, session_id)
        return document_id, len(chunks)

    def process_all_uploaded_files(self, session_id: Optional[str] = None) -> Tuple[List[str], int]:
        """
        Scans uploaded_documents directory (or session subdirectory) and processes any unindexed files.
        """
        processed_files = []
        total_chunks = 0
        search_dir = os.path.join(self.upload_dir, session_id) if session_id else self.upload_dir

        if not os.path.exists(search_dir):
            return processed_files, total_chunks

        for filename in os.listdir(search_dir):
            file_path = os.path.join(search_dir, filename)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(filename)
                if ext.lower() in {".pdf", ".docx", ".txt", ".csv"}:
                    try:
                        doc_id, chunks_added = self.process_file(file_path, filename, session_id=session_id)
                        processed_files.append(filename)
                        total_chunks += chunks_added
                    except Exception as exc:
                        logger.error("Error processing %s during batch process: %s", filename, str(exc))

        return processed_files, total_chunks

    def list_documents(self, session_id: Optional[str] = None) -> DocumentListResponse:
        """
        Returns indexed documents and summary metrics filtered by session_id.
        """
        indexed_map = vector_store_manager.get_indexed_documents(session_id=session_id)
        total_chunks = vector_store_manager.get_total_chunks(session_id=session_id)

        docs_list: List[DocumentInfo] = []
        base_dir = os.path.join(self.upload_dir, session_id) if session_id else self.upload_dir
        for doc_id, info in indexed_map.items():
            source_path = os.path.join(base_dir, info["document_name"])
            docs_list.append(
                DocumentInfo(
                    document_name=info["document_name"],
                    document_id=doc_id,
                    chunk_count=info["chunk_count"],
                    source_path=source_path if os.path.exists(source_path) else None,
                    session_id=session_id,
                )
            )

        return DocumentListResponse(
            total_documents=len(docs_list),
            total_chunks=total_chunks,
            documents=docs_list,
            session_id=session_id,
        )


document_service = DocumentService()
