import os
from typing import List, Optional
from fastapi import APIRouter, File, Form, Header, UploadFile, Query, HTTPException, status
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentProcessResponse,
    HealthResponse,
)
from app.services.document_service import document_service
from app.services.rag_service import rag_service
from app.rag.vectorstore import vector_store_manager

logger = get_logger("api.routes")

router = APIRouter()


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="Uploads a PDF, DOCX, TXT, or CSV file for knowledge base processing.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    session_id: Optional[str] = Form(None, description="Optional session ID for user-isolated documents"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    effective_session_id = session_id or x_session_id
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename must not be empty.",
        )

    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in {".pdf", ".docx", ".txt", ".csv"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: .pdf, .docx, .txt, .csv",
        )

    file_path, file_hash, file_size = document_service.save_uploaded_file(file, session_id=effective_session_id)

    return DocumentUploadResponse(
        filename=file.filename,
        document_id=file_hash,
        file_size_bytes=file_size,
        session_id=effective_session_id,
        status="uploaded",
        message=f"File '{file.filename}' uploaded successfully. Call /documents/process to index it into ChromaDB.",
    )


@router.post(
    "/documents/process",
    response_model=DocumentProcessResponse,
    summary="Process and index documents into ChromaDB",
    description="Splits and indexes uploaded documents into vector embeddings in ChromaDB.",
)
async def process_documents(
    filename: Optional[str] = Query(
        None,
        description="Optional specific filename to process. If omitted, all unindexed documents are processed.",
    ),
    session_id: Optional[str] = Query(None, description="Optional session ID for isolation"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    effective_session_id = session_id or x_session_id
    base_dir = os.path.join(settings.UPLOAD_DIR, effective_session_id) if effective_session_id else settings.UPLOAD_DIR

    if filename:
        file_path = os.path.join(base_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File '{filename}' not found in upload directory.",
            )

        doc_id, chunk_count = document_service.process_file(file_path, filename, session_id=effective_session_id)
        total_in_index = len(vector_store_manager.get_indexed_documents(session_id=effective_session_id))

        return DocumentProcessResponse(
            processed_documents=[filename],
            total_chunks_added=chunk_count,
            total_documents_in_index=total_in_index,
            session_id=effective_session_id,
            status="success",
            message=f"Document '{filename}' processed ({chunk_count} chunks indexed).",
        )

    # Process all files
    processed_files, total_chunks = document_service.process_all_uploaded_files(session_id=effective_session_id)
    total_in_index = len(vector_store_manager.get_indexed_documents(session_id=effective_session_id))

    return DocumentProcessResponse(
        processed_documents=processed_files,
        total_chunks_added=total_chunks,
        total_documents_in_index=total_in_index,
        session_id=effective_session_id,
        status="success",
        message=f"Processed {len(processed_files)} document(s) with {total_chunks} total chunks added.",
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the QA assistant",
    description="Submit questions. Automatically routed through intent classifier, guardrails, and RAG retrieval.",
)
async def query_knowledge_base(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    if not request.session_id and x_session_id:
        request.session_id = x_session_id
    return rag_service.handle_query(request)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
    description="Retrieves inventory of documents and chunk counts stored in ChromaDB (scoped by session).",
)
async def list_documents(
    session_id: Optional[str] = Query(None, description="Optional session ID to scope documents"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
):
    effective_session_id = session_id or x_session_id
    return document_service.list_documents(session_id=effective_session_id)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Checks the health of the API, vector database, and configured LLM models.",
)
async def health_check():
    chroma_ok = False
    total_chunks = 0
    try:
        total_chunks = vector_store_manager.get_total_chunks()
        chroma_ok = True
    except Exception as exc:
        logger.error("Health check ChromaDB probe failed: %s", str(exc))

    return HealthResponse(
        status="healthy" if chroma_ok else "degraded",
        chroma_connected=chroma_ok,
        total_chunks=total_chunks,
        primary_model=settings.PRIMARY_MODEL,
        fallback_model=settings.FALLBACK_MODEL,
        langsmith_tracing=settings.LANGSMITH_TRACING,
        version=settings.APP_VERSION,
    )
