from typing import List, Optional, Union
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query or question")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")
    selected_document: Optional[str] = Field(default=None, description="Optional filename to restrict search to a specific document")
    session_id: Optional[str] = Field(default=None, description="Unique session ID to isolate queries to user-specific documents")


class SourceDocument(BaseModel):
    document_name: str = Field(..., description="Filename of the source document")
    page_number: Union[int, str] = Field(default="N/A", description="Page number where the content appeared")
    relevance_score: Optional[float] = Field(default=None, description="Similarity or distance score from vector store")
    snippet: Optional[str] = Field(default=None, description="Excerpt of the retrieved chunk content")


class QueryResponse(BaseModel):
    query: str
    intent: str = Field(..., description="Detected intent: greeting, casual, appreciation, farewell, rag_query, unsafe")
    answer: str = Field(..., description="Generated answer or conversational response")
    retrieved_sources: List[SourceDocument] = Field(default_factory=list, description="Retrieved document sources")
    model_used: str = Field(..., description="LLM model name that generated the response")
    is_fallback: bool = Field(default=False, description="Whether fallback model was activated")
    confidence: Optional[str] = Field(default="grounded", description="Confidence or grounding assessment")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")


class DocumentInfo(BaseModel):
    document_name: str
    document_id: str
    chunk_count: int
    source_path: Optional[str] = None
    session_id: Optional[str] = None


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]
    session_id: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    filename: str
    document_id: str
    file_size_bytes: int
    session_id: Optional[str] = None
    status: str = "uploaded"
    message: str


class DocumentProcessResponse(BaseModel):
    processed_documents: List[str]
    total_chunks_added: int
    total_documents_in_index: int
    session_id: Optional[str] = None
    status: str = "success"
    message: str


class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    total_chunks: int
    primary_model: str
    fallback_model: str
    langsmith_tracing: bool
    version: str
