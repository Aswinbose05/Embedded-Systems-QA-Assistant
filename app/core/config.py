import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded from environment variables
    and .env files with sensible production defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Embedded Systems QA Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Groq & LLM Gateway
    GROQ_API_KEY: str = ""
    PRIMARY_MODEL: str = "openai/gpt-oss-120b"
    FALLBACK_MODEL: str = "openai/gpt-oss-20b"
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5
    REQUEST_TIMEOUT: float = 30.0
    TEMPERATURE: float = 0.0
    MAX_TOKENS: int = 500

    # Embeddings & ChromaDB
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_DIRECTORY: str = "./chroma_db"
    COLLECTION_NAME: str = "academic_qa"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 5

    # Storage
    UPLOAD_DIR: str = "./uploaded_documents"

    # LangSmith Observability
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_PROJECT: str = "embedded-systems-qa"
    LANGSMITH_TRACING: bool = False
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    # HuggingFace (Optional)
    HUGGINGFACE_API_KEY: Optional[str] = None

    def initialize_runtime_environment(self) -> None:
        """
        Synchronizes configuration with os.environ for third-party libraries
        (LangChain, LangSmith, Groq) and ensures working directories exist.
        """
        os.makedirs(self.CHROMA_DIRECTORY, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

        if self.GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = self.GROQ_API_KEY

        # LangSmith integration
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGSMITH_PROJECT"] = self.LANGSMITH_PROJECT
        else:
            # If not enabled or no key, disable LangSmith tracing to avoid connection warnings
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            os.environ["LANGSMITH_TRACING"] = "false"

        if self.HUGGINGFACE_API_KEY:
            os.environ["HUGGINGFACE_API_KEY"] = self.HUGGINGFACE_API_KEY


settings = Settings()
settings.initialize_runtime_environment()
