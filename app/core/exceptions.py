from typing import Any, Dict, Optional


class AppException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentProcessingError(AppException):
    """Raised when document ingestion or parsing fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class LLMGatewayError(AppException):
    """Raised when LLM calls fail across all gateway retries and fallbacks."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=502, details=details)


class GuardrailViolationError(AppException):
    """Raised when user input violates safety or prompt-injection guardrails."""
    def __init__(self, message: str, category: str = "safety_violation", details: Optional[Dict[str, Any]] = None):
        d = details or {}
        d["category"] = category
        super().__init__(message, status_code=400, details=d)


class RetrieverError(AppException):
    """Raised when vector database retrieval fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ConfigError(AppException):
    """Raised when application configuration is invalid or missing required keys."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)
