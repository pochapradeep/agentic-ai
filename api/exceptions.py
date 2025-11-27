"""Custom exceptions for the API."""
from typing import Optional, Dict, Any


class DeepRAGException(Exception):
    """Base exception for Deep RAG API."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ServiceNotReadyException(DeepRAGException):
    """Raised when the service is not ready to handle requests."""
    def __init__(self, message: str = "Service is not ready", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=503, details=details)


class InvalidRequestException(DeepRAGException):
    """Raised when the request is invalid."""
    def __init__(self, message: str = "Invalid request", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, details=details)


class ProcessingException(DeepRAGException):
    """Raised when processing fails."""
    def __init__(self, message: str = "Processing failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class TimeoutException(DeepRAGException):
    """Raised when processing times out."""
    def __init__(self, message: str = "Processing timeout", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=504, details=details)

