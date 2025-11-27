"""Configuration for the API server."""
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


class APIConfig:
    """API server configuration."""
    
    # Server settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    API_TITLE: str = "Deep RAG API"
    API_DESCRIPTION: str = "Agentic AI system for complex question answering with multi-step reasoning"
    API_VERSION: str = "0.1.0"
    
    # CORS settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    CORS_ALLOW_METHODS: list = ["GET", "POST", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Request settings
    MAX_REQUEST_SIZE: int = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json or text
    
    # Azure settings
    AZURE_ENVIRONMENT: bool = os.getenv("WEBSITE_SITE_NAME") is not None or os.getenv("CONTAINER_APP_NAME") is not None
    
    @classmethod
    def get_docs_url(cls) -> Optional[str]:
        """Get docs URL based on environment."""
        if cls.AZURE_ENVIRONMENT:
            # In Azure, docs might be disabled or require authentication
            return None
        return "/docs"
    
    @classmethod
    def get_redoc_url(cls) -> Optional[str]:
        """Get ReDoc URL based on environment."""
        if cls.AZURE_ENVIRONMENT:
            return None
        return "/redoc"


# Global config instance
config = APIConfig()

