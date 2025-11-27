"""FastAPI application for Deep RAG API."""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import config
from .logging_config import setup_logging
from .service import get_service
from .models import HealthResponse, ErrorResponse
from .routers import query
from .exceptions import DeepRAGException

# Set up logging
setup_logging(log_level=config.LOG_LEVEL, log_format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    logger.info("Starting Deep RAG API server...")
    try:
        service = get_service()
        service.initialize()
        logger.info("Deep RAG service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Deep RAG service: {e}", exc_info=True)
        # Don't raise - let the health endpoint show the error
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deep RAG API server...")


# Create FastAPI app
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url=config.get_docs_url(),
    redoc_url=config.get_redoc_url(),
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=config.CORS_ALLOW_METHODS,
    allow_headers=config.CORS_ALLOW_HEADERS,
)

# Include routers
app.include_router(query.router, prefix=config.API_V1_PREFIX)


# Exception handlers
@app.exception_handler(DeepRAGException)
async def deep_rag_exception_handler(request: Request, exc: DeepRAGException):
    """Handle custom Deep RAG exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.message,
            error_type=type(exc).__name__,
            details=exc.details
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_type="HTTPException"
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            error_type="ValidationError",
            details={"errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_type=type(exc).__name__,
            details={"message": str(exc)}
        ).model_dump()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns the status of the API and Deep RAG service.
    """
    service = get_service()
    system_info = service.get_system_info()
    
    if service.is_ready():
        status_value = "healthy"
    elif system_info.get("error"):
        status_value = "unhealthy"
    else:
        status_value = "degraded"
    
    return HealthResponse(
        status=status_value,
        version=config.API_VERSION,
        system_info=system_info
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": config.API_TITLE,
        "version": config.API_VERSION,
        "description": config.API_DESCRIPTION,
        "docs_url": config.get_docs_url(),
        "health_url": "/health",
        "api_url": config.API_V1_PREFIX
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )

