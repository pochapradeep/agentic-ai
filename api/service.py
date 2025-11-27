"""Service wrapper for Deep RAG system."""
import os
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Fix OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.config import get_config, ensure_directories
from src.embedding_pipeline import load_or_generate_embeddings
from src.embeddings import create_embedding_function
from src.document_loader import load_documents_from_data_folder
from src.utils import process_documents_with_metadata
from src.deep_rag import DeepRAGSystem
from .exceptions import ServiceNotReadyException

logger = logging.getLogger(__name__)


class DeepRAGService:
    """Service class to manage Deep RAG system lifecycle."""
    
    def __init__(self):
        """Initialize the service."""
        self._deep_rag: Optional[DeepRAGSystem] = None
        self._config: Optional[Dict[str, Any]] = None
        self._initialized: bool = False
        self._initialization_error: Optional[str] = None
    
    def initialize(self) -> None:
        """Initialize the Deep RAG system."""
        if self._initialized:
            logger.info("Deep RAG service already initialized")
            return
        
        try:
            logger.info("Initializing Deep RAG service...")
            start_time = time.time()
            
            # Get configuration
            self._config = get_config()
            ensure_directories(self._config)
            
            # Load or generate embeddings
            logger.info("Loading embeddings...")
            vector_store = load_or_generate_embeddings(
                self._config, 
                force_regenerate=False
            )
            
            # Load documents
            logger.info("Loading documents...")
            documents = load_documents_from_data_folder(self._config["data_dir"])
            doc_chunks = process_documents_with_metadata(
                documents,
                chunk_size=self._config.get("chunk_size", 1000),
                chunk_overlap=self._config.get("chunk_overlap", 150)
            )
            logger.info(f"Loaded {len(doc_chunks)} document chunks")
            
            # Create embedding function
            logger.info("Creating embedding function...")
            embedding_function = create_embedding_function(self._config)
            
            # Create Deep RAG system
            logger.info("Initializing Deep RAG system...")
            self._deep_rag = DeepRAGSystem(
                config=self._config,
                vector_store=vector_store,
                documents=doc_chunks,
                embedding_function=embedding_function
            )
            self._deep_rag.compile()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Deep RAG service initialized successfully in {elapsed_time:.2f}s")
            
            self._initialized = True
            self._initialization_error = None
            
        except Exception as e:
            error_msg = f"Failed to initialize Deep RAG service: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._initialization_error = error_msg
            self._initialized = False
            raise ServiceNotReadyException(error_msg)
    
    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self._initialized and self._deep_rag is not None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "initialized": self._initialized,
            "embeddings_loaded": self._initialized,
            "vector_store_ready": self._initialized,
            "error": self._initialization_error
        }
    
    def answer(self, question: str, max_steps: Optional[int] = None) -> str:
        """
        Answer a question using the Deep RAG system.
        
        Args:
            question: The question to answer
            max_steps: Optional maximum reasoning steps
            
        Returns:
            The answer string
            
        Raises:
            ServiceNotReadyException: If service is not initialized
        """
        if not self.is_ready():
            raise ServiceNotReadyException(
                "Service is not ready. Please check the health endpoint."
            )
        
        # Update max_steps if provided
        if max_steps is not None:
            original_max = self._config.get("max_reasoning_iterations", 7)
            self._config["max_reasoning_iterations"] = max_steps
            # Recompile if max_steps changed significantly
            if abs(max_steps - original_max) > 2:
                logger.info(f"Recompiling graph with max_steps={max_steps}")
                self._deep_rag.compile()
        
        try:
            return self._deep_rag.answer(question)
        finally:
            # Restore original max_steps
            if max_steps is not None:
                self._config["max_reasoning_iterations"] = original_max
    
    def stream_answer(self, question: str, max_steps: Optional[int] = None):
        """
        Stream answer generation using the Deep RAG system.
        
        Args:
            question: The question to answer
            max_steps: Optional maximum reasoning steps
            
        Yields:
            Dictionary with chunk type and content
            
        Raises:
            ServiceNotReadyException: If service is not initialized
        """
        if not self.is_ready():
            raise ServiceNotReadyException(
                "Service is not ready. Please check the health endpoint."
            )
        
        # Update max_steps if provided
        if max_steps is not None:
            original_max = self._config.get("max_reasoning_iterations", 7)
            self._config["max_reasoning_iterations"] = max_steps
            if abs(max_steps - original_max) > 2:
                logger.info(f"Recompiling graph with max_steps={max_steps}")
                self._deep_rag.compile()
        
        try:
            # Import here to avoid circular imports
            from .streaming import stream_deep_rag_response
            yield from stream_deep_rag_response(self._deep_rag, question)
        finally:
            # Restore original max_steps
            if max_steps is not None:
                self._config["max_reasoning_iterations"] = original_max
    
    @property
    def deep_rag(self) -> DeepRAGSystem:
        """Get the Deep RAG system instance."""
        if not self.is_ready():
            raise ServiceNotReadyException("Service is not ready")
        return self._deep_rag
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the configuration."""
        if self._config is None:
            self._config = get_config()
        return self._config


# Global service instance (singleton pattern)
_service_instance: Optional[DeepRAGService] = None


def get_service() -> DeepRAGService:
    """Get the global service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DeepRAGService()
    return _service_instance

