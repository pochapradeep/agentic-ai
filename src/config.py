"""Configuration management for the RAG system."""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_config() -> Dict[str, Any]:
    """Get configuration dictionary."""
    return {
        "data_dir": str(PROJECT_ROOT / "data"),
        "vector_store_dir": str(PROJECT_ROOT / "vector_store"),
        "llm_provider": os.getenv("LLM_PROVIDER", "azure_openai"),
        "reasoning_llm": os.getenv("REASONING_LLM", "gpt-3.5-turbo-0125"),
        "fast_llm": os.getenv("FAST_LLM", "gpt-3.5-turbo-0125"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "max_reasoning_iterations": int(os.getenv("MAX_REASONING_ITERATIONS", "7")),
        "top_k_retrieval": int(os.getenv("TOP_K_RETRIEVAL", "10")),
        "top_n_rerank": int(os.getenv("TOP_N_RERANK", "3")),
        "azure_deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    }


def ensure_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["vector_store_dir"], exist_ok=True)

