"""Embedding function creation and management."""
import os
import requests
from typing import Union
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Try to import from langchain-ollama (newer), fallback to langchain-community
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        OllamaEmbeddings = None


def check_ollama_service(base_url: str = "http://localhost:11434") -> bool:
    """
    Check if Ollama service is running.
    
    Args:
        base_url: Ollama base URL
        
    Returns:
        True if service is available, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def create_embedding_function(config: dict) -> Union[OpenAIEmbeddings, AzureOpenAIEmbeddings, HuggingFaceBgeEmbeddings, OllamaEmbeddings]:
    """
    Create an embedding function based on configuration.
    Supports Ollama, OpenAI, Azure OpenAI, and HuggingFace.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Embedding function instance
    """
    embedding_provider = config.get('embedding_provider', 'ollama')
    embedding_model = config.get('embedding_model', 'nomic-embed-text')
    llm_provider = config.get('llm_provider', 'azure_openai')
    
    # Check for Ollama embeddings first
    if embedding_provider == "ollama" or (embedding_provider is None and embedding_model in ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]):
        if OllamaEmbeddings is None:
            print("⚠ OllamaEmbeddings not available. Install with: pip install langchain-ollama")
            print("⚠ Falling back to HuggingFace embeddings...")
            return HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                encode_kwargs={'normalize_embeddings': True}
            )
        
        ollama_base_url = config.get('ollama_base_url', 'http://localhost:11434')
        ollama_model = config.get('ollama_embedding_model', embedding_model)
        
        # Check if Ollama service is running
        if not check_ollama_service(ollama_base_url):
            print(f"⚠ Ollama service not available at {ollama_base_url}")
            print("⚠ Falling back to HuggingFace embeddings...")
            # Fallback to HuggingFace
            return HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                encode_kwargs={'normalize_embeddings': True}
            )
        
        print(f"Using Ollama embeddings with model: {ollama_model}")
        return OllamaEmbeddings(
            model=ollama_model,
            base_url=ollama_base_url
        )
    
    # Check if using OpenAI embeddings
    if embedding_model.startswith('text-embedding') or 'openai' in embedding_model.lower() or embedding_provider in ["openai", "azure_openai"]:
        if llm_provider == "azure_openai" or embedding_provider == "azure_openai":
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            azure_endpoint = config.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_version = config.get("azure_api_version") or os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            
            if api_key and azure_endpoint:
                print("Using Azure OpenAI embeddings...")
                return AzureOpenAIEmbeddings(
                    azure_deployment=embedding_model,
                    azure_endpoint=azure_endpoint,
                    api_version=api_version,
                    api_key=api_key
                )
            else:
                print("⚠ Azure OpenAI credentials not found, falling back to HuggingFace embeddings...")
        else:
            print("Using OpenAI embeddings...")
            return OpenAIEmbeddings(model=embedding_model)
    
    # Use HuggingFace embeddings as default fallback
    print("Using HuggingFace embeddings...")
    return HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        encode_kwargs={'normalize_embeddings': True}
    )

