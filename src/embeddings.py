"""Embedding function creation and management."""
import os
from typing import Union
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def create_embedding_function(config: dict) -> Union[OpenAIEmbeddings, AzureOpenAIEmbeddings, HuggingFaceBgeEmbeddings]:
    """
    Create an embedding function based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Embedding function instance
    """
    embedding_model = config.get('embedding_model', 'BAAI/bge-small-en-v1.5')
    llm_provider = config.get('llm_provider', 'azure_openai')
    
    # Check if using OpenAI embeddings
    if embedding_model.startswith('text-embedding') or 'openai' in embedding_model.lower():
        if llm_provider == "azure_openai":
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
    
    # Use HuggingFace embeddings
    print("Using HuggingFace embeddings...")
    return HuggingFaceBgeEmbeddings(
        model_name=embedding_model,
        encode_kwargs={'normalize_embeddings': True}
    )

