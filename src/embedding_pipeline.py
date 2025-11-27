"""Embedding generation pipeline for processing all documents."""
import os
# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import get_config, ensure_directories
from .document_loader import load_documents_from_data_folder
from .embeddings import create_embedding_function, check_ollama_service
from .utils import process_documents_with_metadata
from .vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
    get_vector_store_info
)


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def generate_embeddings(
    config: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False
) -> FAISS:
    """
    Generate embeddings for all documents in the data folder.
    
    This function:
    1. Loads all documents from the data folder
    2. Processes them with metadata (for deep RAG compatibility)
    3. Creates embeddings using the configured embedding function
    4. Saves the vector store with metadata
    
    Args:
        config: Configuration dictionary (uses get_config() if None)
        force_regenerate: If True, regenerate even if embeddings exist
        
    Returns:
        FAISS vector store instance
    """
    if config is None:
        config = get_config()
    
    ensure_directories(config)
    
    # Get paths
    data_dir = config["data_dir"]
    vector_store_dir = config["vector_store_dir"]
    embedding_store_name = config.get("embedding_store_name", "embeddings")
    persist_directory = Path(vector_store_dir) / embedding_store_name
    
    # Check if embeddings already exist
    if not force_regenerate and vector_store_exists(str(persist_directory)):
        print(f"Embeddings already exist at {persist_directory}")
        print("Use force_regenerate=True to regenerate, or use load_or_generate_embeddings()")
        raise ValueError(
            f"Embeddings already exist. Use load_or_generate_embeddings() to load them, "
            f"or set force_regenerate=True to regenerate."
        )
    
    print("=" * 60)
    print("EMBEDDING GENERATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load documents
    print("\n[1/4] Loading documents from data folder...")
    documents = load_documents_from_data_folder(data_dir)
    print(f"✓ Loaded {len(documents)} document(s)")
    
    if not documents:
        raise ValueError(f"No documents found in {data_dir}")
    
    # Step 2: Process documents with metadata
    print("\n[2/4] Processing documents with metadata...")
    chunk_size = config.get("chunk_size", 1000)
    chunk_overlap = config.get("chunk_overlap", 150)
    
    doc_chunks = process_documents_with_metadata(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    print(f"✓ Created {len(doc_chunks)} chunks with metadata")
    
    # Step 3: Create embedding function
    print("\n[3/4] Creating embedding function...")
    
    # Check Ollama service if using Ollama
    embedding_provider = config.get("embedding_provider", "ollama")
    if embedding_provider == "ollama":
        ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        if not check_ollama_service(ollama_base_url):
            raise ConnectionError(
                f"Ollama service is not running at {ollama_base_url}. "
                f"Please start Ollama or use a different embedding provider."
            )
    
    embedding_function = create_embedding_function(config)
    print(f"✓ Embedding function created")
    
    # Step 4: Generate vector store
    print("\n[4/4] Generating vector store...")
    
    # Compute file hashes for metadata
    data_path = Path(data_dir)
    document_info = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        if source:
            file_path = Path(source)
            if file_path.exists():
                file_hash = compute_file_hash(file_path)
                # Count chunks for this file
                file_chunks = [chunk for chunk in doc_chunks 
                             if chunk.metadata.get("source_doc") == file_path.name]
                document_info.append({
                    "file_name": file_path.name,
                    "file_hash": file_hash,
                    "chunk_count": len(file_chunks)
                })
    
    # Create metadata
    metadata = {
        "embedding_provider": config.get("embedding_provider", "ollama"),
        "embedding_model": config.get("embedding_model", "nomic-embed-text"),
        "ollama_base_url": config.get("ollama_base_url", "http://localhost:11434"),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(doc_chunks),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "documents": document_info
    }
    
    # Create and save vector store
    vector_store = create_vector_store(
        documents=doc_chunks,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
        metadata=metadata
    )
    
    print("\n" + "=" * 60)
    print("✓ EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Vector store saved to: {persist_directory}")
    print(f"Total chunks: {len(doc_chunks)}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Provider: {metadata['embedding_provider']}")
    
    return vector_store


def load_or_generate_embeddings(
    config: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False
) -> FAISS:
    """
    Load existing embeddings or generate new ones if they don't exist.
    
    This function:
    1. Checks if embeddings exist and are up-to-date
    2. Loads existing embeddings if available
    3. Generates new embeddings if needed or if force_regenerate=True
    4. Validates embedding model matches configuration
    
    Args:
        config: Configuration dictionary (uses get_config() if None)
        force_regenerate: If True, regenerate even if embeddings exist
        
    Returns:
        FAISS vector store instance
    """
    if config is None:
        config = get_config()
    
    ensure_directories(config)
    
    # Get paths
    vector_store_dir = config["vector_store_dir"]
    embedding_store_name = config.get("embedding_store_name", "embeddings")
    persist_directory = Path(vector_store_dir) / embedding_store_name
    
    # Check if embeddings exist
    if not force_regenerate and vector_store_exists(str(persist_directory)):
        print(f"Loading existing embeddings from {persist_directory}...")
        
        # Load metadata to validate
        metadata = get_vector_store_info(str(persist_directory))
        if metadata:
            print(f"Found embeddings generated at: {metadata.get('generated_at', 'unknown')}")
            print(f"Embedding model: {metadata.get('embedding_model', 'unknown')}")
            print(f"Provider: {metadata.get('embedding_provider', 'unknown')}")
            
            # Validate embedding model matches config
            config_model = config.get("embedding_model", "nomic-embed-text")
            stored_model = metadata.get("embedding_model", "")
            if stored_model != config_model:
                print(f"⚠ Warning: Stored model ({stored_model}) differs from config ({config_model})")
                print("⚠ Consider regenerating embeddings with force_regenerate=True")
        
        # Create embedding function and load vector store
        embedding_function = create_embedding_function(config)
        vector_store = load_vector_store(
            str(persist_directory),
            embedding_function
        )
        return vector_store
    
    # Generate new embeddings
    print("No existing embeddings found or force_regenerate=True. Generating new embeddings...")
    return generate_embeddings(config, force_regenerate=force_regenerate)

