"""Vector store creation and management."""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure FAISS is available
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    # Try to install faiss-cpu if not available
    try:
        import faiss
        from langchain_community.vectorstores import FAISS
    except ImportError:
        print("Installing faiss-cpu...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
        from langchain_community.vectorstores import FAISS
        print("✓ faiss-cpu installed successfully")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def create_vector_store(
    documents: List[Document],
    embedding_function: Embeddings,
    persist_directory: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> FAISS:
    """
    Create a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects
        embedding_function: Embedding function to use
        persist_directory: Optional directory to persist the vector store
        metadata: Optional metadata dictionary to save alongside the vector store
        
    Returns:
        FAISS vector store instance
    """
    # Ensure FAISS is available
    try:
        import faiss
    except ImportError:
        print("Installing faiss-cpu...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
        import faiss
        print("✓ faiss-cpu installed successfully")
    
    print("Creating vector store with FAISS...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_function
    )
    
    if persist_directory:
        vector_store.save_local(persist_directory)
        print(f"Vector store saved to {persist_directory}")
        
        # Save metadata if provided
        if metadata:
            metadata_path = Path(persist_directory) / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to {metadata_path}")
    
    print(f"Vector store created with {len(documents)} documents.")
    return vector_store


def load_vector_store(
    persist_directory: str,
    embedding_function: Embeddings
) -> FAISS:
    """
    Load an existing FAISS vector store from disk.
    
    Args:
        persist_directory: Directory where the vector store is saved
        embedding_function: Embedding function to use (must match the one used for creation)
        
    Returns:
        FAISS vector store instance
        
    Raises:
        FileNotFoundError: If vector store files are not found
    """
    # Ensure FAISS is available
    try:
        import faiss
    except ImportError:
        print("Installing faiss-cpu...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])
        import faiss
        print("✓ faiss-cpu installed successfully")
    
    persist_path = Path(persist_directory)
    
    # Check if vector store exists
    index_file = persist_path / "index.faiss"
    pkl_file = persist_path / "index.pkl"
    
    if not index_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_directory}. "
            f"Expected files: {index_file}, {pkl_file}"
        )
    
    print(f"Loading vector store from {persist_directory}...")
    vector_store = FAISS.load_local(
        persist_directory,
        embedding_function,
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded successfully.")
    return vector_store


def vector_store_exists(persist_directory: str) -> bool:
    """
    Check if a vector store exists at the given directory.
    
    Args:
        persist_directory: Directory to check
        
    Returns:
        True if vector store exists, False otherwise
    """
    persist_path = Path(persist_directory)
    index_file = persist_path / "index.faiss"
    pkl_file = persist_path / "index.pkl"
    return index_file.exists() and pkl_file.exists()


def get_vector_store_info(persist_directory: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata information about a saved vector store.
    
    Args:
        persist_directory: Directory where the vector store is saved
        
    Returns:
        Metadata dictionary if found, None otherwise
    """
    metadata_path = Path(persist_directory) / "metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None


def create_retriever(
    vector_store: FAISS,
    k: int = 3,
    search_type: str = "similarity"
) -> BaseRetriever:
    """
    Create a retriever from a vector store.
    
    Args:
        vector_store: FAISS vector store instance
        k: Number of documents to retrieve
        search_type: Type of search ("similarity" or "mmr")
        
    Returns:
        BaseRetriever instance
    """
    return vector_store.as_retriever(
        search_kwargs={"k": k},
        search_type=search_type
    )

