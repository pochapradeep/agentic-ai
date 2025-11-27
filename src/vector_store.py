"""Vector store creation and management."""
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever


def create_vector_store(
    documents: List[Document],
    embedding_function: Embeddings,
    persist_directory: Optional[str] = None
) -> FAISS:
    """
    Create a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects
        embedding_function: Embedding function to use
        persist_directory: Optional directory to persist the vector store
        
    Returns:
        FAISS vector store instance
    """
    print("Creating vector store with FAISS...")
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_function
    )
    
    if persist_directory:
        vector_store.save_local(persist_directory)
        print(f"Vector store saved to {persist_directory}")
    
    print(f"Vector store created with {len(documents)} documents.")
    return vector_store


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

