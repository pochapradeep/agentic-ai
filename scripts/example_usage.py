"""Example usage of the RAG system."""
import sys
from pathlib import Path

# Add project root to path (not src, so we can import src as a package)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, ensure_directories
from src.document_loader import load_documents_from_data_folder
from src.embeddings import create_embedding_function
from src.vector_store import create_vector_store, create_retriever
from src.rag_chain import create_baseline_rag_chain
from src.utils import process_documents_with_metadata


def main():
    """Example usage of the RAG system."""
    print("=" * 60)
    print("AGENTIC AI DEEP RAG - Example Usage")
    print("=" * 60)
    
    # Get configuration
    config = get_config()
    ensure_directories(config)
    
    print("\n1. Loading documents...")
    documents = load_documents_from_data_folder(config["data_dir"])
    print(f"   Loaded {len(documents)} documents")
    
    print("\n2. Processing documents with metadata...")
    doc_chunks = process_documents_with_metadata(documents)
    print(f"   Created {len(doc_chunks)} chunks with metadata")
    
    print("\n3. Creating embedding function...")
    embedding_function = create_embedding_function(config)
    
    print("\n4. Creating vector store...")
    vector_store = create_vector_store(doc_chunks, embedding_function)
    
    print("\n5. Creating retriever...")
    retriever = create_retriever(vector_store, k=3)
    
    print("\n6. Creating RAG chain...")
    rag_chain = create_baseline_rag_chain(retriever, config)
    
    print("\n7. Testing with a query...")
    query = "What are the key cost benchmarks for green hydrogen production in India?"
    print(f"   Query: {query}")
    
    try:
        result = rag_chain.invoke(query)
        print("\n   Answer:")
        print(f"   {result}")
    except Exception as e:
        print(f"\n   Error: {e}")
        print("   Please check your Azure OpenAI configuration")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

