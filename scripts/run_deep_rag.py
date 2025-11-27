#!/usr/bin/env python3
"""Deep RAG program for complex question answering with multi-step reasoning.

This script implements the advanced Deep RAG pipeline using LangGraph:
1. Load pre-generated embeddings
2. Create Deep RAG system with planning, retrieval, and reflection
3. Answer complex questions using multi-step reasoning

Usage:
    python scripts/run_deep_rag.py "What are the cost benchmarks for green hydrogen?"
    python scripts/run_deep_rag.py --interactive
    python scripts/run_deep_rag.py --query "Your complex question here"
"""
import os
# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, ensure_directories
from src.embedding_pipeline import load_or_generate_embeddings
from src.embeddings import create_embedding_function
from src.document_loader import load_documents_from_data_folder
from src.utils import process_documents_with_metadata
from src.deep_rag import DeepRAGSystem


def setup_deep_rag(config=None):
    """
    Set up the deep RAG system.
    
    Args:
        config: Configuration dictionary (uses get_config() if None)
        
    Returns:
        DeepRAGSystem instance
    """
    if config is None:
        config = get_config()
    
    ensure_directories(config)
    
    print("=" * 60)
    print("DEEP RAG SETUP")
    print("=" * 60)
    
    # Load or generate embeddings
    print("\n[1/4] Loading embeddings...")
    vector_store = load_or_generate_embeddings(config, force_regenerate=False)
    
    # Load documents for retrieval
    print("\n[2/4] Loading documents...")
    documents = load_documents_from_data_folder(config["data_dir"])
    doc_chunks = process_documents_with_metadata(
        documents,
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 150)
    )
    print(f"✓ Loaded {len(doc_chunks)} document chunks")
    
    # Create embedding function
    print("\n[3/4] Creating embedding function...")
    embedding_function = create_embedding_function(config)
    print("✓ Embedding function created")
    
    # Create Deep RAG system
    print("\n[4/4] Initializing Deep RAG system...")
    deep_rag = DeepRAGSystem(
        config=config,
        vector_store=vector_store,
        documents=doc_chunks,
        embedding_function=embedding_function
    )
    deep_rag.compile()
    print("✓ Deep RAG system ready")
    
    print("\n" + "=" * 60)
    print("✓ DEEP RAG READY")
    print("=" * 60)
    
    return deep_rag, config


def answer_question(deep_rag: DeepRAGSystem, question: str) -> str:
    """
    Answer a question using the Deep RAG system.
    
    Args:
        deep_rag: The Deep RAG system
        question: The question to answer
        
    Returns:
        The answer string
    """
    print(f"\nQuestion: {question}")
    print("\nGenerating answer with multi-step reasoning...")
    print("=" * 60)
    
    try:
        answer = deep_rag.answer(question)
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"


def main():
    """Main entry point for deep RAG."""
    parser = argparse.ArgumentParser(
        description="Run deep RAG to answer complex questions with multi-step reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a single question
  python scripts/run_deep_rag.py "What are green hydrogen cost benchmarks?"

  # Interactive mode
  python scripts/run_deep_rag.py --interactive

  # Using query flag
  python scripts/run_deep_rag.py --query "Your complex question here"
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to answer"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Question to answer (alternative to positional argument)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of embeddings"
    )
    
    args = parser.parse_args()
    
    # Get question
    question = args.query or args.question
    
    # Setup Deep RAG
    try:
        config = get_config()
        if args.force_regenerate:
            from src.embedding_pipeline import generate_embeddings
            print("Force regenerating embeddings...")
            generate_embeddings(config, force_regenerate=True)
        
        deep_rag, config = setup_deep_rag(config)
    except Exception as e:
        print(f"❌ Error setting up Deep RAG: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Enter questions (type 'quit' or 'exit' to stop)\n")
        
        while True:
            try:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                answer = answer_question(deep_rag, question)
                print(f"\n{'=' * 60}")
                print("ANSWER")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                import traceback
                traceback.print_exc()
        
        sys.exit(0)
    
    # Single question mode
    if not question:
        print("❌ Error: No question provided")
        print("Use --interactive for interactive mode or provide a question")
        parser.print_help()
        sys.exit(1)
    
    # Answer the question
    answer = answer_question(deep_rag, question)
    print(f"\n{'=' * 60}")
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

