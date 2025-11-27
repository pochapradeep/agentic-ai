#!/usr/bin/env python3
"""Basic RAG program for simple question answering.

This script implements a simple RAG pipeline:
1. Load pre-generated embeddings (or generate if needed)
2. Create a baseline RAG chain
3. Answer questions using the chain

Usage:
    python scripts/run_basic_rag.py "What are the cost benchmarks for green hydrogen?"
    python scripts/run_basic_rag.py --interactive
    python scripts/run_basic_rag.py --query "Your question here"
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
from src.vector_store import create_retriever
from src.rag_chain import create_baseline_rag_chain


def setup_basic_rag(config=None):
    """
    Set up the basic RAG system.
    
    Args:
        config: Configuration dictionary (uses get_config() if None)
        
    Returns:
        Tuple of (rag_chain, config)
    """
    if config is None:
        config = get_config()
    
    ensure_directories(config)
    
    print("=" * 60)
    print("BASIC RAG SETUP")
    print("=" * 60)
    
    # Load or generate embeddings
    print("\n[1/3] Loading embeddings...")
    vector_store = load_or_generate_embeddings(config, force_regenerate=False)
    
    # Create embedding function for retriever
    print("\n[2/3] Creating retriever...")
    embedding_function = create_embedding_function(config)
    retriever = create_retriever(vector_store, k=config.get("top_k_retrieval", 3))
    print(f"✓ Retriever created (k={config.get('top_k_retrieval', 3)})")
    
    # Create RAG chain
    print("\n[3/3] Creating RAG chain...")
    rag_chain = create_baseline_rag_chain(retriever, config)
    print("✓ RAG chain created")
    
    print("\n" + "=" * 60)
    print("✓ BASIC RAG READY")
    print("=" * 60)
    
    return rag_chain, config


def answer_question(rag_chain, question: str) -> str:
    """
    Answer a question using the RAG chain.
    
    Args:
        rag_chain: The RAG chain
        question: The question to answer
        
    Returns:
        The answer string
    """
    print(f"\nQuestion: {question}")
    print("\nGenerating answer...")
    
    try:
        answer = rag_chain.invoke(question)
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"


def main():
    """Main entry point for basic RAG."""
    parser = argparse.ArgumentParser(
        description="Run basic RAG to answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Answer a single question
  python scripts/run_basic_rag.py "What are green hydrogen cost benchmarks?"

  # Interactive mode
  python scripts/run_basic_rag.py --interactive

  # Using query flag
  python scripts/run_basic_rag.py --query "Your question here"
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
    
    # Setup RAG
    try:
        config = get_config()
        if args.force_regenerate:
            from src.embedding_pipeline import generate_embeddings
            print("Force regenerating embeddings...")
            generate_embeddings(config, force_regenerate=True)
        
        rag_chain, config = setup_basic_rag(config)
    except Exception as e:
        print(f"❌ Error setting up RAG: {e}")
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
                
                answer = answer_question(rag_chain, question)
                print(f"\nAnswer:\n{answer}\n")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        sys.exit(0)
    
    # Single question mode
    if not question:
        print("❌ Error: No question provided")
        print("Use --interactive for interactive mode or provide a question")
        parser.print_help()
        sys.exit(1)
    
    # Answer the question
    answer = answer_question(rag_chain, question)
    print(f"\n{'=' * 60}")
    print("ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

