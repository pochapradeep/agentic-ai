#!/usr/bin/env python3
"""Evaluation program to compare Basic RAG vs Deep RAG approaches.

This script:
1. Runs both Basic RAG and Deep RAG on the same questions
2. Compares their answers using comprehensive metrics
3. Generates a detailed comparison report

Usage:
    python scripts/evaluate_rag.py --questions questions.json
    python scripts/evaluate_rag.py --question "Your question" --ground-truth "Expected answer"
    python scripts/evaluate_rag.py --interactive
"""
import os
# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config, ensure_directories
from src.embedding_pipeline import load_or_generate_embeddings
from src.embeddings import create_embedding_function
from src.vector_store import create_retriever
from src.rag_chain import create_baseline_rag_chain
from src.document_loader import load_documents_from_data_folder
from src.utils import process_documents_with_metadata
from src.deep_rag import DeepRAGSystem
from src.evaluation import comprehensive_evaluation, create_comparison_table


def setup_both_rag_systems(config=None):
    """
    Set up both Basic RAG and Deep RAG systems.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (basic_rag_chain, deep_rag_system, config)
    """
    if config is None:
        config = get_config()
    
    ensure_directories(config)
    
    print("=" * 60)
    print("SETTING UP RAG SYSTEMS FOR EVALUATION")
    print("=" * 60)
    
    # Load embeddings (shared by both systems)
    print("\n[1/3] Loading embeddings...")
    vector_store = load_or_generate_embeddings(config, force_regenerate=False)
    embedding_function = create_embedding_function(config)
    
    # Setup Basic RAG
    print("\n[2/3] Setting up Basic RAG...")
    retriever = create_retriever(vector_store, k=config.get("top_k_retrieval", 3))
    basic_rag_chain = create_baseline_rag_chain(retriever, config)
    print("✓ Basic RAG ready")
    
    # Setup Deep RAG
    print("\n[3/3] Setting up Deep RAG...")
    documents = load_documents_from_data_folder(config["data_dir"])
    doc_chunks = process_documents_with_metadata(
        documents,
        chunk_size=config.get("chunk_size", 1000),
        chunk_overlap=config.get("chunk_overlap", 150)
    )
    deep_rag = DeepRAGSystem(
        config=config,
        vector_store=vector_store,
        documents=doc_chunks,
        embedding_function=embedding_function
    )
    deep_rag.compile()
    print("✓ Deep RAG ready")
    
    print("\n" + "=" * 60)
    print("✓ BOTH SYSTEMS READY")
    print("=" * 60)
    
    return basic_rag_chain, deep_rag, config


def evaluate_question(
    question: str,
    ground_truth: str,
    basic_rag_chain,
    deep_rag: DeepRAGSystem
) -> Dict[str, Any]:
    """
    Evaluate a single question with both systems.
    
    Args:
        question: The question to evaluate
        ground_truth: Ground truth answer
        basic_rag_chain: Basic RAG chain
        deep_rag: Deep RAG system
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'=' * 60}")
    print(f"EVALUATING QUESTION")
    print("=" * 60)
    print(f"Question: {question}")
    
    # Get answers
    print("\n[1/2] Running Basic RAG...")
    try:
        basic_answer = basic_rag_chain.invoke(question)
        print("✓ Basic RAG completed")
    except Exception as e:
        print(f"✗ Basic RAG error: {e}")
        basic_answer = f"Error: {e}"
    
    print("\n[2/2] Running Deep RAG...")
    try:
        deep_answer = deep_rag.answer(question)
        print("✓ Deep RAG completed")
    except Exception as e:
        print(f"✗ Deep RAG error: {e}")
        deep_answer = f"Error: {e}"
    
    # Extract contexts (simplified - in production, track contexts)
    basic_contexts = [basic_answer]  # Placeholder
    deep_contexts = [deep_answer]  # Placeholder
    
    # Evaluate
    print("\nEvaluating metrics...")
    basic_metrics = comprehensive_evaluation(
        question, basic_answer, ground_truth, basic_contexts, "Basic RAG"
    )
    deep_metrics = comprehensive_evaluation(
        question, deep_answer, ground_truth, deep_contexts, "Deep RAG"
    )
    
    return {
        "question": question,
        "ground_truth": ground_truth,
        "basic_answer": basic_answer,
        "deep_answer": deep_answer,
        "basic_metrics": basic_metrics,
        "deep_metrics": deep_metrics
    }


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare Basic RAG vs Deep RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from JSON file
  python scripts/evaluate_rag.py --questions questions.json

  # Evaluate single question
  python scripts/evaluate_rag.py --question "Q?" --ground-truth "Answer"

  # Interactive mode
  python scripts/evaluate_rag.py --interactive
        """
    )
    
    parser.add_argument(
        "--questions",
        type=str,
        help="JSON file with questions and ground truth answers"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to evaluate"
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth answer (required with --question)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Setup systems
    try:
        basic_rag_chain, deep_rag, config = setup_both_rag_systems()
    except Exception as e:
        print(f"❌ Error setting up RAG systems: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load questions
    questions = []
    
    if args.questions:
        # Load from JSON file
        with open(args.questions, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            else:
                questions = [data]
    
    elif args.question:
        # Single question
        if not args.ground_truth:
            print("❌ Error: --ground-truth required with --question")
            sys.exit(1)
        questions = [{"question": args.question, "ground_truth": args.ground_truth}]
    
    elif args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("INTERACTIVE EVALUATION MODE")
        print("=" * 60)
        print("Enter questions and ground truth answers\n")
        
        while True:
            try:
                question = input("Question (or 'quit' to finish): ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                ground_truth = input("Ground truth answer: ").strip()
                if not ground_truth:
                    print("⚠ Skipping question without ground truth")
                    continue
                
                result = evaluate_question(question, ground_truth, basic_rag_chain, deep_rag)
                questions.append({
                    "question": question,
                    "ground_truth": ground_truth
                })
                
                # Display comparison
                print("\n" + "=" * 60)
                print("COMPARISON")
                print("=" * 60)
                comparison_df = create_comparison_table(
                    result["basic_metrics"],
                    result["deep_metrics"]
                )
                print(comparison_df.to_string())
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        if not questions:
            print("No questions evaluated.")
            sys.exit(0)
    
    else:
        print("❌ Error: Must provide --questions, --question, or --interactive")
        parser.print_help()
        sys.exit(1)
    
    # Evaluate all questions
    results = []
    for q_data in questions:
        question = q_data["question"]
        ground_truth = q_data.get("ground_truth", "")
        
        result = evaluate_question(question, ground_truth, basic_rag_chain, deep_rag)
        results.append(result)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    # Aggregate metrics
    basic_scores = [r["basic_metrics"] for r in results]
    deep_scores = [r["deep_metrics"] for r in results]
    
    # Calculate averages
    avg_basic = {}
    avg_deep = {}
    
    for key in basic_scores[0].keys():
        if isinstance(basic_scores[0][key], (int, float)):
            avg_basic[key] = sum(s[key] for s in basic_scores) / len(basic_scores)
            avg_deep[key] = sum(s[key] for s in deep_scores) / len(deep_scores)
    
    print("\nAverage Metrics:")
    print(f"Basic RAG: {avg_basic}")
    print(f"Deep RAG: {avg_deep}")
    
    # Save results
    if args.output:
        output_data = {
            "results": results,
            "summary": {
                "basic_rag_avg": avg_basic,
                "deep_rag_avg": avg_deep
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

