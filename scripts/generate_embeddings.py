#!/usr/bin/env python3
"""Standalone script for generating embeddings from documents.

This script can be run independently or called from server-side code.
It processes all documents in the data folder and generates embeddings
that can be reused by both basic RAG and deep RAG systems.
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
from src.embedding_pipeline import generate_embeddings, load_or_generate_embeddings
from src.embeddings import check_ollama_service


def main():
    """Main entry point for the embedding generation script."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for all documents in the data folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings using default configuration
  python scripts/generate_embeddings.py

  # Force regeneration even if embeddings exist
  python scripts/generate_embeddings.py --force

  # Override data directory
  python scripts/generate_embeddings.py --data-dir /path/to/data

  # Override embedding model
  python scripts/generate_embeddings.py --embedding-model mxbai-embed-large

  # Override Ollama base URL
  python scripts/generate_embeddings.py --ollama-base-url http://localhost:11434
        """
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if embeddings exist"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override data directory path"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override vector store output directory"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Override embedding model (e.g., nomic-embed-text, mxbai-embed-large)"
    )
    
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        help="Override Ollama base URL (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["ollama", "openai", "azure_openai", "huggingface"],
        help="Override embedding provider"
    )
    
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing embeddings if available, otherwise generate new ones"
    )
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config()
    
    # Override config with command-line arguments
    if args.data_dir:
        config["data_dir"] = args.data_dir
    
    if args.output_dir:
        config["vector_store_dir"] = args.output_dir
    
    if args.embedding_model:
        config["embedding_model"] = args.embedding_model
        # If using Ollama model names, set provider to ollama
        if args.embedding_model in ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]:
            config["embedding_provider"] = "ollama"
            config["ollama_embedding_model"] = args.embedding_model
    
    if args.ollama_base_url:
        config["ollama_base_url"] = args.ollama_base_url
    
    if args.embedding_provider:
        config["embedding_provider"] = args.embedding_provider
    
    # Validate configuration
    ensure_directories(config)
    
    # Check Ollama service if using Ollama
    if config.get("embedding_provider", "ollama") == "ollama":
        ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")
        print(f"Checking Ollama service at {ollama_base_url}...")
        if not check_ollama_service(ollama_base_url):
            print(f"❌ Error: Ollama service is not running at {ollama_base_url}")
            print("\nTo start Ollama:")
            print("  1. Install Ollama from https://ollama.com")
            print("  2. Start the Ollama service")
            print("  3. Pull an embedding model: ollama pull nomic-embed-text")
            print("\nOr use a different embedding provider with --embedding-provider")
            sys.exit(1)
        print("✓ Ollama service is running")
    
    try:
        # Generate or load embeddings
        if args.load_existing:
            print("\nAttempting to load existing embeddings...")
            vector_store = load_or_generate_embeddings(
                config=config,
                force_regenerate=args.force
            )
        else:
            vector_store = generate_embeddings(
                config=config,
                force_regenerate=args.force
            )
        
        print("\n✅ Success! Embeddings are ready to use.")
        print("\nYou can now use these embeddings in your notebook or RAG systems.")
        sys.exit(0)
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        if "already exist" in str(e).lower():
            print("\nTo regenerate embeddings, use --force flag:")
            print("  python scripts/generate_embeddings.py --force")
        sys.exit(1)
        
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

