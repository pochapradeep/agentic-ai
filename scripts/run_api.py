#!/usr/bin/env python3
"""Script to run the Deep RAG API server locally."""
import os
import sys
from pathlib import Path

# Fix OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import uvicorn
from api.config import config


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(
        description="Run the Deep RAG API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in development mode (auto-reload)
  python scripts/run_api.py --reload

  # Run in production mode
  python scripts/run_api.py --host 0.0.0.0 --port 8000

  # Run with custom workers
  python scripts/run_api.py --workers 4
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=config.HOST,
        help=f"Host to bind to (default: {config.HOST})"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=config.PORT,
        help=f"Port to bind to (default: {config.PORT})"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=config.WORKERS,
        help=f"Number of worker processes (default: {config.WORKERS})"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Log level (default: {config.LOG_LEVEL})"
    )
    
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level.lower()
    )


if __name__ == "__main__":
    main()

