#!/usr/bin/env python3
"""Entry point for the Deep RAG API server."""
import os

# Fix OpenMP conflict on macOS (must be before any imports that use OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import uvicorn
from api.config import config

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        workers=config.WORKERS if not config.RELOAD else 1,
        log_level=config.LOG_LEVEL.lower()
    )

