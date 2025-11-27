# Project Structure

This document describes the structure of the Agentic AI Deep RAG project.

## Directory Structure

```
agentic-ai-deep-rag/
├── .gitignore              # Git ignore rules
├── .env.example            # Environment variables template
├── README.md               # Main project documentation
├── PROJECT_STRUCTURE.md    # This file
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup configuration
│
├── data/                   # Document data folder
│   ├── .gitkeep           # Keeps directory in git
│   └── *.pdf              # PDF documents (not tracked)
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── document_loader.py # Document loading utilities
│   ├── embeddings.py      # Embedding function creation
│   ├── vector_store.py    # Vector store management
│   ├── retrieval.py       # Advanced retrieval strategies
│   ├── rag_chain.py       # RAG chain creation
│   ├── evaluation.py      # Evaluation metrics
│   └── utils.py           # Utility functions
│
├── notebooks/             # Jupyter notebooks
│   └── deepRAG.ipynb     # Main notebook (to be moved here)
│
├── scripts/               # Utility scripts
│   ├── __init__.py
│   ├── setup_environment.py  # Environment setup check
│   ├── example_usage.py      # Example usage script
│   └── initialize_git.sh     # Git initialization script
│
└── tests/                 # Test files
    └── __init__.py
```

## Module Descriptions

### `src/config.py`
- Configuration management
- Environment variable loading
- Directory creation utilities

### `src/document_loader.py`
- Loads PDF and text files from data directory
- Adds metadata to documents
- Handles errors gracefully

### `src/embeddings.py`
- Creates embedding functions (OpenAI, Azure OpenAI, or HuggingFace)
- Handles different provider configurations
- Automatic fallback to HuggingFace if needed

### `src/vector_store.py`
- Creates FAISS vector stores
- Creates retrievers with configurable parameters
- Supports persistence

### `src/retrieval.py`
- Hybrid retrieval combining BM25 and vector search
- Reciprocal Rank Fusion (RRF) for combining results
- Section-based filtering support

### `src/rag_chain.py`
- Creates baseline RAG chains
- LLM creation (Azure OpenAI or OpenAI)
- Customizable prompts

### `src/evaluation.py`
- Comprehensive evaluation metrics
- RAGAS-style metrics (precision, recall, faithfulness, correctness)
- Comparison utilities

### `src/utils.py`
- Document processing with metadata
- Section detection
- Chunking utilities

## Getting Started

1. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup:**
   ```bash
   python scripts/setup_environment.py
   ```

4. **Run example:**
   ```bash
   python scripts/example_usage.py
   ```

5. **Initialize git (optional):**
   ```bash
   bash scripts/initialize_git.sh
   ```

## Next Steps

1. Move your notebook to `notebooks/deepRAG.ipynb`
2. Update imports in notebook to use `from src import ...`
3. Test the modules individually
4. Commit and push to your git repository

