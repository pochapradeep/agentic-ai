# Embedding Generation Pipeline

## Overview

The embedding generation pipeline allows you to generate embeddings once for all documents in your `data/` folder and reuse them across both basic RAG and deep RAG systems. This is especially useful for server-side deployments where you want to generate embeddings separately from running the RAG systems.

## Features

- **Ollama Support**: Primary embedding provider using local Ollama models (nomic-embed-text, mxbai-embed-large, etc.)
- **Multi-Provider Support**: Also supports OpenAI, Azure OpenAI, and HuggingFace embeddings
- **Automatic Loading**: Notebook automatically loads pre-generated embeddings if available
- **Metadata Tracking**: Saves metadata about embedding model, documents, and generation time
- **Server-Side Ready**: CLI script suitable for cron jobs, task queues, or API endpoints

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (Recommended)

```bash
# Install Ollama from https://ollama.com
# Then pull an embedding model:
ollama pull nomic-embed-text
```

### 3. Configure Environment

Edit your `.env` file:

```bash
# Embedding Configuration
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434
```

### 4. Generate Embeddings

```bash
# Generate embeddings for all documents in data/ folder
python scripts/generate_embeddings.py
```

### 5. Use in Notebook

The notebook will automatically detect and load pre-generated embeddings. Just run the cells normally!

## Usage

### Command-Line Script

```bash
# Basic usage (uses config from .env)
python scripts/generate_embeddings.py

# Force regeneration
python scripts/generate_embeddings.py --force

# Override embedding model
python scripts/generate_embeddings.py --embedding-model mxbai-embed-large

# Override Ollama URL
python scripts/generate_embeddings.py --ollama-base-url http://localhost:11434

# Use different provider
python scripts/generate_embeddings.py --embedding-provider huggingface

# Load existing or generate new
python scripts/generate_embeddings.py --load-existing
```

### Python API

```python
from src.embedding_pipeline import generate_embeddings, load_or_generate_embeddings
from src.config import get_config

config = get_config()

# Generate new embeddings
vector_store = generate_embeddings(config, force_regenerate=True)

# Load existing or generate if needed
vector_store = load_or_generate_embeddings(config, force_regenerate=False)
```

### In Notebook

The notebook has been updated to automatically check for pre-generated embeddings. Two new cells have been added:

1. **Cell 2.1.5**: Markdown cell explaining the embedding pipeline
2. **Cell 2.1.6**: Code cell that loads or generates embeddings

The notebook will:
- Try to load pre-generated embeddings first
- Fall back to generating embeddings in the notebook if not found
- Use the same embedding function for both basic and deep RAG

## File Structure

```
vector_store/
  └── embeddings/
      ├── index.faiss          # FAISS index file
      ├── index.pkl            # FAISS pickle file
      └── metadata.json        # Metadata about embeddings
```

## Metadata Structure

The `metadata.json` file contains:

```json
{
  "embedding_provider": "ollama",
  "embedding_model": "nomic-embed-text",
  "ollama_base_url": "http://localhost:11434",
  "chunk_size": 1000,
  "chunk_overlap": 150,
  "document_count": 149,
  "generated_at": "2024-11-27T12:00:00Z",
  "documents": [
    {
      "file_name": "document.pdf",
      "file_hash": "abc123...",
      "chunk_count": 45
    }
  ]
}
```

## Supported Embedding Models

### Ollama Models
- `nomic-embed-text` (default) - Good balance of quality and speed
- `mxbai-embed-large` - Higher quality, slower
- `all-minilm` - Faster, lower quality

### Other Providers
- OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
- Azure OpenAI: Same as OpenAI
- HuggingFace: Any BGE model (e.g., `BAAI/bge-small-en-v1.5`)

## Troubleshooting

### Ollama Service Not Running

```
Error: Ollama service is not running at http://localhost:11434
```

**Solution:**
1. Install Ollama from https://ollama.com
2. Start the Ollama service
3. Pull a model: `ollama pull nomic-embed-text`
4. Verify: `curl http://localhost:11434/api/tags`

### Embeddings Already Exist

```
Error: Embeddings already exist
```

**Solution:**
- Use `--force` flag to regenerate
- Or use `load_or_generate_embeddings()` function
- Or delete the `vector_store/embeddings/` directory

### Import Errors in Notebook

If you get import errors in the notebook:

1. Make sure you're running from the project root
2. Check that `src/` is in Python path
3. Install dependencies: `pip install -r requirements.txt`

## Server-Side Integration

The script is designed for server-side use:

```python
# Example: API endpoint
@app.post("/generate-embeddings")
def generate_embeddings_endpoint():
    import subprocess
    result = subprocess.run(
        ["python", "scripts/generate_embeddings.py", "--force"],
        capture_output=True,
        text=True
    )
    return {"status": "success" if result.returncode == 0 else "error"}
```

```bash
# Example: Cron job (daily at 2 AM)
0 2 * * * cd /path/to/project && python scripts/generate_embeddings.py
```

## Best Practices

1. **Generate Once, Use Many Times**: Generate embeddings separately and reuse them
2. **Version Control**: Don't commit `vector_store/` directory (already in `.gitignore`)
3. **Monitor Changes**: Check metadata to see when embeddings were last generated
4. **Incremental Updates**: Future enhancement will support updating embeddings when new documents are added
5. **Same Model**: Always use the same embedding model for generation and retrieval

## Next Steps

- Add incremental update support (detect new documents)
- Add embedding versioning
- Add batch processing for large document sets
- Add embedding quality metrics

