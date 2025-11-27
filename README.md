# Agentic AI Deep RAG

A production-grade Retrieval-Augmented Generation (RAG) system with advanced reasoning capabilities, specifically designed for energy sector document analysis.

## Features

- **Multi-stage Retrieval**: Hybrid search combining semantic (vector) and keyword (BM25) retrieval
- **Advanced Reasoning**: LangGraph-based agentic system with planning, retrieval, and reflection
- **Energy Sector Focus**: Optimized for green hydrogen and energy transition documents
- **Comprehensive Evaluation**: Multiple metrics including RAGAS-style evaluation
- **Azure OpenAI Integration**: Supports Azure OpenAI for LLM and embeddings
- **REST API**: FastAPI-based REST API with synchronous and streaming endpoints
- **Azure Deployment**: Ready for deployment to Azure Container Apps
- **Production Ready**: Docker containerization, logging, error handling, and health checks

## Project Structure

```
agentic-ai-deep-rag/
├── api/                     # FastAPI application
│   ├── main.py             # FastAPI app initialization
│   ├── models.py           # Pydantic models
│   ├── service.py          # Deep RAG service wrapper
│   ├── streaming.py        # SSE streaming implementation
│   ├── config.py           # API configuration
│   ├── exceptions.py       # Custom exceptions
│   ├── logging_config.py   # Logging setup
│   └── routers/            # API routers
│       └── query.py        # Query endpoints
├── azure/                   # Azure deployment files
│   ├── container-app.yaml  # Container Apps config
│   ├── deploy.sh           # Deployment script
│   └── README.md           # Deployment guide
├── data/                    # Document data folder
├── src/                     # Source code modules
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # Document loading utilities
│   ├── embeddings.py       # Embedding function creation
│   ├── vector_store.py     # Vector store management
│   ├── retrieval.py        # Advanced retrieval strategies
│   ├── rag_chain.py        # RAG chain creation
│   ├── deep_rag.py         # Deep RAG system
│   ├── evaluation.py       # Evaluation metrics
│   └── utils.py            # Utility functions
├── notebooks/              # Jupyter notebooks
├── scripts/                 # Utility scripts
│   ├── run_basic_rag.py    # Basic RAG script
│   ├── run_deep_rag.py     # Deep RAG script
│   ├── run_api.py          # API server script
│   └── evaluate_rag.py     # Evaluation script
├── tests/                   # Test files
├── Dockerfile               # Docker container definition
└── main.py                  # API entry point
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd agentic-ai-deep-rag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

```env
# Azure OpenAI
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name

# Optional
LANGSMITH_API_KEY=your-langsmith-key
TAVILY_API_KEY=your-tavily-key
```

## Usage

### API Server (Recommended for Production)

The Deep RAG system is available as a FastAPI REST API that can be deployed to Azure Container Apps.

#### Local Development

1. **Start the API server:**
```bash
python scripts/run_api.py --reload
```

Or using the main entry point:
```bash
python main.py
```

2. **Access the API:**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- API Base: http://localhost:8000/api/v1

3. **Test the API:**
```bash
# Using the test script (recommended)
python scripts/test_api.py

# Or using curl
# Synchronous query
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are green hydrogen cost benchmarks?"}'

# Streaming query
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are green hydrogen cost benchmarks?"}' \
  --no-buffer
```

#### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t deep-rag-api .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 \
  -e AZURE_OPENAI_API_KEY=your-key \
  -e AZURE_OPENAI_ENDPOINT=your-endpoint \
  deep-rag-api
```

#### Azure Container Apps Deployment

See [azure/README.md](azure/README.md) for detailed deployment instructions.

Quick deployment:
```bash
cd azure
./deploy.sh
```

For full API documentation, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

### Running the Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/deepRAG.ipynb`

3. Run cells sequentially

### Using as a Python Package

```python
from src.config import get_config, ensure_directories
from src.document_loader import load_documents_from_data_folder
from src.embeddings import create_embedding_function
from src.vector_store import create_vector_store, create_retriever
from src.rag_chain import create_baseline_rag_chain

# Get configuration
config = get_config()
ensure_directories(config)

# Load documents
documents = load_documents_from_data_folder(config["data_dir"])

# Create embedding function
embedding_function = create_embedding_function(config)

# Create vector store
vector_store = create_vector_store(documents, embedding_function)

# Create retriever
retriever = create_retriever(vector_store, k=3)

# Create RAG chain
rag_chain = create_baseline_rag_chain(retriever, config)

# Query
result = rag_chain.invoke("What are the cost benchmarks for green hydrogen?")
print(result)
```

## Data

Place your PDF documents in the `data/` folder. The system supports:
- PDF files (`.pdf`)
- Text files (`.txt`)

## Evaluation

The system includes comprehensive evaluation metrics:
- Basic metrics (length, word count, etc.)
- Content quality metrics
- RAGAS-style metrics (context precision, recall, faithfulness, correctness)
- Context usage metrics
- Answer quality metrics

## Requirements

- Python 3.9+
- See `requirements.txt` for full list of dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

