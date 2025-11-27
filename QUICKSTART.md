# Quick Start Guide

## 1. Initial Setup

```bash
# Clone or navigate to the project
cd agentic-ai-deep-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your actual API keys
```

## 2. Verify Setup

```bash
python scripts/setup_environment.py
```

This will check:
- Environment variables are set
- Configuration loads correctly
- Data directory exists
- Required files are present

## 3. Run Example

```bash
python scripts/example_usage.py
```

This demonstrates:
- Loading documents from the data folder
- Creating embeddings and vector store
- Setting up a RAG chain
- Running a query

## 4. Use in Your Code

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
result = rag_chain.invoke("Your question here")
print(result)
```

## 5. Git Setup (Optional)

```bash
# Initialize git (if not already done)
bash scripts/initialize_git.sh

# Or manually:
git init
git add .
git commit -m "Initial commit: Agentic AI Deep RAG project"

# Add remote and push
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running from the project root and the virtual environment is activated.

### Missing API Keys
Make sure your `.env` file is properly configured with all required Azure OpenAI credentials.

### Document Loading Issues
- Ensure PDF files are in the `data/` folder
- Check that `pypdf` is installed: `pip install pypdf`
- Verify file permissions

### Vector Store Issues
- Ensure `faiss-cpu` is installed: `pip install faiss-cpu`
- Check that embeddings are working correctly

