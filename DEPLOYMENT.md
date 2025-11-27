# Deployment Guide

## Project Status

✅ **Project structure created and ready for git**

All module files have been extracted from the notebook and organized into a proper Python package structure.

## Created Files

### Core Configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `.env.example` - Environment variables template
- ✅ `requirements.txt` - Python dependencies
- ✅ `setup.py` - Package setup

### Documentation
- ✅ `README.md` - Main project documentation
- ✅ `PROJECT_STRUCTURE.md` - Project structure details
- ✅ `QUICKSTART.md` - Quick start guide
- ✅ `DEPLOYMENT.md` - This file

### Source Code Modules (`src/`)
- ✅ `__init__.py` - Package initialization
- ✅ `config.py` - Configuration management
- ✅ `document_loader.py` - Document loading
- ✅ `embeddings.py` - Embedding functions
- ✅ `vector_store.py` - Vector store management
- ✅ `retrieval.py` - Hybrid retrieval strategies
- ✅ `rag_chain.py` - RAG chain creation
- ✅ `evaluation.py` - Evaluation metrics
- ✅ `utils.py` - Utility functions

### Scripts (`scripts/`)
- ✅ `setup_environment.py` - Environment verification
- ✅ `example_usage.py` - Usage example
- ✅ `initialize_git.sh` - Git initialization script

### Directories
- ✅ `data/` - Document storage (with .gitkeep)
- ✅ `notebooks/` - Jupyter notebooks
- ✅ `tests/` - Test files

## Next Steps

### 1. Move Your Notebook
```bash
# Move your notebook to the notebooks directory
mv /Users/pradeepkumar/Downloads/Copy_of_deepRAG.ipynb notebooks/deepRAG.ipynb
```

### 2. Update Notebook Imports
In your notebook, update imports to use the new module structure:
```python
# Old (from notebook)
from langchain_community.document_loaders import PyPDFLoader

# New (using modules)
from src.document_loader import load_documents_from_data_folder
from src.config import get_config
from src.embeddings import create_embedding_function
# etc.
```

### 3. Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your actual API keys
nano .env  # or use your preferred editor
```

### 4. Verify Setup
```bash
python scripts/setup_environment.py
```

### 5. Test the Modules
```bash
python scripts/example_usage.py
```

### 6. Initialize Git (if not already done)
```bash
# Check git status
git status

# If not initialized:
bash scripts/initialize_git.sh

# Or manually:
git init
git add .
git commit -m "Initial commit: Agentic AI Deep RAG project structure"
```

### 7. Push to Remote
```bash
# Add remote repository
git remote add origin <your-repo-url>

# Push to remote
git branch -M main
git push -u origin main
```

## Module Usage Examples

### Loading Documents
```python
from src.config import get_config
from src.document_loader import load_documents_from_data_folder

config = get_config()
documents = load_documents_from_data_folder(config["data_dir"])
```

### Creating Vector Store
```python
from src.embeddings import create_embedding_function
from src.vector_store import create_vector_store, create_retriever

embedding_function = create_embedding_function(config)
vector_store = create_vector_store(documents, embedding_function)
retriever = create_retriever(vector_store, k=3)
```

### Creating RAG Chain
```python
from src.rag_chain import create_baseline_rag_chain

rag_chain = create_baseline_rag_chain(retriever, config)
result = rag_chain.invoke("Your question here")
```

### Evaluation
```python
from src.evaluation import comprehensive_evaluation, create_comparison_table

baseline_metrics = comprehensive_evaluation(question, baseline_answer, ground_truth, contexts)
advanced_metrics = comprehensive_evaluation(question, advanced_answer, ground_truth, contexts)
comparison_df = create_comparison_table(baseline_metrics, advanced_metrics)
```

## Notes

- All modules are self-contained and can be imported independently
- Configuration is centralized in `src/config.py`
- Environment variables are loaded automatically via `python-dotenv`
- The project follows Python package best practices
- All code is ready for production use

## Troubleshooting

If you encounter import errors:
1. Make sure you're in the project root directory
2. Ensure virtual environment is activated
3. Check that `src/` directory is in Python path
4. Verify all dependencies are installed: `pip install -r requirements.txt`

