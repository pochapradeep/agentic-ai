# RAG Programs Documentation

## Overview

Three server-side programs have been created for running and evaluating RAG systems:

1. **Basic RAG** (`scripts/run_basic_rag.py`) - Simple retrieve-augment-generate pipeline
2. **Deep RAG** (`scripts/run_deep_rag.py`) - Advanced multi-step reasoning with LangGraph
3. **Evaluation** (`scripts/evaluate_rag.py`) - Compare both approaches with comprehensive metrics

## Basic RAG Program

### Purpose
Simple RAG pipeline for straightforward question answering. Uses a single retrieval step followed by generation.

### Usage

```bash
# Answer a single question
python scripts/run_basic_rag.py "What are green hydrogen cost benchmarks?"

# Interactive mode
python scripts/run_basic_rag.py --interactive

# Using query flag
python scripts/run_basic_rag.py --query "Your question here"

# Force regenerate embeddings
python scripts/run_basic_rag.py --force-regenerate "Your question"
```

### Features
- Loads pre-generated embeddings (or generates if needed)
- Simple retrieve → augment → generate pipeline
- Fast response times
- Good for simple, fact-based questions

### Limitations
- Single retrieval step
- No query planning or decomposition
- Limited to information in retrieved context
- May fail on complex, multi-hop questions

## Deep RAG Program

### Purpose
Advanced RAG system with multi-step reasoning, planning, and self-critique. Uses LangGraph to orchestrate complex reasoning workflows.

### Usage

```bash
# Answer a complex question
python scripts/run_deep_rag.py "What are the key cost benchmarks for green hydrogen production in India?"

# Interactive mode
python scripts/run_deep_rag.py --interactive

# Using query flag
python scripts/run_deep_rag.py --query "Your complex question here"
```

### Features
- **Planning**: Breaks complex questions into sub-questions
- **Multi-step Retrieval**: Iteratively retrieves information
- **Adaptive Strategies**: Chooses between vector, keyword, or hybrid search
- **Self-Critique**: Decides when to continue or stop
- **Context Distillation**: Compresses and focuses retrieved information
- **Reflection**: Summarizes findings at each step
- **Final Synthesis**: Combines all findings into comprehensive answer

### Architecture
- **LangGraph**: Orchestrates the reasoning workflow
- **Multiple Agents**: Planner, Query Rewriter, Retrieval Supervisor, Reflection, Distiller, Policy
- **State Management**: RAGState tracks progress through reasoning steps
- **Conditional Routing**: Dynamic tool selection based on plan

### When to Use
- Complex questions requiring multiple information sources
- Questions needing synthesis and comparison
- Multi-hop reasoning questions
- When accuracy and completeness are critical

## Evaluation Program

### Purpose
Compare Basic RAG vs Deep RAG on the same questions using comprehensive metrics.

### Usage

```bash
# Evaluate from JSON file
python scripts/evaluate_rag.py --questions questions.json

# Evaluate single question
python scripts/evaluate_rag.py --question "Q?" --ground-truth "Answer"

# Interactive mode
python scripts/evaluate_rag.py --interactive

# Save results to file
python scripts/evaluate_rag.py --questions questions.json --output results.json
```

### Questions JSON Format

```json
[
  {
    "question": "What are green hydrogen cost benchmarks?",
    "ground_truth": "Expected answer here..."
  },
  {
    "question": "Another question?",
    "ground_truth": "Another expected answer..."
  }
]
```

### Metrics Evaluated

1. **RAGAS-style Metrics**:
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Correctness

2. **Basic Metrics**:
   - Answer length
   - Word count
   - Key terms coverage
   - Technical terms count

3. **Quality Metrics**:
   - Context usage
   - Ground truth similarity
   - Specificity
   - Readability
   - Structure

### Output
- Side-by-side comparison table
- Average metrics for both systems
- Detailed per-question results
- JSON export option

## Server-Side Integration

All three programs are designed for server-side use:

### API Integration Example

```python
from scripts.run_basic_rag import setup_basic_rag, answer_question
from scripts.run_deep_rag import setup_deep_rag, answer_question

# Setup (do once, reuse)
basic_rag_chain, config = setup_basic_rag()
deep_rag, config = setup_deep_rag()

# API endpoint
@app.post("/basic-rag")
def basic_rag_endpoint(question: str):
    answer = answer_question(basic_rag_chain, question)
    return {"answer": answer}

@app.post("/deep-rag")
def deep_rag_endpoint(question: str):
    answer = answer_question(deep_rag, question)
    return {"answer": answer}
```

### Background Jobs

```python
# Scheduled evaluation
import subprocess

subprocess.run([
    "python", "scripts/evaluate_rag.py",
    "--questions", "questions.json",
    "--output", "results.json"
])
```

### Docker Integration

```dockerfile
# In Dockerfile
COPY scripts/ /app/scripts/
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "scripts/run_basic_rag.py", "--interactive"]
```

## Dependencies

All programs require:
- Pre-generated embeddings (use `scripts/generate_embeddings.py`)
- Configuration in `.env` file
- Required Python packages (see `requirements.txt`)

## Performance Considerations

### Basic RAG
- **Speed**: Fast (~1-3 seconds)
- **Cost**: Low (single LLM call)
- **Use Case**: Simple questions, high throughput

### Deep RAG
- **Speed**: Slower (~10-30 seconds)
- **Cost**: Higher (multiple LLM calls)
- **Use Case**: Complex questions, accuracy critical

## Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd /path/to/agentic-ai-deep-rag

# Install dependencies
pip install -r requirements.txt
```

### Embeddings Not Found
```bash
# Generate embeddings first
python scripts/generate_embeddings.py
```

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Or use different embedding provider
export EMBEDDING_PROVIDER=huggingface
```

## Next Steps

1. **Web Search Integration**: Add Tavily API for web search node
2. **Reranking**: Implement cross-encoder reranking
3. **Caching**: Add response caching for common questions
4. **Monitoring**: Add logging and metrics collection
5. **API Server**: Create FastAPI/Flask wrapper for REST API

