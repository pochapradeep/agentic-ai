# Deep RAG Notebook Explanation

## Overview

This notebook is a **comprehensive guide to building a "Deep Thinking RAG" (Retrieval-Augmented Generation) system**. It demonstrates how to build an advanced, agentic RAG system that can handle complex, multi-step queries that traditional "shallow" RAG systems fail on.

## The Problem: Why "Shallow" RAG Fails

Traditional RAG systems follow a simple linear pipeline:
1. **Retrieve** → Get relevant documents from a vector store
2. **Augment** → Add retrieved context to the prompt
3. **Generate** → LLM generates an answer

This works well for simple, fact-based questions but fails when:
- Questions require information from multiple document sections
- Questions need synthesis and comparison
- Questions require multi-step reasoning
- Information is spread across different sources

## The Solution: Deep Thinking RAG

The notebook builds an **autonomous reasoning agent** that can:
- **Plan** complex queries by breaking them into sub-questions
- **Adaptively retrieve** information using multiple strategies
- **Self-critique** its progress and decide when to continue or stop
- **Synthesize** information from multiple sources
- **Use tools** like web search when needed

## Notebook Structure (8 Parts)

### Part 1: Setting the Stage
- **1.1**: Introduction explaining the limits of shallow RAG
- **1.2**: Environment setup (API keys, imports, configuration)
- **1.3**: Loading documents from the `data/` folder (your green hydrogen PDF)
- **1.4**: Defining a challenging multi-hop query to test the system

### Part 2: Building a Baseline RAG System
- **2.1**: Document loading and chunking (splitting PDF into smaller pieces)
- **2.2**: Creating a vector store using FAISS (for semantic search)
- **2.3**: Assembling a simple RAG chain
- **2.4**: Demonstrating how the baseline fails on complex queries
- **2.5**: Analyzing why it failed

**Key Takeaway**: Shows that simple RAG isn't enough for complex questions.

### Part 3: Building the "Deep Thinking" Components

This is the core of the advanced system, with 4 main components:

#### **Component 1: Dynamic Planning and Query Formulation**
- **3.2.1**: **Planner Agent** - An LLM that breaks complex questions into a step-by-step plan
  - Example: "What are the cost benchmarks for green hydrogen?" 
  - Plan: 
    1. Search for cost data in documents
    2. Search for benchmark methodologies
    3. Compare different cost components
- **3.2.2**: **Query Rewriter** - Optimizes search queries for better retrieval
- **3.2.3**: **Metadata Extraction** - Extracts document sections and entities

#### **Component 2: Multi-Stage Retrieval Funnel**
- **3.3.1**: **Retrieval Supervisor** - Decides which retrieval strategy to use
- **3.3.2**: **Hybrid Retrieval** - Combines:
  - **Vector Search** (semantic similarity using embeddings)
  - **BM25** (keyword-based search)
  - **Reciprocal Rank Fusion (RRF)** - Merges results from both methods
- **3.3.3**: **Reranker** - Uses a cross-encoder model to reorder results by relevance
- **3.3.4**: **Context Distillation** - Compresses and focuses the retrieved context

#### **Component 3: Tool Augmentation**
- **3.4**: **Web Search Integration** - Uses Tavily API to search the web when:
  - Information isn't in the local documents
  - Need for up-to-date information
  - Cross-referencing external sources

#### **Component 4: Self-Critique and Control Flow**
- **3.5.1**: **Reflection Step** - Agent reviews what it's learned so far
- **3.5.2**: **Policy Agent** - An LLM "judge" that decides:
  - ✅ Continue searching (need more information)
  - ✅ Stop and answer (have enough information)
  - ✅ Try different approach (current strategy isn't working)
- **3.5.3**: **Stopping Criteria** - Prevents infinite loops

### Part 4: Assembly with LangGraph

**LangGraph** is used to orchestrate the entire workflow as a state machine:

- **4.1**: Defines graph nodes (functions):
  - `plan_node` - Creates the initial plan
  - `retrieval_node` - Searches documents
  - `web_search_node` - Searches the web
  - `rerank_node` - Reorders results
  - `reflection_node` - Reviews progress
  - `final_answer_node` - Generates final answer

- **4.2**: Defines conditional edges (routing logic):
  - Based on the policy agent's decision, route to next step
  - Example: If policy says "continue", go back to retrieval

- **4.3**: Builds the StateGraph connecting all nodes

- **4.4**: Compiles and visualizes the workflow

**The Graph Flow**:
```
Start → Plan → Retrieve → Rerank → Reflect → Policy Decision
                                    ↓
                              Continue? → Back to Retrieve
                              Stop? → Final Answer
```

### Part 5: Running and Comparing

- **5.1**: Invokes the Deep Thinking RAG graph with a complex query
- **5.2**: Analyzes the high-quality output
- **5.3**: Side-by-side comparison:
  - Baseline RAG answer (simple, incomplete)
  - Deep Thinking RAG answer (comprehensive, well-reasoned)

### Part 6: Evaluation Framework

- **6.1**: Overview of evaluation metrics
- **6.2**: Implements comprehensive evaluation using:
  - **RAGAS metrics** (Context Precision, Context Recall, Faithfulness, Answer Correctness)
  - **Custom metrics** (answer length, key terms coverage, specificity, etc.)
- **6.3**: Interprets evaluation scores

### Part 7: Production Optimizations

- **7.1**: Caching strategies
- **7.2**: Provenance and citations (tracking sources)
- **7.3**: Future improvements (learned policies)
- **7.4**: Error handling and fallbacks

### Part 8: Conclusion

- Summary of the journey
- Key architectural principles
- Future directions

## Key Technologies Used

1. **LangChain** - Framework for building LLM applications
2. **LangGraph** - Library for building stateful, multi-agent workflows
3. **FAISS** - Vector database for semantic search
4. **BM25** - Keyword-based search algorithm
5. **Cross-Encoder** - Reranking model (ms-marco-MiniLM-L-6-v2)
6. **Azure OpenAI** - LLM provider (GPT models)
7. **Tavily** - Web search API
8. **LangSmith** - Tracing and debugging tool

## How It Works: Step-by-Step Example

**Query**: "What are the key cost benchmarks for green hydrogen production in India?"

1. **Planning Phase**:
   - Planner breaks this into:
     - "Find cost data for green hydrogen production"
     - "Find benchmark methodologies"
     - "Find India-specific data"

2. **Retrieval Phase** (iterative):
   - **Iteration 1**: Vector search finds documents about green hydrogen costs
   - **Rerank**: Cross-encoder reorders by relevance
   - **Reflection**: "I found cost data but need India-specific benchmarks"
   - **Policy**: "Continue - need more specific information"
   
   - **Iteration 2**: BM25 search for "India" + "benchmark"
   - **Rerank**: New results added
   - **Reflection**: "I have cost data and India-specific information"
   - **Policy**: "Stop - have enough information"

3. **Answer Generation**:
   - Final answer synthesizes all retrieved information
   - Includes citations to source documents

## Why This Approach Works

1. **Adaptive**: Changes strategy based on what it finds
2. **Iterative**: Doesn't give up after first retrieval
3. **Self-aware**: Knows when it has enough information
4. **Multi-modal**: Uses both semantic and keyword search
5. **Tool-augmented**: Can search web when needed
6. **Explainable**: Tracks its reasoning process

## Your Specific Use Case

The notebook has been adapted for your **green hydrogen energy document**:
- Loads PDFs from `data/` folder
- Uses energy-sector-specific prompts
- Queries are tailored to green hydrogen topics
- Evaluation uses energy-sector ground truth

## Next Steps

1. **Run the notebook** cell by cell to understand each component
2. **Experiment** with different queries
3. **Modify** the prompts for your specific needs
4. **Extend** with additional tools or data sources
5. **Deploy** using the modular Python project structure

## Key Files Created

The notebook code has been extracted into reusable modules:
- `src/document_loader.py` - Loads your PDFs
- `src/embeddings.py` - Creates embedding functions
- `src/vector_store.py` - Manages FAISS vector store
- `src/retrieval.py` - Hybrid retrieval strategies
- `src/rag_chain.py` - Baseline RAG chain
- `src/evaluation.py` - Evaluation metrics

This modular structure makes it easy to use components independently or integrate into production systems.

