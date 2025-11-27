"""Streaming implementation for Deep RAG responses."""
import logging
from typing import Dict, Any, Iterator
from datetime import datetime

from src.deep_rag import DeepRAGSystem
from src.graph_nodes import RAGState

logger = logging.getLogger(__name__)


def stream_deep_rag_response(deep_rag: DeepRAGSystem, question: str) -> Iterator[Dict[str, Any]]:
    """
    Stream Deep RAG response as it processes.
    
    Args:
        deep_rag: The Deep RAG system instance
        question: The question to answer
        
    Yields:
        Dictionary with chunk information
    """
    if deep_rag.compiled_graph is None:
        deep_rag.compile()
    
    # Prepare graph input
    graph_input: RAGState = {
        "original_question": question,
        "question": question,
        "plan": None,
        "past_steps": [],
        "current_step_index": 0,
        "retrieved_docs": [],
        "web_results": [],
        "reranked_docs": [],
        "compressed_context": "",
        "synthesized_context": "",
        "research_history": "",
        "final_answer": "",
        "current_step": 0,
        "max_steps": deep_rag.config.get("max_reasoning_iterations", 7)
    }
    
    recursion_limit = 200
    step_count = 0
    plan_generated = False
    
    try:
        for chunk in deep_rag.compiled_graph.stream(
            graph_input,
            stream_config={"recursion_limit": recursion_limit},
            stream_mode="values"
        ):
            state = chunk
            
            # Yield plan when it's generated
            if state.get("plan") and not plan_generated:
                plan = state["plan"]
                plan_generated = True
                yield {
                    "type": "plan",
                    "content": f"Generated plan with {len(plan.steps)} steps",
                    "step": 0,
                    "metadata": {
                        "steps": [step.sub_question for step in plan.steps],
                        "total_steps": len(plan.steps)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Yield retrieval events
            current_step_index = state.get("current_step_index", 0)
            if current_step_index > step_count:
                step_count = current_step_index
                retrieved_docs = state.get("retrieved_docs", [])
                if retrieved_docs:
                    yield {
                        "type": "retrieval",
                        "content": f"Retrieved {len(retrieved_docs)} documents for step {current_step_index + 1}",
                        "step": current_step_index + 1,
                        "metadata": {
                            "doc_count": len(retrieved_docs),
                            "sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs[:3]]
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
            
            # Yield reflection events
            past_steps = state.get("past_steps", [])
            if past_steps and len(past_steps) > step_count:
                reflection = past_steps[-1]
                yield {
                    "type": "reflection",
                    "content": f"Step {len(past_steps)} reflection: {reflection.summary[:100]}...",
                    "step": len(past_steps),
                    "metadata": {
                        "summary_length": len(reflection.summary)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Yield final answer when available
            final_answer = state.get("final_answer", "")
            if final_answer:
                yield {
                    "type": "answer",
                    "content": final_answer,
                    "step": current_step_index + 1,
                    "metadata": {
                        "answer_length": len(final_answer),
                        "total_steps": current_step_index + 1
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                # Break after final answer
                break
        
        # Yield completion event
        yield {
            "type": "complete",
            "content": "Processing complete",
            "step": step_count + 1,
            "metadata": {
                "total_steps": step_count + 1
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during streaming: {e}", exc_info=True)
        yield {
            "type": "error",
            "content": f"Error: {str(e)}",
            "step": step_count + 1,
            "metadata": {
                "error_type": type(e).__name__
            },
            "timestamp": datetime.utcnow().isoformat()
        }

