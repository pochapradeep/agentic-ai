"""LangGraph nodes and state definitions for deep RAG."""
from typing import List, Dict, TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Pydantic models for structured outputs
class Step(BaseModel):
    """A single step in the reasoning plan."""
    sub_question: str = Field(description="A clear, specific sub-question to answer.")
    justification: str = Field(description="Why this step is necessary.")
    tool: Literal["search_documents", "search_web"]
    keywords: List[str] = Field(description="Key terms to search for.")
    document_section: Optional[str] = Field(default=None, description="Relevant document section.")


class Plan(BaseModel):
    """Multi-step plan for answering a complex question."""
    steps: List[Step] = Field(description="Ordered list of steps to execute.")


class PastStep(TypedDict):
    """Record of a completed step."""
    step_index: int
    sub_question: str
    retrieved_docs: List[Document]
    summary: str


class RetrievalDecision(BaseModel):
    """Decision from retrieval supervisor."""
    strategy: Literal["vector_search", "keyword_search", "hybrid_search"]
    justification: str


class PolicyDecision(BaseModel):
    """Decision from policy agent."""
    decision: Literal["continue", "stop"]
    reasoning: str


# Main state dictionary
class RAGState(TypedDict):
    """State that flows through the LangGraph."""
    original_question: str
    question: str
    plan: Optional[Plan]
    past_steps: List[PastStep]
    current_step_index: int
    retrieved_docs: List[Document]
    web_results: List[Document]
    reranked_docs: List[Document]
    compressed_context: str
    synthesized_context: str
    research_history: str
    final_answer: str
    current_step: int
    max_steps: int


def get_past_context_str(past_steps: List[PastStep]) -> str:
    """Format past steps as context string."""
    return "\n\n".join([
        f"Step {s['step_index']}: {s['sub_question']}\nSummary: {s['summary']}"
        for s in past_steps
    ])


def format_docs(docs: List[Document]) -> str:
    """Format documents as string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

