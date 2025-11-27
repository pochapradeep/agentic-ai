"""Pydantic models for API requests and responses."""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="The question to answer", min_length=1)
    max_steps: Optional[int] = Field(None, description="Maximum reasoning steps", ge=1, le=20)
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the key cost benchmarks for green hydrogen production in India?",
                "max_steps": 5,
                "temperature": 0.0
            }
        }


class SourceDocument(BaseModel):
    """Model for source document metadata."""
    content: str = Field(..., description="Document content snippet")
    source: Optional[str] = Field(None, description="Document source/file name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="The generated answer")
    question: str = Field(..., description="The original question")
    steps_taken: int = Field(..., description="Number of reasoning steps taken")
    sources: Optional[List[SourceDocument]] = Field(None, description="Source documents used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Green hydrogen cost benchmarks in India range from $2.50 to $3.00 per kilogram...",
                "question": "What are green hydrogen cost benchmarks?",
                "steps_taken": 3,
                "sources": [],
                "processing_time": 12.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    type: Literal["plan", "retrieval", "reflection", "answer", "error", "complete"] = Field(
        ..., description="Type of chunk"
    )
    content: str = Field(..., description="Chunk content")
    step: Optional[int] = Field(None, description="Current step number")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "retrieval",
                "content": "Retrieving documents for step 1...",
                "step": 1,
                "metadata": {"strategy": "hybrid_search"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    system_info: Optional[Dict[str, Any]] = Field(None, description="System information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "system_info": {
                    "embeddings_loaded": True,
                    "vector_store_ready": True
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type/class")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

