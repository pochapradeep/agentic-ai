"""Query endpoints for the Deep RAG API."""
import time
import logging
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..models import QueryRequest, QueryResponse, StreamChunk, ErrorResponse, SourceDocument
from ..service import get_service
from ..exceptions import (
    ServiceNotReadyException,
    InvalidRequestException,
    ProcessingException,
    TimeoutException
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a question using the Deep RAG system (synchronous).
    
    This endpoint processes the question and returns the complete answer.
    For real-time updates, use the /query/stream endpoint.
    """
    service = get_service()
    
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Please check the health endpoint."
        )
    
    start_time = time.time()
    
    try:
        # Answer the question
        answer = service.answer(
            question=request.question,
            max_steps=request.max_steps
        )
        
        processing_time = time.time() - start_time
        
        # Extract sources from the deep RAG system if available
        sources = []
        try:
            # Try to get sources from the last state
            # Note: This is a simplified version - you may need to enhance
            # the DeepRAGSystem to expose source documents
            pass
        except Exception as e:
            logger.warning(f"Could not extract sources: {e}")
        
        return QueryResponse(
            answer=answer,
            question=request.question,
            steps_taken=service.config.get("max_reasoning_iterations", 7),
            sources=sources if sources else None,
            processing_time=processing_time
        )
        
    except ServiceNotReadyException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutException as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/stream")
async def query_stream(request: QueryRequest) -> EventSourceResponse:
    """
    Answer a question using the Deep RAG system (streaming).
    
    This endpoint streams real-time updates as the agent processes the question.
    Uses Server-Sent Events (SSE) for streaming.
    """
    service = get_service()
    
    if not service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service is not ready. Please check the health endpoint."
        )
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from Deep RAG streaming."""
        try:
            for chunk in service.stream_answer(
                question=request.question,
                max_steps=request.max_steps
            ):
                # Format as SSE
                chunk_data = StreamChunk(**chunk)
                yield f"data: {chunk_data.model_dump_json()}\n\n"
            
            # Send completion event
            yield "data: [DONE]\n\n"
            
        except ServiceNotReadyException as e:
            error_chunk = StreamChunk(
                type="error",
                content=f"Service error: {str(e)}",
                metadata={"error_type": "ServiceNotReadyException"}
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            error_chunk = StreamChunk(
                type="error",
                content=f"Processing error: {str(e)}",
                metadata={"error_type": type(e).__name__}
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
    
    return EventSourceResponse(event_generator())

