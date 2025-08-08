"""
HackRx API endpoints for the intelligent query-retrieval system
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import verify_bearer_token
from app.models.request_models import HackrxRunRequest, QueryRequest
from app.models.response_models import (
    HackrxRunResponse, 
    DetailedHackrxResponse, 
    QueryResponse, 
    HealthCheckResponse,
    ErrorResponse
)
from app.services.query_processor import QueryProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize the query processor
query_processor = QueryProcessor()

@router.post(
    "/run",
    response_model=HackrxRunResponse,
    summary="Process document and answer questions",
    description="Main endpoint that processes a document from URL and answers multiple questions"
)
async def hackrx_run(
    request: HackrxRunRequest,
    token: str = Depends(verify_bearer_token)
) -> HackrxRunResponse:
    """
    Process a document and answer multiple questions
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Extracts text and creates chunks
    3. Generates embeddings for all chunks
    4. Stores embeddings in Pinecone vector database
    5. Processes each question by finding relevant chunks
    6. Generates answers using GPT-4
    
    Args:
        request: HackrxRunRequest containing document URL and questions
        token: Bearer token for authentication
        
    Returns:
        HackrxRunResponse with answers to all questions
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info(f"Processing HackRx run request with {len(request.questions)} questions")
        
        # Validate input
        if not request.documents.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL cannot be empty"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        # Process the request
        result = await query_processor.process_hackrx_request(
            document_url=request.documents,
            questions=request.questions
        )
        
        logger.info(f"HackRx run completed successfully in {result['processing_time']:.2f} seconds")
        
        return HackrxRunResponse(answers=result["answers"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hackrx_run: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post(
    "/run/detailed",
    response_model=DetailedHackrxResponse,
    summary="Process document and answer questions with detailed metadata",
    description="Extended version of /run that includes additional metadata and processing information"
)
async def hackrx_run_detailed(
    request: HackrxRunRequest,
    token: str = Depends(verify_bearer_token)
) -> DetailedHackrxResponse:
    """
    Process a document and answer questions with detailed metadata
    
    Args:
        request: HackrxRunRequest containing document URL and questions
        token: Bearer token for authentication
        
    Returns:
        DetailedHackrxResponse with answers and additional metadata
    """
    try:
        logger.info(f"Processing detailed HackRx run request with {len(request.questions)} questions")
        
        # Process the request (this gives us more detailed information)
        result = await query_processor.process_hackrx_request(
            document_url=request.documents,
            questions=request.questions
        )
        
        # For detailed response, we need to process questions individually to get more metadata
        detailed_answers = []
        
        # Note: This is a simplified version. In a production system, you might want to
        # modify the query_processor to return more detailed information per question
        for i, (question, answer) in enumerate(zip(request.questions, result["answers"])):
            detailed_answer = {
                "question": question,
                "answer": answer,
                "confidence_score": 0.8,  # Would come from actual processing
                "supporting_chunks": [],  # Would come from actual processing
                "page_references": [],    # Would come from actual processing
                "clause_references": []   # Would come from actual processing
            }
            detailed_answers.append(detailed_answer)
        
        return DetailedHackrxResponse(
            answers=detailed_answers,
            document_metadata=result["metadata"]["document_metadata"],
            processing_time=result["processing_time"],
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Would come from actual processing
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hackrx_run_detailed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Process a single query against a document",
    description="Process a single question against a document (either by ID or URL)"
)
async def process_query(
    request: QueryRequest,
    token: str = Depends(verify_bearer_token)
) -> QueryResponse:
    """
    Process a single query against a document
    
    Args:
        request: QueryRequest containing the query and document information
        token: Bearer token for authentication
        
    Returns:
        QueryResponse with the answer and supporting information
    """
    try:
        logger.info(f"Processing single query: {request.query}")
        
        result = await query_processor.process_single_query(
            query=request.query,
            document_id=request.document_id,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health check endpoint",
    description="Check the health status of all system components"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        HealthCheckResponse with system health status
    """
    try:
        health_status = await query_processor.health_check()
        
        return HealthCheckResponse(
            status=health_status["status"],
            version="1.0.0",
            timestamp=str(health_status["timestamp"]),
            services=health_status.get("services", {})
        )
        
    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp="",
            services={"error": str(e)}
        )

@router.get(
    "/metrics",
    summary="Get processing metrics",
    description="Get system metrics and statistics"
)
async def get_metrics(
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Get processing metrics
    
    Args:
        token: Bearer token for authentication
        
    Returns:
        Dictionary with system metrics
    """
    try:
        metrics = await query_processor.get_processing_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error in get_metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.get(
    "/document/{document_id}/summary",
    summary="Get document summary",
    description="Get a summary of a processed document"
)
async def get_document_summary(
    document_id: str,
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Get document summary
    
    Args:
        document_id: Document identifier
        token: Bearer token for authentication
        
    Returns:
        Dictionary with document summary
    """
    try:
        summary = await query_processor.get_document_summary(document_id)
        return summary
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in get_document_summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@router.delete(
    "/document/{document_id}",
    summary="Delete a document",
    description="Delete a document and all its associated data from the system"
)
async def delete_document(
    document_id: str,
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Delete a document
    
    Args:
        document_id: Document identifier
        token: Bearer token for authentication
        
    Returns:
        Dictionary with deletion status
    """
    try:
        result = await query_processor.delete_document(document_id)
        return result
        
    except Exception as e:
        logger.error(f"Error in delete_document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )