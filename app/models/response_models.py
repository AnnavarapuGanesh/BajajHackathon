"""
Pydantic models for API response serialization
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class HackrxRunResponse(BaseModel):
    """
    Response model for the /hackrx/run endpoint
    """
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the questions",
        example=[
            "A grace period of thirty days is provided for premium payment after the due date...",
            "There is a waiting period of thirty-six (36) months of continuous coverage..."
        ]
    )

class DetailedAnswer(BaseModel):
    """
    Detailed answer with supporting information
    """
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer to the question")
    confidence_score: float = Field(..., description="Confidence score (0.0 to 1.0)")
    supporting_chunks: List[str] = Field(..., description="Text chunks that support the answer")
    page_references: List[int] = Field(default=[], description="Page numbers where information was found")
    clause_references: List[str] = Field(default=[], description="Specific clauses or sections referenced")

class DetailedHackrxResponse(BaseModel):
    """
    Detailed response with additional metadata
    """
    answers: List[DetailedAnswer] = Field(..., description="Detailed answers with supporting information")
    document_metadata: Dict[str, Any] = Field(..., description="Metadata about the processed document")
    processing_time: float = Field(..., description="Total processing time in seconds")
    token_usage: Dict[str, int] = Field(..., description="Token usage statistics")

class DocumentProcessResponse(BaseModel):
    """
    Response for document processing
    """
    document_id: str = Field(..., description="Unique identifier for the processed document")
    status: str = Field(..., description="Processing status")
    chunks_count: int = Field(..., description="Number of text chunks created")
    embeddings_count: int = Field(..., description="Number of embeddings generated")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default={}, description="Document metadata")

class QueryResponse(BaseModel):
    """
    Response for individual query
    """
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated answer")
    confidence_score: float = Field(..., description="Confidence score")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Retrieved document chunks")
    reasoning: str = Field(..., description="Explanation of the decision process")

class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    services: Dict[str, str] = Field(..., description="Status of dependent services")

class ErrorResponse(BaseModel):
    """
    Error response model
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")