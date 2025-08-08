"""
Pydantic models for API request validation
"""
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field, validator

class HackrxRunRequest(BaseModel):
    """
    Request model for the /hackrx/run endpoint
    """
    documents: str = Field(
        ..., 
        description="Document URL (PDF blob URL)",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03..."
    )
    questions: List[str] = Field(
        ..., 
        description="List of questions to be answered from the document",
        min_items=1,
        max_items=20,
        example=[
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    )
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list"""
        if not v:
            raise ValueError("At least one question is required")
        
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
            if len(question) > 500:
                raise ValueError("Question too long (max 500 characters)")
        
        return v
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Validate document URL"""
        if not v.strip():
            raise ValueError("Document URL cannot be empty")
        
        # Basic URL validation
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Document URL must be a valid HTTP/HTTPS URL")
        
        return v

class DocumentProcessRequest(BaseModel):
    """
    Request model for document processing
    """
    document_url: str = Field(..., description="URL of the document to process")
    document_type: Optional[str] = Field(None, description="Type of document (pdf, docx, etc.)")
    chunk_size: Optional[int] = Field(1000, description="Size of text chunks for processing")
    chunk_overlap: Optional[int] = Field(200, description="Overlap between chunks")

class QueryRequest(BaseModel):
    """
    Request model for individual query processing
    """
    query: str = Field(..., description="The question to be answered")
    document_id: str = Field(..., description="Document identifier")
    top_k: Optional[int] = Field(5, description="Number of top similar chunks to retrieve")
    similarity_threshold: Optional[float] = Field(0.7, description="Minimum similarity score")