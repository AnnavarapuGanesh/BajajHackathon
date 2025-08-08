"""
Test cases for API endpoints
"""
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app
from app.core.config import settings

client = TestClient(app)

# Test data
TEST_DOCUMENT_URL = "https://example.com/test.pdf"
TEST_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?"
]

@pytest.fixture
def auth_headers():
    """Fixture for authentication headers"""
    return {"Authorization": f"Bearer {settings.BEARER_TOKEN}"}

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == settings.VERSION

def test_health_check():
    """Test the basic health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_hackrx_run_without_auth():
    """Test hackrx/run endpoint without authentication"""
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    response = client.post("/api/v1/hackrx/run", json=payload)
    assert response.status_code == 401

def test_hackrx_run_with_invalid_auth():
    """Test hackrx/run endpoint with invalid authentication"""
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post("/api/v1/hackrx/run", json=payload, headers=headers)
    assert response.status_code == 401

@patch('app.services.query_processor.QueryProcessor.process_hackrx_request')
def test_hackrx_run_success(mock_process, auth_headers):
    """Test successful hackrx/run request"""
    # Mock the response
    mock_process.return_value = {
        "answers": ["Test answer 1", "Test answer 2"],
        "document_id": "test_doc_id",
        "processing_time": 5.0,
        "metadata": {
            "document_metadata": {"file_type": ".pdf"},
            "total_questions": 2,
            "total_chunks": 10
        }
    }
    
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": TEST_QUESTIONS
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload, headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "answers" in data
    assert len(data["answers"]) == 2
    assert data["answers"][0] == "Test answer 1"

def test_hackrx_run_empty_document_url(auth_headers):
    """Test hackrx/run with empty document URL"""
    payload = {
        "documents": "",
        "questions": TEST_QUESTIONS
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload, headers=auth_headers)
    assert response.status_code == 400

def test_hackrx_run_empty_questions(auth_headers):
    """Test hackrx/run with empty questions list"""
    payload = {
        "documents": TEST_DOCUMENT_URL,
        "questions": []
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload, headers=auth_headers)
    assert response.status_code == 400

def test_hackrx_run_invalid_payload():
    """Test hackrx/run with invalid payload"""
    payload = {
        "invalid_field": "value"
    }
    
    response = client.post("/api/v1/hackrx/run", json=payload)
    assert response.status_code == 401  # Will fail on auth first

@patch('app.services.query_processor.QueryProcessor.health_check')
def test_hackrx_health_check(mock_health):
    """Test hackrx health check endpoint"""
    mock_health.return_value = {
        "status": "healthy",
        "timestamp": 1234567890.0,
        "services": {
            "pinecone": "healthy",
            "openai_embeddings": "healthy",
            "openai_llm": "healthy"
        }
    }
    
    response = client.get("/api/v1/hackrx/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data

@patch('app.services.query_processor.QueryProcessor.get_processing_metrics')
def test_get_metrics(mock_metrics, auth_headers):
    """Test get metrics endpoint"""
    mock_metrics.return_value = {
        "timestamp": 1234567890.0,
        "pinecone_stats": {"total_vector_count": 100},
        "supported_formats": [".pdf", ".docx"],
        "embedding_dimension": 1536
    }
    
    response = client.get("/api/v1/hackrx/metrics", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert "timestamp" in data
    assert "pinecone_stats" in data

@patch('app.services.query_processor.QueryProcessor.get_document_summary')
def test_get_document_summary(mock_summary, auth_headers):
    """Test get document summary endpoint"""
    mock_summary.return_value = {
        "document_id": "test_doc_id",
        "total_chunks": 10,
        "total_characters": 5000,
        "average_chunk_size": 500
    }
    
    response = client.get("/api/v1/hackrx/document/test_doc_id/summary", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["document_id"] == "test_doc_id"
    assert data["total_chunks"] == 10

@patch('app.services.query_processor.QueryProcessor.get_document_summary')
def test_get_document_summary_not_found(mock_summary, auth_headers):
    """Test get document summary for non-existent document"""
    mock_summary.side_effect = ValueError("Document not_found_id not found")
    
    response = client.get("/api/v1/hackrx/document/not_found_id/summary", headers=auth_headers)
    assert response.status_code == 404

@patch('app.services.query_processor.QueryProcessor.delete_document')
def test_delete_document(mock_delete, auth_headers):
    """Test delete document endpoint"""
    mock_delete.return_value = {
        "document_id": "test_doc_id",
        "deleted": True
    }
    
    response = client.delete("/api/v1/hackrx/document/test_doc_id", headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["document_id"] == "test_doc_id"
    assert data["deleted"] is True

@patch('app.services.query_processor.QueryProcessor.process_single_query')
def test_process_query(mock_query, auth_headers):
    """Test process single query endpoint"""
    mock_query.return_value = {
        "query": "Test query",
        "answer": "Test answer",
        "confidence_score": 0.9,
        "retrieved_chunks": [],
        "reasoning": "Test reasoning",
        "supporting_chunks": [],
        "token_usage": {"total_tokens": 100},
        "processing_time": 2.0,
        "document_id": "test_doc_id"
    }
    
    payload = {
        "query": "Test query",
        "document_id": "test_doc_id",
        "top_k": 5,
        "similarity_threshold": 0.7
    }
    
    response = client.post("/api/v1/hackrx/query", json=payload, headers=auth_headers)
    assert response.status_code == 200
    
    data = response.json()
    assert data["query"] == "Test query"
    assert data["answer"] == "Test answer"
    assert data["confidence_score"] == 0.9

class TestRequestValidation:
    """Test request validation"""
    
    def test_hackrx_request_validation(self):
        """Test HackrxRunRequest validation"""
        from app.models.request_models import HackrxRunRequest
        
        # Valid request
        valid_request = HackrxRunRequest(
            documents="https://example.com/test.pdf",
            questions=["Test question?"]
        )
        assert valid_request.documents == "https://example.com/test.pdf"
        assert len(valid_request.questions) == 1
        
        # Test validation errors
        with pytest.raises(ValueError):
            HackrxRunRequest(
                documents="",
                questions=["Test question?"]
            )
        
        with pytest.raises(ValueError):
            HackrxRunRequest(
                documents="https://example.com/test.pdf",
                questions=[]
            )
    
    def test_query_request_validation(self):
        """Test QueryRequest validation"""
        from app.models.request_models import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(
            query="Test query?",
            document_id="test_doc_id"
        )
        assert valid_request.query == "Test query?"
        assert valid_request.document_id == "test_doc_id"
        assert valid_request.top_k == 5  # default value
        assert valid_request.similarity_threshold == 0.7  # default value

if __name__ == "__main__":
    pytest.main([__file__])