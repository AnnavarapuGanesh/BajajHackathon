"""
Test cases for service modules
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import io

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

class TestDocumentProcessor:
    """Test DocumentProcessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = DocumentProcessor()
    
    def test_get_file_extension_from_url(self):
        """Test file extension detection from URL"""
        # Test PDF URL
        pdf_url = "https://example.com/document.pdf"
        ext = self.processor._get_file_extension_from_url(pdf_url, "application/pdf")
        assert ext == ".pdf"
        
        # Test DOCX URL
        docx_url = "https://example.com/document.docx"
        ext = self.processor._get_file_extension_from_url(docx_url, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert ext == ".docx"
        
        # Test content type fallback
        unknown_url = "https://example.com/document"
        ext = self.processor._get_file_extension_from_url(unknown_url, "application/pdf")
        assert ext == ".pdf"
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        raw_text = "This   is    a  test\n\nwith   multiple    spaces and \t tabs."
        cleaned = self.processor._clean_text(raw_text)
        
        assert "  " not in cleaned  # No double spaces
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
        assert "test with" in cleaned
    
    def test_split_into_sentences(self):
        """Test sentence splitting"""
        text = "This is sentence one. This is sentence two! And this is sentence three?"
        sentences = self.processor._split_into_sentences(text)
        
        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]
        assert "This is sentence two" in sentences[1]
        assert "And this is sentence three" in sentences[2]
    
    def test_create_text_chunks(self):
        """Test text chunking functionality"""
        # Create a long text that will be split into multiple chunks
        text = "This is a test sentence. " * 100  # Long text
        
        chunks = self.processor._create_text_chunks(text)
        
        assert len(chunks) > 1  # Should create multiple chunks
        assert all(chunk['text'] for chunk in chunks)  # All chunks should have text
        assert all('chunk_index' in chunk for chunk in chunks)  # All chunks should have index
    
    @patch('httpx.AsyncClient.get')
    async def test_download_document_success(self, mock_get):
        """Test successful document download"""
        mock_response = MagicMock()
        mock_response.content = b"test content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        content, content_type = await self.processor._download_document("https://example.com/test.pdf")
        
        assert content == b"test content"
        assert content_type == "application/pdf"
    
    def test_extract_text_from_pdf(self):
        """Test PDF text extraction (mock)"""
        # This would require a real PDF file, so we'll mock it
        with patch('PyPDF2.PdfReader') as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Test PDF content"
            mock_reader.return_value.pages = [mock_page]
            
            result = asyncio.run(self.processor._extract_text_from_pdf(b"fake pdf content"))
            
            assert "Test PDF content" in result
            assert "[Page 1]" in result

class TestEmbeddingService:
    """Test EmbeddingService class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('openai.AsyncOpenAI'):
            self.service = EmbeddingService()
    
    def test_validate_embedding(self):
        """Test embedding validation"""
        # Valid embedding
        valid_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        assert self.service.validate_embedding(valid_embedding)
        
        # Invalid embedding - wrong dimension
        invalid_embedding = [0.1, 0.2, 0.3]
        assert not self.service.validate_embedding(invalid_embedding)
        
        # Invalid embedding - not a list
        assert not self.service.validate_embedding("not a list")
        
        # Invalid embedding - contains NaN
        import math
        nan_embedding = [0.1, math.nan, 0.3] * 512
        assert not self.service.validate_embedding(nan_embedding)
    
    def test_calculate_similarity(self):
        """Test similarity calculation"""
        query_embedding = [1.0, 0.0, 0.0]
        doc_embeddings = [
            [1.0, 0.0, 0.0],  # Identical - should have high similarity
            [0.0, 1.0, 0.0],  # Orthogonal - should have low similarity
            [0.5, 0.5, 0.0]   # Partial match
        ]
        
        similarities = self.service.calculate_similarity(query_embedding, doc_embeddings)
        
        assert len(similarities) == 3
        assert similarities[0] > similarities[1]  # First should be more similar
        assert similarities[0] > similarities[2]  # First should be most similar
    
    def test_find_most_similar_chunks(self):
        """Test finding most similar chunks"""
        query_embedding = [1.0, 0.0, 0.0]
        chunks_with_embeddings = [
            {"text": "chunk 1", "embedding": [1.0, 0.0, 0.0]},
            {"text": "chunk 2", "embedding": [0.0, 1.0, 0.0]},
            {"text": "chunk 3", "embedding": [0.9, 0.1, 0.0]}
        ]
        
        results = self.service.find_most_similar_chunks(
            query_embedding=query_embedding,
            chunks_with_embeddings=chunks_with_embeddings,
            top_k=2,
            similarity_threshold=0.5
        )
        
        assert len(results) <= 2  # Should return at most top_k results
        assert all('similarity_score' in result for result in results)
        
        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            assert results[0]['similarity_score'] >= results[1]['similarity_score']
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_query_embedding(self, mock_openai):
        """Test query embedding generation"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3] * 512)]
        mock_client.embeddings.create.return_value = mock_response
        
        with patch.object(self.service, 'client', mock_client):
            embedding = await self.service.generate_query_embedding("test query")
            
            assert len(embedding) == 1536  # OpenAI embedding dimension
            assert all(isinstance(x, float) for x in embedding)

class TestLLMService:
    """Test LLMService class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('openai.AsyncOpenAI'):
            self.service = LLMService()
    
    def test_get_system_prompt(self):
        """Test system prompt generation"""
        prompt = self.service._get_system_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "document analyst" in prompt.lower()
        assert "json" in prompt.lower()
    
    def test_create_context_from_chunks(self):
        """Test context creation from chunks"""
        chunks = [
            {"text": "First chunk content", "similarity_score": 0.9, "chunk_index": 0},
            {"text": "Second chunk content", "similarity_score": 0.8, "chunk_index": 1}
        ]
        
        context = self.service._create_context_from_chunks(chunks)
        
        assert "First chunk content" in context
        assert "Second chunk content" in context
        assert "Context 1" in context
        assert "Context 2" in context
        assert "Similarity: 0.900" in context
    
    def test_create_answer_prompt(self):
        """Test answer prompt creation"""
        query = "What is the waiting period?"
        context = "The waiting period is 30 days."
        metadata = {"file_type": ".pdf", "total_chunks": 5}
        
        prompt = self.service._create_answer_prompt(query, context, metadata)
        
        assert query in prompt
        assert context in prompt
        assert ".pdf" in prompt
        assert "total_chunks" in prompt.lower()
    
    @patch('openai.AsyncOpenAI')
    async def test_generate_answer(self, mock_openai):
        """Test answer generation"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "Test answer", "confidence_score": 0.9, "reasoning": "Test reasoning"}'
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response
        
        relevant_chunks = [
            {"text": "Test chunk", "similarity_score": 0.9, "chunk_index": 0}
        ]
        
        with patch.object(self.service, 'client', mock_client):
            result = await self.service.generate_answer("Test query", relevant_chunks)
            
            assert result["answer"] == "Test answer"
            assert result["confidence_score"] == 0.9
            assert result["reasoning"] == "Test reasoning"
            assert "token_usage" in result
            assert result["token_usage"]["total_tokens"] == 150
    
    async def test_generate_answer_no_chunks(self):
        """Test answer generation with no relevant chunks"""
        result = await self.service.generate_answer("Test query", [])
        
        assert "couldn't find relevant information" in result["answer"].lower()
        assert result["confidence_score"] == 0.0
        assert len(result["supporting_chunks"]) == 0

class TestIntegration:
    """Integration tests for service interactions"""
    
    @patch('app.services.document_processor.DocumentProcessor.process_document_from_url')
    @patch('app.services.embedding_service.EmbeddingService.embed_document_chunks')
    @patch('app.services.llm_service.LLMService.process_multiple_queries')
    async def test_full_pipeline_mock(self, mock_llm, mock_embed, mock_process):
        """Test the full processing pipeline with mocks"""
        from app.services.query_processor import QueryProcessor
        
        # Mock document processing
        mock_process.return_value = {
            "text_content": "Test document content",
            "chunks": [{"text": "Test chunk", "chunk_index": 0}],
            "metadata": {"file_type": ".pdf"}
        }
        
        # Mock embedding
        mock_embed.return_value = [
            {"text": "Test chunk", "chunk_index": 0, "embedding": [0.1] * 1536}
        ]
        
        # Mock LLM processing
        mock_llm.return_value = ["Test answer"]
        
        # Test the query processor
        with patch('app.services.pinecone_service.PineconeService.upsert_document_chunks'):
            processor = QueryProcessor()
            result = await processor.process_hackrx_request(
                "https://example.com/test.pdf",
                ["Test question?"]
            )
            
            assert "answers" in result
            assert len(result["answers"]) == 1
            assert result["answers"][0] == "Test answer"

if __name__ == "__main__":
    pytest.main([__file__])