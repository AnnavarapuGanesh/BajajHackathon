"""
Query processing service that orchestrates the entire query-retrieval pipeline
"""
import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Main service that orchestrates the entire query-retrieval pipeline
    """
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
        self.llm_service = LLMService()
    
    async def process_hackrx_request(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Process the main HackRx request: document processing + question answering
        
        Args:
            document_url: URL of the document to process
            questions: List of questions to answer
            
        Returns:
            Dictionary with answers and metadata
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting HackRx request processing for document: {document_url}")
            
            # Step 1: Process document
            logger.info("Step 1: Processing document...")
            document_data = await self.document_processor.process_document_from_url(document_url)
            
            # Step 2: Generate embeddings for chunks
            logger.info("Step 2: Generating embeddings...")
            chunks_with_embeddings = await self.embedding_service.embed_document_chunks(
                document_data["chunks"]
            )
            
            # Step 3: Store in Pinecone
            logger.info("Step 3: Storing embeddings in Pinecone...")
            await self.pinecone_service.upsert_document_chunks(
                document_id=document_id,
                chunks_with_embeddings=chunks_with_embeddings
            )
            
            # Step 4: Process questions
            logger.info("Step 4: Processing questions...")
            answers = await self.llm_service.process_multiple_queries(
                queries=questions,
                document_chunks=chunks_with_embeddings,
                document_metadata=document_data["metadata"]
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"HackRx request completed in {processing_time:.2f} seconds")
            
            return {
                "answers": answers,
                "document_id": document_id,
                "processing_time": processing_time,
                "metadata": {
                    "document_metadata": document_data["metadata"],
                    "total_questions": len(questions),
                    "total_chunks": len(chunks_with_embeddings)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing HackRx request: {str(e)}")
            # Clean up Pinecone data if error occurred after upsert
            try:
                await self.pinecone_service.delete_document(document_id)
            except:
                pass
            raise
    
    async def process_single_query(
        self, 
        query: str, 
        document_id: Optional[str] = None,
        document_url: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Process a single query against a document
        
        Args:
            query: The question to answer
            document_id: Optional existing document ID in Pinecone
            document_url: Optional document URL to process
            top_k: Number of top similar chunks to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with answer and supporting information
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing single query: {query}")
            
            # If no document_id provided, process document from URL
            if not document_id and document_url:
                document_id = await self._process_and_store_document(document_url)
            elif not document_id and not document_url:
                raise ValueError("Either document_id or document_url must be provided")
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(query)
            
            # Retrieve similar chunks from Pinecone
            similar_chunks = await self.pinecone_service.query_similar_chunks(
                query_embedding=query_embedding,
                document_id=document_id,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if not similar_chunks:
                return {
                    "query": query,
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence_score": 0.0,
                    "retrieved_chunks": [],
                    "reasoning": "No relevant chunks found above the similarity threshold.",
                    "processing_time": time.time() - start_time
                }
            
            # Generate answer using LLM
            answer_result = await self.llm_service.generate_answer(
                query=query,
                relevant_chunks=similar_chunks
            )
            
            processing_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": answer_result["answer"],
                "confidence_score": answer_result["confidence_score"],
                "retrieved_chunks": similar_chunks,
                "reasoning": answer_result["reasoning"],
                "supporting_chunks": answer_result["supporting_chunks"],
                "token_usage": answer_result["token_usage"],
                "processing_time": processing_time,
                "document_id": document_id
            }
            
            logger.info(f"Single query processed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing single query: {str(e)}")
            raise
    
    async def _process_and_store_document(self, document_url: str) -> str:
        """
        Process a document and store it in Pinecone
        
        Args:
            document_url: URL of the document
            
        Returns:
            Document ID
        """
        document_id = str(uuid.uuid4())
        
        try:
            # Process document
            document_data = await self.document_processor.process_document_from_url(document_url)
            
            # Generate embeddings
            chunks_with_embeddings = await self.embedding_service.embed_document_chunks(
                document_data["chunks"]
            )
            
            # Store in Pinecone
            await self.pinecone_service.upsert_document_chunks(
                document_id=document_id,
                chunks_with_embeddings=chunks_with_embeddings
            )
            
            return document_id
            
        except Exception as e:
            logger.error(f"Error processing and storing document: {str(e)}")
            raise
    
    async def get_document_summary(self, document_id: str) -> Dict[str, Any]:
        """
        Get a summary of a processed document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with document summary
        """
        try:
            # Check if document exists
            exists = await self.pinecone_service.check_document_exists(document_id)
            if not exists:
                raise ValueError(f"Document {document_id} not found")
            
            # Get document chunks
            chunks = await self.pinecone_service.get_document_chunks(document_id)
            
            # Get index stats
            index_stats = await self.pinecone_service.get_index_stats()
            
            # Calculate summary statistics
            total_text_length = sum(chunk.get("character_count", 0) for chunk in chunks)
            
            summary = {
                "document_id": document_id,
                "total_chunks": len(chunks),
                "total_characters": total_text_length,
                "average_chunk_size": total_text_length / len(chunks) if chunks else 0,
                "first_chunk_preview": chunks[0]["text"][:200] + "..." if chunks else "",
                "index_stats": index_stats
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting document summary: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from the system
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with deletion status
        """
        try:
            result = await self.pinecone_service.delete_document(document_id)
            logger.info(f"Document {document_id} deleted successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all services
        
        Returns:
            Dictionary with health status
        """
        try:
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "services": {}
            }
            
            # Check Pinecone
            try:
                await self.pinecone_service.get_index_stats()
                health_status["services"]["pinecone"] = "healthy"
            except Exception as e:
                health_status["services"]["pinecone"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check OpenAI (simple embedding test)
            try:
                await self.embedding_service.generate_query_embedding("test")
                health_status["services"]["openai_embeddings"] = "healthy"
            except Exception as e:
                health_status["services"]["openai_embeddings"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check OpenAI LLM (simple completion test)
            try:
                test_response = await self.llm_service.generate_answer(
                    query="test query",
                    relevant_chunks=[{"text": "test context", "similarity_score": 1.0}]
                )
                health_status["services"]["openai_llm"] = "healthy"
            except Exception as e:
                health_status["services"]["openai_llm"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e)
            }
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """
        Get processing metrics and statistics
        
        Returns:
            Dictionary with metrics
        """
        try:
            # Get Pinecone index stats
            index_stats = await self.pinecone_service.get_index_stats()
            
            metrics = {
                "timestamp": time.time(),
                "pinecone_stats": index_stats,
                "supported_formats": list(self.document_processor.supported_formats),
                "embedding_dimension": self.embedding_service.embedding_dimension,
                "max_chunk_size": self.document_processor.max_chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting processing metrics: {str(e)}")
            raise