"""
Embedding service for generating and managing text embeddings using Gemini or OpenAI
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

# Provider Selection Logic
if settings.LLM_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
else:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating and managing text embeddings using Gemini or OpenAI
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.batch_size = 100  # Process embeddings in batches
        
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.embedding_model = "text-embedding-ada-002"
        else:
            # Gemini embedding model
            self.embedding_model = "models/embedding-001"
        
        logger.info(f"Embedding Service initialized with provider: {self.provider}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Process in batches to avoid API limits
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = await self._generate_batch_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: Batch of text strings
            
        Returns:
            List of embedding vectors for the batch
        """
        try:
            # Clean texts - remove empty strings and excessive whitespace
            cleaned_texts = [text.strip() for text in texts if text.strip()]
            
            if not cleaned_texts:
                return []
            
            if self.provider == "gemini":
                embeddings = await self._generate_gemini_embeddings(cleaned_texts)
            else:
                embeddings = await self._generate_openai_embeddings(cleaned_texts)
            
            # Pad with zeros if we had empty texts
            if len(embeddings) < len(texts):
                zero_embedding = [0.0] * self.embedding_dimension
                result = []
                cleaned_index = 0
                
                for original_text in texts:
                    if original_text.strip():
                        result.append(embeddings[cleaned_index])
                        cleaned_index += 1
                    else:
                        result.append(zero_embedding)
                
                return result
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    async def _generate_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Gemini"""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            logger.error(f"Gemini embedding error: {str(e)}")
            raise
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI"""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a single query
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        try:
            if not query.strip():
                return [0.0] * self.embedding_dimension
            
            if self.provider == "gemini":
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=query.strip(),
                    task_type="retrieval_query"
                )
                return result['embedding']
            else:
                response = await self.client.embeddings.create(
                    model=self.embedding_model,
                    input=[query.strip()]
                )
                return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
    
    def calculate_similarity(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Query embedding vector
            document_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            if not document_embeddings:
                return []
            
            query_array = np.array(query_embedding).reshape(1, -1)
            doc_arrays = np.array(document_embeddings)
            
            similarities = cosine_similarity(query_array, doc_arrays)[0]
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            raise
    
    def find_most_similar_chunks(
        self, 
        query_embedding: List[float], 
        chunks_with_embeddings: List[Dict[str, Any]], 
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar chunks to a query
        
        Args:
            query_embedding: Query embedding vector
            chunks_with_embeddings: List of chunks with their embeddings
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of most similar chunks with similarity scores
        """
        try:
            if not chunks_with_embeddings:
                return []
            
            # Extract embeddings from chunks
            embeddings = [chunk['embedding'] for chunk in chunks_with_embeddings]
            
            # Calculate similarities
            similarities = self.calculate_similarity(query_embedding, embeddings)
            
            # Create results with similarity scores
            results = []
            for i, (chunk, similarity) in enumerate(zip(chunks_with_embeddings, similarities)):
                if similarity >= similarity_threshold:
                    result_chunk = chunk.copy()
                    result_chunk['similarity_score'] = float(similarity)
                    results.append(result_chunk)
            
            # Sort by similarity score (descending) and take top_k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar chunks: {str(e)}")
            raise
    
    async def embed_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings added
        """
        try:
            if not chunks:
                return []
            
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embedding
                chunk_with_embedding['provider'] = self.provider
                chunks_with_embeddings.append(chunk_with_embedding)
            
            return chunks_with_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding document chunks: {str(e)}")
            raise
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate an embedding vector
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) != self.embedding_dimension:
                return False
            
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # Check for NaN or infinite values
            if any(np.isnan(x) or np.isinf(x) for x in embedding):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_embedding_stats(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Get statistics about a list of embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Dictionary with embedding statistics
        """
        try:
            if not embeddings:
                return {"count": 0}
            
            embeddings_array = np.array(embeddings)
            
            stats = {
                "count": len(embeddings),
                "dimension": embeddings_array.shape[1],
                "mean_norm": float(np.mean(np.linalg.norm(embeddings_array, axis=1))),
                "std_norm": float(np.std(np.linalg.norm(embeddings_array, axis=1))),
                "min_value": float(np.min(embeddings_array)),
                "max_value": float(np.max(embeddings_array)),
                "mean_value": float(np.mean(embeddings_array)),
                "provider": self.provider
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating embedding stats: {str(e)}")
            return {"count": len(embeddings), "error": str(e)}
