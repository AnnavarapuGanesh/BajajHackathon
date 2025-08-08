"""
Pinecone service for vector database operations
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings

logger = logging.getLogger(__name__)


class PineconeService:
    """
    Service for managing vector storage and retrieval using Pinecone
    """
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """
        Initialize or connect to the Pinecone index
        """
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=settings.PINECONE_ENVIRONMENT  # Use from settings
                    )
                )
                # Wait for index to be ready
                import time
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    async def upsert_document_chunks(
        self, 
        document_id: str, 
        chunks_with_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Upsert document chunks with embeddings to Pinecone
        
        Args:
            document_id: Unique identifier for the document
            chunks_with_embeddings: List of chunks with embeddings
            
        Returns:
            Dictionary with upsert results
        """
        try:
            if not chunks_with_embeddings:
                return {"upserted_count": 0}
            
            # Prepare vectors for upsert
            vectors = []
            for i, chunk in enumerate(chunks_with_embeddings):
                vector_id = f"{document_id}_chunk_{i}"
                
                # Validate embedding exists and is correct dimension
                if "embedding" not in chunk:
                    logger.warning(f"Chunk {i} missing embedding, skipping")
                    continue
                
                embedding = chunk["embedding"]
                if len(embedding) != self.dimension:
                    logger.warning(f"Chunk {i} embedding dimension mismatch: {len(embedding)} vs {self.dimension}")
                    continue
                
                # Metadata to store with the vector
                metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk.get("chunk_index", i),
                    "text": chunk["text"][:1000],  # Limit text length for metadata
                    "character_count": chunk.get("character_count", len(chunk["text"])),
                    "sentence_count": chunk.get("sentence_count", 0)
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            if not vectors:
                logger.warning("No valid vectors to upsert")
                return {"upserted_count": 0}
            
            # Upsert in batches (Pinecone has batch size limits)
            batch_size = 100
            upserted_count = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    result = self.index.upsert(vectors=batch)
                    upserted_count += result.upserted_count
                except Exception as batch_error:
                    logger.error(f"Error upserting batch {i//batch_size}: {str(batch_error)}")
                    # Continue with next batch
                    continue
            
            logger.info(f"Upserted {upserted_count} vectors for document {document_id}")
            
            return {
                "upserted_count": upserted_count,
                "document_id": document_id,
                "total_chunks": len(chunks_with_embeddings),
                "valid_vectors": len(vectors)
            }
            
        except Exception as e:
            logger.error(f"Error upserting document chunks: {str(e)}")
            raise
    
    async def query_similar_chunks(
        self, 
        query_embedding: List[float], 
        document_id: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query for similar chunks using vector similarity
        
        Args:
            query_embedding: Query embedding vector
            document_id: Optional document ID to filter results
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of similar chunks with scores
        """
        try:
            # Validate query embedding
            if len(query_embedding) != self.dimension:
                raise ValueError(f"Query embedding dimension {len(query_embedding)} doesn't match index dimension {self.dimension}")
            
            # Prepare query parameters
            query_params = {
                "vector": query_embedding,
                "top_k": min(top_k, 10000),  # Pinecone limit
                "include_metadata": include_metadata,
                "include_values": False
            }
            
            # Add filter if document_id is specified
            if document_id:
                query_params["filter"] = {"document_id": {"$eq": document_id}}
            
            # Execute query
            query_result = self.index.query(**query_params)
            
            # Process results
            similar_chunks = []
            for match in query_result.matches:
                if match.score >= similarity_threshold:
                    chunk_data = {
                        "id": match.id,
                        "similarity_score": float(match.score),
                        "metadata": match.metadata if include_metadata else {}
                    }
                    
                    if include_metadata and match.metadata:
                        chunk_data.update({
                            "text": match.metadata.get("text", ""),
                            "document_id": match.metadata.get("document_id", ""),
                            "chunk_index": match.metadata.get("chunk_index", 0),
                            "character_count": match.metadata.get("character_count", 0),
                            "sentence_count": match.metadata.get("sentence_count", 0)
                        })
                    
                    similar_chunks.append(chunk_data)
            
            logger.info(f"Found {len(similar_chunks)} similar chunks above threshold {similarity_threshold}")
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error querying similar chunks: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary with deletion results
        """
        try:
            # Delete vectors with the specified document_id
            delete_result = self.index.delete(
                filter={"document_id": {"$eq": document_id}}
            )
            
            logger.info(f"Deleted document {document_id} from Pinecone")
            
            return {
                "document_id": document_id,
                "deleted": True,
                "operation": "delete_by_filter"
            }
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {},
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise
    
    async def check_document_exists(self, document_id: str) -> bool:
        """
        Check if a document exists in the index
        
        Args:
            document_id: Document identifier
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            # Query for any vector with this document_id
            query_result = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=1,
                filter={"document_id": {"$eq": document_id}},
                include_metadata=False
            )
            
            exists = len(query_result.matches) > 0
            logger.debug(f"Document {document_id} exists: {exists}")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return False
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of document chunks
        """
        try:
            # Query for all chunks of the document
            query_result = self.index.query(
                vector=[0.0] * self.dimension,  # Dummy vector
                top_k=10000,  # Large number to get all chunks
                filter={"document_id": {"$eq": document_id}},
                include_metadata=True,
                include_values=False
            )
            
            chunks = []
            for match in query_result.matches:
                if match.metadata:
                    chunk_data = {
                        "id": match.id,
                        "text": match.metadata.get("text", ""),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "document_id": match.metadata.get("document_id", ""),
                        "character_count": match.metadata.get("character_count", 0),
                        "sentence_count": match.metadata.get("sentence_count", 0)
                    }
                    chunks.append(chunk_data)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])
            
            logger.info(f"Retrieved {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
    
    async def update_chunk_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a specific chunk
        
        Args:
            vector_id: Vector identifier
            metadata: New metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Pinecone doesn't support direct metadata updates
            # We would need to fetch the vector, update metadata, and upsert again
            logger.warning("Metadata updates require vector re-upsert in Pinecone")
            
            # Alternative: You could implement this by fetching the vector,
            # updating metadata, and upserting again
            # For now, return False to indicate unsupported operation
            return False
            
        except Exception as e:
            logger.error(f"Error updating chunk metadata: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the Pinecone service
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Try to get index stats as a health check
            stats = await self.get_index_stats()
            
            return {
                "status": "healthy",
                "index_name": self.index_name,
                "vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", self.dimension),
                "connection": "active"
            }
            
        except Exception as e:
            logger.error(f"Pinecone health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "index_name": self.index_name,
                "error": str(e),
                "connection": "failed"
            }
    