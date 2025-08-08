"""
Large Language Model service for query processing and answer generation
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.config import settings

# Provider Selection Logic (UPDATED)
if settings.LLM_PROVIDER == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
else:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for processing queries and generating answers using Gemini or OpenAI
    """
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
        if self.provider == "openai":
            self.client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # For Gemini compatibility
            )
            self.model = settings.OPENAI_MODEL
        else:
            # Gemini is configured globally above
            self.model = settings.GEMINI_MODEL
        
        logger.info(f"LLM Service initialized with provider: {self.provider}")

    async def generate_answer(
        self, 
        query: str, 
        relevant_chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer to a query based on relevant document chunks
        """
        try:
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence_score": 0.0,
                    "reasoning": "No relevant chunks found in the document.",
                    "supporting_chunks": [],
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            
            # Create context from relevant chunks
            context = self._create_context_from_chunks(relevant_chunks)
            
            # Generate the prompt
            prompt = self._create_answer_prompt(query, context, document_metadata)
            
            # Call appropriate LLM API
            if self.provider == "gemini":
                response_content = await self._call_gemini_api(prompt)
                token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # Gemini doesn't provide token counts
            else:
                response_content, token_usage = await self._call_openai_api(prompt)
            
            # Parse the response
            try:
                parsed_response = json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback if response is not JSON
                parsed_response = {
                    "answer": response_content,
                    "confidence_score": 0.8,
                    "reasoning": "Response generated but not in expected JSON format"
                }
            
            # Prepare the result
            result = {
                "answer": parsed_response.get("answer", ""),
                "confidence_score": float(parsed_response.get("confidence_score", 0.0)),
                "reasoning": parsed_response.get("reasoning", ""),
                "supporting_chunks": [chunk["text"] for chunk in relevant_chunks[:3]],
                "chunk_references": [f"Chunk {chunk.get('chunk_index', i)}" for i, chunk in enumerate(relevant_chunks[:3])],
                "token_usage": token_usage,
                "provider": self.provider
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    async def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            full_prompt = f"{self._get_system_prompt()}\n\nUser Query:\n{prompt}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def _call_openai_api(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Call OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return response.choices[0].message.content, token_usage
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for answer generation"""
        return """You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. 
        Your task is to provide accurate, detailed answers based on the provided document context.

        Guidelines:
        1. Answer only based on the provided context
        2. If information is not available in the context, clearly state so
        3. Provide specific details, numbers, and conditions when available
        4. Reference specific clauses or sections when applicable
        5. Be precise about waiting periods, coverage limits, and conditions
        6. Always provide a confidence score (0.0 to 1.0) based on how well the context supports your answer
        7. Explain your reasoning process

        Response format must be valid JSON with the following structure:
        {
            "answer": "Your detailed answer here",
            "confidence_score": 0.95,
            "reasoning": "Explanation of how you arrived at this answer"
        }"""
    
    def _create_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Create context string from document chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            similarity_score = chunk.get("similarity_score", 0.0)
            chunk_text = chunk.get("text", "")
            chunk_index = chunk.get("chunk_index", i)
            
            context_part = f"""
[Context {i+1} - Chunk {chunk_index} - Similarity: {similarity_score:.3f}]
{chunk_text}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_answer_prompt(
        self, 
        query: str, 
        context: str, 
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create the prompt for answer generation"""
        metadata_info = ""
        if document_metadata:
            metadata_info = f"""
Document Information:
- File Type: {document_metadata.get('file_type', 'Unknown')}
- Total Chunks: {document_metadata.get('total_chunks', 'Unknown')}
- Source: {document_metadata.get('source_url', 'Unknown')}
"""
        
        prompt = f"""
{metadata_info}

Document Context:
{context}

Question: {query}

Based on the provided document context, please answer the question. Follow these guidelines:
1. Provide a comprehensive answer using only the information from the context
2. Include specific details, numbers, conditions, and time periods when available
3. If the context doesn't contain enough information, clearly state what's missing
4. Reference specific parts of the context that support your answer
5. Provide a confidence score based on how well the context supports your answer
6. Explain your reasoning process

Remember to respond in valid JSON format as specified in the system prompt.
"""
        
        return prompt

    async def process_multiple_queries(
        self, 
        queries: List[str], 
        document_chunks: List[Dict[str, Any]],
        document_metadata: Optional[Dict[str, Any]] = None,
        top_k_per_query: int = 5
    ) -> List[str]:
        """
        Process multiple queries against the same document
        
        Args:
            queries: List of questions
            document_chunks: All document chunks with embeddings
            document_metadata: Optional document metadata
            top_k_per_query: Number of top chunks to use per query
            
        Returns:
            List of answers corresponding to the queries
        """
        try:
            from app.services.embedding_service import EmbeddingService
            
            embedding_service = EmbeddingService()
            answers = []
            
            for query in queries:
                try:
                    # Generate query embedding
                    query_embedding = await embedding_service.generate_query_embedding(query)
                    
                    # Find relevant chunks
                    relevant_chunks = embedding_service.find_most_similar_chunks(
                        query_embedding=query_embedding,
                        chunks_with_embeddings=document_chunks,
                        top_k=top_k_per_query,
                        similarity_threshold=0.5  # Lower threshold for broader search
                    )
                    
                    # Generate answer
                    answer_result = await self.generate_answer(
                        query=query,
                        relevant_chunks=relevant_chunks,
                        document_metadata=document_metadata
                    )
                    
                    answers.append(answer_result["answer"])
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {str(e)}")
                    answers.append(f"Error processing query: {str(e)}")
            
            return answers
            
        except Exception as e:
            logger.error(f"Error processing multiple queries: {str(e)}")
            raise

    async def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent and key information from a user query"""
        try:
            prompt = f"""
Analyze the following query and extract key information:

Query: "{query}"

Please identify:
1. The main topic or subject being asked about
2. The type of information requested (coverage, conditions, limits, periods, etc.)
3. Key terms and concepts
4. The expected answer format (yes/no, amount, duration, description, etc.)

Respond in JSON format:
{{
    "main_topic": "topic here",
    "information_type": "type here",
    "key_terms": ["term1", "term2"],
    "expected_format": "format here",
    "complexity": "simple/medium/complex"
}}
"""
            
            if self.provider == "gemini":
                response_content = await self._call_gemini_api(prompt)
            else:
                response_content, _ = await self._call_openai_api(prompt)
            
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback if response is not JSON
                return {
                    "main_topic": "unknown",
                    "information_type": "general",
                    "key_terms": [],
                    "expected_format": "description",
                    "complexity": "medium"
                }
            
        except Exception as e:
            logger.error(f"Error extracting query intent: {str(e)}")
            return {
                "main_topic": "unknown",
                "information_type": "general",
                "key_terms": [],
                "expected_format": "description",
                "complexity": "medium"
            }

    async def validate_answer_quality(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Validate the quality of a generated answer
        
        Args:
            query: Original question
            answer: Generated answer
            context: Document context used
            
        Returns:
            Dictionary with quality assessment
        """
        try:
            prompt = f"""
Evaluate the quality of this answer:

Question: {query}
Answer: {answer}
Context Available: {len(context)} characters

Rate the answer on:
1. Relevance (0-10): How well does it address the question?
2. Accuracy (0-10): Based on the context, is the information correct?
3. Completeness (0-10): Does it fully answer the question?
4. Clarity (0-10): Is the answer clear and understandable?
5. Specificity (0-10): Does it provide specific details when needed?

Respond in JSON format:
{{
    "relevance": 8,
    "accuracy": 9,
    "completeness": 7,
    "clarity": 8,
    "specificity": 6,
    "overall_score": 7.6,
    "feedback": "Brief feedback on the answer quality"
}}
"""
            
            if self.provider == "gemini":
                response_content = await self._call_gemini_api(prompt)
            else:
                response_content, _ = await self._call_openai_api(prompt)
            
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Fallback if response is not JSON
                return {
                    "relevance": 5,
                    "accuracy": 5,
                    "completeness": 5,
                    "clarity": 5,
                    "specificity": 5,
                    "overall_score": 5.0,
                    "feedback": "Error parsing quality assessment"
                }
            
        except Exception as e:
            logger.error(f"Error validating answer quality: {str(e)}")
            return {
                "relevance": 5,
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "specificity": 5,
                "overall_score": 5.0,
                "feedback": f"Error during evaluation: {str(e)}"
            }

    async def summarize_document(self, document_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of a document based on its chunks
        
        Args:
            document_chunks: List of document chunks
            
        Returns:
            Dictionary with document summary
        """
        try:
            # Combine first few chunks for summary
            summary_chunks = document_chunks[:10]  # Use first 10 chunks
            combined_text = " ".join([chunk.get("text", "") for chunk in summary_chunks])
            
            prompt = f"""
Based on the following document content, provide a comprehensive summary:

Document Content:
{combined_text[:3000]}  # Limit to avoid token limits

Please provide:
1. Main topics covered
2. Key information and details
3. Document type/domain (insurance, legal, HR, compliance)
4. Important numbers, dates, or conditions mentioned

Respond in JSON format:
{{
    "summary": "Detailed summary here",
    "main_topics": ["topic1", "topic2"],
    "document_type": "insurance/legal/hr/compliance",
    "key_information": ["info1", "info2"],
    "important_details": ["detail1", "detail2"]
}}
"""
            
            if self.provider == "gemini":
                response_content = await self._call_gemini_api(prompt)
            else:
                response_content, _ = await self._call_openai_api(prompt)
            
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                return {
                    "summary": response_content,
                    "main_topics": [],
                    "document_type": "unknown",
                    "key_information": [],
                    "important_details": []
                }
            
        except Exception as e:
            logger.error(f"Error summarizing document: {str(e)}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "main_topics": [],
                "document_type": "unknown", 
                "key_information": [],
                "important_details": []
            }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the LLM service
        
        Returns:
            Dictionary with health check results
        """
        try:
            test_prompt = "Respond with 'OK' if you can process this request."
            
            if self.provider == "gemini":
                response = await self._call_gemini_api(test_prompt)
            else:
                response, _ = await self._call_openai_api(test_prompt)
            
            return {
                "status": "healthy",
                "provider": self.provider,
                "model": self.model,
                "response_test": "passed" if response else "failed"
            }
            
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "provider": self.provider,
                "model": self.model,
                "error": str(e),
                "response_test": "failed"
            }
