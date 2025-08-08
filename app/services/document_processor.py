"""
Document processing service for handling PDFs, DOCX, and other document formats
"""
import asyncio
import logging
import io
import re
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import aiofiles
import httpx
import PyPDF2
from docx import Document
import email
from email.mime.text import MIMEText

from app.core.config import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Service for processing various document formats
    """
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.doc', '.txt', '.eml'}
        self.max_chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    async def process_document_from_url(self, document_url: str) -> Dict[str, Any]:
        """
        Process a document from a URL
        
        Args:
            document_url: URL of the document to process
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Download document
            document_content, content_type = await self._download_document(document_url)
            
            # Determine file type
            file_extension = self._get_file_extension_from_url(document_url, content_type)
            
            # Extract text based on file type
            if file_extension == '.pdf':
                text_content = await self._extract_text_from_pdf(document_content)
            elif file_extension in ['.docx', '.doc']:
                text_content = await self._extract_text_from_docx(document_content)
            elif file_extension == '.txt':
                text_content = document_content.decode('utf-8', errors='ignore')
            elif file_extension == '.eml':
                text_content = await self._extract_text_from_email(document_content)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Create chunks
            chunks = self._create_text_chunks(text_content)
            
            # Extract metadata
            metadata = {
                'source_url': document_url,
                'file_type': file_extension,
                'content_type': content_type,
                'total_characters': len(text_content),
                'total_chunks': len(chunks),
                'chunk_size': self.max_chunk_size,
                'chunk_overlap': self.chunk_overlap
            }
            
            return {
                'text_content': text_content,
                'chunks': chunks,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document from URL {document_url}: {str(e)}")
            raise
    
    async def _download_document(self, url: str) -> Tuple[bytes, str]:
        """
        Download document from URL
        
        Args:
            url: Document URL
            
        Returns:
            Tuple of (document_content, content_type)
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                
                content_type = response.headers.get('content-type', '')
                return response.content, content_type
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error downloading document: {str(e)}")
                raise
    
    def _get_file_extension_from_url(self, url: str, content_type: str) -> str:
        """
        Determine file extension from URL or content type
        
        Args:
            url: Document URL
            content_type: HTTP content type
            
        Returns:
            File extension
        """
        # Try to get extension from URL path
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        for ext in self.supported_formats:
            if path.endswith(ext):
                return ext
        
        # Try to determine from content type
        content_type_mapping = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/msword': '.doc',
            'text/plain': '.txt',
            'message/rfc822': '.eml'
        }
        
        return content_type_mapping.get(content_type.split(';')[0], '.txt')
    
    async def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted text
        """
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                    continue
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    async def _extract_text_from_docx(self, docx_content: bytes) -> str:
        """
        Extract text from DOCX content
        
        Args:
            docx_content: DOCX file content as bytes
            
        Returns:
            Extracted text
        """
        try:
            docx_file = io.BytesIO(docx_content)
            doc = Document(docx_file)
            
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise
    
    async def _extract_text_from_email(self, email_content: bytes) -> str:
        """
        Extract text from email content
        
        Args:
            email_content: Email file content as bytes
            
        Returns:
            Extracted text
        """
        try:
            email_text = email_content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(email_text)
            
            text_parts = []
            
            # Add headers
            text_parts.append(f"Subject: {msg.get('Subject', 'N/A')}")
            text_parts.append(f"From: {msg.get('From', 'N/A')}")
            text_parts.append(f"To: {msg.get('To', 'N/A')}")
            text_parts.append(f"Date: {msg.get('Date', 'N/A')}")
            text_parts.append("-" * 50)
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text_parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
            else:
                text_parts.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
            
            return '\n\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text from email: {str(e)}")
            raise
    
    def _create_text_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Create overlapping text chunks from the document text
        
        Args:
            text: Full document text
            
        Returns:
            List of text chunks with metadata
        """
        # Clean and normalize text
        cleaned_text = self._clean_text(text)
        
        # Split into sentences for better chunk boundaries
        sentences = self._split_into_sentences(cleaned_text)
        
        chunks = []
        current_chunk = ""
        current_sentence_count = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                current_chunk = potential_chunk
                current_sentence_count += 1
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_index': len(chunks),
                        'sentence_count': current_sentence_count,
                        'character_count': len(current_chunk)
                    })
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(chunks) > 0:
                    # Create overlap from the end of previous chunk
                    overlap_text = self._create_overlap(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                
                current_sentence_count = 1
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_index': len(chunks),
                'sentence_count': current_sentence_count,
                'character_count': len(current_chunk)
            })
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', ' ', text)
        
        # Remove page markers but keep the content
        text = re.sub(r'\[Page \d+\]\n?', '', text)
        
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with NLTK
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        
        return sentences
    
    def _create_overlap(self, text: str, overlap_size: int) -> str:
        """
        Create overlap text from the end of a chunk
        
        Args:
            text: Source text
            overlap_size: Size of overlap in characters
            
        Returns:
            Overlap text
        """
        if len(text) <= overlap_size:
            return text
        
        # Take the last overlap_size characters and find word boundary
        overlap_text = text[-overlap_size:]
        
        # Find the first space to avoid cutting words
        first_space = overlap_text.find(' ')
        if first_space != -1:
            overlap_text = overlap_text[first_space:].strip()
        
        return overlap_text