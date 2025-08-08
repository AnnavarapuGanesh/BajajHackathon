"""
Helper utilities for the application
"""
import hashlib
import time
from typing import Any, Dict, List, Optional
import re

def generate_document_id(url: str) -> str:
    """
    Generate a unique document ID based on URL
    
    Args:
        url: Document URL
        
    Returns:
        Unique document ID
    """
    return hashlib.md5(url.encode()).hexdigest()

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"\\|?*]', '_', filename)
    sanitized = re.sub(r'[^\w\s-.]', '', sanitized)
    return sanitized.strip()

def calculate_processing_time(start_time: float) -> float:
    """
    Calculate processing time from start time
    
    Args:
        start_time: Start timestamp
        
    Returns:
        Processing time in seconds
    """
    return time.time() - start_time

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def extract_domain_from_url(url: str) -> str:
    """
    Extract domain from URL
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc

def validate_url(url: str) -> bool:
    """
    Validate if string is a valid URL
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def clean_text_for_processing(text: str) -> str:
    """
    Clean text for better processing
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\/]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

def create_error_response(error_type: str, message: str, details: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_type: Type of error
        message: Error message
        details: Optional additional details
        
    Returns:
        Error response dictionary
    """
    response = {
        "error": error_type,
        "message": message,
        "timestamp": time.time()
    }
    
    if details:
        response["details"] = details
    
    return response

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def calculate_similarity_score(score: float) -> str:
    """
    Convert similarity score to descriptive text
    
    Args:
        score: Similarity score (0.0 to 1.0)
        
    Returns:
        Descriptive text
    """
    if score >= 0.9:
        return "Excellent"
    elif score >= 0.8:
        return "Very Good"
    elif score >= 0.7:
        return "Good"
    elif score >= 0.6:
        return "Fair"
    elif score >= 0.5:
        return "Poor"
    else:
        return "Very Poor"

def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."