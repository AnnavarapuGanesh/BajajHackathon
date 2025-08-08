"""
Configuration settings for the HackRx Intelligent Query System
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "HackRx Intelligent Query System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "LLM-Powered Intelligent Query-Retrieval System for document processing"
    
    # Server Settings
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Gemini AI Settings (Updated from OpenAI)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.1
    
    # OpenAI Settings (Kept for backward compatibility - optional)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-4"
    
    # Pinecone Settings
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    EMBEDDING_DIMENSION: int = 1536  # Compatible with both OpenAI and Gemini embeddings
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/hackrx_db")
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Bearer Token from document
    BEARER_TOKEN: str = os.getenv("BEARER_TOKEN", "4097f4513d2abce6e34765287189ca01fc5d42f566fdebecaa1aa41c52d83cca")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Document Processing
    MAX_DOCUMENT_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_DOCUMENT_TYPES: list = [".pdf", ".docx", ".doc", ".txt"]
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # LLM Provider Selection (NEW)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "openai"
    
    # Gemini-specific settings (NEW)
    GEMINI_SAFETY_SETTINGS: dict = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
    }
    
    # Rate limiting (NEW)
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "15"))
    
    class Config:
        case_sensitive = True


# Global settings instance
settings = Settings()
