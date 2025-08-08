"""
Logging configuration for the application
"""
import logging
import sys
from typing import Any, Dict
from app.core.config import settings

def setup_logging():
    """
    Setup logging configuration for the application
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hackrx_system.log')
        ]
    )
    
    # Set specific logger levels
    loggers = {
        'uvicorn': logging.INFO,
        'fastapi': logging.INFO,
        'httpx': logging.WARNING,
        'openai': logging.WARNING,
        'pinecone': logging.INFO,
    }
    
    for logger_name, level in loggers.items():
        logging.getLogger(logger_name).setLevel(level)
    
    # Create application logger
    logger = logging.getLogger("hackrx")
    logger.info("Logging configured successfully")

class LoggerMixin:
    """
    Mixin class to add logging functionality to other classes
    """
    
    @property
    def logger(self):
        """Get logger instance for the class"""
        return logging.getLogger(self.__class__.__name__)