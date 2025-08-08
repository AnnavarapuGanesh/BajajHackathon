"""
API v1 router configuration
"""
from fastapi import APIRouter

from app.api.v1.endpoints import hackrx
from app.core.config import settings

api_router = APIRouter()

# Include HackRx endpoints
api_router.include_router(
    hackrx.router, 
    prefix="/hackrx", 
    tags=["hackrx"]
)