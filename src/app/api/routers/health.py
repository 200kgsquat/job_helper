# src/app/api/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["health"])
def health_check():
    """
    Simple healthcheck endpoint.
    Adjust body/status as you need.
    """
    return {"status": "ok"}
