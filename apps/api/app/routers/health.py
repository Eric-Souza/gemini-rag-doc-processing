from fastapi import APIRouter

from app.config import get_settings
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        gemini_configured=settings.gemini_configured,
        langsmith_configured=settings.langsmith_configured,
        langfuse_configured=settings.langfuse_configured,
        chat_model=settings.gemini_chat_model,
        embed_model=settings.gemini_embed_model,
        retrieval_mode=settings.retrieval_mode,
    )
