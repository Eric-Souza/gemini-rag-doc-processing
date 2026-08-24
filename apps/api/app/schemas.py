from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models import DocumentStatus


class HealthResponse(BaseModel):
    status: str
    gemini_configured: bool
    chat_model: str = ""
    embed_model: str = ""


class DocumentOut(BaseModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    error_message: str | None = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    document_ids: list[str] | None = None
    mode: Literal["rag", "agent"] = "rag"


class CitationOut(BaseModel):
    document_id: str
    filename: str
    excerpt: str
    score: float | None = None
