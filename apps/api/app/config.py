from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve paths relative to apps/api (this file lives in apps/api/app/)
API_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = API_ROOT.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(REPO_ROOT / ".env"),
            str(API_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-3.6-flash"
    # Evals (answers + judge) run on a lite model: the free tier allows far more requests per day
    gemini_eval_model: str = "gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"
    # Postgres + pgvector (see docker-compose.yml at the repo root)
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"
    # gemini-embedding-001 supports reduced output sizes; 768 keeps HNSW indexes small
    embed_dim: int = 768

    # Retrieval: "vector", "hybrid" (full-text + vector, RRF), or "hybrid_rerank"
    retrieval_mode: str = "hybrid_rerank"
    retrieval_candidates: int = 20
    rrf_k: int = 60
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    upload_dir: str = str(API_ROOT / "data" / "uploads")
    cors_origins: str = "http://localhost:5173"

    langsmith_api_key: str = ""
    langchain_api_key: str = ""
    langsmith_tracing: bool | None = None
    langsmith_project: str = "doc-processing-ai"

    @model_validator(mode="after")
    def _resolve_langsmith(self) -> "Settings":
        if not self.langsmith_api_key.strip() and self.langchain_api_key.strip():
            self.langsmith_api_key = self.langchain_api_key
        if self.langsmith_tracing is None:
            object.__setattr__(
                self,
                "langsmith_tracing",
                bool(self.langsmith_api_key.strip()),
            )
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def langsmith_configured(self) -> bool:
        return bool(self.langsmith_api_key.strip()) and bool(self.langsmith_tracing)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def apply_langsmith_env(settings: Settings | None = None) -> None:
    """Export LangSmith env vars so LangChain/LangGraph auto-trace."""
    import os

    s = settings or get_settings()
    if s.langsmith_configured:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = s.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = s.langsmith_project
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
