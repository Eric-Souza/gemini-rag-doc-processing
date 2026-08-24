from functools import lru_cache
from pathlib import Path

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
    gemini_embed_model: str = "gemini-embedding-001"
    database_url: str = f"sqlite+aiosqlite:///{(API_ROOT / 'data' / 'app.db').as_posix()}"
    chroma_path: str = str(API_ROOT / "data" / "chroma")
    upload_dir: str = str(API_ROOT / "data" / "uploads")
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
