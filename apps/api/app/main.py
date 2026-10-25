import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import apply_langsmith_env, get_settings
from app.db import init_db
from app.routers import chat, documents, health
from app.services.retrieval.rerank import warm_up_reranker
from app.services.tracing import flush as flush_traces


@asynccontextmanager
async def lifespan(_app: FastAPI):
    apply_langsmith_env()
    await init_db()
    if get_settings().retrieval_mode == "hybrid_rerank":
        # Load the cross-encoder now so the first chat request isn't slow
        await asyncio.to_thread(warm_up_reranker)
    yield
    # Send buffered Langfuse spans before the process exits
    flush_traces()


def create_app() -> FastAPI:
    settings = get_settings()
    apply_langsmith_env(settings)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="Doc Processing AI", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()
