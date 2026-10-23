from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings


def clear_llm_caches() -> None:
    get_chat_model.cache_clear()
    get_embeddings.cache_clear()


@lru_cache
def get_chat_model(model: str | None = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    if not settings.gemini_configured:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see apps/api/README.md).")
    return ChatGoogleGenerativeAI(
        model=model or settings.gemini_chat_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.2,
    )


@lru_cache
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    if not settings.gemini_configured:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env (see apps/api/README.md).")
    model = settings.gemini_embed_model
    if model.startswith("models/"):
        model = model.removeprefix("models/")
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embed_dim,
    )


async def embed_query(text: str) -> list[float]:
    return await get_embeddings().aembed_query(text)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await get_embeddings().aembed_documents(texts)
