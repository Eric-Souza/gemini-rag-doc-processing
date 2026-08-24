from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.services.llm import get_embeddings

COLLECTION_NAME = "documents"


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_path,
    )


def delete_document_vectors(document_id: str) -> None:
    store = get_vectorstore()
    store.delete(where={"document_id": document_id})
