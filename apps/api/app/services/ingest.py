from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import delete

from app.config import get_settings
from app.db import AsyncSessionLocal
from app.models import Chunk, Document, DocumentStatus
from app.services.llm import embed_documents


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    return path.read_text(encoding="utf-8", errors="replace").strip()


async def ingest_document(document_id: str) -> None:
    settings = get_settings()
    async with AsyncSessionLocal() as db:
        doc = await db.get(Document, document_id)
        if not doc:
            return
        doc.status = DocumentStatus.indexing
        doc.error_message = None
        doc.updated_at = _utcnow()
        await db.commit()

        suffix = Path(doc.filename).suffix.lower() or ".bin"
        file_path = Path(settings.upload_dir) / f"{doc.id}{suffix}"

        try:
            text = _load_text(file_path)
            if not text:
                raise ValueError("No extractable text in file")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_text(text)
            embeddings = await embed_documents(chunks)

            # Re-ingest safety: replace any chunks from a previous run
            await db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            db.add_all(
                Chunk(
                    id=f"{doc.id}_{i}",
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk,
                    embedding=embedding,
                )
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            )

            doc.chunk_count = len(chunks)
            doc.status = DocumentStatus.ready
            doc.error_message = None
            doc.updated_at = _utcnow()
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            await db.rollback()
            # Rollback expires loaded objects; reload before recording the failure
            doc = await db.get(Document, document_id)
            doc.status = DocumentStatus.failed
            doc.error_message = str(exc)
            doc.updated_at = _utcnow()
            await db.commit()
