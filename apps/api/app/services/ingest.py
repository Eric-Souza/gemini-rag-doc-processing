from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from app.config import get_settings
from app.db import AsyncSessionLocal
from app.models import Document, DocumentStatus
from app.services.vectorstore import get_vectorstore


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
            lc_docs = [
                LCDocument(
                    page_content=chunk,
                    metadata={
                        "document_id": doc.id,
                        "filename": doc.filename,
                        "chunk_index": i,
                    },
                )
                for i, chunk in enumerate(chunks)
            ]

            store = get_vectorstore()
            # Remove any prior vectors for this doc (re-ingest safety)
            try:
                store.delete(where={"document_id": doc.id})
            except Exception:
                pass
            ids = [f"{doc.id}_{i}" for i in range(len(lc_docs))]
            store.add_documents(lc_docs, ids=ids)

            doc.chunk_count = len(lc_docs)
            doc.status = DocumentStatus.ready
            doc.error_message = None
            doc.updated_at = _utcnow()
            await db.commit()
        except Exception as exc:  # noqa: BLE001
            doc.status = DocumentStatus.failed
            doc.error_message = str(exc)
            doc.updated_at = _utcnow()
            await db.commit()
