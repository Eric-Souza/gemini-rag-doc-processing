import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas import ChatRequest
from app.services.agent import stream_agent
from app.services.rag import stream_rag

router = APIRouter(tags=["chat"])


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = {"event": event, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            if body.mode == "agent":
                stream = stream_agent(body.message, body.document_ids)
            else:
                stream = stream_rag(body.message, body.document_ids)
            async for event_name, payload in stream:
                yield _sse(event_name, payload)
        except Exception as exc:  # noqa: BLE001 — surface to client over SSE
            yield _sse("error", {"message": str(exc)})
            yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
