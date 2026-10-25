"""Optional Langfuse tracing.

Enabled when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set. Each request becomes one
trace: retrieval stages are spans (embed, vector, keyword, rrf, rerank) and LLM calls are
generations recorded by Langfuse's LangChain callback, with token usage and cost. With the
keys unset every helper here is a no-op, so the app runs the same without Langfuse.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from app.config import get_settings


class _NoopObservation:
    def update(self, **_kwargs: Any) -> "_NoopObservation":
        return self


_NOOP = _NoopObservation()


@lru_cache
def get_langfuse():
    """The Langfuse client, or None when tracing isn't configured."""
    settings = get_settings()
    if not settings.langfuse_configured:
        return None
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
    )


@contextmanager
def observe(name: str, as_type: str = "span", **fields: Any) -> Iterator[Any]:
    """A Langfuse observation nested under the current one. Call .update(output=...) on the
    yielded object to record results."""
    client = get_langfuse()
    if client is None:
        yield _NOOP
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **fields) as observation:
        yield observation


@contextmanager
def trace(name: str, *, tags: list[str] | None = None, metadata: dict[str, Any] | None = None, **fields: Any) -> Iterator[Any]:
    """The root observation of a request, with trace-level name, tags, and metadata."""
    client = get_langfuse()
    if client is None:
        yield _NOOP
        return
    from langfuse import propagate_attributes

    # Propagated values are capped at 200 characters, so lists (document_ids) stay on the root only
    attributes = {k: v for k, v in (metadata or {}).items() if isinstance(v, str | int | float | bool)}
    with client.start_as_current_observation(name=name, as_type="span", metadata=metadata, **fields) as root:
        with propagate_attributes(trace_name=name, tags=tags, metadata=attributes):
            yield root


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """Callbacks that record LangChain LLM calls as generations in the current trace."""
    if get_langfuse() is None:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        client.flush()
