from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.services.llm import get_chat_model
from app.services.rag import _run_config, retrieve

_active_document_ids: list[str] | None = None
_collected_citations: list[dict[str, Any]] = []


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def retrieve_documents(query: str) -> str:
    """Search uploaded documents for passages relevant to the query."""
    global _collected_citations
    hits = retrieve(query, document_ids=_active_document_ids, k=4)
    if not hits:
        return "No relevant passages found."
    parts: list[str] = []
    for i, (doc, score) in enumerate(hits, start=1):
        citation = {
            "document_id": doc.metadata.get("document_id", ""),
            "filename": doc.metadata.get("filename", "unknown"),
            "excerpt": doc.page_content[:400],
            "score": float(score) if score is not None else None,
        }
        _collected_citations.append(citation)
        parts.append(
            f"[{i}] {citation['filename']} (score={score:.3f})\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def _build_graph():
    tools = [retrieve_documents]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)

    def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_graph = None


def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


async def stream_agent(
    message: str,
    document_ids: list[str] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    global _active_document_ids, _collected_citations, _graph
    _active_document_ids = document_ids
    _collected_citations = []
    # Rebuild graph so tool closure / model pick up current settings after key changes
    _graph = _build_graph()

    yield ("status", {"message": "Planning and retrieving…"})

    system = SystemMessage(
        content=(
            "You are a document Q&A agent. Use the retrieve_documents tool to find evidence "
            "before answering. If retrieval finds nothing useful, say so clearly. "
            "Do not invent facts not present in retrieved passages."
        )
    )
    config = _run_config("agent", document_ids=document_ids)
    result = get_agent_graph().invoke(
        {"messages": [system, HumanMessage(content=message)]},
        config=config,
    )

    seen: set[tuple[str, str]] = set()
    for c in _collected_citations:
        key = (c.get("document_id", ""), c.get("excerpt", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        yield ("citation", c)

    final = result["messages"][-1]
    text = final.content if isinstance(final, AIMessage) else str(final.content)
    if isinstance(text, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in text
        )

    yield ("status", {"message": "Streaming answer…"})

    # Stream in small chunks for UX parity with RAG
    content = str(text or "")
    chunk_size = 24
    for i in range(0, len(content), chunk_size):
        yield ("token", {"text": content[i : i + chunk_size]})

    yield ("done", {})
