from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.services.llm import get_chat_model
from app.services.rag import _run_config
from app.services.retrieval import retrieve

_active_document_ids: list[str] | None = None
_collected_citations: list[dict[str, Any]] = []


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
async def retrieve_documents(query: str) -> str:
    """Search uploaded documents for passages relevant to the query."""
    hits = await retrieve(query, k=4, document_ids=_active_document_ids)
    if not hits:
        return "No relevant passages found."
    parts: list[str] = []
    for i, hit in enumerate(hits, start=1):
        _collected_citations.append(
            {
                "document_id": hit.document_id,
                "filename": hit.filename,
                "excerpt": hit.content[:400],
                "score": hit.score,
            }
        )
        parts.append(f"[{i}] {hit.filename} (score={hit.score:.3f})\n{hit.content}")
    return "\n\n".join(parts)


def _build_graph():
    tools = [retrieve_documents]
    llm = get_chat_model().bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent_node(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = await llm.ainvoke(state["messages"])
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
    global _active_document_ids, _graph
    _active_document_ids = document_ids
    _collected_citations.clear()
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
    result = await get_agent_graph().ainvoke(
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
