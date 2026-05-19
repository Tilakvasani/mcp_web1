"""
rag/selector.py
===============
Given a user message + all tools for an app,
returns only the top-k most relevant tools using RAG.

Flow:
  1. index_tools() — called once when tools are loaded (idempotent)
  2. select_tools() — called on every message, returns top-k tools
"""

from config.settings import RAG_TOP_K
from rag.store import index_tools, search_tools
from core.logger import log


def prepare(app_name: str, tools: list) -> None:
    """
    Index tools for an app into ChromaDB.
    Safe to call multiple times — re-indexes cleanly.
    """
    index_tools(app_name, tools)


def select(app_name: str, query: str, all_tools: list, top_k: int = RAG_TOP_K) -> list:
    """
    Return the top_k most relevant tools for this query.

    Falls back to returning all tools if RAG returns nothing.
    Always includes 'generic' tools (search, list, get_properties etc.)
    that work across all object types.
    """
    if not all_tools:
        return []

    # Get top-k tool names from vector search
    top_names = set(search_tools(app_name, query, top_k=top_k))

    # Always include generic tools (they work for any query)
    generic    = _get_generic_tools(all_tools)
    rag_tools  = [t for t in all_tools if getattr(t, "name", "") in top_names]
    combined   = _dedupe(rag_tools + generic)

    if not combined:
        log("warn", f"RAG returned nothing for '{query[:40]}' — using all tools")
        return all_tools

    log("rag", f"{app_name}: {len(all_tools)} tools → {len(combined)} selected")
    return combined


def _get_generic_tools(tools: list) -> list:
    """
    Tools that work across all object types — always include these.
    Examples: search_crm_objects, get_properties, list_modules
    """
    generic_patterns = [
        "search", "list", "get_propert", "get_field",
        "get_module", "query", "find",
    ]
    result = []
    for t in tools:
        name = getattr(t, "name", "").lower()
        if any(p in name for p in generic_patterns):
            result.append(t)
    return result


def _dedupe(tools: list) -> list:
    seen  = set()
    deduped = []
    for t in tools:
        name = getattr(t, "name", id(t))
        if name not in seen:
            seen.add(name)
            deduped.append(t)
    return deduped
