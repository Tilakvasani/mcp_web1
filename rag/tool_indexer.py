"""
rag/tool_indexer.py
===================
Indexes tool descriptions into Chroma once per agent type (not per session).

Called from:
  core/agent.py -> after tools are loaded for hubspot / zoho_people
  web_app.py    -> on disconnect (deindex)
"""

from __future__ import annotations
from rag.embeddings   import embed_tools
from rag.chroma_store import index_tools, remove_agent, is_indexed, _tool_doc
from crm_logger import log


async def index_agent_tools(agent_key: str, tools: list) -> bool:
    """
    Embed all tool descriptions and push into Chroma in ONE batch call.
    Skips if already indexed for this agent (global, not per-session).
    Returns True on success.
    """
    if not tools:
        log("rag", f"no tools to index for agent={agent_key}")
        return False

    if is_indexed(agent_key):
        log("rag", f"already indexed for agent={agent_key} -- skip")
        return True

    log("rag", f"indexing {len(tools)} tools for agent={agent_key}...")

    # Build text documents for embedding
    docs = [_tool_doc(t) for t in tools]

    try:
        embeddings = await embed_tools(docs)
    except Exception as e:
        log("error", f"embed_tools failed for agent={agent_key}: {e}")
        return False

    count = index_tools(agent_key, tools, embeddings)
    log("rag", f"indexed {count} tools for agent={agent_key}")
    return count > 0


async def reindex_agent_tools(agent_key: str, tools: list) -> bool:
    """Force re-index -- used when tools change (e.g. HubSpot scope change)."""
    remove_agent(agent_key)
    return await index_agent_tools(agent_key, tools)


def deindex_agent_tools(agent_key: str):
    """Remove all tool vectors for this agent from Chroma. Called on disconnect."""
    remove_agent(agent_key)
    log("rag", f"deindexed agent={agent_key}")
