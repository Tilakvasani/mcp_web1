"""
rag/tool_indexer.py
===================
Indexes tool descriptions into Chroma when a user connects,
removes them when the user disconnects or the session ends.

Called from:
  web_app.py → after HubSpot / Zoho People client connects
  web_app.py → on disconnect / session end
"""

from __future__ import annotations
from rag.embeddings   import embed_tools
from rag.chroma_store import index_tools, remove_session, is_indexed, _tool_doc
from crm_logger import log


async def index_session_tools(session_id: str, tools: list) -> bool:
    """
    Embed all tool descriptions and push into Chroma in ONE batch call.
    Skips if already indexed (won't re-index unless force=True).
    Returns True on success.
    """
    if not tools:
        log("rag", f"no tools to index for session={session_id[:8]}")
        return False

    if is_indexed(session_id):
        log("rag", f"already indexed for session={session_id[:8]} — skip")
        return True

    log("rag", f"indexing {len(tools)} tools for session={session_id[:8]}…")

    # Build text documents for embedding
    docs = [_tool_doc(t) for t in tools]

    try:
        embeddings = await embed_tools(docs)
    except Exception as e:
        log("error", f"embed_tools failed for session={session_id[:8]}: {e}")
        return False

    count = index_tools(session_id, tools, embeddings)
    log("rag", f"✅ indexed {count} tools for session={session_id[:8]}")
    return count > 0


async def reindex_session_tools(session_id: str, tools: list) -> bool:
    """Force re-index — used when tools change (e.g. HubSpot scope change)."""
    remove_session(session_id)
    return await index_session_tools(session_id, tools)


def deindex_session_tools(session_id: str):
    """Remove all tool vectors for this session from Chroma. Called on disconnect."""
    remove_session(session_id)
    log("rag", f"🗑️  deindexed session={session_id[:8]}")
