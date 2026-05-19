"""
rag/chroma_store.py
===================
ChromaDB wrapper for tool vector storage.

Rules:
  - ONE collection per agent type -> "tools__hubspot", "tools__zoho_people"
  - ONLY tool vectors live here -- no other data ever
  - On first connect -> index_tools()   (create collection, push embeddings)
  - On disconnect    -> remove_agent()  (delete collection entirely)

Local persistent storage in ./chroma_data/ -- no cloud needed.
"""

import os
import re
import chromadb
from chromadb.config import Settings
from crm_logger import log

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_data")
_client: chromadb.ClientAPI | None = None

# In-memory set of agents whose tools are already indexed in Chroma.
# Avoids a Chroma round-trip on every request just to check.
_INDEXED: set[str] = set()


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

def _get_chroma() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path     = os.path.abspath(_DATA_DIR),
            settings = Settings(anonymized_telemetry=False),
        )
    return _client


def _col_name(agent_key: str) -> str:
    """Chroma collection names: alphanumeric + underscore, max 63 chars."""
    safe = re.sub(r"[^a-zA-Z0-9]", "_", agent_key)
    return f"tools__{safe}"[:63]


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def index_tools(
    agent_key: str,
    tools: list,
    embeddings: list[list[float]],
) -> int:
    """
    Store tool vectors in Chroma for this agent.
    Always replaces any existing collection for this agent.
    Returns number of tools indexed.
    """
    col_name = _col_name(agent_key)
    db       = _get_chroma()

    # Drop existing collection (clean slate on reconnect)
    try:
        db.delete_collection(col_name)
    except Exception:
        pass

    col = db.create_collection(
        name     = col_name,
        metadata = {"hnsw:space": "cosine"},
    )

    if not tools or not embeddings:
        log("rag", f"no tools to index for agent={agent_key}")
        return 0

    ids       = [f"t{i}" for i in range(len(tools))]
    documents = [_tool_doc(t) for t in tools]
    metadatas = [_tool_meta(t, agent_key) for t in tools]

    col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    _INDEXED.add(agent_key)
    log("rag", f"indexed {len(tools)} tools -> collection={col_name}")
    return len(tools)


def remove_agent(agent_key: str):
    """Delete the Chroma collection for this agent. Called on disconnect."""
    col_name = _col_name(agent_key)
    db       = _get_chroma()
    _INDEXED.discard(agent_key)
    try:
        db.delete_collection(col_name)
        log("rag", f"removed collection for agent={agent_key}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def search_tools(
    agent_key: str,
    query_vec: list[float],
    n_results: int = 15,
    agent_filter: str | None = None,
) -> list[str]:
    """
    Vector similarity search. Returns ranked list of tool names.
    agent_filter: 'hubspot' | 'zoho_people' | None (search all)
    """
    col_name = _col_name(agent_key)
    db       = _get_chroma()

    try:
        col = db.get_collection(col_name)
    except Exception:
        log("rag", f"no collection found for agent={agent_key}")
        return []

    count = col.count()
    if count == 0:
        return []

    k      = min(n_results, count)
    kwargs: dict = dict(
        query_embeddings = [query_vec],
        n_results        = k,
        include          = ["metadatas", "distances"],
    )
    if agent_filter:
        kwargs["where"] = {"agent": {"$eq": agent_filter}}

    results    = col.query(**kwargs)
    tool_names = [m["tool_name"] for m in results["metadatas"][0]]
    log("rag", f"vector search -> top {len(tool_names)} tools (filter={agent_filter})")
    return tool_names


def is_indexed(agent_key: str) -> bool:
    """True if tools are already indexed for this agent."""
    if agent_key in _INDEXED:
        return True
    # Fallback: check Chroma (cold start after restart)
    col_name = _col_name(agent_key)
    db       = _get_chroma()
    try:
        if db.get_collection(col_name).count() > 0:
            _INDEXED.add(agent_key)
            return True
    except Exception:
        pass
    return False


def get_index_count(agent_key: str) -> int:
    """Return number of tools indexed for this agent."""
    col_name = _col_name(agent_key)
    db       = _get_chroma()
    try:
        return db.get_collection(col_name).count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------

def _tool_doc(tool) -> str:
    """Text fed to the embedding model -- richer = better retrieval."""
    name = getattr(tool, "name", "")
    desc = (getattr(tool, "description", "") or "").strip()[:400]
    return f"Tool: {name}\nDescription: {desc}"


def _tool_meta(tool, agent_key: str = "hubspot") -> dict:
    """Metadata stored alongside the vector -- used for filtering."""
    name  = getattr(tool, "name", "")
    desc  = (getattr(tool, "description", "") or "").strip()[:200]
    return {
        "tool_name": name,
        "agent":     agent_key,
        "desc":      desc,
    }
