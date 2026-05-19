"""
rag/store.py
============
Simple in-memory tool store — no chromadb needed.
Searches by keyword overlap between query and tool names/descriptions.
"""

from core.logger import log

# In-memory: app_name → list of {name, keywords}
_store: dict[str, list[dict]] = {}


def index_tools(app_name: str, tools: list) -> None:
    entries = []
    for tool in tools:
        name = getattr(tool, "name", str(tool))
        desc = (getattr(tool, "description", "") or "").strip()
        text = f"{name} {desc}".lower().replace("_", " ")
        keywords = set(text.split())
        entries.append({"name": name, "keywords": keywords})
    _store[app_name] = entries
    log("rag", f"Indexed {len(entries)} tools for {app_name}")


def search_tools(app_name: str, query: str, top_k: int = 5) -> list[str]:
    entries = _store.get(app_name, [])
    if not entries:
        return []
    query_words = set(query.lower().replace("_", " ").split())
    scored = sorted(entries, key=lambda e: len(query_words & e["keywords"]), reverse=True)
    top = [e["name"] for e in scored[:top_k]]
    log("rag", f"'{query[:40]}' → {top}")
    return top