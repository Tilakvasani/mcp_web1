"""
core/session_cache.py
=====================
Session-level tool cache.

Loads ALL tools from an MCP client ONCE per session (not per request).
TTL = 10 minutes. Stale or disconnected clients evict automatically.

Structure:
  _CACHE[session_id][agent_key] = SessionEntry(tools, loaded_at)
"""

from __future__ import annotations
import time
import asyncio
from dataclasses import dataclass, field
from crm_logger import log

_TTL = 600  # 10 minutes


@dataclass
class _Entry:
    tools     : list
    loaded_at : float = field(default_factory=time.time)

    def is_stale(self) -> bool:
        return time.time() - self.loaded_at > _TTL


# session_id → { agent_key → _Entry }
_CACHE: dict[str, dict[str, _Entry]] = {}
_LOCK  = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def get_or_load_tools(
    session_id : str,
    agent_key  : str,
    client,            # MCPClient
) -> list:
    """
    Return cached tools for (session_id, agent_key).
    If not cached or stale, connect to MCP and load fresh tools.
    """
    async with _LOCK:
        session = _CACHE.setdefault(session_id, {})
        entry   = session.get(agent_key)

        if entry and not entry.is_stale():
            log("cache", f"tool cache HIT  session={session_id[:8]} agent={agent_key} ({len(entry.tools)} tools)")
            return entry.tools

    # Outside lock — load tools (may be slow)
    log("cache", f"tool cache MISS session={session_id[:8]} agent={agent_key} — loading…")
    try:
        raw_tools = await client.list_tools()
    except Exception as e:
        log("error", f"list_tools failed for {agent_key}: {e}")
        return []

    lc_tools = _to_langchain_tools(raw_tools, client)

    async with _LOCK:
        _CACHE.setdefault(session_id, {})[agent_key] = _Entry(tools=lc_tools)

    log("ok", f"loaded {len(lc_tools)} tools for session={session_id[:8]} agent={agent_key}")
    return lc_tools


async def evict_session(session_id: str):
    """Evict all tool cache entries for a session (on disconnect)."""
    async with _LOCK:
        _CACHE.pop(session_id, None)
    log("cache", f"evicted session={session_id[:8]}")


async def evict_stale_sessions():
    """Periodic cleanup — remove sessions where all entries are stale."""
    async with _LOCK:
        stale_sessions = []
        for sid, agents in _CACHE.items():
            if all(e.is_stale() for e in agents.values()):
                stale_sessions.append(sid)
        for sid in stale_sessions:
            del _CACHE[sid]
    if stale_sessions:
        log("cache", f"evicted {len(stale_sessions)} stale sessions")


async def evict_all_sessions():
    async with _LOCK:
        count = len(_CACHE)
        _CACHE.clear()
    log("cache", f"evicted all {count} sessions")


def get_cache_stats() -> dict:
    total_tools = sum(
        len(entry.tools)
        for agents in _CACHE.values()
        for entry in agents.values()
    )
    return {
        "sessions"   : len(_CACHE),
        "total_tools": total_tools,
        "ttl_seconds": _TTL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool wrapper
# ─────────────────────────────────────────────────────────────────────────────

def _to_langchain_tools(mcp_tools: list, client) -> list:
    """Wrap raw MCP tool definitions as LangChain-compatible callable tools."""
    from langchain_core.tools import StructuredTool
    import json

    lc_tools = []
    for tool in mcp_tools:
        name = getattr(tool, "name", "")
        desc = getattr(tool, "description", "") or ""
        schema = getattr(tool, "inputSchema", {}) or {}

        async def _call(client=client, name=name, **kwargs) -> str:
            try:
                result = await client.call_tool(name, kwargs)
                if result is None:
                    return "No result returned."
                content = getattr(result, "content", result)
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        text = getattr(c, "text", None)
                        if text:
                            parts.append(text)
                        elif isinstance(c, dict):
                            parts.append(json.dumps(c))
                    return "\n".join(parts) if parts else str(content)
                return str(content)
            except Exception as e:
                return f"Tool error: {e}"

        # Build args_schema from inputSchema if available
        props    = schema.get("properties", {})
        required = schema.get("required", [])

        try:
            lc_tool = StructuredTool.from_function(
                coroutine    = _call,
                name         = name,
                description  = desc,
                infer_schema = not bool(props),
            )
            lc_tools.append(lc_tool)
        except Exception as e:
            log("warn", f"could not wrap tool {name}: {e}")

    return lc_tools
