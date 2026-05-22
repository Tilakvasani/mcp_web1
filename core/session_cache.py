"""
core/session_cache.py
=====================
Session-level tool cache.

Loads ALL tools from an MCP client ONCE per session (not per request).
TTL = 10 minutes. Stale or disconnected clients evict automatically.

Key design:
  - Tool closures capture a *client_resolver* callable, not the client
    directly. This was used by the old MCPClient pool. Now kept for context only.

Structure:
  _CACHE[session_id][agent_key] = SessionEntry(tools, loaded_at)
"""

from __future__ import annotations
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass, field

from pydantic import create_model, Field
from langchain_core.tools import StructuredTool
from crm_logger import log

_TTL = 600  # 10 minutes


@dataclass
class _Entry:
    tools     : list
    loaded_at : float = field(default_factory=time.time)

    def is_stale(self) -> bool:
        return time.time() - self.loaded_at > _TTL


# session_id -> { agent_key -> _Entry }
_CACHE: dict[str, dict[str, _Entry]] = {}
_LOCK  = asyncio.Lock()

# Global tool schema cache — raw MCP tool schemas keyed by agent_key.
# Avoids calling list_tools() on the MCP server for every new session.
_TOOL_SCHEMAS: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_or_load_tools(
    session_id      : str,
    agent_key       : str,
    client,                          # MCPManager instance
    client_resolver : Callable = None,  # kept for API compat, unused
) -> list:
    async with _LOCK:
        session = _CACHE.setdefault(session_id, {})
        entry   = session.get(agent_key)

        if entry and not entry.is_stale():
            log("cache", f"tool cache HIT  session={session_id[:8]} agent={agent_key} ({len(entry.tools)} tools)")
            return entry.tools

    log("cache", f"tool cache MISS session={session_id[:8]} agent={agent_key} -- loading...")

    # Load tools via MCPManager (LangChain MCP adapters — reconnects per call)
    # _TOOL_SCHEMAS is now keyed by tool name for schema dedup logging
    try:
        lc_tools = await client.get_tools([agent_key])
        # Update schema cache with tool names for stats/logging
        tool_names = [getattr(t, "name", "") for t in lc_tools]
        _TOOL_SCHEMAS[agent_key] = tool_names
        log("cache", f"tool schema cache STORE for {agent_key} ({len(lc_tools)} tools)")
    except Exception as e:
        log("error", f"get_tools failed for {agent_key}: {e}")
        return []

    # Inject synthetic tools for agents with limited MCP tool sets
    synthetic = _build_synthetic_tools(agent_key)
    if synthetic:
        lc_tools.extend(synthetic)
        log("tool", f"injected {len(synthetic)} synthetic tool(s) for {agent_key}")

    async with _LOCK:
        _CACHE.setdefault(session_id, {})[agent_key] = _Entry(tools=lc_tools)

    log("ok", f"loaded {len(lc_tools)} tools for session={session_id[:8]} agent={agent_key}")
    return lc_tools


async def evict_session(session_id: str):
    async with _LOCK:
        _CACHE.pop(session_id, None)
    log("cache", f"evicted session={session_id[:8]}")


async def evict_stale_sessions():
    async with _LOCK:
        stale_sessions = [
            sid for sid, agents in _CACHE.items()
            if all(e.is_stale() for e in agents.values())
        ]
        for sid in stale_sessions:
            del _CACHE[sid]
    if stale_sessions:
        log("cache", f"evicted {len(stale_sessions)} stale sessions")


async def evict_all_sessions():
    async with _LOCK:
        count = len(_CACHE)
        _CACHE.clear()
        _TOOL_SCHEMAS.clear()
    log("cache", f"evicted all {count} sessions + tool schema cache")


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




# ---------------------------------------------------------------------------
# Synthetic tools (injected for agents with limited MCP tool sets)
# ---------------------------------------------------------------------------

def _build_synthetic_tools(agent_key: str) -> list:
    """
    Build extra LangChain tools that aren't in the MCP server.
    Currently adds ZohoPeople_callAPI for the zoho_people agent
    when OAuth credentials are configured.
    """
    if agent_key != "zoho_people":
        return []

    try:
        from apps.zoho_people.zoho_people_oauth import is_oauth_configured
        if not is_oauth_configured():
            log("tool", "Zoho People OAuth not configured -- skipping ZohoPeople_callAPI")
            return []
    except ImportError:
        return []

    from apps.zoho_people.zoho_people_api import zoho_people_call_api

    # Build Pydantic args schema
    CallAPIArgs = create_model(
        "ZohoPeople_callAPI_Args",
        method=(str, Field(
            default="GET",
            description="HTTP method: GET, POST, PUT, DELETE"
        )),
        path=(str, Field(
            ...,
            description=(
                "Zoho People REST API path (after /people/api). Examples: "
                "/forms/employee/getRecords, /forms/department/getRecords, "
                "/forms/leave/getRecords, /forms/employee/getRecords?searchColumn=Employeestatus&searchValue=Active"
            )
        )),
        params=(Optional[dict], Field(
            default=None,
            description=(
                "Query parameters dict. Examples: "
                "{\"sIndex\": 1, \"limit\": 200}, "
                "{\"searchColumn\": \"Department\", \"searchValue\": \"Engineering\"}"
            )
        )),
        data=(Optional[dict], Field(
            default=None,
            description="JSON body for POST/PUT requests"
        )),
    )

    tool = StructuredTool(
        name="ZohoPeople_callAPI",
        description=(
            "Call any Zoho People REST API endpoint directly. Use this for bulk queries "
            "the other tools cannot handle. Path is relative to /people/api. "
            "ALWAYS make only ONE call with the correct parameters. "
            "EXACT API reference: "
            "List all employees: path=/forms/employee/getRecords, params={}. "
            "Active employees: path=/forms/employee/getRecords, params={\"searchColumn\": \"Employeestatus\", \"searchValue\": \"Active\"}. "
            "Employees by dept: path=/forms/employee/getRecords, params={\"searchColumn\": \"Department\", \"searchValue\": \"<dept_name>\"}. "
            "Search by name: path=/forms/employee/getRecords, params={\"searchColumn\": \"EMPLOYEENAME\", \"searchValue\": \"<name>\"}. "
            "List departments: path=/forms/department/getRecords, params={}. "
            "List leave records: path=/forms/leave/getRecords, params={}. "
            "Pagination: add \"sIndex\": 1, \"limit\": 200 to params. "
            "IMPORTANT: searchColumn is case-SENSITIVE. Use exactly: Employeestatus, Department, EMPLOYEENAME. "
            "Do NOT guess or retry with different casing."
        ),
        args_schema=CallAPIArgs,
        coroutine=zoho_people_call_api,
    )

    log("tool", "built synthetic: ZohoPeople_callAPI")
    return [tool]