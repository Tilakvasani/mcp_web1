"""
core/session_cache.py
=====================
Session-level tool cache.

Loads ALL tools from an MCP client ONCE per session (not per request).
TTL = 10 minutes. Stale or disconnected clients evict automatically.

Key design:
  - Tool closures capture a *client_resolver* callable, not the client
    directly. This means tools survive MCPClient.reconnect() and even
    full pool eviction + recreation.

Structure:
  _CACHE[session_id][agent_key] = SessionEntry(tools, loaded_at)
"""

from __future__ import annotations
import time
import asyncio
import json
from typing import Any, Optional, Callable
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
    client,                          # MCPClient (used only for list_tools)
    client_resolver : Callable = None,  # () -> MCPClient  (used by tool closures)
) -> list:
    async with _LOCK:
        session = _CACHE.setdefault(session_id, {})
        entry   = session.get(agent_key)

        if entry and not entry.is_stale():
            log("cache", f"tool cache HIT  session={session_id[:8]} agent={agent_key} ({len(entry.tools)} tools)")
            return entry.tools

    log("cache", f"tool cache MISS session={session_id[:8]} agent={agent_key} -- loading...")

    # Use global schema cache if available (skip MCP round-trip)
    raw_tools = _TOOL_SCHEMAS.get(agent_key)
    schema_cached = False
    if raw_tools:
        log("cache", f"tool schema cache HIT for {agent_key} ({len(raw_tools)} schemas)")
        schema_cached = True
    else:
        try:
            raw_tools = await client.list_tools()
            _TOOL_SCHEMAS[agent_key] = raw_tools
            log("cache", f"tool schema cache STORE for {agent_key} ({len(raw_tools)} schemas)")
        except Exception as e:
            log("error", f"list_tools failed for {agent_key}: {e}")
            return []

    # If no resolver provided, fall back to direct client reference
    resolver = client_resolver or (lambda: client)
    lc_tools = _to_langchain_tools(raw_tools, resolver, verbose=not schema_cached)

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
# JSON Schema -> Pydantic model (no MCPClient in schema)
# ---------------------------------------------------------------------------

def _json_type_to_python(prop: dict) -> type:
    type_str = prop.get("type", "string")
    if isinstance(type_str, list):
        non_null = [t for t in type_str if t != "null"]
        type_str = non_null[0] if non_null else "string"
    return {
        "string" : str,
        "integer": int,
        "number" : float,
        "boolean": bool,
        "array"  : list,
        "object" : dict,
    }.get(type_str, Any)


def _build_args_schema(schema: dict, tool_name: str):
    """
    Build a proper Pydantic model from the MCP tool's inputSchema.
    This gives the LLM a real parameter spec so it knows exactly what to send.
    No MCPClient reference in this model.
    """
    properties = schema.get("properties", {})
    required   = set(schema.get("required", []))
    fields: dict[str, tuple] = {}

    for param_name, prop in properties.items():
        py_type     = _json_type_to_python(prop)
        description = prop.get("description", "")
        enum_vals   = prop.get("enum")

        # Enrich description with allowed values if present
        if enum_vals:
            description = f"{description} Allowed values: {', '.join(str(v) for v in enum_vals)}"

        if param_name in required:
            fields[param_name] = (py_type, Field(..., description=description))
        else:
            fields[param_name] = (Optional[py_type], Field(default=None, description=description))

    # Fallback: at least one field so the model is valid
    if not fields:
        fields["input"] = (Optional[str], Field(default=None, description="Tool input"))

    safe_name = tool_name.replace("-", "_").replace(".", "_")
    return create_model(f"{safe_name}_Args", **fields)


# ---------------------------------------------------------------------------
# LangChain tool wrapper
# ---------------------------------------------------------------------------

def _make_tool_fn(client_resolver: Callable, tool_name: str):
    """
    Return an async callable that calls client.call_tool(tool_name, kwargs).

    IMPORTANT: client_resolver is a callable that returns the *current*
    MCPClient at call-time.  This means the tool keeps working even after
    MCPClient.reconnect() creates a new transport, or after the pool
    evicts and recreates the client entirely.
    """
    async def _call(**kwargs) -> str:
        # Strip None values so MCP server doesn't complain about missing fields
        clean = {k: v for k, v in kwargs.items() if v is not None and k != "input"}
        try:
            client = client_resolver()
            result = await client.call_tool(tool_name, clean)
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
            return f"Tool error ({tool_name}): {e}"

    return _call


def _to_langchain_tools(mcp_tools: list, client_resolver: Callable, verbose: bool = True) -> list:
    """
    Wrap raw MCP tool definitions as LangChain StructuredTools.

    Key design choices:
    - client_resolver is called at tool invocation time (not wrapping time)
    - args_schema is built from inputSchema -> LLM gets real param spec
    - verbose=False suppresses per-tool logs (used on schema cache HIT)
    """
    lc_tools = []
    for tool in mcp_tools:
        name   = getattr(tool, "name",        "") or ""
        desc   = getattr(tool, "description", "") or ""
        schema = getattr(tool, "inputSchema", {}) or {}

        if not name:
            continue

        try:
            args_schema = _build_args_schema(schema, name)
            call_fn     = _make_tool_fn(client_resolver, name)

            lc_tool = StructuredTool(
                name        = name,
                description = desc,
                args_schema = args_schema,
                coroutine   = call_fn,
            )
            lc_tools.append(lc_tool)
            if verbose:
                log("tool", f"wrapped: {name} ({len(schema.get('properties', {}))} params)")

        except Exception as e:
            log("warn", f"could not wrap tool {name}: {e}")

    return lc_tools


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