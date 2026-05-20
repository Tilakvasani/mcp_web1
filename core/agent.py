"""
core/agent.py
=============
Central agent dispatcher.

Agents
------
  hubspot      -> HubSpot CRM only
  zoho_people  -> Zoho People HRMS only
  cross        -> HubSpot + Zoho People combined

Pipeline per request
--------------------
  1. pre_router    -- instant reply for greetings/help/off-topic (zero LLM cost)
  2. response_cache -- return cached answer for repeated read queries
  3. session_cache  -- load all tools from MCP (once per session, TTL 10 min)
  4. RAG index      -- embed tools once per agent type (global, not per-session)
  5. hybrid_search  -- keyword + Chroma vector search -> top K relevant tools
  6. LangGraph ReAct agent (runs selected tools, returns answer)
  7. response_cache.set -- store answer for next time
"""

from __future__ import annotations
import time
from crm_logger import log

from core.pre_router     import pre_route
from core.response_cache import get_cached, set_cached, clear_session as _clear_cache, cache_stats
from core.session_cache  import (
    get_or_load_tools,
    evict_session        as _cache_evict,
    evict_stale_sessions as _cache_evict_stale,
    evict_all_sessions   as _cache_evict_all,
    get_cache_stats      as _session_stats,
)
from rag.hybrid_search   import hybrid_search
from rag.tool_indexer    import index_agent_tools

from typing import AsyncGenerator
from agents.hubspot_agent      import run_hubspot_agent, run_hubspot_agent_stream
from agents.zoho_people_agent  import run_zoho_people_agent, run_zoho_people_agent_stream
from agents.cross_agent        import run_cross_agent, run_cross_agent_stream


# ---------------------------------------------------------------------------
# Private helpers for tools management
# ---------------------------------------------------------------------------

async def _load_and_index_tools(session_id: str, agent: str, clients: dict) -> tuple[list, list]:
    """
    Loads tools from session cache for HubSpot and Zoho People, indexes them in Chroma,
    and returns a tuple of (hs_tools, zp_tools).
    """
    hs_tools: list = []
    zp_tools: list = []

    if agent in ("hubspot", "cross") and "hubspot" in clients:
        hs_tools = await get_or_load_tools(
            session_id,
            "hubspot",
            clients["hubspot"],
            client_resolver=lambda: clients["hubspot"],
        )
        # Index into Chroma once per agent type (global)
        await index_agent_tools("hubspot", hs_tools)

    if agent in ("zoho_people", "cross") and "zoho_people" in clients:
        zp_tools = await get_or_load_tools(
            session_id,
            "zoho_people",
            clients["zoho_people"],
            client_resolver=lambda: clients["zoho_people"],
        )
        await index_agent_tools("zoho_people", zp_tools)

    return hs_tools, zp_tools


async def _select_relevant_tools(message: str, agent: str, hs_tools: list, zp_tools: list) -> list:
    """
    Selects the most relevant tools using hybrid search and force-injects ZohoPeople_callAPI
    if it is available.
    """
    all_tools = hs_tools + zp_tools
    if not all_tools:
        return []

    # Determine which agent-level index to search
    rag_key = "hubspot" if agent == "hubspot" else ("zoho_people" if agent == "zoho_people" else "cross")

    relevant_tools = await hybrid_search(
        message   = message,
        agent_key = rag_key,
        all_tools = all_tools,
    )

    # Force-inject ZohoPeople_callAPI if it exists in zp_tools
    call_api_tool = next((t for t in zp_tools if getattr(t, "name", "") == "ZohoPeople_callAPI"), None)
    if call_api_tool and call_api_tool not in relevant_tools:
        relevant_tools.append(call_api_tool)

    log("ai", f"hybrid -> {len(relevant_tools)}/{len(all_tools)} tools selected")
    return relevant_tools


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

async def run_agent(
    message      : str,
    history      : list[dict],
    clients      : dict,          # {"hubspot": MCPClient, "zoho_people": MCPClient}
    agent        : str,           # "hubspot" | "zoho_people" | "cross"
    granted_scopes: list[str],
    is_admin     : bool,
    session_id   : str = "default",
) -> str:

    t0 = time.time()

    # -- Step 1: pre-router (instant, zero cost) ----------------------------
    instant = pre_route(message)
    if instant:
        log("ai", f"pre_router handled '{message[:40]}'")
        return instant

    # -- Step 2: response cache ---------------------------------------------
    cached = get_cached(message, agent, session_id)
    if cached:
        log("ai", f"cache HIT in {time.time()-t0:.2f}s")
        return cached

    # -- Step 3: load tools from session cache ------------------------------
    hs_tools, zp_tools = await _load_and_index_tools(session_id, agent, clients)

    # -- Step 4: hybrid tool selection --------------------------------------
    relevant_tools = await _select_relevant_tools(message, agent, hs_tools, zp_tools)

    if not relevant_tools:
        return _no_tools_msg(agent, clients)

    # -- Step 5: run agent --------------------------------------------------
    if agent == "hubspot":
        answer = await run_hubspot_agent(
            message  = message,
            history  = history,
            tools    = relevant_tools,
            scopes   = granted_scopes,
            is_admin = is_admin,
            session_id = session_id,
        )

    elif agent == "zoho_people":
        answer = await run_zoho_people_agent(
            message = message,
            history = history,
            tools   = relevant_tools,
            session_id = session_id,
        )

    else:  # cross
        hs_set = set(id(t) for t in hs_tools)
        zp_set = set(id(t) for t in zp_tools)
        hs_relevant = [t for t in relevant_tools if id(t) in hs_set]
        zp_relevant = [t for t in relevant_tools if id(t) in zp_set]

        # Ensure each side has at least some tools
        if not hs_relevant:
            hs_relevant = hs_tools[:8]
        if not zp_relevant:
            zp_relevant = zp_tools[:8]

        answer = await run_cross_agent(
            message  = message,
            history  = history,
            hs_tools = hs_relevant,
            zp_tools = zp_relevant,
            scopes   = granted_scopes,
            is_admin = is_admin,
            session_id = session_id,
        )

    elapsed = time.time() - t0
    log("ok", f"agent={agent} | {len(answer)} chars | {elapsed:.1f}s")

    # -- Step 6: cache the answer -------------------------------------------
    set_cached(message, agent, session_id, answer)
    return answer


async def run_agent_stream(
    message      : str,
    history      : list[dict],
    clients      : dict,          # {"hubspot": MCPClient, "zoho_people": MCPClient}
    agent        : str,           # "hubspot" | "zoho_people" | "cross"
    granted_scopes: list[str],
    is_admin     : bool,
    session_id   : str = "default",
) -> AsyncGenerator[str, None]:
    """
    Core agent stream dispatcher. Walks through pre-router, cache checks, tool loading,
    semantic selection, and yields LLM tokens in real-time as they generate.
    """
    t0 = time.time()

    # -- Step 1: pre-router (instant, zero cost) ----------------------------
    instant = pre_route(message)
    if instant:
        log("ai", f"pre_router handled '{message[:40]}'")
        yield instant
        return

    # -- Step 2: response cache ---------------------------------------------
    cached = get_cached(message, agent, session_id)
    if cached:
        log("ai", f"cache HIT in {time.time()-t0:.2f}s")
        yield cached
        return

    # -- Step 3: load tools from session cache ------------------------------
    hs_tools, zp_tools = await _load_and_index_tools(session_id, agent, clients)

    # -- Step 4: hybrid tool selection --------------------------------------
    relevant_tools = await _select_relevant_tools(message, agent, hs_tools, zp_tools)

    if not relevant_tools:
        yield _no_tools_msg(agent, clients)
        return

    # -- Step 5: run agent stream -------------------------------------------
    accumulated_chunks = []
    if agent == "hubspot":
        async for chunk in run_hubspot_agent_stream(
            message  = message,
            history  = history,
            tools    = relevant_tools,
            scopes   = granted_scopes,
            is_admin = is_admin,
            session_id = session_id,
        ):
            accumulated_chunks.append(chunk)
            yield chunk

    elif agent == "zoho_people":
        async for chunk in run_zoho_people_agent_stream(
            message = message,
            history = history,
            tools   = relevant_tools,
            session_id = session_id,
        ):
            accumulated_chunks.append(chunk)
            yield chunk

    else:  # cross
        hs_set = set(id(t) for t in hs_tools)
        zp_set = set(id(t) for t in zp_tools)
        hs_relevant = [t for t in relevant_tools if id(t) in hs_set]
        zp_relevant = [t for t in relevant_tools if id(t) in zp_set]

        # Ensure each side has at least some tools
        if not hs_relevant:
            hs_relevant = hs_tools[:8]
        if not zp_relevant:
            zp_relevant = zp_tools[:8]

        async for chunk in run_cross_agent_stream(
            message  = message,
            history  = history,
            hs_tools = hs_relevant,
            zp_tools = zp_relevant,
            scopes   = granted_scopes,
            is_admin = is_admin,
            session_id = session_id,
        ):
            accumulated_chunks.append(chunk)
            yield chunk

    answer = "".join(accumulated_chunks)
    elapsed = time.time() - t0
    log("ok", f"agent_stream done | agent={agent} | {len(answer)} chars | {elapsed:.1f}s")

    # -- Step 6: cache the answer -------------------------------------------
    if answer:
        set_cached(message, agent, session_id, answer)


# ---------------------------------------------------------------------------
# Session lifecycle helpers (called from web_app.py)
# ---------------------------------------------------------------------------

async def evict_session(session_id: str):
    await _cache_evict(session_id)
    _clear_cache(session_id)


async def evict_stale_sessions():
    await _cache_evict_stale()


async def evict_all_sessions():
    await _cache_evict_all()


def get_cache_stats() -> dict:
    return {
        "session_cache" : _session_stats(),
        "response_cache": cache_stats(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_tools_msg(agent: str, clients: dict) -> str:
    if agent == "hubspot" and "hubspot" not in clients:
        return "HubSpot is not connected. Click **Connect HubSpot** in the sidebar."
    if agent == "zoho_people" and "zoho_people" not in clients:
        return "Zoho People is not connected. Add your MCP URL in the sidebar."
    if agent == "cross" and not clients:
        return "Neither HubSpot nor Zoho People is connected. Connect at least one from the sidebar."
    return "No tools available for this query. Please reconnect and try again."
