"""
core/agent.py
=============
Central agent dispatcher.

Agents
------
  hubspot      → HubSpot CRM only
  zoho_people  → Zoho People HRMS only
  cross        → HubSpot + Zoho People combined

Pipeline per request
--------------------
  1. pre_router    — instant reply for greetings/help/off-topic (zero LLM cost)
  2. response_cache — return cached answer for repeated read queries
  3. session_cache  — load all tools from MCP (once per session, TTL 10 min)
  4. hybrid_search  — keyword + Chroma vector search → top K relevant tools
  5. LangGraph ReAct agent (runs selected tools, returns answer)
  6. response_cache.set — store answer for next time
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
from rag.tool_indexer    import index_session_tools

from agents.hubspot_agent      import run_hubspot_agent
from agents.zoho_people_agent  import run_zoho_people_agent
from agents.cross_agent        import run_cross_agent


# ─────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ─────────────────────────────────────────────────────────────────────────────

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

    # ── Step 1: pre-router (instant, zero cost) ───────────────────────────────
    instant = pre_route(message)
    if instant:
        log("ai", f"pre_router handled '{message[:40]}'")
        return instant

    # ── Step 2: response cache ────────────────────────────────────────────────
    cached = get_cached(message, agent, session_id)
    if cached:
        log("ai", f"cache HIT in {time.time()-t0:.2f}s")
        return cached

    # ── Step 3: load tools from session cache ─────────────────────────────────
    hs_tools: list = []
    zp_tools: list = []

    if agent in ("hubspot", "cross") and "hubspot" in clients:
        hs_tools = await get_or_load_tools(session_id, "hubspot", clients["hubspot"])
        # Index into Chroma if not already done (once per session)
        await index_session_tools(session_id, hs_tools)

    if agent in ("zoho_people", "cross") and "zoho_people" in clients:
        zp_tools = await get_or_load_tools(session_id, "zoho_people", clients["zoho_people"])
        await index_session_tools(session_id, zp_tools)

    # ── Step 4: hybrid tool selection ─────────────────────────────────────────
    all_tools = hs_tools + zp_tools
    if all_tools:
        relevant_tools = await hybrid_search(
            message    = message,
            session_id = session_id,
            all_tools  = all_tools,
        )
        log("ai", f"hybrid → {len(relevant_tools)}/{len(all_tools)} tools selected")
    else:
        relevant_tools = []

    if not relevant_tools:
        return _no_tools_msg(agent, clients)

    # ── Step 5: run agent ─────────────────────────────────────────────────────
    if agent == "hubspot":
        answer = await run_hubspot_agent(
            message  = message,
            history  = history,
            tools    = relevant_tools,
            scopes   = granted_scopes,
            is_admin = is_admin,
        )

    elif agent == "zoho_people":
        answer = await run_zoho_people_agent(
            message = message,
            history = history,
            tools   = relevant_tools,
        )

    else:  # cross
        hs_relevant = [t for t in relevant_tools if t in hs_tools or _is_hs_tool(t)]
        zp_relevant = [t for t in relevant_tools if t in zp_tools or _is_zp_tool(t)]

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
        )

    elapsed = time.time() - t0
    log("ok", f"agent={agent} | {len(answer)} chars | {elapsed:.1f}s")

    # ── Step 6: cache the answer ───────────────────────────────────────────────
    set_cached(message, agent, session_id, answer)
    return answer


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle helpers (called from web_app.py)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _no_tools_msg(agent: str, clients: dict) -> str:
    if agent == "hubspot" and "hubspot" not in clients:
        return "🔑 HubSpot is not connected. Click **Connect HubSpot** in the sidebar."
    if agent == "zoho_people" and "zoho_people" not in clients:
        return "🔗 Zoho People is not connected. Add your MCP URL in the sidebar."
    if agent == "cross" and not clients:
        return "⚠️ Neither HubSpot nor Zoho People is connected. Connect at least one from the sidebar."
    return "⚠️ No tools available for this query. Please reconnect and try again."


def _is_hs_tool(tool) -> bool:
    name = getattr(tool, "name", "").lower()
    return not any(k in name for k in ("leave", "attendance", "employee", "people", "shift", "payroll"))


def _is_zp_tool(tool) -> bool:
    return not _is_hs_tool(tool)
