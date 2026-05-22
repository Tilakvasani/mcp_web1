"""
core/agent.py
=============
Simplified dispatcher.

Pipeline:
  1. pre_router     — instant reply for greetings/help/off-topic (zero LLM cost)
  2. response_cache — return cached answer for repeat queries
  3. session_cache  — load all tools from MCP (once per session, TTL 10 min)
  4. RAG hybrid_search — select top K relevant tools
  5. ONE unified ReAct agent
  6. Grader agent   — score the response (async, non-blocking)
  7. response_cache.set — store answer
"""

from __future__ import annotations
import time
from typing import AsyncGenerator

from crm_logger import log
from core.pre_router import pre_route
from core.response_cache import get_cached, set_cached, clear_session as _clear_cache, cache_stats
from core.session_cache import (
    get_or_load_tools,
    evict_session        as _cache_evict,
    evict_stale_sessions as _cache_evict_stale,
    evict_all_sessions   as _cache_evict_all,
    get_cache_stats      as _session_stats,
)
from rag.hybrid_search import hybrid_search
from rag.tool_indexer  import index_agent_tools
from agents.unified_agent import run_unified_agent, run_unified_agent_stream
from agents.grader_agent  import grade_response


async def run_agent_stream(
    message       : str,
    history       : list[dict],
    clients       : dict,         # {name: MCPManager ref}
    agent         : str,          # ignored — kept for API compatibility, always "unified"
    granted_scopes: list[str],
    is_admin      : bool,
    session_id    : str = "default",
) -> AsyncGenerator[str, None]:

    t0 = time.time()

    # Step 1: pre-router (zero cost)
    instant = pre_route(message)
    if instant:
        log("ai", f"pre_router handled '{message[:40]}'")
        yield instant
        return

    # Step 2: response cache
    cached = get_cached(message, "unified", session_id)
    if cached:
        log("ai", f"cache HIT in {time.time()-t0:.2f}s")
        yield cached
        return

    # Step 3: load tools from session cache
    # clients dict maps name -> mcp_manager instance
    all_tools: list = []
    for agent_key, client in clients.items():
        tools = await get_or_load_tools(
            session_id,
            agent_key,
            client,
            client_resolver=lambda c=client, k=agent_key: c,
        )
        await index_agent_tools(agent_key, tools)
        all_tools.extend(tools)

    if not all_tools:
        yield "No tools available. Please connect HubSpot or Zoho People from the sidebar."
        return

    # Step 4: hybrid tool selection (RAG)
    relevant_tools = await hybrid_search(
        message   = message,
        agent_key = "cross" if len(clients) > 1 else list(clients.keys())[0],
        all_tools = all_tools,
    )

    # Force-inject ZohoPeople_callAPI if available
    from core.tools import get_synthetic_tool
    call_api = get_synthetic_tool("ZohoPeople_callAPI", all_tools)
    if call_api and call_api not in relevant_tools:
        relevant_tools.append(call_api)

    log("ai", f"hybrid -> {len(relevant_tools)}/{len(all_tools)} tools selected")

    # Step 5: run unified agent
    accumulated = []
    async for chunk in run_unified_agent_stream(
        message    = message,
        history    = history,
        tools      = relevant_tools,
        session_id = session_id,
    ):
        accumulated.append(chunk)
        yield chunk

    answer = "".join(accumulated)
    elapsed = time.time() - t0
    log("ok", f"agent_stream done | unified | {len(answer)} chars | {elapsed:.1f}s")

    # Step 6: cache + grade (fire and forget)
    if answer:
        set_cached(message, "unified", session_id, answer)
        import asyncio
        asyncio.create_task(_grade_and_log(message, answer))


async def run_agent(
    message       : str,
    history       : list[dict],
    clients       : dict,
    agent         : str,
    granted_scopes: list[str],
    is_admin      : bool,
    session_id    : str = "default",
) -> str:
    chunks = []
    async for chunk in run_agent_stream(
        message        = message,
        history        = history,
        clients        = clients,
        agent          = agent,
        granted_scopes = granted_scopes,
        is_admin       = is_admin,
        session_id     = session_id,
    ):
        chunks.append(chunk)
    return "".join(chunks)


async def _grade_and_log(question: str, answer: str):
    scores = await grade_response(question, answer)
    log("grade", f"scores -> {scores}")


# Session lifecycle helpers
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
