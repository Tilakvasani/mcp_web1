"""
agents/runner.py
================
Main entry point for running the AI agent.

Flow per message:
  1. Router detects intent → which app
  2. Load tools for that app (cached per session)
  3. RAG selects top-k relevant tools
  4. LLM agent runs with those tools
  5. Return answer
"""

import asyncio
from mcp_bridge.client import MCPClient
from core.tools import get_langchain_tools
from core.logger import log
from rag.selector import prepare, select
from agents.router import (
    detect_intent,
    APP_HUBSPOT_CRM, APP_HUBSPOT_TICKETS, APP_ZOHO_HRMS, APP_BOTH
)
from agents.prompts import (
    hubspot_crm_prompt,
    hubspot_tickets_prompt,
    zoho_hrms_prompt,
    both_prompt,
)
from agents.base import run

# Session tool cache: {session_id: {app_name: [tools]}}
_tool_cache: dict[str, dict[str, list]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Tool loading
# ─────────────────────────────────────────────────────────────────────────────

async def _load_tools(
    app_name: str,
    client: MCPClient,
    session_id: str,
    granted_scopes: list[str],
) -> list:
    """Load tools for an app, caching per session."""
    session = _tool_cache.setdefault(session_id, {})
    if app_name in session:
        return session[app_name]

    try:
        tools = await get_langchain_tools(client, granted_scopes=granted_scopes)
        prepare(app_name, tools)   # index into RAG
        session[app_name] = tools
        log("mcp", f"Loaded {len(tools)} tools for {app_name} | session={session_id[:8]}")
        return tools
    except Exception as exc:
        log("error", f"Failed to load tools for {app_name}: {exc}")
        return []


def evict_session(session_id: str) -> None:
    """Clear tool cache for a session (on disconnect/logout)."""
    _tool_cache.pop(session_id, None)
    log("bye", f"Tool cache cleared | session={session_id[:8]}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def run_agent(
    message: str,
    history: list[dict],
    clients: dict[str, MCPClient],   # {"hubspot": client, "zoho_hrms": client}
    granted_scopes: list[str] | None = None,
    session_id: str = "default",
) -> str:
    """
    Main entry point.
    clients keys: "hubspot" | "zoho_hrms"
    """
    scopes = granted_scopes or []

    # ── Step 1: Route ─────────────────────────────────────────────────────
    app = await detect_intent(message)
    log("agent", f"Routing to: {app}")

    # ── Step 2: Load tools ────────────────────────────────────────────────
    if app == APP_BOTH:
        hs_tools   = await _load_tools("hubspot",   clients.get("hubspot"),   session_id, scopes) if "hubspot"   in clients else []
        zoho_tools = await _load_tools("zoho_hrms", clients.get("zoho_hrms"), session_id, scopes) if "zoho_hrms" in clients else []

        if not hs_tools and not zoho_tools:
            return "⚠️ No apps connected. Please connect HubSpot and/or Zoho from the sidebar."

        # RAG select from each, then combine
        hs_sel   = select("hubspot",   message, hs_tools)   if hs_tools   else []
        zoho_sel = select("zoho_hrms", message, zoho_tools) if zoho_tools else []
        tools    = hs_sel + zoho_sel
        prompt   = both_prompt(tools)

    elif app == APP_HUBSPOT_TICKETS:
        if "hubspot" not in clients:
            return "⚠️ HubSpot not connected. Please connect from the sidebar."
        all_tools = await _load_tools("hubspot", clients["hubspot"], session_id, scopes)
        # Filter to ticket-related tools only
        ticket_tools = [t for t in all_tools if "ticket" in getattr(t, "name", "").lower()]
        tools  = select("hubspot", message, ticket_tools or all_tools)
        prompt = hubspot_tickets_prompt(tools)

    elif app == APP_ZOHO_HRMS:
        if "zoho_hrms" not in clients:
            return "⚠️ Zoho HRMS not connected. Please paste your Zoho MCP URL in the sidebar."
        all_tools = await _load_tools("zoho_hrms", clients["zoho_hrms"], session_id, scopes)
        tools  = select("zoho_hrms", message, all_tools)
        prompt = zoho_hrms_prompt(tools)

    else:  # APP_HUBSPOT_CRM (default)
        if "hubspot" not in clients:
            return "⚠️ HubSpot not connected. Please connect from the sidebar."
        all_tools = await _load_tools("hubspot", clients["hubspot"], session_id, scopes)
        tools  = select("hubspot", message, all_tools)
        prompt = hubspot_crm_prompt(tools)

    # ── Step 3: Run agent ─────────────────────────────────────────────────
    return await run(message, history, tools, prompt)