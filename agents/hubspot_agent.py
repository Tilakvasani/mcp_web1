"""
agents/hubspot_agent.py
=======================
HubSpot CRM agent using LangGraph ReAct agent with real-time streaming and a minified system prompt.
"""

from __future__ import annotations
import os
import json
from typing import AsyncGenerator
from langgraph.prebuilt      import create_react_agent
from crm_logger import log
from core.utils import get_agent_llm, build_agent_messages

_RECURSION_LIMIT    = 20


def build_system_prompt(tools: list, scopes: list[str], is_admin: bool, memory_context: str = "") -> str:
    """Build a compact, token-efficient system prompt for the HubSpot CRM agent."""
    tool_lines = []
    for t in tools:
        schema = t.args_schema.schema() if t.args_schema else {}
        props = schema.get("properties", {})
        tool_lines.append(f"  • {t.name}: {t.description}\n    Args: {json.dumps(props)}")
    tool_str = "\n".join(tool_lines)

    scope_str  = ", ".join(scopes[:12]) + ("…" if len(scopes) > 12 else "")
    admin_note = "Super Admin (all permitted)" if is_admin else "Standard User"

    return f"""You are a HubSpot CRM assistant.
Permission: {admin_note} | Scopes: {scope_str or 'unknown'}
{memory_context}

Tools available:
{tool_str}

STRICT INSTRUCTIONS:
1. Never ask the user for record/object IDs. Search by name/email/subject first to get the ID, then fetch or update.
2. Smart Defaults:
   - "today": use current date in YYYY-MM-DD format.
   - "this week": Monday to today. "this month": 1st to last day of month.
   - "all deals/contacts/companies": fetch list without filters (default limit=20, default sort closedate asc for deals).
   - "resolve/close tickets": set the ticket's property `hs_pipeline_stage` to `"4"` (Closed/Resolved). Do NOT use properties named `status`, `ticketstatus`, or `stage` for support tickets.
3. If missing required scopes, explain which scopes are required.
4. Output format:
   - Deal lists: Table with | Deal Name | Stage | Amount | Close Date | Owner |
   - Contacts: Table with | Name | Email | Company | Last Activity |
   - Tickets: Table with | Ticket | Status | Priority | Owner |
   - Single records: Clear bullet list of fields.
   - Summaries: Bold counts/metrics, e.g. "**3** deals in closed-won".
5. Confirm write operations (create/update/delete) with a short 1-line summary.
6. When presenting statistical, reporting, or counting data (e.g., deals by stage, employees by department, leave counts), ALWAYS append a JSON block at the very end of your response inside a standard code block labeled json-chart in this exact format:
```json-chart
{{
  "type": "bar" | "line" | "area",
  "x": "column_for_x_axis",
  "y": "column_for_y_axis",
  "data": [
    {{"Stage": "Closed Won", "Deals Count": 12}},
    {{"Stage": "Closed Lost", "Deals Count": 4}}
  ]
}}
```
7. Use the `save_user_preference` tool to remember user facts or preferences (e.g., name, managed department, view preferences) when asked.
"""


async def run_hubspot_agent_stream(
    message: str,
    history: list[dict],
    tools: list,
    scopes: list[str],
    is_admin: bool,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Asynchronously yields real-time text tokens from the HubSpot ReAct agent.
    """
    from core.memory import get_preferences, create_save_preference_tool

    # Add memory tool to tools list
    memory_tool = create_save_preference_tool(session_id)
    all_tools = list(tools) + [memory_tool]

    # Retrieve user preferences for memory context
    prefs = get_preferences(session_id)
    if prefs:
        prefs_str = "\n".join(f"  - {k}: {v}" for k, v in prefs.items())
        memory_context = f"\nRemembered User Preferences:\n{prefs_str}\n"
    else:
        memory_context = ""

    llm    = get_agent_llm(message)
    system = build_system_prompt(all_tools, scopes, is_admin, memory_context)
    messages = build_agent_messages(system, history, message)

    agent = create_react_agent(llm, all_tools)
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
        config={"recursion_limit": _RECURSION_LIMIT},
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Skip chunks that represent tool calls
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                continue
            text = getattr(chunk, "content", "")
            if text:
                yield text


async def run_hubspot_agent(
    message: str,
    history: list[dict],
    tools: list,
    scopes: list[str],
    is_admin: bool,
    session_id: str = "default",
) -> str:
    """
    Runs the HubSpot ReAct agent to completion and returns the final response string.
    Matches the original sync signature by aggregating the stream.
    """
    chunks = []
    async for chunk in run_hubspot_agent_stream(message, history, tools, scopes, is_admin, session_id):
        chunks.append(chunk)
    content = "".join(chunks)
    log("ai", f"HubSpot agent -> {len(content)} chars")
    return content