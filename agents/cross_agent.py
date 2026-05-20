"""
agents/cross_agent.py
=====================
Cross-system agent — HubSpot CRM + Zoho People HRMS combined with real-time streaming and a minified prompt.
"""

from __future__ import annotations
import os
from typing import AsyncGenerator
from langgraph.prebuilt      import create_react_agent
from crm_logger import log
from core.utils import get_agent_llm, build_agent_messages

_RECURSION   = 22


def _build_system_prompt(hs_tools: list, zp_tools: list, memory_context: str = "") -> str:
    """Build a compact, token-efficient system prompt for the cross-system agent."""
    hs_lines = "\n".join(
        f"  [HubSpot] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in hs_tools
    )
    zp_lines = "\n".join(
        f"  [ZohoPeople] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in zp_tools
    )
    
    call_api_section = ""
    has_call_api = any(getattr(t, "name", "") == "ZohoPeople_callAPI" for t in zp_tools)
    if has_call_api:
        call_api_section = """
## ZohoPeople_callAPI - Dynamic REST API
Use `/forms/{formName}/getRecords` to fetch list/bulk Zoho records (e.g. "/forms/employee/getRecords").
"""

    return f"""You are a cross-system assistant with access to HubSpot CRM and Zoho People HRMS.
Answer queries needing data from BOTH systems. Call both, then merge into ONE response.
{call_api_section}
{memory_context}

HubSpot Tools:
{hs_lines}

Zoho People Tools:
{zp_lines}

STRICT INSTRUCTIONS:
1. Cross-System Mapping:
   - "sales reps on leave today": 1) Zoho leaves today -> get names. 2) HubSpot owners -> match names. 3) Show matched reps.
   - "tickets assigned to absent employees": 1) Zoho leaves today -> names. 2) HubSpot tickets -> match owner names.
   - "deals where owner is on leave": 1) Zoho leaves today -> names. 2) HubSpot deals -> match owner names.
2. Match employees across systems by: Full Name (case-insensitive) or Email. Be spelling-tolerant.
3. If one system has no data, report it in 1 line, and show the other system's data.
4. Output format: Merge into a single markdown table with a 'Source' or status column.
5. End with a 1-line summary (e.g. "**2** of **5** sales reps are on leave today").
6. When presenting statistical, reporting, or counting data (e.g., deals by stage, employees by department, leave counts), ALWAYS append a JSON block at the very end of your response inside a standard code block labeled json-chart in this exact format:
```json-chart
{{
  "type": "bar" | "line" | "area",
  "x": "column_for_x_axis",
  "y": "column_for_y_axis",
  "data": [
    {{"Label": "Active", "Count": 12}},
    {{"Label": "Inactive", "Count": 4}}
  ]
}}
```
7. Use the `save_user_preference` tool to remember user facts or preferences (e.g., name, managed department, view preferences) when asked.
"""


async def run_cross_agent_stream(
    message: str,
    history: list[dict],
    hs_tools: list,
    zp_tools: list,
    scopes: list[str],
    is_admin: bool,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Asynchronously yields real-time text tokens from the Cross-system ReAct agent.
    """
    from core.memory import get_preferences, create_save_preference_tool

    # Add memory tool to tools list (associated with both sides)
    memory_tool = create_save_preference_tool(session_id)
    all_hs_tools = list(hs_tools)
    all_zp_tools = list(zp_tools) + [memory_tool]
    all_tools = all_hs_tools + all_zp_tools

    if not all_tools:
        yield "⚠️ No CRM or HR tools available. Please connect both HubSpot and Zoho People."
        return

    # Retrieve user preferences for memory context
    prefs = get_preferences(session_id)
    if prefs:
        prefs_str = "\n".join(f"  - {k}: {v}" for k, v in prefs.items())
        memory_context = f"\nRemembered User Preferences:\n{prefs_str}\n"
    else:
        memory_context = ""

    llm    = get_agent_llm(max_tokens=3500)
    system = _build_system_prompt(all_hs_tools, all_zp_tools, memory_context)
    messages = build_agent_messages(system, history, message)

    agent  = create_react_agent(llm, all_tools)
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
        config={"recursion_limit": _RECURSION},
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # Skip chunks representing tool calls
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                continue
            text = getattr(chunk, "content", "")
            if text:
                yield text


async def run_cross_agent(
    message: str,
    history: list[dict],
    hs_tools: list,
    zp_tools: list,
    scopes: list[str],
    is_admin: bool,
    session_id: str = "default",
) -> str:
    """
    Runs the Cross-system ReAct agent to completion and returns the final response string.
    Matches the original sync signature by aggregating the stream.
    """
    chunks = []
    async for chunk in run_cross_agent_stream(message, history, hs_tools, zp_tools, scopes, is_admin, session_id):
        chunks.append(chunk)
    content = "".join(chunks)
    log("ai", f"cross-agent -> {len(content)} chars ({len(hs_tools)} HS + {len(zp_tools)} ZP tools)")
    return content