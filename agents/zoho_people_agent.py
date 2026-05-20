"""
agents/zoho_people_agent.py
============================
Zoho People HRMS agent using LangGraph ReAct agent with real-time streaming and a minified system prompt.
"""

from __future__ import annotations
import os
from typing import AsyncGenerator
from langgraph.prebuilt      import create_react_agent
from crm_logger import log
from core.utils import get_agent_llm, build_agent_messages

_RECURSION_LIMIT    = 20


def build_system_prompt(tools: list, memory_context: str = "") -> str:
    """Build a compact, token-efficient system prompt for the Zoho People HRMS agent."""
    tool_lines = "\n".join(
        f"  • {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:140]}"
        for t in tools
    )

    # Detect if the dynamic callAPI tool is available
    has_call_api = any(getattr(t, "name", "") == "ZohoPeople_callAPI" for t in tools)

    call_api_section = ""
    if has_call_api:
        call_api_section = """
## ZohoPeople_callAPI - Dynamic REST API
Use this for ALL bulk, listing, and filtering queries.
- method: "GET" (reading data)
- path: e.g. "/forms/employee/getRecords" (all employees), "/forms/department/getRecords" (all departments), "/forms/leave/getRecords" (all leaves)
- params: e.g. {"searchColumn": "Employeestatus", "searchValue": "Active"}, {"searchColumn": "Department", "searchValue": "Engineering"}, {"searchColumn": "ApprovalStatus", "searchValue": "Pending"}
"""
    else:
        call_api_section = """
## Limited Tools Warning
Only single record operations by ID are supported.
"""

    return f"""You are a Zoho People HRMS assistant.
{memory_context}

Tools available:
{tool_lines}
{call_api_section}

STRICT INSTRUCTIONS:
1. Never ask the user for employee/record/department IDs. Use ZohoPeople_callAPI to search/list records and find IDs.
2. Smart Defaults:
   - "today": use today's actual date (YYYY-MM-DD).
   - "active employees": filter by Employeestatus = Active.
   - Default time range for leaves/attendance: today.
3. Use ZohoPeople_callAPI for any list/bulk queries. Other tools like fetchEmployeeRecordById only support single records.
4. Output format:
   - Employee lists: Table with | Name | Department | Status |
   - Leaves: Table with | Employee | Leave Type | From | To | Status |
   - Attendance: Table with | Employee | Date | Check-in | Check-out | Hours |
   - Single profiles: Bullet list of key fields.
5. Confirm write operations (e.g. approve/reject leave) with a short 1-line summary.
6. When presenting statistical, reporting, or counting data (e.g., deals by stage, employees by department, leave counts), ALWAYS append a JSON block at the very end of your response inside a standard code block labeled json-chart in this exact format:
```json-chart
{{
  "type": "bar" | "line" | "area",
  "x": "column_for_x_axis",
  "y": "column_for_y_axis",
  "data": [
    {{"Department": "Engineering", "Employees": 12}},
    {{"Department": "Sales", "Employees": 4}}
  ]
}}
```
7. Use the `save_user_preference` tool to remember user facts or preferences (e.g., name, managed department, view preferences) when asked.
"""


async def run_zoho_people_agent_stream(
    message: str,
    history: list[dict],
    tools: list,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    """
    Asynchronously yields real-time text tokens from the Zoho People ReAct agent.
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
    system = build_system_prompt(all_tools, memory_context)
    messages = build_agent_messages(system, history, message)

    agent  = create_react_agent(llm, all_tools)
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


async def run_zoho_people_agent(
    message: str,
    history: list[dict],
    tools: list,
    session_id: str = "default",
) -> str:
    """
    Runs the Zoho People ReAct agent to completion and returns the final response string.
    Matches the original sync signature by aggregating the stream.
    """
    chunks = []
    async for chunk in run_zoho_people_agent_stream(message, history, tools, session_id):
        chunks.append(chunk)
    content = "".join(chunks)
    log("ai", f"Zoho People agent -> {len(content)} chars")
    return content