"""
agents/unified_agent.py
=======================
Single ReAct agent with access to ALL connected MCP tools.
No routing. No intent detection. The agent picks its own tools.

Works for:
  - HubSpot-only queries    ("Show all deals closing this month")
  - Zoho-only queries       ("List employees on leave today")
  - Cross-system queries    ("Which sales reps are on leave?")
  - Off-topic / help        (handled by pre_router before reaching here)
"""

from __future__ import annotations
from typing import AsyncGenerator
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.utils import get_agent_llm, build_agent_messages
from crm_logger import log

_RECURSION_LIMIT = 20

SYSTEM_PROMPT = """You are a business AI assistant with access to tools from:
- HubSpot CRM (deals, contacts, companies, tickets, tasks, owners, pipelines)
- Zoho People HRMS (employees, leave, attendance, departments, timesheets)

RULES:
1. Use the right tools for the query. CRM question → HubSpot tools. HR question → Zoho tools.
   Cross-system question → call both, merge the results into one answer.
2. Never ask the user for IDs. Search by name/email first to find the ID.
3. Dates: "today" = current date YYYY-MM-DD. "this week" = Monday to today.
4. For mutating actions (create/update/delete), confirm with a 1-line summary after.
5. Output format:
   - Lists: markdown table
   - Single record: bullet list
   - Stats/counts: bold the numbers, add a json-chart block at the end (see below)
6. When data has counts or metrics (deals by stage, employees by dept, leave counts),
   append this at the very end of the response:
   ```json-chart
   {
     "type": "bar",
     "x": "column_name",
     "y": "value_column",
     "data": [{"Label": "X", "Count": 5}]
   }
   ```
7. Cross-system matching: match employees across HubSpot and Zoho by full name
   (case-insensitive) or email.
"""


async def run_unified_agent_stream(
    message: str,
    history: list[dict],
    tools: list,  # LangChain tools from MCPManager + synthetic tools
    session_id: str = "default",
) -> AsyncGenerator[str, None]:

    from core.memory import get_preferences, create_save_preference_tool

    # Add memory tool
    memory_tool = create_save_preference_tool(session_id)
    all_tools = list(tools) + [memory_tool]

    # Inject remembered preferences into system prompt
    prefs = get_preferences(session_id)
    prefs_section = ""
    if prefs:
        lines = "\n".join(f"  - {k}: {v}" for k, v in prefs.items())
        prefs_section = f"\nRemembered about this user:\n{lines}\n"

    system = SYSTEM_PROMPT + prefs_section

    llm = get_agent_llm(message)
    messages = build_agent_messages(system, history, message)

    agent = create_react_agent(llm, all_tools, handle_tool_errors=True)
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
        config={"recursion_limit": _RECURSION_LIMIT},
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                continue
            text = getattr(chunk, "content", "")
            if text:
                yield text


async def run_unified_agent(
    message: str,
    history: list[dict],
    tools: list,
    session_id: str = "default",
) -> str:
    chunks = []
    async for chunk in run_unified_agent_stream(message, history, tools, session_id):
        chunks.append(chunk)
    return "".join(chunks) 