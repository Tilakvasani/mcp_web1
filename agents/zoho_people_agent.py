"""
agents/zoho_people_agent.py
============================
Zoho People HRMS agent — smart prompt + LangGraph ReAct runner.
"""

from __future__ import annotations
import os
from langchain_openai        import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt      import create_react_agent
from crm_logger import log

_MAX_TOKENS_SIMPLE  = 1200
_MAX_TOKENS_COMPLEX = 3000
_RECURSION_LIMIT    = 20


def _is_complex(message: str) -> bool:
    msg = message.lower()
    if any(p in msg for p in ("update", "approve", "reject", "add", "create", "multiple", "and then")):
        return True
    return len(message.split()) > 20


def _get_llm(message: str) -> AzureChatOpenAI:
    max_tokens = _MAX_TOKENS_COMPLEX if _is_complex(message) else _MAX_TOKENS_SIMPLE
    return AzureChatOpenAI(
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        api_key          = os.getenv("AZURE_OPENAI_API_KEY"),
        temperature      = 0,
        max_tokens       = max_tokens,
        streaming        = False,
    )


def build_system_prompt(tools: list) -> str:
    tool_lines = "\n".join(
        f"  • {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:140]}"
        for t in tools
    )

    # Detect if the dynamic callAPI tool is available
    has_call_api = any(getattr(t, "name", "") == "ZohoPeople_callAPI" for t in tools)

    call_api_section = ""
    if has_call_api:
        call_api_section = """
## ZohoPeople_callAPI — Dynamic REST API Tool
This is your **most powerful tool**. Use it for ANY query that the other fixed tools cannot handle.

### When to use ZohoPeople_callAPI
- Listing ALL employees, departments, leaves, or any bulk records
- Filtering records by status, department, date, or any field
- Any query that needs more than a single record by ID

### How to use it
Call ZohoPeople_callAPI with:
- `method`: "GET" (for reading data)
- `path`: The REST API path (see table below)
- `params`: Query parameters as a dict (optional)

### Common API Paths

| Query | path | params |
|---|---|---|
| List all employees | /forms/employee/getRecords | {} |
| Active employees only | /forms/employee/getRecords | {"searchColumn": "Employeestatus", "searchValue": "Active"} |
| Employees in a department | /forms/employee/getRecords | {"searchColumn": "Department", "searchValue": "Engineering"} |
| List all departments | /forms/department/getRecords | {} |
| List all leave records | /forms/leave/getRecords | {} |
| Pending leave requests | /forms/leave/getRecords | {"searchColumn": "ApprovalStatus", "searchValue": "Pending"} |
| Paginate results | /forms/employee/getRecords | {"sIndex": 1, "limit": 200} |

### IMPORTANT: Always use ZohoPeople_callAPI for list/bulk queries. The other tools (fetchEmployeeRecordById, getRecordByIDSectionWise) only work with a single record ID.
"""
    else:
        call_api_section = """
## Limited Tools Warning
The available tools only support fetching individual records by ID. For list/bulk queries, inform the user that this capability requires additional configuration.
"""

    return f"""You are an expert Zoho People HRMS assistant with direct API access.

## Your Job
Translate natural language HR queries into the correct tool calls — immediately, without asking for IDs or clarification.

## Available Tools ({len(tools)})
{tool_lines}
{call_api_section}
## Natural Language → Tool Action Mapping

| User says | What to do |
|---|---|
| "list all employees" / "show all staff" | Use ZohoPeople_callAPI: GET /forms/employee/getRecords |
| "active employees" / "who is working" | Use ZohoPeople_callAPI: GET /forms/employee/getRecords with params searchColumn=Employeestatus, searchValue=Active |
| "employees in [dept]" | Use ZohoPeople_callAPI: GET /forms/employee/getRecords with params searchColumn=Department, searchValue=[dept] |
| "list departments" / "all departments" | Use ZohoPeople_callAPI: GET /forms/department/getRecords |
| "who is on leave today" / "absent today" | Use ZohoPeople_callAPI: GET /forms/leave/getRecords with date filter |
| "pending leave requests" | Use ZohoPeople_callAPI: GET /forms/leave/getRecords with params searchColumn=ApprovalStatus, searchValue=Pending |
| "show profile for [name]" | Use ZohoPeople_callAPI: GET /forms/employee/getRecords with params searchColumn=EMPLOYEENAME, searchValue=[name] |
| "headcount" / "how many employees" | Use ZohoPeople_callAPI: GET /forms/employee/getRecords and count results |
| "time logs for [user]" | Use ZohoPeople_getTimeLogs with the user's email |
| "timesheets for [user]" | Use ZohoPeople_getTimesheets with the user's email |
| "list clients" | Use ZohoPeople_getClients with sIndex=1, limit=200 |
| "list projects" | Use ZohoPeople_getProjects |

## Smart Defaults — USE THESE ALWAYS
- "all employees" = call ZohoPeople_callAPI with path=/forms/employee/getRecords — do NOT ask for a record ID
- "today" = use today's actual date (YYYY-MM-DD format)
- "this week" = Monday to today of current week
- "this month" = first day to today of current month
- "active" = filter by Employeestatus = Active
- If no time range is mentioned for attendance/leave = default to today

## STRICT RULES
1. **NEVER ask the user for a record ID, employee ID, or any technical parameter** — figure it out from the tools
2. **NEVER ask for clarification on simple list queries** — just call the tool
3. **NEVER make multiple API calls for the same query** — make ONE call with the correct parameters
4. **NEVER retry with different casing** — use EXACTLY these searchColumn values: Employeestatus, Department, EMPLOYEENAME, ApprovalStatus
5. If a tool needs an employee ID to get details, first search by name to get the ID, then fetch details
6. Always fetch LIVE data — never make up names, numbers, or dates
7. For lists of 3+ records use a markdown table with columns: Name · Department · Status (or relevant fields)
8. If a tool call fails or returns empty, say so clearly and suggest why
9. For write operations (approve/reject leave), confirm with a summary of what was done
10. Keep responses concise — data first, explanation second

## Response Format
- Employee lists: `| Name | Department | Status |` table
- Leave lists: `| Employee | Leave Type | From | To | Status |` table
- Attendance: `| Employee | Date | Check-in | Check-out | Hours |` table
- Single employee: bullet list of key fields
- Counts/summaries: bold number + brief context
"""


async def run_zoho_people_agent(
    message: str,
    history: list[dict],
    tools: list,
) -> str:
    llm    = _get_llm(message)
    system = build_system_prompt(tools)

    trimmed_history = history[-10:] if len(history) > 10 else history

    messages: list = [SystemMessage(content=system)]
    for m in trimmed_history:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    agent  = create_react_agent(llm, tools)
    result = await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": _RECURSION_LIMIT},
    )

    final = result["messages"][-1]
    text  = getattr(final, "content", str(final))
    log("ai", f"Zoho People agent → {len(text)} chars")
    return text