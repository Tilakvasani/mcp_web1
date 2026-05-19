"""
agents/hubspot_agent.py
=======================
HubSpot CRM agent — smart prompt + LangGraph ReAct runner.
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
    if any(p in msg for p in ("create", "update", "delete", "and then", "also", "multiple")):
        return True
    return len(message.split()) > 20


def _get_llm(message: str) -> AzureChatOpenAI:
    max_tokens = _MAX_TOKENS_COMPLEX if _is_complex(message) else _MAX_TOKENS_SIMPLE
    return AzureChatOpenAI(
        azure_endpoint       = os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version          = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        api_key              = os.getenv("AZURE_OPENAI_API_KEY"),
        temperature          = 0,
        max_tokens           = max_tokens,
        streaming            = False,
    )


def build_system_prompt(tools: list, scopes: list[str], is_admin: bool) -> str:
    tool_lines = "\n".join(
        f"  • {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:140]}"
        for t in tools
    )
    scope_str  = ", ".join(scopes[:12]) + ("…" if len(scopes) > 12 else "")
    admin_note = "Super Admin — all operations permitted." if is_admin else "Standard User."

    return f"""You are an expert HubSpot CRM assistant with direct API access via MCP tools.

## Your Job
Translate natural language CRM queries into the correct tool calls — immediately, without asking for IDs or clarification.

## Available Tools ({len(tools)})
{tool_lines}

## Session Info
- Permission level : {admin_note}
- Granted scopes   : {scope_str or 'unknown'}

## Natural Language → Tool Action Mapping

| User says | What to do |
|---|---|
| "show all deals" / "list deals" | Fetch all deals — no filter needed |
| "open deals" / "active deals" | Fetch deals with stage NOT = closed |
| "deals closing this month" | Fetch deals with closedate in current month |
| "closed won this quarter" | Fetch deals with stage = closedwon in current quarter |
| "pipeline overview" / "pipeline by stage" | Fetch all deals, group by dealstage |
| "find contact [email/name]" | Search contacts by email or name — no ID needed |
| "all contacts" / "list contacts" | Fetch contacts list with no filter |
| "companies" / "top companies" | Fetch companies list, sort by deal value |
| "unresolved tickets" / "open tickets" | Fetch tickets where status != closed |
| "tasks due today" | Fetch tasks with duedate = today |
| "recent emails" / "emails last 7 days" | Fetch email engagements for last 7 days |
| "show meetings" | Fetch meeting engagements |
| "who owns [deal/contact]" | Fetch the record and return the hubspot_owner_id resolved to name |
| "create deal [name]" | Create a new deal — ask only for missing required fields |

## Smart Defaults — USE THESE ALWAYS
- "all deals/contacts/companies" = fetch the object list with NO filter — do NOT ask for an ID
- "today" = use today's actual date in YYYY-MM-DD format
- "this week" = Monday to today of current week
- "this month" = first to last day of current month
- "this quarter" = first day of current quarter to today
- "open" / "active" = not closed / not resolved
- Default sort for deals = closedate ascending
- Default limit for lists = 20 records (unless user specifies more)

## STRICT RULES
1. **NEVER ask the user for a record ID or internal HubSpot ID** — search by name/email instead
2. **NEVER ask for clarification on simple list or show queries** — just call the tool and return data
3. If you need an ID to fetch details, first search by name/email to get the ID, then fetch
4. Always use tools to get live data — never make up deal names, amounts, or contacts
5. For lists of 3+ records use a markdown table with the most relevant columns
6. For pipeline queries, group and summarise by stage with deal count and total value
7. If a scope is missing for an operation, clearly tell the user which scope they need
8. For write operations (create/update/delete), confirm with a 1-line summary

## Response Format
- Deal lists: `| Deal Name | Stage | Amount | Close Date | Owner |`
- Contact lists: `| Name | Email | Company | Last Activity |`
- Ticket lists: `| Subject | Status | Priority | Owner | Created |`
- Pipeline summary: `| Stage | # Deals | Total Value |`
- Single record: bullet list of key fields
"""


async def run_hubspot_agent(
    message: str,
    history: list[dict],
    tools: list,
    scopes: list[str],
    is_admin: bool,
) -> str:
    llm    = _get_llm(message)
    system = build_system_prompt(tools, scopes, is_admin)

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
    log("ai", f"HubSpot agent → {len(text)} chars")
    return text