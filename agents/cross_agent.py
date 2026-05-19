"""
agents/cross_agent.py
=====================
Cross-system agent — HubSpot CRM + Zoho People HRMS combined.
"""

from __future__ import annotations
import os
from langchain_openai        import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt      import create_react_agent
from crm_logger import log

_MAX_TOKENS  = 3500
_RECURSION   = 22


def _get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        api_key          = os.getenv("AZURE_OPENAI_API_KEY"),
        temperature      = 0,
        max_tokens       = _MAX_TOKENS,
        streaming        = False,
    )


def _build_system_prompt(hs_tools: list, zp_tools: list) -> str:
    hs_lines = "\n".join(
        f"  [HubSpot] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in hs_tools
    )
    zp_lines = "\n".join(
        f"  [ZohoPeople] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in zp_tools
    )
    return f"""You are a cross-system business intelligence assistant with access to both HubSpot CRM and Zoho People HRMS.

## Your Job
Answer queries that need data from both systems. Call both systems, then merge into ONE unified answer.
Never ask the user for IDs or clarification — figure it out from the tools.

## HubSpot Tools ({len(hs_tools)})
{hs_lines}

## Zoho People Tools ({len(zp_tools)})
{zp_lines}

## Natural Language → Cross-System Mapping

| User says | Strategy |
|---|---|
| "sales reps on leave today" | 1) Fetch Zoho leave records for today → get employee names. 2) Fetch HubSpot owners/contacts → match by name. 3) Show matched reps |
| "tickets assigned to absent employees" | 1) Fetch Zoho absent employees today. 2) Fetch HubSpot open tickets. 3) Match owner names and show tickets |
| "deals where owner is on leave" | 1) Fetch Zoho leave today → names. 2) Fetch HubSpot deals → filter by owner in leave list |
| "team availability this week" | 1) Fetch Zoho leave for this week. 2) Fetch HubSpot owners. 3) Mark each as available/on-leave |
| "compare deals vs headcount by dept" | 1) Fetch Zoho dept headcount. 2) Fetch HubSpot deals grouped by team/owner. 3) Merge into table |
| "who manages [client] — are they available?" | 1) Fetch HubSpot contact/company → get owner. 2) Check Zoho leave for that person today |

## Smart Defaults
- "today" = today's actual date YYYY-MM-DD
- "this week" = Monday to today
- "available" = not on leave in Zoho + exists as owner in HubSpot
- Match employees across systems by: full name (case-insensitive) or email

## STRICT RULES
1. **NEVER ask the user for any ID or parameter** — derive everything from tool calls
2. **NEVER return two separate answers** — always merge into one unified response
3. Call both systems even if only one seems relevant — the user asked for cross-system
4. Match records by full name (lowercase) or email — be tolerant of minor spelling differences
5. If one system returns no data, say so in one line and show what the other system found
6. Use markdown tables for all merged results
7. For "on leave" queries: if no leave records found, explicitly say "no one is on leave today"

## Response Format
- Cross-system tables: include a "Source" or status column showing which system the data came from
- Always end with a 1-line summary: e.g. "2 of 5 sales reps are on leave today"
"""


async def run_cross_agent(
    message: str,
    history: list[dict],
    hs_tools: list,
    zp_tools: list,
    scopes: list[str],
    is_admin: bool,
) -> str:
    all_tools = hs_tools + zp_tools
    if not all_tools:
        return "⚠️ No CRM or HR tools available. Please connect both HubSpot and Zoho People."

    llm    = _get_llm()
    system = _build_system_prompt(hs_tools, zp_tools)

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

    agent  = create_react_agent(llm, all_tools)
    result = await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": _RECURSION},
    )

    final = result["messages"][-1]
    text  = getattr(final, "content", str(final))
    log("ai", f"cross-agent → {len(text)} chars ({len(hs_tools)} HS + {len(zp_tools)} ZP tools)")
    return text