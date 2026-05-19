"""
agents/cross_agent.py
=====================
Cross-system agent — queries HubSpot CRM + Zoho People HRMS in parallel
and merges the results into one unified response.

Example queries:
  "Which sales reps are on leave today?"
  "Show support tickets assigned to absent employees"
  "Compare active deals with department headcount"
"""

from __future__ import annotations
import os
import asyncio
from langchain_openai        import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt      import create_react_agent
from crm_logger import log

_MAX_TOKENS  = 3000
_RECURSION   = 20


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
        f"  [HubSpot] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:100]}"
        for t in hs_tools
    )
    zp_lines = "\n".join(
        f"  [ZohoPeople] {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:100]}"
        for t in zp_tools
    )
    return f"""You are a cross-system business intelligence assistant with access to both HubSpot CRM and Zoho People HRMS.

## Role
Answer queries that require data from both systems. Fetch from both, then merge into one clear answer.

## HubSpot Tools ({len(hs_tools)})
{hs_lines}

## Zoho People Tools ({len(zp_tools)})
{zp_lines}

## Rules
1. For cross-system queries, call BOTH systems and correlate the data.
2. Match records by: full name, email, or employee ID where possible.
3. Present merged data in a single unified table or list.
4. If one system has no matching data, say so explicitly.
5. For "sales reps on leave" style queries:
   - Fetch HubSpot owners/contacts who are sales reps
   - Fetch Zoho People leave records for today
   - Cross-reference by name/email and list who matches
6. Use markdown tables for structured data (3+ rows).
7. Be concise — one combined answer, not two separate sections.
"""


async def _run_single_agent(
    llm: AzureChatOpenAI,
    tools: list,
    messages: list,
    label: str,
) -> str:
    agent  = create_react_agent(llm, tools)
    result = await agent.ainvoke(
        {"messages": messages},
        config={"recursion_limit": _RECURSION},
    )
    final = result["messages"][-1]
    text  = getattr(final, "content", str(final))
    log("ai", f"{label} sub-agent → {len(text)} chars")
    return text


async def run_cross_agent(
    message: str,
    history: list[dict],
    hs_tools: list,
    zp_tools: list,
    scopes: list[str],
    is_admin: bool,
) -> str:
    """
    Run a combined HubSpot + Zoho People agent over all tools in one LLM call.
    The single agent has all tools available and figures out which to call.
    """
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
