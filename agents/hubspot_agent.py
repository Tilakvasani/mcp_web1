"""
agents/hubspot_agent.py
=======================
HubSpot CRM agent — prompt builder + LangGraph ReAct runner.
"""

from __future__ import annotations
import os
from langchain_openai        import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt      import create_react_agent
from crm_logger import log

_MAX_TOKENS_SIMPLE  = 800
_MAX_TOKENS_COMPLEX = 2500
_RECURSION_LIMIT    = 18

_SIMPLE_PATTERNS = [
    "show", "list", "get", "find", "fetch", "display",
    "what", "who", "how many", "count", "search",
]


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
        f"  • {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in tools
    )
    scope_str  = ", ".join(scopes[:12]) + ("…" if len(scopes) > 12 else "")
    admin_note = "Super Admin — all operations permitted." if is_admin else "Standard User."

    return f"""You are an expert HubSpot CRM assistant with direct access to the HubSpot API via MCP tools.

## Role
Answer the user's question by calling the right HubSpot tools. Be concise and accurate.

## Available Tools ({len(tools)})
{tool_lines}

## Session Info
- Permission level : {admin_note}
- Granted scopes   : {scope_str or 'unknown'}

## Rules
1. Always use tools to fetch live data — never invent records.
2. For lists, show key fields only (name, amount, stage, owner, date).
3. If a tool call fails, explain why briefly and suggest next steps.
4. For write operations, confirm what you did with a short summary.
5. Use markdown tables for lists of 3+ records.
6. If you can't complete a task due to missing scope, tell the user which scope is needed.
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

    # Trim history — keep last 10 messages to avoid prompt bloat
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
