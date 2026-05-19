"""
agents/zoho_people_agent.py
============================
Zoho People HRMS agent — prompt builder + LangGraph ReAct runner.
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
        f"  • {getattr(t,'name','')}: {(getattr(t,'description','') or '')[:120]}"
        for t in tools
    )
    return f"""You are an expert Zoho People HRMS assistant with direct access to Zoho People via MCP tools.

## Role
Answer HR queries by calling the right Zoho People tools. Be concise, accurate, and professional.

## Available Tools ({len(tools)})
{tool_lines}

## Rules
1. Always fetch live data using tools — never make up employee names, leave counts, or dates.
2. For employee lists, show: Name · Department · Status.
3. For leave requests, show: Employee · Type · Dates · Status.
4. For attendance, show: Employee · Date · Check-in · Check-out · Duration.
5. Use markdown tables for lists of 3+ records.
6. If a tool call fails, explain what went wrong briefly.
7. For sensitive operations (approve/reject leave), confirm what you did with a summary.
8. Be mindful of HR data privacy — don't surface sensitive salary data unless explicitly asked.
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
