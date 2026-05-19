"""
agents/base.py
==============
Shared agent runner used by all 3 app agents.
Handles: LLM setup, history conversion, tool calling, response extraction.
"""

import re
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment = AZURE_OPENAI_DEPLOYMENT,
        azure_endpoint   = AZURE_OPENAI_ENDPOINT,
        api_key          = AZURE_OPENAI_API_KEY or None,
        api_version      = AZURE_OPENAI_API_VERSION,
        temperature      = 0.1,
        max_tokens       = 2500,
    )


def _history_to_lc(history: list[dict]) -> list:
    result = []
    for msg in history:
        role, content = msg.get("role", ""), msg.get("content", "")
        if role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


def _strip_html(text: str) -> str:
    clean = re.sub(r"<(?!\!\[)[^>]+>", "", text)
    return clean.replace("&nbsp;", " ").replace("&amp;", "&").strip()


def _extract_final_text(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return _strip_html(content)
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                joined = "\n".join(texts).strip()
                if joined:
                    return _strip_html(joined)
    return "I could not generate a response. Please try again."


_CONFIRM_RE = re.compile(
    r"\b(yes|yep|yup|yeah|sure|ok|okay|do it|go ahead|confirm|proceed|"
    r"just do it|agreed|fine|correct|please do|create it|add it)\b",
    re.IGNORECASE,
)


def _has_prior_confirmation(history: list[dict]) -> bool:
    for msg in reversed(history):
        if msg.get("role") == "user":
            return bool(_CONFIRM_RE.search(msg.get("content", "")))
    return False


def _augment_message(message: str, history: list[dict]) -> str:
    hints = []
    if _has_prior_confirmation(history):
        hints.append("(The user has already confirmed this action.)")
    return message + ("\n\n" + "\n".join(hints) if hints else "")


async def run(
    message: str,
    history: list[dict],
    tools: list,
    system_prompt: str,
) -> str:
    """Core agent runner — used by all 3 app agents."""
    if not tools:
        return "⚠️ No tools available. Please check your connection."

    llm         = get_llm()
    react_agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=_augment_message(message, history)))

    result = await react_agent.ainvoke(
        {"messages": lc_messages},
        config={"recursion_limit": 20},
    )
    return _extract_final_text(result)
