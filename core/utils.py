"""
core/utils.py
=============
Shared agent utility helpers to eliminate duplicate code across agents.
Includes query complexity classification, standard Azure OpenAI LLM setup,
and history-to-messages list conversions.
"""

from __future__ import annotations
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

_MAX_TOKENS_SIMPLE = 1200
_MAX_TOKENS_COMPLEX = 3000

def is_complex_query(message: str) -> bool:
    """
    Determine if a query requires a complex context window based on keywords.
    
    Args:
        message: The raw user message.
        
    Returns:
        True if the query matches complex keywords or length, False otherwise.
    """
    msg = message.lower()
    complex_keywords = (
        "create", "update", "delete", "approve", "reject", 
        "add", "and then", "also", "multiple"
    )
    if any(p in msg for p in complex_keywords):
        return True
    return len(message.split()) > 20


def get_agent_llm(message: str | None = None, max_tokens: int | None = None) -> AzureChatOpenAI:
    """
    Return a configured Azure OpenAI model with streaming enabled.
    
    Args:
        message: Optional user message to dynamically adjust token limit.
        max_tokens: Optional explicit token limit.
        
    Returns:
        An instantiated AzureChatOpenAI instance.
    """
    if max_tokens is None:
        if message is not None:
            max_tokens = _MAX_TOKENS_COMPLEX if is_complex_query(message) else _MAX_TOKENS_SIMPLE
        else:
            max_tokens = _MAX_TOKENS_COMPLEX

    return AzureChatOpenAI(
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        api_key          = os.getenv("AZURE_OPENAI_API_KEY"),
        temperature      = 0,
        max_tokens       = max_tokens,
        streaming        = True,
    )


def build_agent_messages(system_prompt: str, history: list[dict], message: str) -> list:
    """
    Trims, maps roles, and formats conversation history into a list of LangChain Messages.
    
    Args:
        system_prompt: The agent system instructions.
        history: Cumulative list of previous messages in the session.
        message: The latest user message.
        
    Returns:
        A list of SystemMessage, HumanMessage, and AIMessage objects.
    """
    trimmed_history = history[-10:] if len(history) > 10 else history

    messages = [SystemMessage(content=system_prompt)]
    for m in trimmed_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    messages.append(HumanMessage(content=message))
    return messages
