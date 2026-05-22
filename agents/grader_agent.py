"""
agents/grader_agent.py
======================
Evaluates every agent response for quality.
Runs AFTER the main agent produces its answer.
Returns a score dict — used for logging, UI badges, future learning.

Scores (0.0 - 1.0):
  relevance    — did the answer address the question?
  completeness — is the answer complete or did it give up early?
  format       — is it well-structured (tables, lists, clear)?
  overall      — weighted average

This is lightweight: fast LLM call with JSON output.
Do NOT stream this — we want structured JSON back, not text.
"""

from __future__ import annotations
import json
from langchain_core.messages import HumanMessage, SystemMessage
from core.utils import get_agent_llm
from crm_logger import log

_GRADER_PROMPT = """You are an output quality evaluator for a business AI assistant.

Given a user question and an assistant response, score the response on:
- relevance (0.0-1.0): Did the response directly address what was asked?
- completeness (0.0-1.0): Is the answer complete? Did it actually fetch/show data or just explain it couldn't?
- format (0.0-1.0): Is the output well-structured (tables for lists, bullets for records, clear markdown)?

Return ONLY valid JSON, no explanation, no markdown:
{"relevance": 0.9, "completeness": 0.8, "format": 0.9, "overall": 0.87, "note": "one short observation"}

Be strict: if the agent said "I cannot retrieve" when tools were available, completeness = 0.2.
If the response is a table with headers and data, format = 0.9+.
"""


async def grade_response(question: str, answer: str) -> dict:
    """
    Grades a response. Returns score dict.
    Falls back gracefully if grading fails.
    """
    if not answer or len(answer) < 10:
        return {"relevance": 0.0, "completeness": 0.0, "format": 0.0, "overall": 0.0, "note": "empty response"}

    try:
        llm = get_agent_llm(max_tokens=200)  # small, fast call
        messages = [
            SystemMessage(content=_GRADER_PROMPT),
            HumanMessage(content=f"QUESTION:\n{question}\n\nRESPONSE:\n{answer[:2000]}")
        ]

        result = await llm.ainvoke(messages)
        raw = result.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        scores = json.loads(raw)
        log("grade", f"relevance={scores.get('relevance')} completeness={scores.get('completeness')} format={scores.get('format')}")
        return scores

    except Exception as e:
        log("warn", f"grader failed: {e}")
        return {"relevance": None, "completeness": None, "format": None, "overall": None, "note": "grader error"}
