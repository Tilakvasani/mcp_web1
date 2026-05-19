"""
rag/hybrid_search.py
====================
Hybrid tool selection: keyword pre-filter → Chroma vector search.

Flow per query
--------------
1. Keyword scan (zero cost) → detect agent + topic substrings
2. Embed user query → Azure OpenAI (cached — same query = zero API cost)
3. Chroma vector search (fast local lookup)
   - If keyword matched an agent → filter by that agent in Chroma
   - If cross-system keywords   → search both (no filter)
   - If no keyword match        → full semantic search (no filter)
4. Re-rank: tools matching BOTH keyword AND vector float to the top
5. Always append generic/utility tools (search_crm, manage_crm etc.)
6. Return top K LangChain tool objects
"""

from __future__ import annotations
from rag.embeddings  import embed_query
from rag.chroma_store import search_tools, is_indexed
from crm_logger import log

TOP_K = 14  # max tools sent to LLM per request


# ─────────────────────────────────────────────────────────────────────────────
# Expanded intent map
# ─────────────────────────────────────────────────────────────────────────────
# Format: (user_keywords, agent, tool_name_substrings)

_INTENT_MAP: list[tuple[list[str], str, list[str]]] = [

    # ── HubSpot CRM ──────────────────────────────────────────────────────────
    (["deal", "opportunity", "revenue", "pipeline", "dael", "opport"],
     "hubspot", ["deal", "Deal"]),

    (["contact", "person", "contct", "conatct", "client"],
     "hubspot", ["contact", "Contact"]),

    (["company", "account", "organisation", "organization", "firm", "business"],
     "hubspot", ["company", "Company", "account", "Account"]),

    (["ticket", "support", "issue", "case", "complaint"],
     "hubspot", ["ticket", "Ticket"]),

    (["task", "todo", "to-do", "to do", "action item", "reminder"],
     "hubspot", ["task", "Task"]),

    (["meeting", "appointment", "schedule", "calendar", "call log"],
     "hubspot", ["meeting", "Meeting", "call", "Call"]),

    (["email", "message", "inbox", "outreach", "mail"],
     "hubspot", ["email", "Email"]),

    (["note", "comment", "log", "activity"],
     "hubspot", ["note", "Note", "engagement", "Engagement"]),

    (["owner", "rep", "salesperson", "assigned to", "assignee"],
     "hubspot", ["owner", "Owner"]),

    (["lead", "prospect", "laed", "hot lead"],
     "hubspot", ["lead", "Lead"]),

    (["quote", "proposal", "invoice", "estimate"],
     "hubspot", ["quote", "Quote", "invoice", "Invoice"]),

    (["product", "item", "sku", "catalogue", "catalog"],
     "hubspot", ["product", "Product", "line_item"]),

    (["list", "segment", "audience", "filter"],
     "hubspot", ["list", "List"]),

    (["workflow", "automation", "sequence", "enrol", "enroll"],
     "hubspot", ["workflow", "Workflow", "automation", "sequence"]),

    (["property", "field", "attribute", "custom field"],
     "hubspot", ["propert", "field", "Field"]),

    (["report", "analytics", "stats", "metric", "hubsql", "insight"],
     "hubspot", ["report", "analytic", "hubsql"]),

    (["association", "link", "relate", "relation"],
     "hubspot", ["associat"]),

    # ── Zoho People HRMS ─────────────────────────────────────────────────────
    (["employee", "staff", "worker", "member", "employ", "headcount"],
     "zoho_people", ["employee", "Employee"]),

    (["leave", "time off", "vacation", "holiday", "absent", "pto", "day off"],
     "zoho_people", ["leave", "Leave"]),

    (["attendance", "check in", "check out", "present", "log in", "login time"],
     "zoho_people", ["attendance", "Attendance"]),

    (["department", "team", "division", "group", "unit", "dept"],
     "zoho_people", ["department", "Department"]),

    (["salary", "payroll", "compensation", "pay", "ctc", "wage"],
     "zoho_people", ["salary", "Salary", "payroll", "Payroll"]),

    (["shift", "roster", "timing", "schedule", "work hours"],
     "zoho_people", ["shift", "Shift"]),

    (["appraisal", "review", "performance", "rating", "kra", "kpi"],
     "zoho_people", ["appraisal", "performance", "review", "Performance"]),

    (["onboarding", "joining", "new hire", "induction", "joining date"],
     "zoho_people", ["onboard", "join", "Onboard"]),

    (["training", "course", "learning", "lms"],
     "zoho_people", ["training", "Training", "course", "Course"]),

    (["document", "file", "certificate", "form"],
     "zoho_people", ["document", "Document", "form", "Form"]),

    # ── Cross-system ──────────────────────────────────────────────────────────
    (["sales", "sales rep", "salesperson", "account manager"],
     "both", ["contact", "Contact", "owner", "Owner", "employee", "Employee"]),

    (["absent", "on leave", "not available", "out of office", "ooo"],
     "both", ["leave", "Leave", "contact", "Contact", "owner", "Owner"]),
]

# Generic tools always included regardless of intent (work across all objects)
_GENERIC_SUBSTRINGS = [
    "search_crm", "manage_crm", "get_crm", "get_propert",
    "list_propert", "list_objects", "search_objects",
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def hybrid_search(
    message: str,
    session_id: str,
    all_tools: list,
    n_results: int = TOP_K,
) -> list:
    """
    Main entry point.
    Returns a filtered, ranked list of LangChain tool objects.
    Falls back to full tool list if Chroma is not yet indexed.
    """
    if not all_tools:
        return all_tools

    tool_map = {getattr(t, "name", ""): t for t in all_tools}

    # ── Step 1: keyword intent detection ─────────────────────────────────────
    agent_hint, kw_substrings = _detect_intent(message)

    # ── Step 2: Chroma availability check ────────────────────────────────────
    if not is_indexed(session_id):
        log("rag", "Chroma not indexed — using keyword fallback")
        return _keyword_fallback(message, all_tools, kw_substrings)

    # ── Step 3: embed query (cached) ─────────────────────────────────────────
    query_vec = await embed_query(message)

    # ── Step 4: vector search ────────────────────────────────────────────────
    # Filter by agent when we have a clear single-agent signal
    agent_filter = agent_hint if agent_hint and agent_hint != "both" else None

    ranked_names = search_tools(
        session_id   = session_id,
        query_vec    = query_vec,
        n_results    = n_results * 2,   # get extra, we'll trim after re-rank
        agent_filter = agent_filter,
    )

    if not ranked_names:
        log("rag", "Chroma returned empty — falling back to keyword filter")
        return _keyword_fallback(message, all_tools, kw_substrings)

    # ── Step 5: re-rank (keyword ∩ vector first) ─────────────────────────────
    if kw_substrings:
        priority = [n for n in ranked_names if any(s in n for s in kw_substrings)]
        rest     = [n for n in ranked_names if n not in set(priority)]
        ranked_names = (priority + rest)[:n_results]
    else:
        ranked_names = ranked_names[:n_results]

    # ── Step 6: build result — ranked + generic ───────────────────────────────
    selected = set(ranked_names)
    result   = [tool_map[n] for n in ranked_names if n in tool_map]

    for t in _generic_tools(all_tools):
        if getattr(t, "name", "") not in selected:
            result.append(t)
            selected.add(getattr(t, "name", ""))

    log("rag", f"hybrid_search → {len(result)} tools (agent={agent_hint}, kw={len(kw_substrings)})")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_intent(message: str) -> tuple[str | None, list[str]]:
    """
    Returns (agent_hint, tool_name_substrings).
    agent_hint: 'hubspot' | 'zoho_people' | 'both' | None
    """
    msg     = message.lower()
    matched: dict[str, list[str]] = {}  # agent → substrings

    for keywords, agent, substrings in _INTENT_MAP:
        if any(kw in msg for kw in keywords):
            if agent not in matched:
                matched[agent] = []
            matched[agent].extend(substrings)

    if not matched:
        return None, []

    if len(matched) > 1 or "both" in matched:
        # Multiple agents or explicit cross-system → both
        all_subs = []
        for subs in matched.values():
            all_subs.extend(subs)
        return "both", list(set(all_subs))

    agent = next(iter(matched))
    return agent, list(set(matched[agent]))


def _generic_tools(tools: list) -> list:
    """Tools that work across all object types via parameters — always include."""
    return [t for t in tools if any(s in getattr(t, "name", "").lower() for s in _GENERIC_SUBSTRINGS)]


def _keyword_fallback(message: str, tools: list, substrings: list[str]) -> list:
    """Pure keyword filter — used when Chroma isn't ready yet."""
    if not substrings:
        return tools
    filtered = [t for t in tools if any(s in getattr(t, "name", "") for s in substrings)]
    if not filtered:
        return tools
    # Always add generic tools
    generic_names = {getattr(t, "name", "") for t in _generic_tools(tools)}
    result = filtered + [t for t in tools if getattr(t, "name", "") in generic_names and t not in filtered]
    log("rag", f"keyword fallback → {len(result)}/{len(tools)} tools")
    return result
