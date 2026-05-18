"""
Multi-Agent LangGraph System  (Fully Dynamic)
=============================================
v3 — Session-based tool cache + intent filtering

Changes vs v2:
  - All 95 tools are loaded ONCE per session (not on every message).
  - Per message, only 6-12 relevant tools are passed to the LLM.
  - Session ends → cache entry deleted automatically (TTL or explicit evict).
  - Keeps all v2 improvements: demo data, confirmation rules, casual language.
"""

import os
import re
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from mcp_client import MCPClient
from core.tools import get_langchain_tools, get_tool_scope_map, describe_scopes
from core.session_cache import SessionToolCache

load_dotenv()

# Global session cache — one instance for the whole app lifetime
_session_cache = SessionToolCache()


# =============================================================================
# LLM factory
# =============================================================================

def get_llm() -> AzureChatOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or None
    return AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key          = api_key,
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        temperature      = 0.1,
        max_tokens       = 2500,
    )


# =============================================================================
# Dynamic prompt builders
# =============================================================================

def _build_tool_table(tools: list) -> str:
    if not tools:
        return "_No tools loaded._"
    lines = ["| Tool | Description |", "|------|-------------|"]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = (getattr(t, "description", "") or "").strip().split("\n")[0][:120]
        lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


def _classify_tools(tools: list) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "read": [], "create": [], "update": [], "delete": [],
        "search": [], "convert": [], "other": [],
    }
    for t in tools:
        name = getattr(t, "name", "").lower()
        if any(x in name for x in ("get", "list", "fetch", "read")):
            groups["read"].append(t.name)
        elif any(x in name for x in ("create", "insert", "add", "post")):
            groups["create"].append(t.name)
        elif any(x in name for x in ("update", "edit", "patch", "put", "upsert")):
            groups["update"].append(t.name)
        elif any(x in name for x in ("delete", "remove", "archive")):
            groups["delete"].append(t.name)
        elif any(x in name for x in ("search", "find", "query", "coql")):
            groups["search"].append(t.name)
        elif any(x in name for x in ("convert", "transform", "merge")):
            groups["convert"].append(t.name)
        else:
            groups["other"].append(t.name)
    return groups


def _infer_modules_from_tools(tool_names: list[str]) -> list[str]:
    prefixes = r"^(ZohoCRM_|hs_|crm_)?(get|create|update|upsert|delete|clone|search|list|fetch|manage)"
    suffixes = r"(Records?|Items?|Objects?|Entries?)$"
    skip_words = {
        "crm", "object", "field", "module", "record", "property",
        "connection", "workflow", "configuration", "template", "email",
        "inventory", "territory", "blueprint", "mass", "lite", "deleted",
    }
    seen: set[str] = set()
    modules: list[str] = []
    for name in tool_names:
        stripped = re.sub(prefixes, "", name, flags=re.IGNORECASE)
        stripped = re.sub(suffixes, "", stripped, flags=re.IGNORECASE)
        if stripped and stripped.lower() not in skip_words and len(stripped) > 2:
            if stripped not in seen:
                seen.add(stripped)
                modules.append(stripped)
    return sorted(modules)


def _build_intent_table(tools: list) -> str:
    groups = _classify_tools(tools)

    def _first(lst, n=2):
        return ", ".join(f"`{x}`" for x in lst[:n]) if lst else "_none_"

    lines = [
        "**Intent → Tool mapping (auto-built from available tools)**",
        "",
        f'- "show", "list", "get", "find", "what are", "display"  → READ   — e.g. {_first(groups["read"])}',
        f'- "search", "query", "find by", "look up"                → SEARCH — e.g. {_first(groups["search"])}',
        f'- "create", "add", "new", "make", "put", "insert"        → CREATE — e.g. {_first(groups["create"])}',
        f'- "update", "change", "edit", "fix", "modify"            → UPDATE — e.g. {_first(groups["update"])}',
        f'- "delete", "remove", "get rid of"                       → DELETE — e.g. {_first(groups["delete"])}',
    ]
    if groups["convert"]:
        lines.append(f'- "convert", "move to"                                    → CONVERT — e.g. {_first(groups["convert"])}')
    return "\n".join(lines)


def _build_crm_guide(crm_label: str, tools: list) -> str:
    groups     = _classify_tools(tools)
    tool_names = [getattr(t, "name", "") for t in tools]
    modules    = _infer_modules_from_tools(tool_names)

    def _find(pattern: str) -> str | None:
        return next((n for n in tool_names if pattern in n.lower()), None)

    has_fields     = _find("field")
    has_modules    = _find("module")
    has_search     = _find("search") or _find("query")
    has_coql       = _find("coql")
    has_convert    = _find("convert")
    has_properties = _find("propert")

    lines = [f"## {crm_label} Guide", ""]
    lines.append("**Available tool categories:**")
    lines.append("")
    for cat, lst in groups.items():
        if lst:
            lines.append(f"- **{cat.title()}** — " + ", ".join(f"`{x}`" for x in lst[:5]) +
                         (f" + {len(lst)-5} more" if len(lst) > 5 else ""))
    lines.append("")

    if modules:
        lines.append("**Detected record types** (inferred from tool names):")
        lines.append(", ".join(f"`{m}`" for m in modules))
        lines.append("")

    lines.append("**Key workflows:**")
    lines.append("")
    if has_modules:
        lines.append(f"- To list all available modules/objects → `{has_modules}`")
    if has_fields:
        lines.append(f"- Before create/update, ALWAYS get exact field names first → `{has_fields}`")
    if has_properties:
        lines.append(f"- To list object properties → `{has_properties}`")
    if has_search:
        lines.append(f"- To find a specific record → `{has_search}`")
    if has_coql:
        lines.append(f"- For complex filter queries → `{has_coql}` (SQL-like syntax)")
    if has_convert:
        lines.append(f"- To convert a record (e.g. lead→contact) → find ID first, then `{has_convert}`")

    lines += [
        "",
        "**Field Discovery Workflow (before create/update):**",
        "1. Call the fields/modules tool to get exact field API names for the target module.",
        "2. Use only field names returned by that tool.",
        "3. Then call the create/update tool with those exact field names.",
        "",
        "**Best practices:**",
        "- Avoid calling the same tool with the same arguments twice.",
        "- If a tool returns no useful data, report to the user and stop.",
        "- Prefer search/query tools over listing all records when looking for a specific one.",
    ]
    return "\n".join(lines)


def _build_multi_crm_section(clients_tools: dict[str, list]) -> str:
    lines = ["## Multi-CRM Guide", ""]
    lines.append("You have access to tools from **multiple CRM systems**:")
    lines.append("")
    for crm_key, tools in clients_tools.items():
        tool_names = [getattr(t, "name", "") for t in tools]
        sample = tool_names[:4]
        label  = crm_key.replace("_", " ").title()
        suffix = f" + {len(tool_names)-4} more" if len(tool_names) > 4 else ""
        lines.append(f"- **{label}** — {', '.join(f'`{x}`' for x in sample)}{suffix}")
    lines += [
        "",
        '**When the user asks about multiple CRMs:**',
        "1. Query each CRM in sequence using its own tools.",
        "2. Present combined results in a Markdown table with a **Source** column.",
        "",
        "When the user asks about one CRM specifically, only use that CRM's tools.",
    ]
    return "\n".join(lines)


# =============================================================================
# System prompt
# =============================================================================

_REACT_PREAMBLE = """\
You are a CRM assistant that helps users manage their data using available tools.

## How You Work

You follow a structured approach:
1. Understand what the user is asking for.
2. Call the appropriate tool with correct parameters.
3. Read the result and decide if more information is needed.
4. Respond with a clear, formatted answer in Markdown.

## Understanding User Input

Users may write casually or with typos. Interpret their intent:
- "create", "make", "add", "new" → they want to create a record.
- "show", "list", "get", "fetch" → they want to read data.
- "update", "change", "edit" → they want to modify a record.
- "delete", "remove" → they want to delete a record.
- "yes", "ok", "sure", "do it", "go ahead" → they are confirming a previous action.
- "sample", "demo", "test data", "you decide" → they want you to fill in reasonable values.

## Working with Sample Data

When asked to create sample or demo records, use plausible values such as:
- Names: "John Demo", "Jane Sample"
- Emails: "john.demo@acme.com"
- Companies: "Acme Corp", "Test Co"
- Amounts: 10000, 25000

## Confirmations

- For create, update, or delete actions: confirm with the user once before proceeding.
- If the user already said "yes" or "confirm" in their previous message, proceed directly.
- Read-only actions (list, search, get) do not need confirmation.

## Error Handling

- If a tool returns a permission error, let the user know which scope is missing.
- If a tool returns a format error, try once more with corrected parameters.
- If a tool returns no useful data, let the user know.

## Tool Usage

- Avoid calling the same tool with identical arguments more than once per request.
- Limit retries to 2 attempts per action.

{intent_section}

**Output format:** Markdown. Use tables for record lists.

**Available Tools:**
{tool_table}

{crm_specific_section}
"""


# =============================================================================
# Prompt builders
# =============================================================================

def build_single_crm_prompt(crm_label: str, tools: list, extra_section: str = "") -> str:
    tool_table     = _build_tool_table(tools)
    intent_section = _build_intent_table(tools)
    crm_guide      = _build_crm_guide(crm_label, tools)
    crm_section    = crm_guide + ("\n\n" + extra_section if extra_section else "")
    return (
        _REACT_PREAMBLE
        .replace("{tool_table}",           tool_table)
        .replace("{intent_section}",       intent_section)
        .replace("{crm_specific_section}", crm_section)
    )


def build_multi_crm_prompt(clients_tools: dict[str, list], extra_section: str = "") -> str:
    all_tools      = [t for tools in clients_tools.values() for t in tools]
    tool_table     = _build_tool_table(all_tools)
    intent_section = _build_intent_table(all_tools)
    crm_section    = _build_multi_crm_section(clients_tools)
    if extra_section:
        crm_section += "\n\n" + extra_section
    return (
        _REACT_PREAMBLE
        .replace("{tool_table}",           tool_table)
        .replace("{intent_section}",       intent_section)
        .replace("{crm_specific_section}", crm_section)
    )


def build_hubspot_prompt(tools: list, granted_scopes: list, is_admin: bool) -> str:
    tool_scope_map = get_tool_scope_map(granted_scopes)
    accessible = [t for t, s in tool_scope_map.items() if s]
    blocked    = [t for t, s in tool_scope_map.items() if not s]
    perm_lines = [f"**Your role:** {'🔑 Super Admin' if is_admin else '👤 Standard User'}"]
    if accessible:
        perm_lines.append("**Accessible tools:** " + ", ".join(f"`{t}`" for t in accessible))
    if blocked:
        perm_lines.append("**Blocked tools:** "    + ", ".join(f"`{t}`" for t in blocked))
    return build_single_crm_prompt("HubSpot", tools, extra_section="\n".join(perm_lines))


def  build_zoho_prompt(tools: list) -> str:
    return build_single_crm_prompt("Zoho CRM", tools)


def build_both_prompt(tools: list, granted_scopes: list, is_admin: bool) -> str:
    hs_tools   = [t for t in tools if not getattr(t, "name", "").startswith("ZohoCRM_")]
    zoho_tools = [t for t in tools if     getattr(t, "name", "").startswith("ZohoCRM_")]
    clients_tools = {}
    if hs_tools:   clients_tools["hubspot"]  = hs_tools
    if zoho_tools: clients_tools["zoho_crm"] = zoho_tools
    if not clients_tools: clients_tools["crm"] = tools
    role_note = f"\n**HubSpot role:** {'Super Admin' if is_admin else 'Standard User'}" if granted_scopes else ""
    return build_multi_crm_prompt(clients_tools, extra_section=role_note)


# =============================================================================
# Utilities
# =============================================================================

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
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
    clean = clean.replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    return clean.strip()


def _extract_final_text(result: dict) -> str:
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return _strip_html(content)
            if isinstance(content, list):
                texts  = [b.get("text", "") for b in content
                          if isinstance(b, dict) and b.get("type") == "text"]
                joined = "\n".join(texts).strip()
                if joined:
                    return _strip_html(joined)
    return "I could not generate a response. Please try again."


_CONFIRM_RE = re.compile(
    r"\b(yes|yep|yup|yeah|ya|sure|ok|okay|do it|go ahead|confirm|proceed|"
    r"just do it|go for it|agreed|fine|correct|right|please do|make it|"
    r"create it|add it)\b",
    re.IGNORECASE,
)

_DEMO_RE = re.compile(
    r"\b(demo|test[\s_-]?data|sample|anything|whatever|you\s+(can\s+)?decide|"
    r"put\s+anything|make\s+(it\s+)?up|fill\s+it|use\s+(any|test|sample|fake|dummy)|"
    r"don.?t\s+(care|matter)|just\s+(do|create|add|make))\b",
    re.IGNORECASE,
)


def _has_prior_confirmation(history: list[dict]) -> bool:
    # B15: only check the immediately preceding user message, not all history
    for msg in reversed(history):
        if msg.get("role") == "user":
            return bool(_CONFIRM_RE.search(msg.get("content", "")))
    return False


def _augment_message(message: str, history: list[dict]) -> str:
    hints: list[str] = []
    if _has_prior_confirmation(history):
        hints.append(
            "(The user has already confirmed this action in a previous message.)"
        )
    recent_user = " ".join(
        m.get("content", "") for m in history[-3:] if m.get("role") == "user"
    )
    if _DEMO_RE.search(message) or _DEMO_RE.search(recent_user):
        hints.append(
            "(The user wants sample or demo data. Use reasonable placeholder values.)"
        )
    return message + ("\n\n" + "\n".join(hints) if hints else "")


# =============================================================================
# Cache management — exposed for web_app.py
# =============================================================================

async def evict_session(session_id: str):
    """Call this when a session ends (disconnect, logout, browser close)."""
    await _session_cache.evict(session_id)


async def evict_stale_sessions():
    """Call this from a background task periodically."""
    await _session_cache.evict_stale()


async def evict_all_sessions():
    """Call this on disconnect / account switch to flush stale scope guards."""
    await _session_cache.evict_all()


def get_cache_stats() -> dict:
    """For /api/cache-stats debug endpoint."""
    return _session_cache.stats()


# =============================================================================
# Public agent runners
# =============================================================================

async def run_agent(
    message: str,
    history: list[dict],
    clients: dict[str, "MCPClient"],
    agent: str = "hubspot",
    granted_scopes: list[str] | None = None,
    is_admin: bool = False,
    session_id: str = "default",
) -> str:
    scopes = granted_scopes or []
    llm    = get_llm()

    crm_clients = (
        {k: v for k, v in clients.items() if k == "zoho_crm"}
        if agent == "zoho_crm" else
        {k: v for k, v in clients.items() if k == "hubspot"}
    )

    # Load ALL tools once per session, then filter per message
    all_tools = await _session_cache.get_or_load(
        session_id, agent, crm_clients, granted_scopes=scopes
    )
    if not all_tools:
        label = "Zoho CRM" if agent == "zoho_crm" else "HubSpot"
        return (
            f"⚠️ **{label} not connected.**\n\n"
            f"Please connect your {label} account using the **Connect** button in the sidebar."
        )

    # Filter to only the relevant tools for this message
    tools = _session_cache.filter_for_message(message, all_tools)

    system_prompt = (
        build_zoho_prompt(tools)     if agent == "zoho_crm" else
        build_hubspot_prompt(tools, scopes, is_admin)
    )

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


async def run_agent_both(
    message: str,
    history: list[dict],
    clients: dict[str, "MCPClient"],
    granted_scopes: list[str] | None = None,
    is_admin: bool = False,
    session_id: str = "default",
) -> str:
    scopes      = granted_scopes or []
    llm         = get_llm()
    crm_clients = {k: v for k, v in clients.items() if k in ("hubspot", "zoho_crm")}

    all_tools = await _session_cache.get_or_load(
        session_id, "both", crm_clients, granted_scopes=scopes
    )
    if not all_tools:
        return (
            "⚠️ **No CRM systems connected.**\n\n"
            "Connect at least one CRM from the sidebar."
        )

    tools = _session_cache.filter_for_message(message, all_tools)

    system_prompt = build_both_prompt(tools, scopes, is_admin)
    react_agent   = create_react_agent(
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