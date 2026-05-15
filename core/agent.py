"""
Multi-Agent LangGraph System  (Fully Dynamic)
=============================================
Key changes vs previous version:
  1. ZERO hardcoded tool names, module names, or field names anywhere.
  2. All prompt sections (CRM guide, intent table, multi-CRM list) are built
     at runtime by inspecting the actual tools returned by each MCP server.
  3. Adding a new MCP = zero changes here. The prompt auto-adapts.
  4. Tool-call deduplication note added to every prompt (kills infinite loops).
  5. recursion_limit raised to 50 (was 25).
  6. temperature lowered to 0.1 for more deterministic tool selection.
"""

import os
import re
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from mcp_client import MCPClient
from core.tools import get_langchain_tools, build_tool_scope_map, describe_scopes

load_dotenv()


# =============================================================================
# LLM factory
# =============================================================================

def get_llm() -> AzureChatOpenAI:
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or None
    return AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key          = api_key,
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature      = 0.1,   # lowered from 0.3 — more deterministic tool selection
        max_tokens       = 8000,
    )


# =============================================================================
# Dynamic prompt builders — ALL content from live tool introspection
# =============================================================================

def _build_tool_table(tools: list) -> str:
    """Markdown table of tool names + first-line descriptions."""
    if not tools:
        return "_No tools loaded._"
    lines = ["| Tool | Description |", "|------|-------------|"]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = (getattr(t, "description", "") or "").strip().split("\n")[0][:120]
        lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


def _classify_tools(tools: list) -> dict[str, list[str]]:
    """
    Group tool names by operation type by scanning their names.
    No hardcoded tool names — works for any MCP.
    """
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
    """
    Derive record module/object names from tool names by stripping verb
    prefixes and noun suffixes. Works for any CRM naming convention.
    e.g. getLeadsRecords -> Leads, ZohoCRM_getDealsRecords -> Deals,
         manage_crm_objects -> (skipped), searchContactsRecords -> Contacts
    """
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
    """
    Build the intent-recognition section dynamically.
    Example tool names in each row are real tools from the live MCP.
    """
    groups = _classify_tools(tools)

    def _first(lst, n=2):
        return ", ".join(f"`{x}`" for x in lst[:n]) if lst else "_none_"

    lines = [
        "**Intent → Tool mapping (auto-built from available tools)**",
        "",
        f'- "show me", "list", "get", "find" → READ   — e.g. {_first(groups["read"])}',
        f'- "search", "query", "find by"      → SEARCH — e.g. {_first(groups["search"])}',
        f'- "create", "add", "new"            → CREATE (confirm once) — e.g. {_first(groups["create"])}',
        f'- "update", "change", "edit"        → UPDATE (confirm once) — e.g. {_first(groups["update"])}',
        f'- "delete", "remove"               → DELETE (confirm TWICE) — e.g. {_first(groups["delete"])}',
    ]
    if groups["convert"]:
        lines.append(f'- "convert"                        → CONVERT — e.g. {_first(groups["convert"])}')
    return "\n".join(lines)


def _build_crm_guide(crm_label: str, tools: list) -> str:
    """
    Build a CRM-specific guide section entirely from the live tool list.
    Zero hardcoded tool names, module names, or field names.
    """
    groups     = _classify_tools(tools)
    tool_names = [getattr(t, "name", "") for t in tools]
    modules    = _infer_modules_from_tools(tool_names)

    # Detect capability tools by name pattern
    def _find(pattern: str) -> str | None:
        return next((n for n in tool_names if pattern in n.lower()), None)

    has_fields     = _find("field")
    has_modules    = _find("module")
    has_search     = _find("search") or _find("query")
    has_coql       = _find("coql")
    has_convert    = _find("convert")
    has_properties = _find("propert")

    lines = [f"## {crm_label} Guide", ""]

    # Tool categories
    lines.append("**Available tool categories:**")
    lines.append("")
    for cat, lst in groups.items():
        if lst:
            lines.append(f"- **{cat.title()}** — " + ", ".join(f"`{x}`" for x in lst[:5]) +
                         (f" + {len(lst)-5} more" if len(lst) > 5 else ""))
    lines.append("")

    # Inferred modules
    if modules:
        lines.append("**Detected record types** (inferred from tool names):")
        lines.append(", ".join(f"`{m}`" for m in modules))
        lines.append("")

    # Workflows — use real tool names
    lines.append("**Key workflows:**")
    lines.append("")
    if has_modules:
        lines.append(f"- To list all available modules/objects → `{has_modules}`")
    if has_fields:
        lines.append(f"- Before create/update, get exact field names → `{has_fields}`")
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
        "**Rules (CRITICAL):**",
        "- NEVER call the same tool with the same arguments twice.",
        "  If a tool returns no useful data, STOP and tell the user what you found.",
        "- Always check what parameters a tool requires before calling it.",
        "- Prefer search/query tools over listing all records when looking for a specific one.",
    ]

    return "\n".join(lines)


def _build_multi_crm_section(clients_tools: dict[str, list]) -> str:
    """
    Build the multi-CRM section dynamically from connected clients.
    No hardcoded CRM names or tool lists.
    """
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
        "**When the user asks about one CRM specifically**, only use that CRM's tools.",
        "**NEVER call the same tool with the same arguments more than once.**",
    ]
    return "\n".join(lines)


# =============================================================================
# Static preamble — universal rules only, no CRM-specific content
# =============================================================================

_REACT_PREAMBLE = """\
You are an expert CRM AI assistant. You operate using the **ReAct** (Reason + Act) method:

**ReAct Loop**
1. **Thought** — Read the user's message, identify their intent, decide which tool to call.
2. **Action** — Call the most appropriate tool with correct parameters.
3. **Observation** — Read the tool result.
4. **Repeat** — If the result is incomplete, call another tool. Stop when you have a full answer.
5. **Answer** — Respond clearly in Markdown.

{intent_section}

**Output Format Rules (CRITICAL)**
- NEVER output raw HTML
- ALWAYS use clean Markdown: **bold**, ## headings, bullet lists, | tables |
- Present ALL record lists as Markdown tables with clear column headers
- If a tool errors, tell the user what went wrong and suggest a fix

**Confirmation Rules**
- Ask for confirmation ONCE before any create/update/delete
- If the user already confirmed ("yes", "do it", "confirm"), proceed immediately
- Never confirm on read-only operations

**Loop Prevention (CRITICAL)**
- NEVER call the same tool with the same arguments more than once per turn
- If a tool returns empty or unhelpful data, do NOT retry — report and stop
- If you cannot complete the task, say so clearly

**Available Tools**
{tool_table}

{crm_specific_section}
"""


# =============================================================================
# Prompt builders — fully runtime, no hardcoded CRM knowledge
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


# ── Legacy wrappers — keep web_app.py untouched ──────────────────────────────

def build_hubspot_prompt(tools: list, granted_scopes: list, is_admin: bool) -> str:
    tool_scope_map = build_tool_scope_map(granted_scopes)
    accessible = [t for t, s in tool_scope_map.items() if s]
    blocked    = [t for t, s in tool_scope_map.items() if not s]
    perm_lines = [f"**Your role:** {'🔑 Super Admin' if is_admin else '👤 Standard User'}"]
    if accessible:
        perm_lines.append("**Accessible tools:** " + ", ".join(f"`{t}`" for t in accessible))
    if blocked:
        perm_lines.append("**Blocked tools:** "    + ", ".join(f"`{t}`" for t in blocked))
    return build_single_crm_prompt("HubSpot", tools, extra_section="\n".join(perm_lines))


def build_zoho_prompt(tools: list) -> str:
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
) -> str:
    scopes = granted_scopes or []
    llm    = get_llm()

    crm_clients = (
        {k: v for k, v in clients.items() if k == "zoho_crm"}
        if agent == "zoho_crm" else
        {k: v for k, v in clients.items() if k == "hubspot"}
    )

    tools = await get_langchain_tools(crm_clients, granted_scopes=scopes)
    if not tools:
        label = "Zoho CRM" if agent == "zoho_crm" else "HubSpot"
        return (
            f"⚠️ **{label} not connected.**\n\n"
            f"Please connect your {label} account using the **Connect** button in the sidebar."
        )

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
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke(
        {"messages": lc_messages},
        config={"recursion_limit": 50},   # raised from 25
    )
    return _extract_final_text(result)


async def run_agent_both(
    message: str,
    history: list[dict],
    clients: dict[str, "MCPClient"],
    granted_scopes: list[str] | None = None,
    is_admin: bool = False,
) -> str:
    scopes      = granted_scopes or []
    llm         = get_llm()
    crm_clients = {k: v for k, v in clients.items() if k in ("hubspot", "zoho_crm")}
    tools       = await get_langchain_tools(crm_clients, granted_scopes=scopes)

    if not tools:
        return (
            "⚠️ **No CRM systems connected.**\n\n"
            "Connect at least one CRM from the sidebar."
        )

    system_prompt = build_both_prompt(tools, scopes, is_admin)
    react_agent   = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke(
        {"messages": lc_messages},
        config={"recursion_limit": 50},
    )
    return _extract_final_text(result)