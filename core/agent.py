"""
Multi-Agent LangGraph System  (Improved)
=========================================
Key improvements over original:
  1. Smart ReAct master prompt — understands natural language, maps intent to tools
  2. Dynamic tool listing — every prompt includes the actual tool names + descriptions
     fetched at runtime so the LLM always knows exactly what's available
  3. Connection caching — MCP clients are reused across calls (see web_app.py)
  4. Unified error messages in Markdown (no stray HTML)
  5. "Both CRMs" prompt now lists tools from each CRM separately so the LLM
     knows which CRM each tool belongs to
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
        temperature      = 0.3,   # lower = more reliable tool selection
        max_tokens       = 8000,
    )


# =============================================================================
# Dynamic tool description builder
# =============================================================================

def _build_tool_table(tools: list) -> str:
    """
    Returns a Markdown table of tool names + descriptions.
    Included in every system prompt so the LLM knows exactly what it can do.
    """
    if not tools:
        return "_No tools loaded._"
    lines = ["| Tool | Description |", "|------|-------------|"]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = (getattr(t, "description", "") or "").strip().split("\n")[0][:120]
        lines.append(f"| `{name}` | {desc} |")
    return "\n".join(lines)


# =============================================================================
# Master ReAct system prompt builder
# =============================================================================

_REACT_PREAMBLE = """\
You are an expert CRM AI assistant. You operate using the **ReAct** (Reason + Act) method:

**ReAct Loop**
1. **Thought** — Read the user's message, identify their intent, and decide which tool to call.
2. **Action** — Call the most appropriate tool with correct parameters.
3. **Observation** — Read the tool result.
4. **Repeat** — If the result is incomplete, call another tool. Stop when you have a full answer.
5. **Answer** — Respond clearly in Markdown.

**Intent Recognition Rules**
- "show me", "list", "get", "find", "what are" → READ operation (getRecords / search)
- "create", "add", "new" → CREATE operation (confirm once before executing)
- "update", "change", "edit", "set" → UPDATE operation (confirm once before executing)
- "delete", "remove" → DELETE operation (confirm twice before executing)
- "convert" → convertLead or similar conversion tool
- "compare", "both" → query both CRMs

**Output Format Rules (CRITICAL)**
- NEVER output raw HTML: no <div>, <p>, <span>, <table>, <script>
- ALWAYS use clean Markdown: **bold**, *italic*, ## headings, bullet lists, | tables |
- Present ALL record lists as Markdown tables with clear column headers
- Strip any HTML that comes back from tools before showing it to the user
- If a tool errors, tell the user what went wrong and suggest a fix

**Confirmation Rules**
- Ask for confirmation ONCE before any create/update/delete
- If the user has already confirmed (said "yes", "do it", "confirm"), proceed immediately
- Never ask for confirmation on read-only operations

**Available Tools**
{tool_table}

{crm_specific_section}
"""

# ── HubSpot-specific section ─────────────────────────────────────────────────

_HUBSPOT_SECTION = """\
## HubSpot CRM Guide

**Primary tool:** `manage_crm_objects`
- Create: `createRequest: {"objects": [{"objectType": "contacts", "properties": {"firstname": "Jane", "email": "jane@acme.com"}}]}, confirmationStatus: "CONFIRMED"`
- Update: `updateRequest: {"objects": [{"objectType": "contacts", "objectId": 12345, "properties": {"jobtitle": "Manager"}}]}, confirmationStatus: "CONFIRMED"`

**Other tools:**
- `search_crm_objects` — Search/filter any CRM object type
- `get_crm_objects` — Fetch records by ID
- `search_owners` — Find HubSpot users/owners
- `get_user_details` — Current user info
- `get_properties` — List available properties for an object type

**Error handling:**
- `FORMAT_ERROR` → fix parameter format and retry silently
- `PERMISSION_DENIED` → STOP, tell user which scope is missing
- Never expose tokens or credentials

{permission_section}
"""

def _hubspot_permission_block(granted_scopes, is_admin):
    lines = [f"**Your role:** {'🔑 Super Admin' if is_admin else '👤 Standard User'}"]
    tool_scope_map = build_tool_scope_map(granted_scopes)
    accessible = [t for t, s in tool_scope_map.items() if s]
    blocked    = [t for t, s in tool_scope_map.items() if not s]
    if accessible:
        lines.append("**Accessible tools:** " + ", ".join(f"`{t}`" for t in accessible))
    if blocked:
        lines.append("**Blocked tools:** " + ", ".join(f"`{t}`" for t in blocked))
    return "\n".join(lines)


def build_hubspot_prompt(tools: list, granted_scopes: list, is_admin: bool) -> str:
    tool_table = _build_tool_table(tools)
    perm_block = _hubspot_permission_block(granted_scopes, is_admin)
    crm_section = _HUBSPOT_SECTION.replace("{permission_section}", perm_block)
    return _REACT_PREAMBLE.replace("{tool_table}", tool_table).replace("{crm_specific_section}", crm_section)


# ── Zoho-specific section ────────────────────────────────────────────────────

_ZOHO_SECTION = """\
## Zoho CRM Guide

**CRITICAL — `fields` is always required:**
ALL `get*Records` tools require `query_params` with a `fields` key.
Never call them with `query_params: null`. Minimum: `{"fields": "id,Last_Name,First_Name,Email,Phone"}`

**Common operations:**

| Intent | Tool | Notes |
|--------|------|-------|
| List leads | `getLeadsRecords` | fields: id,First_Name,Last_Name,Email,Lead_Status |
| List contacts | `getContactsRecords` | fields: id,First_Name,Last_Name,Email,Phone |
| List deals | `getDealsRecords` | fields: id,Deal_Name,Stage,Amount,Closing_Date |
| List accounts | `getAccountsRecords` | fields: id,Account_Name,Phone,Industry |
| Search | `searchRecords` | criteria: `(Email:equals:user@example.com)` |
| COQL query | `executeCOQLQuery` | `SELECT id,Last_Name FROM Leads WHERE Lead_Status='New'` |
| Create records | `createRecords` / `create*Records` | Run `getFields` first to get correct field names |
| Update records | `updateRecord` / `updateRecords` | Requires record ID |
| Convert lead | `convertLead` | Find lead ID first with searchRecords |
| List modules | `getModules` | Shows all available CRM modules |
| Get fields | `getFields` | Always call before create/update |

**Module API names:** Leads, Contacts, Accounts, Deals, Tasks, Events, Calls, Cases,
Products, Vendors, Quotes, Sales_Orders, Purchase_Orders, Invoices, Campaigns

**Workflow for creating records:**
1. `getFields` with module name → see exact field API names
2. `create*Records` with correct field names
3. Confirm success, show new record ID

**Workflow for finding a contact:**
1. `searchRecords` module=Contacts criteria=(Email:equals:user@example.com)
2. Show result as Markdown table
"""

def build_zoho_prompt(tools: list) -> str:
    tool_table = _build_tool_table(tools)
    return _REACT_PREAMBLE.replace("{tool_table}", tool_table).replace("{crm_specific_section}", _ZOHO_SECTION)


# ── Both CRMs section ────────────────────────────────────────────────────────

_BOTH_SECTION = """\
## Multi-CRM Guide

You have access to tools from **both HubSpot and Zoho CRM**.
Tool names starting with common prefixes indicate their origin:
- HubSpot tools: `manage_crm_objects`, `search_crm_objects`, `get_crm_objects`, `search_owners`, etc.
- Zoho tools: `getLeadsRecords`, `getContactsRecords`, `getDealsRecords`, `searchRecords`, etc.

**When user asks for "both CRMs":**
1. Query HubSpot first (e.g., `search_crm_objects` for contacts)
2. Query Zoho second (e.g., `getContactsRecords`)
3. Combine and present results in a unified Markdown table with a **Source** column

**When user asks about one CRM specifically**, only query that CRM's tools.

Follow the same field/confirmation rules as the individual CRM guides above.
"""

def build_both_prompt(tools: list, granted_scopes: list, is_admin: bool) -> str:
    tool_table = _build_tool_table(tools)
    perm_note  = ""
    if granted_scopes:
        perm_note = f"\n**HubSpot role:** {'Super Admin' if is_admin else 'Standard User'}"
    section = _BOTH_SECTION + perm_note
    return _REACT_PREAMBLE.replace("{tool_table}", tool_table).replace("{crm_specific_section}", section)


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
    """Remove stray HTML tags from LLM response."""
    clean = re.sub(r"<(?!!\\[)[^>]+>", "", text)
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
# Public agent runner
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

    if agent == "zoho_crm":
        crm_clients = {k: v for k, v in clients.items() if k == "zoho_crm"}
    else:
        crm_clients = {k: v for k, v in clients.items() if k == "hubspot"}

    tools = await get_langchain_tools(crm_clients, granted_scopes=scopes)

    if not tools:
        label = "Zoho CRM" if agent == "zoho_crm" else "HubSpot"
        return (
            f"⚠️ **{label} not connected.**\n\n"
            f"Please connect your {label} account using the **Connect** button in the sidebar."
        )

    # Build prompt WITH the actual tool list — LLM now knows exactly what tools exist
    if agent == "zoho_crm":
        system_prompt = build_zoho_prompt(tools)
    else:
        system_prompt = build_hubspot_prompt(tools, scopes, is_admin)

    react_agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke(
        {"messages": lc_messages},
        config={"recursion_limit": 25},
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
            "Connect at least one CRM (HubSpot or Zoho) from the sidebar."
        )

    system_prompt = build_both_prompt(tools, scopes, is_admin)

    react_agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke(
        {"messages": lc_messages},
        config={"recursion_limit": 25},
    )

    return _extract_final_text(result)
