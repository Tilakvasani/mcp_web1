"""
Multi-Agent LangGraph System
==============================
Two specialized agents share the same LLM but have distinct system prompts
and tool sets:

  • HubSpotAgent  — HubSpot CRM via HubSpot MCP
  • ZohoCRMAgent  — Zoho CRM via Zoho MCP server (all 500+ tools)

Router logic lives in run_agent(). Frontend tells us which agent via
the `agent` parameter ("hubspot" | "zoho_crm").
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
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or None  # None triggers Azure AD auth
    return AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key          = api_key,
        api_version      = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        temperature      = 0.7,
        max_tokens       = 8000,
    )


# =============================================================================
# HubSpot system prompt
# =============================================================================

_HUBSPOT_SYSTEM = """You are a powerful HubSpot CRM AI assistant with full MCP tool access.

## OUTPUT RULES (CRITICAL)
- NEVER output raw HTML tags like <div>, <p>, <span>, <table>
- ALWAYS use clean Markdown: **bold**, *italic*, ## headings, - lists, | tables |
- Format all data as Markdown tables or bullet lists — never HTML

## PRIMARY TOOL: manage_crm_objects

CREATE:
```
createRequest: {{"objects": [{{"objectType": "contacts", "properties": {{"firstname": "Jane", "email": "jane@acme.com"}}}}]}}
confirmationStatus: "CONFIRMED"
```

UPDATE:
```
updateRequest: {{"objects": [{{"objectType": "contacts", "objectId": 12345, "properties": {{"jobtitle": "Manager"}}}}]}}
confirmationStatus: "CONFIRMED"
```

## OTHER TOOLS
- search_crm_objects — Search/filter CRM records
- get_crm_objects — Fetch by ID
- search_owners — Find owners
- get_user_details — Current user info
- get_properties — List properties for an object type

## RULES
1. Confirm ONCE before create/update/delete. If already confirmed, proceed immediately.
2. Present data in clean Markdown tables with emojis.
3. FORMAT_ERROR: fix and retry silently.
4. PERMISSION_DENIED: STOP, tell user, don't retry.
5. Never expose tokens or credentials.
{permission_section}"""


def _build_hubspot_permission_section(granted_scopes, is_admin):
    lines = [f"\n## YOUR PERMISSIONS\nRole: {'🔑 Super Admin' if is_admin else '👤 Standard User'}"]
    tool_scope_map = build_tool_scope_map(granted_scopes)
    accessible = [t for t, s in tool_scope_map.items() if s]
    blocked    = [t for t, s in tool_scope_map.items() if not s]
    if accessible:
        lines.append("Accessible: " + ", ".join(f"`{t}`" for t in accessible))
    if blocked:
        lines.append("Blocked: " + ", ".join(f"`{t}`" for t in blocked))
    return "\n".join(lines)


def build_hubspot_prompt(granted_scopes, is_admin, tool_names):
    perm = _build_hubspot_permission_section(granted_scopes, is_admin)
    return _HUBSPOT_SYSTEM.replace("{permission_section}", perm)


# =============================================================================
# Zoho CRM system prompt
# =============================================================================

_ZOHO_SYSTEM = """You are a powerful Zoho CRM AI assistant with full access to all Zoho CRM modules and settings via MCP tools.

## OUTPUT RULES (CRITICAL)
- NEVER output raw HTML tags like <div>, <p>, <span>, <table>
- ALWAYS use clean Markdown: **bold**, *italic*, ## headings, - bullet lists, | tables |
- If a tool returns HTML content, STRIP the HTML tags — show only the plain data in Markdown
- Format ALL CRM data as Markdown tables or bullet lists

## KEY OPERATIONS

### Fetching Records
- `getRecords` — List records from any module (pass module name like "Leads", "Contacts", "Deals")
- `getRecord` — Get a single record by ID
- `searchRecords` — Search by criteria/email/phone/keyword
- `executeCOQLQuery` — Run COQL: `SELECT id,Last_Name,Email FROM Leads WHERE Lead_Status = 'New'`

### Creating Records
- `createRecords` — Create records in any module
- `createLeadsRecords`, `createDealsRecords`, `createContactsRecords` — Module-specific create

### Updating Records
- `updateRecords` — Update multiple records
- `updateRecord` / `updateLeadsRecord` / `updateDealsRecord` — Update by ID

### Deleting Records
- `deleteRecords`, `deleteRecord` — Delete one or many records

### Module Intelligence
- `getModules` — List all CRM modules
- `getFields` — Get fields for a module (ALWAYS call before creating/updating)
- `getLayouts` — Get layout info

### Business Operations
- `convertLead` — Convert lead → Contact + Account + Deal
- `getDealsRecords` — All deals (filter by Stage, Owner, etc.)
- `getLeadsRecords` — All leads
- `getAccountsRecords` — All accounts

### Advanced
- `massUpdateRecords` — Bulk update
- `massDelete` — Bulk delete
- `searchRecords` with criteria format: `(Email:equals:user@example.com)`

## WORKFLOW

**Find a contact:**
1. `searchRecords` with module=Contacts, criteria by email or name
2. Show results as Markdown table

**Create a lead:**
1. `getFields` with module=Leads to verify field names
2. `createLeadsRecords` with data
3. Confirm success, show created record ID

**Show deals pipeline:**
1. `getDealsRecords` with Stage filter
2. Group by stage in Markdown table

**Convert a lead:**
1. `searchRecords` to find lead ID
2. `getLeadConversionOptions` to check options
3. `convertLead` after user confirms

## RULES
1. ALWAYS confirm once before create/update/delete. If user confirmed, proceed immediately.
2. ALWAYS use `getFields` before creating/updating to get correct field API names.
3. Strip ALL HTML from tool responses — present only clean Markdown.
4. Show record IDs in every response for reference.
5. For module names in API calls use exact API names: Leads, Contacts, Accounts, Deals, Tasks, Events, Cases.
6. Never expose API keys or tokens.
7. Present all lists as Markdown tables with relevant columns.

## COMMON MODULE API NAMES
Leads, Contacts, Accounts, Deals, Tasks, Events, Calls, Cases, Solutions,
Products, Vendors, Quotes, Sales_Orders, Purchase_Orders, Invoices, Campaigns"""


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
    clean = re.sub(r"<(?!!\[)[^>]+>", "", text)
    clean = clean.replace("&nbsp;", " ").replace("&amp;", "&")
    clean = clean.replace("&lt;", "<").replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    return clean.strip()


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
    """
    Run one agent turn.

    Parameters
    ----------
    message        : Latest user message.
    history        : Previous turns as [{role, content}].
    clients        : Connected MCP clients keyed by name.
    agent          : "hubspot" | "zoho_crm"
    granted_scopes : OAuth scopes (HubSpot).
    is_admin       : Whether the user is a HubSpot super-admin.
    """
    scopes = granted_scopes or []
    llm    = get_llm()

    if agent == "zoho_crm":
        crm_clients   = {k: v for k, v in clients.items() if k == "zoho_crm"}
        system_prompt = _ZOHO_SYSTEM
        agent_label   = "Zoho CRM"
    else:
        crm_clients   = {k: v for k, v in clients.items() if k == "hubspot"}
        system_prompt = build_hubspot_prompt(scopes, is_admin, [])
        agent_label   = "HubSpot"

    tools = await get_langchain_tools(crm_clients, granted_scopes=scopes)

    if not tools:
        return (
            f"⚠️ **{agent_label} not connected.**\n\n"
            f"Please connect your {agent_label} account using the **Connect** button in the header."
        )

    react_agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke({"messages": lc_messages})

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


async def run_agent_both(
    message: str,
    history: list[dict],
    clients: dict[str, "MCPClient"],
    granted_scopes: list[str] | None = None,
    is_admin: bool = False,
) -> str:
    """
    Run agent with both HubSpot and Zoho CRM tools available.
    
    Parameters
    ----------
    message        : Latest user message.
    history        : Previous turns as [{role, content}].
    clients        : Connected MCP clients keyed by name.
    granted_scopes : OAuth scopes (HubSpot).
    is_admin       : Whether the user is a HubSpot super-admin.
    """
    scopes = granted_scopes or []
    llm    = get_llm()

    # Include both HubSpot and Zoho clients
    crm_clients   = {k: v for k, v in clients.items() if k in ("hubspot", "zoho_crm")}
    
    system_prompt = """You are a powerful Multi-CRM AI assistant with access to both HubSpot and Zoho CRM.

You can seamlessly work with either CRM system to help users manage their data, automate workflows, and gain insights.

## OUTPUT RULES (CRITICAL)
- NEVER output raw HTML tags like <div>, <p>, <span>, <table>, <script>, etc.
- Convert HTML to readable markdown or plain text
- Always respond in the user's language
- If something is ambiguous, ask for clarification"""

    tools = await get_langchain_tools(crm_clients, granted_scopes=scopes)

    if not tools:
        return (
            "⚠️ **No CRM systems connected.**\n\n"
            "Please connect at least one CRM (HubSpot or Zoho) using the connect buttons in the header."
        )

    react_agent = create_react_agent(
        model  = llm,
        tools  = tools,
        prompt = SystemMessage(content=system_prompt),
    )

    lc_messages = _history_to_lc(history)
    lc_messages.append(HumanMessage(content=message))

    result = await react_agent.ainvoke({"messages": lc_messages})

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
