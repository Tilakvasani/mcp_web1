# Multi-CRM AI Agent — LangChain + LangGraph

> **⚠️ Security & Multi-User Notice**
>
> - This app stores **one OAuth token per CRM** in local files (`.zoho_tokens.json`, `.hubspot_tokens.json`).
>   If multiple users connect, each new login **overwrites** the previous user's token.
>   This is a **single-user-per-CRM** app by design. Do not expose it publicly without adding per-user token isolation.
> - **Never commit `.env`, `.zoho_tokens.json`, or `.hubspot_tokens.json`** to version control.
>   All three contain live credentials. They are in `.gitignore` — keep them there.
> - Rate limiting: `/api/chat` allows **15 requests per 60 seconds** per session.


A FastAPI web app that connects to **HubSpot** and/or **Zoho CRM** via MCP
using **LangGraph's ReAct agent** and Azure OpenAI.

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Azure OpenAI (`AzureChatOpenAI`) |
| Agent | LangGraph `create_react_agent` |
| Tools | LangChain `StructuredTool` wrapping CRM MCP servers |
| MCP Transport | Streamable HTTP (`mcp >= 1.8`) |
| Auth | HubSpot OAuth 2.1 PKCE + Zoho OAuth |
| Web | FastAPI + SSE streaming |

## File Structure

```
mcp_web/
├── web_app.py          # FastAPI app — OAuth routes + SSE chat endpoint
├── mcp_client.py       # MCP client for remote MCP servers
├── hubspot_oauth.py    # HubSpot OAuth 2.1 PKCE flow helpers
├── zoho_auth.py        # Zoho OAuth helpers
├── crm_logger.py       # Compact structured logger
├── pyproject.toml      # Dependencies
├── .env                # Secrets (never commit)
└── core/
    ├── agent.py        # LangGraph ReAct agent + system prompt (v2)
    └── tools.py        # MCP → LangChain StructuredTool bridge
```

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   # or with uv:
   uv sync
   ```

2. Fill in `.env` with your Azure OpenAI, HubSpot, and Zoho credentials.

3. Run:
   ```bash
   python web_app.py
   # or
   uvicorn web_app:app --host 0.0.0.0 --port 8000
   ```

4. Open http://localhost:8000 and connect your CRM via the sidebar.

## How It Works

```
User message
     │
     ▼
web_app.py  ──► core/agent.py  (_augment_message adds context hints)
                    │
                    ├── AzureChatOpenAI (LangChain)
                    ├── create_react_agent (LangGraph)
                    │       ↕ tool calls (automatic loop)
                    └── core/tools.py
                            └── MCPClient → HubSpot / Zoho MCP
```

## v2 Prompt Improvements (core/agent.py)

### 1 — Demo / Test Data Handling
When a user says *"create a demo lead"*, *"put anything you want"*,
*"use test data"*, or *"you decide"*, the agent now:
- Generates realistic sample data itself (name, email, company, phone).
- Proceeds without asking the user for field values.
- Reports back exactly what was created.

### 2 — Confirmation Loop Fix
- Confirmation is asked **once only** before create/update/delete.
- If the user already said *"yes"* / *"do it"* / *"ok"* / *"sure"* in **any**
  previous message, the agent proceeds immediately — no re-confirmation.
- `_augment_message()` injects a `[SYSTEM CONTEXT]` hint into the current user
  message so the LLM never forgets prior confirmations across turns.

### 3 — Casual Language & Typo Handling
A lookup table in the system prompt maps informal phrases and common typos
to correct intents:
- *"craete"*, *"mak"*, *"lisain"* → create / list intent
- *"ya"*, *"ok bro"*, *"just do it"* → confirmed

### 4 — Field Discovery Workflow
Before any create/update the agent is instructed to:
1. Call the CRM's fields/modules tool to get exact API field names.
2. Use only those field names — never guess.
3. Retry once with corrections on FORMAT_ERROR, then report and stop.

### 5 — Human-Readable Permission Errors
PERMISSION_DENIED responses are surfaced as:
> ❌ I don't have permission to [action]. Your CRM token is missing the
> required scope: [scope]. Please reconnect with the correct permissions.

### 6 — Loop Prevention Tightened
- Max 2 attempts per action before stopping and reporting.
- NEVER retries a tool with identical arguments.

## Environment Variables

```env
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-02-01

HUBSPOT_CLIENT_ID=...
HUBSPOT_CLIENT_SECRET=...
HUBSPOT_MCP_URL=https://mcp.hubspot.com/

ZOHO_CLIENT_ID=...
ZOHO_CLIENT_SECRET=...

STREAMLIT_URL=http://localhost:8501
```