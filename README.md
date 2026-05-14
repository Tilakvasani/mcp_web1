# HubSpot CRM Agent — LangChain + LangGraph

A FastAPI web app that connects to **HubSpot via MCP** using **LangGraph's ReAct agent**.

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Azure OpenAI (`AzureChatOpenAI`) |
| Agent | LangGraph `create_react_agent` |
| Tools | LangChain `StructuredTool` wrapping HubSpot MCP |
| MCP Transport | Streamable HTTP (`mcp >= 1.8`) |
| Auth | HubSpot OAuth 2.1 PKCE |
| Web | FastAPI + SSE streaming |

## File Structure

```
mcp_web/
├── web_app.py          # FastAPI app — OAuth routes + SSE chat endpoint
├── mcp_client.py       # MCP client for HubSpot's remote MCP server
├── hubspot_oauth.py    # OAuth 2.1 PKCE flow helpers
├── pyproject.toml      # Dependencies
├── .env                # Secrets (never commit)
└── core/
    ├── agent.py        # LangGraph ReAct agent + system prompt
    └── tools.py        # MCP → LangChain StructuredTool bridge
```

## Setup

1. Install dependencies:
   ```bash
   pip install -e .
   # or with uv:
   uv sync
   ```

2. Fill in `.env` with your Azure OpenAI and HubSpot credentials.

3. Run:
   ```bash
   python web_app.py
   # or
   uvicorn web_app:app --host 0.0.0.0 --port 8000
   ```

4. Open http://localhost:8000 and click **Connect HubSpot** to complete OAuth.

## How It Works

```
User message
     │
     ▼
web_app.py  ──► core/agent.py
                    │
                    ├── AzureChatOpenAI (LangChain)
                    ├── create_react_agent (LangGraph)
                    │       ↕ tool calls (automatic loop)
                    └── core/tools.py
                            └── MCPClient → HubSpot MCP
```

The LangGraph `create_react_agent` handles the entire tool-call loop
(think → act → observe → repeat) without any manual agentic loop code.
