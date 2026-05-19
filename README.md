# AI Business Assistant

One chat interface → HubSpot CRM + Zoho People HRMS + Cross-system queries.

**Stack:** FastAPI · Streamlit · LangGraph · ChromaDB · Azure OpenAI · MCP

---

## Quick Start

```bash
# 1. Install dependencies
pip install -e .
# or
uv sync

# 2. Copy and fill environment variables
cp .env.example .env

# 3. Start the backend
uvicorn web_app:app --port 8000

# 4. Start the frontend (new terminal)
streamlit run streamlit_app.py
```

Open http://localhost:8501

---

## What you need in Azure OpenAI

Two deployments in Azure OpenAI Studio:

| Deployment | Model | Used for |
|---|---|---|
| `gpt-4o` (or your name) | GPT-4o / GPT-4.1 | Chat / reasoning |
| `text-embedding-3-small` | text-embedding-3-small | RAG tool selection |

---

## Connecting your apps

**HubSpot** → Click "Connect HubSpot" in sidebar → OAuth flow → done.

**Zoho People** → Go to [mcp.zoho.com](https://mcp.zoho.com) → copy your MCP URL → paste in sidebar → Save & Connect.

---

## How RAG tool selection works

1. User sends a message
2. Message embedded via Azure OpenAI (cached — same query = zero API cost)
3. ChromaDB cosine similarity search on all indexed tools
4. Keyword pre-filter re-ranks results
5. Top 14 tools sent to LLM (not all 95+)

**Tool indexing lifecycle:**
- HubSpot connects → tools embedded + stored in ChromaDB
- Zoho People connects → tools embedded + stored in ChromaDB
- User disconnects / session ends → vectors deleted from ChromaDB

---

## File structure

```
ai-assistant/
├── rag/
│   ├── embeddings.py        Azure OpenAI embeddings + query cache
│   ├── chroma_store.py      ChromaDB wrapper (one collection per session)
│   ├── hybrid_search.py     Keyword pre-filter + vector similarity
│   └── tool_indexer.py      Index on connect, deindex on disconnect
│
├── core/
│   ├── agent.py             Central dispatcher (pre-router → cache → RAG → LLM)
│   ├── session_cache.py     MCP tool cache (once per session, TTL 10 min)
│   ├── response_cache.py    In-memory response cache (TTL 5 min)
│   ├── pre_router.py        Instant replies for greetings/help/off-topic
│   └── tools.py             HubSpot scope guards
│
├── agents/
│   ├── hubspot_agent.py     HubSpot CRM LangGraph agent
│   ├── zoho_people_agent.py Zoho People HRMS LangGraph agent
│   └── cross_agent.py       Cross-system combined agent
│
├── apps/
│   ├── hubspot/
│   │   └── hubspot_oauth.py HubSpot OAuth 2.1 + PKCE
│   └── zoho_people/
│       └── zoho_people_auth.py Zoho People MCP URL manager
│
├── chroma_data/             Local ChromaDB storage (gitignored)
├── web_app.py               FastAPI backend
├── streamlit_app.py         Streamlit frontend
├── mcp_client.py            MCP client (unchanged)
├── crm_logger.py            Logger (unchanged)
├── pyproject.toml
└── .env.example
```
