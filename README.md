# Multi-CRM AI Agent

## Stack
- **LangGraph** — agent orchestration
- **FastAPI** — backend API
- **Streamlit** — frontend UI
- **ChromaDB** — RAG tool store
- **Azure OpenAI** — LLM

## Apps
| App | What it does |
|-----|-------------|
| HubSpot CRM | Deals, contacts, companies, pipelines |
| HubSpot Tickets | Support tickets |
| Zoho HRMS | Employees, leave, attendance, HR |

## Architecture
```
User message
     ↓
Router (intent detector) → which app?
     ↓
RAG selector → top 5 relevant tools
     ↓
LLM agent → calls tool → answers
     ↓
Single chat window
```

## Folder Structure
```
├── config/         settings, env vars
├── auth/           HubSpot OAuth, Zoho MCP URL
├── mcp/            MCP client
├── agents/         router + 3 app agents + prompts
├── rag/            ChromaDB store + tool selector
├── core/           tools bridge, session, logger
├── api/            FastAPI routes
└── ui/             Streamlit frontend
```

## Setup
```bash
cp .env.example .env
# fill in your keys
pip install -r requirements.txt
uvicorn api.routes:app --port 8000
streamlit run ui/app.py --server.port 8501
```
