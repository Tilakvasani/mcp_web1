"""
config/settings.py
==================
Single source of truth for all environment variables.
Import this everywhere — never use os.getenv() directly in other files.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT     = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
AZURE_OPENAI_API_VERSION    = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# ── HubSpot ───────────────────────────────────────────────────────────────────
HUBSPOT_CLIENT_ID           = os.getenv("HUBSPOT_CLIENT_ID", "")
HUBSPOT_CLIENT_SECRET       = os.getenv("HUBSPOT_CLIENT_SECRET", "")
HUBSPOT_REDIRECT_URI        = os.getenv("HUBSPOT_REDIRECT_URI", "http://localhost:8000/oauth/callback")
HUBSPOT_MCP_URL             = os.getenv("HUBSPOT_MCP_URL", "https://mcp.hubspot.com/")

# ── Zoho ──────────────────────────────────────────────────────────────────────
ZOHO_MCP_URL                = os.getenv("ZOHO_MCP_URL", "")   # paste from mcp.zoho.in

# ── App ───────────────────────────────────────────────────────────────────────
STREAMLIT_URL               = os.getenv("STREAMLIT_URL", "http://localhost:8501")
FASTAPI_PORT                = int(os.getenv("FASTAPI_PORT", "8000"))
SESSION_TTL_SECONDS         = int(os.getenv("SESSION_TTL", "600"))

# ── RAG ───────────────────────────────────────────────────────────────────────
RAG_TOP_K                   = int(os.getenv("RAG_TOP_K", "5"))     # tools to send to LLM
CHROMA_PERSIST_DIR          = os.getenv("CHROMA_PERSIST_DIR", ".chromadb")
