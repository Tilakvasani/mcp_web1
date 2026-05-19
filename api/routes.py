"""
api/routes.py
=============
FastAPI backend — handles auth, chat, and session management.
"""

import secrets
import asyncio
from fastapi import FastAPI, Request# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse, RedirectResponse# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware# pyrefly: ignore [missing-import]

from config.settings import STREAMLIT_URL, FASTAPI_PORT
from auth.hubspot import (
    generate_pkce_pair, build_auth_url, exchange_code_for_tokens,
    get_valid_token, get_connection_status, get_granted_scopes,
    check_is_admin, get_token_user_id,
)
from auth.zoho import save_mcp_url, get_mcp_url, disconnect as zoho_disconnect, is_connected as zoho_connected
from mcp_bridge.client import MCPClient
from agents.runner import run_agent, evict_session
from core.logger import log

app = FastAPI(title="Multi-CRM AI Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory PKCE state store  {state: (verifier, session_id)}
_pkce_store: dict[str, tuple[str, str]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_session(request: Request) -> str:
    return request.cookies.get("session_id") or secrets.token_urlsafe(16)


async def _build_clients(request: Request) -> dict[str, MCPClient]:
    """Build MCP clients for connected apps."""
    clients    = {}
    session_id = _get_session(request)

    # HubSpot
    try:
        token = get_valid_token()
        clients["hubspot"] = MCPClient(
            url     = "https://mcp.hubspot.com/",
            headers = {"Authorization": f"Bearer {token}"},
        )
    except Exception:
        pass

    # Zoho HRMS
    mcp_url = get_mcp_url(session_id)
    if mcp_url:
        clients["zoho_hrms"] = MCPClient(url=mcp_url, headers={})

    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status(request: Request):
    session_id  = _get_session(request)
    hs_ok, _    = get_connection_status()
    zo_ok       = zoho_connected(session_id)
    return {
        "hubspot_connected": hs_ok,
        "zoho_connected":    zo_ok,
        "session_id":        session_id[:8],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HubSpot OAuth
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/oauth/connect")
async def hubspot_connect(request: Request):
    session_id       = _get_session(request)
    verifier, challenge = generate_pkce_pair()
    state            = secrets.token_urlsafe(16)
    _pkce_store[state] = (verifier, session_id)
    response = RedirectResponse(build_auth_url(code_challenge=challenge, state=state))
    response.set_cookie("session_id", session_id, httponly=True, max_age=86400)
    return response


@app.get("/oauth/callback")
async def hubspot_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error}&crm=hubspot")

    entry = _pkce_store.pop(state, None)
    if not entry:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=hubspot")

    verifier, session_id = entry
    try:
        exchange_code_for_tokens(code=code, code_verifier=verifier)
        log("ok", f"HubSpot connected | session={session_id[:8]}")
    except Exception as exc:
        log("error", f"HubSpot token exchange failed: {exc}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")

    response = RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")
    response.set_cookie("session_id", session_id, httponly=True, max_age=86400)
    return response


@app.post("/oauth/disconnect")
async def hubspot_disconnect(request: Request):
    session_id = _get_session(request)
    evict_session(session_id)
    log("bye", f"HubSpot disconnected | session={session_id[:8]}")
    return {"disconnected": True}


# ─────────────────────────────────────────────────────────────────────────────
# Zoho MCP URL
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/zoho/save-url")
async def zoho_save_url(request: Request):
    session_id = _get_session(request)
    body       = await request.json()
    url        = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)

    # Test connection before saving
    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    if not ok:
        return JSONResponse({"error": f"Cannot reach Zoho MCP: {msg}"}, status_code=400)

    save_mcp_url(url, session_id)
    response = JSONResponse({"saved": True, "message": "Zoho HRMS connected ✅"})
    response.set_cookie("session_id", session_id, httponly=True, max_age=86400)
    return response


@app.post("/zoho/disconnect")
async def zoho_disconnect_route(request: Request):
    session_id = _get_session(request)
    zoho_disconnect(session_id)
    evict_session(session_id)
    return {"disconnected": True}


# ─────────────────────────────────────────────────────────────────────────────
# Chat
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: Request):
    body       = await request.json()
    message    = body.get("message", "").strip()
    history    = body.get("history", [])
    session_id = _get_session(request)

    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)

    clients = await _build_clients(request)
    if not clients:
        return JSONResponse({
            "reply": "⚠️ No apps connected. Please connect HubSpot or Zoho from the sidebar."
        })

    # Get scopes + admin status
    try:
        scopes   = get_granted_scopes()
        token    = get_valid_token()
        user_id  = get_token_user_id()
        is_admin = await check_is_admin(token, user_id) if user_id else False
    except Exception:
        scopes   = []
        is_admin = False

    try:
        reply = await run_agent(
            message        = message,
            history        = history,
            clients        = clients,
            granted_scopes = scopes,
            session_id     = session_id,
        )
    except Exception as exc:
        log("error", f"Agent error: {exc}")
        reply = f"⚠️ Something went wrong: {exc}"

    response = JSONResponse({"reply": reply})
    response.set_cookie("session_id", session_id, httponly=True, max_age=86400)
    return response