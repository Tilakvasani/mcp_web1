"""
FastAPI Backend — AI Assistant
==============================
Routes
------
GET  /api/status
POST /api/chat             (agent="auto" — detects intent automatically)
GET  /api/permissions
GET  /api/cache-stats
POST /api/session/end
GET  /api/debug-mcp
POST /api/disconnect           (HubSpot)
GET  /oauth/connect            (HubSpot)
GET  /oauth/callback           (HubSpot)
GET  /oauth/zoho-people/connect    (Zoho People OAuth)
GET  /oauth/zoho-people/callback   (Zoho People OAuth)
POST /zoho-people/disconnect
POST /zoho-people/disconnect-oauth
"""

import os
import json
import asyncio
import secrets
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from crm_logger import log, suppress_noisy_libs
suppress_noisy_libs()

from mcp_manager import MCPManager

from core.agent import (
    run_agent,
    evict_session, evict_stale_sessions, evict_all_sessions,
    get_cache_stats,
)
from apps.hubspot.hubspot_oauth import (
    get_valid_token      as hs_get_token,
    get_connection_status as hs_status,
    generate_pkce_pair,
    build_auth_url       as hs_build_auth_url,
    exchange_code_for_tokens as hs_exchange,
    get_token_scopes,
    get_token_user_id,
    check_is_admin,
    TOKEN_FILE           as HS_TOKEN_FILE,
)
from apps.zoho_people.zoho_people_auth import (
    get_connection_status as zp_status,
    get_mcp_url           as zp_get_mcp_url,
    save_mcp_url          as zp_save_mcp_url,
    disconnect            as zp_disconnect_auth,
    reconnect             as zp_reconnect_auth,
    MCP_URL_FILE          as ZP_URL_FILE,
)
from apps.zoho_people.zoho_people_oauth import (
    build_auth_url        as zp_build_auth_url,
    exchange_code_for_tokens as zp_exchange,
    get_oauth_status      as zp_oauth_status,
    disconnect_oauth      as zp_disconnect_oauth,
)
from core.tools import describe_scopes, get_tool_scope_map
from rag.tool_indexer import deindex_agent_tools, index_agent_tools

load_dotenv()

HUBSPOT_MCP_URL = os.getenv("HUBSPOT_MCP_URL", "https://mcp.hubspot.com/")
STREAMLIT_URL   = os.getenv("STREAMLIT_URL",   "http://localhost:8501")


# =============================================================================
# PKCE state store
# =============================================================================

class _TTLDict:
    def __init__(self, max_age: int = 600):
        self._data: dict[str, tuple[str, float]] = {}
        self._max_age = max_age

    def __setitem__(self, key: str, value: str):
        self._prune()
        self._data[key] = (value, time.time())

    def pop(self, key: str, default=None):
        entry = self._data.pop(key, None)
        if not entry:
            return default
        value, ts = entry
        return default if time.time() - ts > self._max_age else value

    def __contains__(self, key: str) -> bool:
        entry = self._data.get(key)
        if not entry:
            return False
        _, ts = entry
        if time.time() - ts > self._max_age:
            del self._data[key]
            return False
        return True

    def _prune(self):
        now   = time.time()
        stale = [k for k, (_, ts) in self._data.items() if now - ts > self._max_age]
        for k in stale:
            del self._data[k]

_hs_pkce = _TTLDict(max_age=600)


# =============================================================================
# MCPManager (replaces MCPPool + keepalive + intent detection)
# =============================================================================

mcp_manager = MCPManager()



# =============================================================================
# Lifespan
# =============================================================================

async def _stale_session_eviction_loop():
    """Periodically evict stale sessions from the session cache."""
    while True:
        await asyncio.sleep(300)  # every 5 minutes
        await evict_stale_sessions()


@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), "AZURE_OPENAI_DEPLOYMENT_NAME missing"
    assert os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), "AZURE_OPENAI_EMBEDDING_DEPLOYMENT missing"

    # Auto-register any already-authenticated services on startup
    _register_hubspot()
    _register_zoho_people()

    task = asyncio.create_task(_stale_session_eviction_loop())
    log("boot", "FastAPI ready -> http://localhost:8000")
    yield
    task.cancel()
    log("bye", "shutdown complete")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = [STREAMLIT_URL, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

@app.middleware("http")
async def access_log(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    ms       = int((time.time() - t0) * 1000)
    status   = response.status_code
    kind     = "ok" if status < 400 else ("warn" if status < 500 else "error")
    log(kind, f"{request.method} {request.url.path} {status} ({ms}ms)")
    return response


# =============================================================================
# MCP connection helpers (MCPManager-based — no persistent pool)
# =============================================================================

def _register_hubspot():
    """Register HubSpot in MCPManager if token is available."""
    try:
        hs_get_token()
        mcp_manager.set_hubspot(access_token_fn=hs_get_token)
        log("connect", "HubSpot registered in MCPManager ✓")
        return True
    except RuntimeError:
        log("error", "HubSpot -> no valid token")
        return False


def _register_zoho_people():
    """Register Zoho People in MCPManager if MCP URL is available."""
    mcp_url = zp_get_mcp_url()
    if not mcp_url:
        return False, "mcp_url_missing"
    from apps.zoho_people.zoho_people_oauth import get_access_token as zp_get_token
    mcp_manager.set_zoho_people(mcp_url=mcp_url, access_token_fn=zp_get_token)
    log("connect", "Zoho People registered in MCPManager ✓")
    return True, ""


# =============================================================================
# Agent runner
# =============================================================================

_VALID_AGENTS = ("hubspot", "zoho_people", "cross", "auto", "unified")


def _build_clients_from_manager() -> dict:
    """
    Return all currently registered MCP names mapped to mcp_manager.
    The unified agent uses mcp_manager.get_tools([name]) per key.
    """
    return mcp_manager.get_active_clients()


def _handle_agent_exception(exc: Exception, detected_agent: str) -> dict:
    """Standardized error handler mapping exceptions to user-friendly UI error objects."""
    import traceback
    tb = traceback.format_exc().strip().splitlines()
    for line in tb[-3:]:
        log("error", line.strip())

    err = str(exc).lower()
    if "content_filter" in err or "responsibleaipolicyviolation" in err or "content safety" in err:
        return {"ok": False, "error": "content_filter",
                "detail": "Content safety filter triggered. Please rephrase.", "agent": detected_agent}
    if "rate_limit" in err or "429" in err:
        return {"ok": False, "error": "rate_limit",
                "detail": "Too many requests — please wait a moment.", "agent": detected_agent}
    if "authentication" in err or "401" in err:
        return {"ok": False, "error": "auth_error",
                "detail": "AI service authentication failed. Check your API key.", "agent": detected_agent}
    return {"ok": False, "error": "agent_error", "detail": str(exc), "agent": detected_agent}


async def _stream_agent(message, history, agent, session_id):
    log("stream", f"start | agent=unified | session={session_id[:8]} | '{message[:60]}'")

    from core.agent import run_agent_stream

    # All connected MCPs — unified agent routes internally
    clients = _build_clients_from_manager()

    # Always emit "unified" so the UI can show a badge
    yield f"data: {json.dumps({'type': 'agent', 'agent': 'unified'})}\n\n"

    if not clients:
        yield f"data: {json.dumps({'type': 'error', 'error': 'no_clients', 'detail': 'No MCP services connected. Please connect HubSpot or Zoho People.'})}\n\n"
        return

    scopes   = get_token_scopes() if mcp_manager.is_connected("hubspot") else []
    is_admin = False
    if mcp_manager.is_connected("hubspot"):
        try:
            token   = hs_get_token()
            user_id = get_token_user_id()
            if user_id and token:
                is_admin = await check_is_admin(token, user_id)
        except Exception:
            pass

    chunks = 0
    try:
        async for chunk in run_agent_stream(
            message        = message,
            history        = history,
            clients        = clients,
            agent          = "unified",
            granted_scopes = scopes,
            is_admin       = is_admin,
            session_id     = session_id,
        ):
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
            chunks += 1
            await asyncio.sleep(0.001)

    except Exception as exc:
        err_res = _handle_agent_exception(exc, "unified")
        yield f"data: {json.dumps({'type': 'error', 'error': err_res['error'], 'detail': err_res['detail']})}\n\n"
        return

    log("stream", f"done | {chunks} chunks | agent=unified")
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# Rate limiter
# =============================================================================

_rate_store = {}
_RATE_LIMIT  = 15
_RATE_WINDOW = 60

def _is_rate_limited(session_id):
    now  = time.time()
    hits = [t for t in _rate_store.get(session_id, []) if now - t < _RATE_WINDOW]
    if len(hits) >= _RATE_LIMIT:
        return True
    hits.append(now)
    _rate_store[session_id] = hits
    return False


# =============================================================================
# API Routes
# =============================================================================

@app.get("/api/status")
async def api_status():
    hs_ok, hs_msg = hs_status()
    zp_ok, zp_msg = zp_status()
    zp_oauth_ok, zp_oauth_msg = zp_oauth_status()

    log("info", f"status -> HS={'ok' if hs_ok else 'off'} ZP={'ok' if zp_ok else 'off'} ZP_API={'ok' if zp_oauth_ok else 'off'}")
    return {
        "hubspot"     : {"connected": hs_ok, "message": hs_msg},
        "zoho_people" : {
            "connected": zp_ok,
            "message": zp_msg,
            "api_connected": zp_oauth_ok,
            "api_message": zp_oauth_msg,
        },
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    body       = await request.json()
    message    = body.get("message", "").strip()
    history    = body.get("history", [])
    agent      = body.get("agent", "auto")   # default is now "auto"
    session_id = body.get("session_id", "default")

    log("user", f"[{agent}] [{session_id[:8]}] '{message[:80]}'")

    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    if agent not in _VALID_AGENTS:
        return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)
    if len(message) > 4000:
        return JSONResponse({"error": "message too long (max 4000 chars)"}, status_code=400)
    if _is_rate_limited(session_id):
        return JSONResponse(
            {"error": "rate_limit", "detail": "Too many requests — please wait a moment."},
            status_code=429,
        )

    return StreamingResponse(
        _stream_agent(message, history, agent, session_id),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/permissions")
async def api_permissions():
    scopes = get_token_scopes()
    if not scopes:
        return {"connected": False, "is_admin": False, "scopes": [], "tools": []}
    is_admin = False
    try:
        token   = hs_get_token()
        user_id = get_token_user_id()
        if user_id and token:
            is_admin = await check_is_admin(token, user_id)
    except Exception:
        pass
    tool_scope_map = get_tool_scope_map(scopes)
    return {
        "connected": True,
        "is_admin" : is_admin,
        "scopes"   : describe_scopes(scopes),
        "tools"    : [
            {"tool": name, "accessible": bool(sc), "requires": sc[:2]}
            for name, sc in tool_scope_map.items()
        ],
    }


@app.post("/api/session/end")
async def api_session_end(request: Request):
    body       = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    await evict_session(session_id)
    # Agent-level RAG index persists -- no deindex on session end
    log("bye", f"session ended: {session_id[:8]}")
    return {"evicted": True, "session_id": session_id}


@app.get("/api/cache-stats")
async def api_cache_stats():
    return get_cache_stats()


@app.get("/api/debug-mcp")
async def api_debug_mcp(crm: str = "hubspot"):
    log("debug", f"MCP test -> crm={crm}")
    if not mcp_manager.is_connected(crm):
        return JSONResponse({"error": f"{crm} not registered in MCPManager"}, status_code=400)
    try:
        tools = await mcp_manager.get_tools([crm])
        return {
            "status"    : "ok",
            "crm"       : crm,
            "tool_count": len(tools),
            "tools"     : [getattr(t, "name", str(t)) for t in tools[:15]],
        }
    except Exception as exc:
        log("warn", f"debug-mcp failed for {crm}: {exc}")
        return {"status": "error", "crm": crm, "detail": str(exc)}


# =============================================================================
# HubSpot OAuth
# =============================================================================

@app.get("/oauth/connect")
async def oauth_connect():
    if not os.getenv("HUBSPOT_CLIENT_ID"):
        return JSONResponse({"error": "HUBSPOT_CLIENT_ID not set"}, status_code=500)
    verifier, challenge = generate_pkce_pair()
    state               = secrets.token_urlsafe(16)
    _hs_pkce[state]     = verifier
    log("oauth", "HubSpot OAuth flow started")
    return RedirectResponse(hs_build_auth_url(code_challenge=challenge, state=state))


@app.get("/oauth/callback")
async def oauth_callback(
    code: str = "", state: str = "",
    error: str = "", error_description: str = "",
):
    if error:
        log("error", f"HubSpot OAuth: {error_description or error}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=hubspot")
    verifier = _hs_pkce.pop(state, None)
    if not verifier:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=hubspot")
    try:
        hs_exchange(code=code, code_verifier=verifier)
        log("ok", "HubSpot tokens exchanged")
    except ValueError as exc:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")
    mcp_manager.remove("hubspot")
    _register_hubspot()  # re-register with fresh token
    deindex_agent_tools("hubspot")  # scopes may have changed -> re-index on next connect
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    mcp_manager.remove("hubspot")
    deindex_agent_tools("hubspot")
    await evict_all_sessions()
    log("bye", "HubSpot disconnected")
    return {"disconnected": True}


# =============================================================================
# Zoho People
# =============================================================================

@app.post("/zoho-people/save-mcp-url")
async def zoho_people_save_url(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    zp_save_mcp_url(url)
    mcp_manager.remove("zoho_people")
    _register_zoho_people()  # re-register with new URL
    deindex_agent_tools("zoho_people")  # URL changed -> force re-index on next connect
    log("info", f"Zoho People MCP URL saved -> {url[:60]}")
    # Quick reachability check via a lightweight tool list
    try:
        tools = await mcp_manager.get_tools(["zoho_people"])
        ok, msg = True, f"{len(tools)} tools available"
    except Exception as exc:
        ok, msg = False, str(exc)
    log("ok" if ok else "warn", f"Zoho People reachable={ok}: {msg}")
    return {"saved": True, "reachable": ok, "detail": msg}


@app.post("/zoho-people/disconnect")
async def zoho_people_disconnect():
    zp_disconnect_auth()
    zp_disconnect_oauth()
    mcp_manager.remove("zoho_people")
    deindex_agent_tools("zoho_people")
    await evict_all_sessions()
    log("bye", "Zoho People fully disconnected (MCP + OAuth)")
    return {"disconnected": True}


# =============================================================================
# Zoho People OAuth (for direct REST API access)
# =============================================================================

_zp_oauth_states = _TTLDict(max_age=600)

@app.get("/oauth/zoho-people/connect")
async def zoho_people_oauth_connect():
    if not os.getenv("ZOHO_PEOPLE_CLIENT_ID"):
        return JSONResponse({"error": "ZOHO_PEOPLE_CLIENT_ID not set"}, status_code=500)
    state = secrets.token_urlsafe(16)
    _zp_oauth_states[state] = "pending"
    log("oauth", "Zoho People OAuth flow started")
    return RedirectResponse(zp_build_auth_url(state=state))


@app.get("/oauth/zoho-people/callback")
async def zoho_people_oauth_callback(
    code: str = "", state: str = "",
    error: str = "", error_description: str = "",
    location: str = "",
):
    if error:
        log("error", f"Zoho People OAuth: {error_description or error}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=zoho_people")
    if state not in _zp_oauth_states:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=zoho_people")
    _zp_oauth_states.pop(state, None)
    try:
        zp_exchange(code=code)
        log("ok", "Zoho People OAuth tokens exchanged")
    except ValueError as exc:
        log("error", f"Zoho People token exchange failed: {exc}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=zoho_people")
    # Clear disconnect sentinel so MCP URL from .env loads again
    zp_reconnect_auth()
    _register_zoho_people()  # re-register with updated OAuth token
    # Clear tool caches so the synthetic callAPI tool gets injected on next request
    deindex_agent_tools("zoho_people")
    await evict_all_sessions()
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=zoho_people")


@app.post("/zoho-people/disconnect-oauth")
async def zoho_people_disconnect_oauth():
    """Disconnect only the OAuth (REST API) part, keep MCP."""
    zp_disconnect_oauth()
    deindex_agent_tools("zoho_people")
    await evict_all_sessions()
    log("bye", "Zoho People OAuth disconnected (MCP still active)")
    return {"disconnected": True}




# =============================================================================
# Response Quality
# =============================================================================

_last_scores: dict[str, dict] = {}  # session_id -> scores


def _store_grade_scores(session_id: str, scores: dict):
    """Called by grader after scoring (stored for UI badge retrieval)."""
    _last_scores[session_id] = scores


@app.get("/api/response-quality/{session_id}")
async def response_quality(session_id: str):
    return _last_scores.get(session_id, {})

if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)