"""
FastAPI Backend — AI Assistant
==============================
Routes
------
GET  /api/status
POST /api/chat
GET  /api/permissions
GET  /api/cache-stats
POST /api/session/end
GET  /api/debug-mcp
POST /api/disconnect           (HubSpot)
GET  /oauth/connect            (HubSpot)
GET  /oauth/callback           (HubSpot)
POST /zoho-people/save-mcp-url
POST /zoho-people/disconnect
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

from mcp_client import MCPClient

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
    MCP_URL_FILE          as ZP_URL_FILE,
)
from core.tools import describe_scopes, get_tool_scope_map
from rag.tool_indexer import index_session_tools, deindex_session_tools

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

    def pop(self, key: str, default=None) -> str | None:
        entry = self._data.pop(key, None)
        if not entry:
            return default
        value, ts = entry
        return default if time.time() - ts > self._max_age else value

    def _prune(self):
        now   = time.time()
        stale = [k for k, (_, ts) in self._data.items() if now - ts > self._max_age]
        for k in stale:
            del self._data[k]

_hs_pkce = _TTLDict(max_age=600)


# =============================================================================
# MCP Connection Pool
# =============================================================================

class _Cached:
    def __init__(self, client: MCPClient):
        self.client  = client
        self.created = time.time()

    def age(self) -> float:
        return time.time() - self.created


class MCPPool:
    MAX_AGE    = 600
    PING_EVERY = 120

    def __init__(self):
        self._pool : dict[str, _Cached] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> MCPClient | None:
        async with self._lock:
            cached = self._pool.get(key)
            if not cached:
                return None
            if cached.age() >= self.MAX_AGE:
                log("pool", f"{key} expired — evicting")
                try:
                    await cached.client.cleanup()
                except Exception:
                    pass
                del self._pool[key]
                return None
            client = cached.client

        try:
            await client.list_tools()
            log("pool", f"{key} reused (age {cached.age():.0f}s)")
            return client
        except Exception as exc:
            log("warn", f"{key} stale ({type(exc).__name__}) — evicting")
            async with self._lock:
                self._pool.pop(key, None)
            try:
                await client.cleanup()
            except Exception:
                pass
            return None

    async def put(self, key: str, client: MCPClient):
        async with self._lock:
            self._pool[key] = _Cached(client)
        log("connect", f"{key} cached in pool")

    async def invalidate(self, key: str):
        async with self._lock:
            cached = self._pool.pop(key, None)
        if cached:
            try:
                await cached.client.cleanup()
            except Exception:
                pass
            log("bye", f"evicted: {key}")

    async def invalidate_all(self):
        async with self._lock:
            snapshot = dict(self._pool)
            self._pool.clear()
        for c in snapshot.values():
            try:
                await c.client.cleanup()
            except Exception:
                pass
        log("bye", "all pool clients evicted")

    async def keepalive(self):
        async with self._lock:
            snapshot = dict(self._pool)
        dead: list[str] = []
        for key, cached in snapshot.items():
            try:
                ok, _ = await cached.client.preflight()
                if ok:
                    log("ping", f"{key} alive")
                else:
                    dead.append(key)
            except Exception:
                dead.append(key)
        for key in dead:
            async with self._lock:
                cached = self._pool.pop(key, None)
            if cached:
                try:
                    await cached.client.cleanup()
                except Exception:
                    pass
                log("warn", f"evicted stale: {key}")


_pool = MCPPool()

async def _keepalive_loop():
    while True:
        await asyncio.sleep(MCPPool.PING_EVERY)
        await _pool.keepalive()
        await evict_stale_sessions()


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), "AZURE_OPENAI_DEPLOYMENT_NAME missing"
    assert os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), "AZURE_OPENAI_EMBEDDING_DEPLOYMENT missing"
    log("boot", "FastAPI ready → http://localhost:8000")
    task = asyncio.create_task(_keepalive_loop())
    yield
    task.cancel()
    await _pool.invalidate_all()
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
# MCP client factories
# =============================================================================

async def _make_hubspot_client() -> MCPClient | None:
    cached = await _pool.get("hubspot")
    if cached:
        return cached
    log("connect", "HubSpot → connecting fresh…")
    try:
        token = hs_get_token()
    except RuntimeError:
        log("error", "HubSpot → no valid token")
        return None
    client = MCPClient(url=HUBSPOT_MCP_URL, headers={"Authorization": f"Bearer {token}"})
    ok, msg = await client.preflight()
    if not ok:
        log("error", f"HubSpot preflight failed: {msg}")
        return None
    try:
        await client.connect()
        await _pool.put("hubspot", client)
        log("ok", "HubSpot connected ✓")
        return client
    except ConnectionError as exc:
        log("error", f"HubSpot connect error: {exc}")
        return None


async def _make_zoho_people_client() -> tuple[MCPClient | None, str]:
    mcp_url = zp_get_mcp_url()
    if not mcp_url:
        return None, "mcp_url_missing"
    cached = await _pool.get("zoho_people")
    if cached:
        return cached, ""
    log("connect", "Zoho People → connecting fresh…")
    client = MCPClient(url=mcp_url, headers={})
    ok, msg = await client.preflight()
    if not ok:
        log("error", f"Zoho People preflight failed: {msg}")
        return None, "preflight_failed"
    try:
        await client.connect()
        await _pool.put("zoho_people", client)
        log("ok", "Zoho People connected ✓")
        return client, ""
    except ConnectionError as exc:
        log("error", f"Zoho People connect error: {exc}")
        return None, "connect_failed"


# =============================================================================
# Agent runner
# =============================================================================

_VALID_AGENTS = ("hubspot", "zoho_people", "cross")

async def _run_agent_turn(
    message    : str,
    history    : list,
    agent      : str,
    session_id : str = "default",
) -> dict:
    clients: dict = {}
    scopes        = []
    is_admin      = False

    # Connect HubSpot
    if agent in ("hubspot", "cross"):
        hs = await _make_hubspot_client()
        if hs:
            clients["hubspot"] = hs
            scopes = get_token_scopes()
            try:
                token   = hs_get_token()
                user_id = get_token_user_id()
                if user_id and token:
                    is_admin = await check_is_admin(token, user_id)
            except Exception:
                pass

    # Connect Zoho People
    zp_err = ""
    if agent in ("zoho_people", "cross"):
        zp, zp_err = await _make_zoho_people_client()
        if zp:
            clients["zoho_people"] = zp

    # Guard — agent-specific failures
    if agent == "hubspot" and "hubspot" not in clients:
        return {"ok": False, "error": "hubspot_unavailable",
                "detail": "HubSpot not connected or token expired."}
    if agent == "zoho_people" and "zoho_people" not in clients:
        return {"ok": False, "error": f"zoho_people_{zp_err}",
                "detail": f"Zoho People unavailable: {zp_err}"}
    if agent == "cross" and not clients:
        return {"ok": False, "error": "no_clients",
                "detail": "Neither HubSpot nor Zoho People is connected."}

    t0 = time.time()
    try:
        text = await run_agent(
            message       = message,
            history       = history,
            clients       = clients,
            agent         = agent,
            granted_scopes= scopes,
            is_admin      = is_admin,
            session_id    = session_id,
        )
        log("ok", f"agent done → {len(text or '')} chars in {time.time()-t0:.1f}s")
        return {"ok": True, "text": text}

    except Exception as exc:
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-3:]:
            log("error", line.strip())
        # Invalidate stale connections
        if agent in ("hubspot", "cross"):
            await _pool.invalidate("hubspot")
        if agent in ("zoho_people", "cross"):
            await _pool.invalidate("zoho_people")

        err = str(exc).lower()
        if "content_filter" in err or "responsibleaipolicyviolation" in err:
            return {"ok": False, "error": "content_filter",
                    "detail": "Content safety filter triggered. Please rephrase."}
        if "rate_limit" in err or "429" in err:
            return {"ok": False, "error": "rate_limit",
                    "detail": "Too many requests — please wait a moment."}
        if "authentication" in err or "401" in err:
            return {"ok": False, "error": "auth_error",
                    "detail": "AI service authentication failed. Check your API key."}
        return {"ok": False, "error": "agent_error", "detail": str(exc)}


async def _stream_agent(message: str, history: list, agent: str, session_id: str):
    log("stream", f"start | agent={agent} | session={session_id[:8]} | '{message[:60]}'")
    result = await _run_agent_turn(message, history, agent, session_id=session_id)

    if not result["ok"]:
        log("error", f"stream failed: {result['error']}")
        yield f"data: {json.dumps({'type': 'error', 'error': result['error'], 'detail': result['detail']})}\n\n"
        return

    text   = result["text"] or ""
    chunks = 0
    for i in range(0, len(text), 160):
        yield f"data: {json.dumps({'type': 'chunk', 'text': text[i:i+160]})}\n\n"
        chunks += 1
        await asyncio.sleep(0.003)

    log("stream", f"done | {chunks} chunks | {len(text)} chars")
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# Rate limiter
# =============================================================================

_rate_store: dict[str, list[float]] = {}
_RATE_LIMIT  = 15
_RATE_WINDOW = 60

def _is_rate_limited(session_id: str) -> bool:
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
    log("info", f"status → HS={'✅' if hs_ok else '○'} ZP={'✅' if zp_ok else '○'}")
    return {
        "hubspot"     : {"connected": hs_ok, "message": hs_msg},
        "zoho_people" : {"connected": zp_ok, "message": zp_msg},
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    body       = await request.json()
    message    = body.get("message", "").strip()
    history    = body.get("history", [])
    agent      = body.get("agent", "hubspot")
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
    deindex_session_tools(session_id)          # ← remove from Chroma on logout
    log("bye", f"session ended: {session_id[:8]}")
    return {"evicted": True, "session_id": session_id}


@app.get("/api/cache-stats")
async def api_cache_stats():
    return get_cache_stats()


@app.get("/api/debug-mcp")
async def api_debug_mcp(crm: str = "hubspot"):
    log("debug", f"MCP test → crm={crm}")
    if crm == "zoho_people":
        mcp_url = zp_get_mcp_url()
        if not mcp_url:
            return JSONResponse({"error": "mcp_url_missing"}, status_code=400)
        client = MCPClient(url=mcp_url, headers={})
    else:
        try:
            token = hs_get_token()
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=401)
        client = MCPClient(url=HUBSPOT_MCP_URL, headers={"Authorization": f"Bearer {token}"})

    ok, msg = await client.preflight()
    if not ok:
        return {"status": "preflight_failed", "detail": msg}
    try:
        await client.connect()
        tools = await client.list_tools()
        result = {
            "status"    : "ok",
            "transport" : client.transport,
            "tool_count": len(tools),
            "tools"     : [t.name for t in tools[:15]],
        }
        await client.cleanup()
        return result
    except ConnectionError as exc:
        return {"status": "connect_failed", "detail": str(exc)}


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
        log("ok", "HubSpot tokens exchanged ✓")
    except ValueError as exc:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")
    await _pool.invalidate("hubspot")
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    await _pool.invalidate("hubspot")
    await evict_all_sessions()
    log("bye", "HubSpot disconnected")
    return {"disconnected": True}


# =============================================================================
# Zoho People — MCP URL only (no OAuth)
# =============================================================================

@app.post("/zoho-people/save-mcp-url")
async def zoho_people_save_url(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)

    zp_save_mcp_url(url)
    await _pool.invalidate("zoho_people")
    log("info", f"Zoho People MCP URL saved → {url[:60]}")

    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    log("ok" if ok else "warn", f"Zoho People reachable={ok}: {msg}")
    return {"saved": True, "reachable": ok, "detail": msg}


@app.post("/zoho-people/disconnect")
async def zoho_people_disconnect():
    zp_disconnect_auth()
    await _pool.invalidate("zoho_people")
    await evict_all_sessions()
    log("bye", "Zoho People disconnected")
    return {"disconnected": True}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
