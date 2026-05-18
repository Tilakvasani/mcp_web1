"""
Multi-CRM AI Agent — FastAPI Backend  (Improved)
=================================================
Speed improvements over original:
  1. MCP client CONNECTION POOL — clients are created once and reused.
  2. Health-check on reuse — if a cached client is stale it reconnects once.
  3. Background keepalive — a background task pings connected clients every 90s.
  4. Streaming chunk size increased (160 chars) → fewer SSE frames.

Routes (unchanged)
---------------------------------
GET  /api/status        GET  /api/permissions   POST /api/chat
GET  /api/debug-mcp     POST /api/disconnect
GET  /oauth/connect     GET  /oauth/callback
GET  /zoho/connect      GET  /zoho/callback
POST /zoho/save-mcp-url POST /zoho/disconnect
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

# ── logger first so suppress fires before any MCP imports ──────────────────
from crm_logger import log, suppress_noisy_libs
suppress_noisy_libs()

from mcp_client import MCPClient
from core.agent import run_agent, run_agent_both, evict_session, evict_stale_sessions, evict_all_sessions, get_cache_stats
from hubspot_oauth import (
    get_valid_token as hs_get_token,
    get_connection_status as hs_status,
    generate_pkce_pair,
    build_auth_url as hs_build_auth_url,
    exchange_code_for_tokens as hs_exchange,
    get_token_scopes,
    get_token_user_id,
    check_is_admin,
    TOKEN_FILE as HS_TOKEN_FILE,
)
from zoho_auth import (
    get_connection_status as zo_status,
    get_mcp_url,
    get_mcp_status,
    save_mcp_url,
    build_auth_url as zo_build_auth_url,
    exchange_code_for_tokens as zo_exchange,
    generate_pkce_pair as zo_pkce,
    TOKEN_FILE as ZO_TOKEN_FILE,
    ZOHO_CLIENT_ID as ZO_CLIENT_ID,
    MCP_URL_FILE,
)
from core.tools import describe_scopes, get_tool_scope_map

load_dotenv()

HUBSPOT_MCP_URL = os.getenv("HUBSPOT_MCP_URL", "https://mcp.hubspot.com/")
STREAMLIT_URL   = os.getenv("STREAMLIT_URL", "http://localhost:8501")

# ---------------------------------------------------------------------------
# PKCE verifier store — TTL-pruned to prevent memory leaks (B6)
# ---------------------------------------------------------------------------

class _TTLDict:
    """Dict that stores (value, timestamp) and prunes entries older than max_age on insert."""
    def __init__(self, max_age: int = 600):
        self._data: dict[str, tuple[str, float]] = {}
        self._max_age = max_age

    def __setitem__(self, key: str, value: str):
        self._prune()
        self._data[key] = (value, time.time())

    def pop(self, key: str, default=None) -> str | None:
        entry = self._data.pop(key, None)
        if entry is None:
            return default
        value, ts = entry
        if time.time() - ts > self._max_age:
            return default  # expired
        return value

    def _prune(self):
        now = time.time()
        stale = [k for k, (_, ts) in self._data.items() if now - ts > self._max_age]
        for k in stale:
            del self._data[k]


_hs_pkce = _TTLDict(max_age=600)   # 10 minute TTL
_zo_pkce = _TTLDict(max_age=600)


# =============================================================================
# MCP Connection Pool
# =============================================================================

class _CachedClient:
    def __init__(self, client: MCPClient):
        self.client  = client
        self.created = time.time()

    def age(self) -> float:
        return time.time() - self.created


class MCPPool:
    MAX_AGE    = 600   # raised from 300 — keep connections alive longer
    PING_EVERY = 120   # raised from 90  — less frequent background pings

    def __init__(self):
        self._pool: dict[str, _CachedClient] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> MCPClient | None:
        """
        Return a cached client only after confirming it is still alive.

        Root cause of ClosedResourceError:
          The old code checked `cached.healthy` (always True) and returned the
          client without touching the actual connection. If the server closed
          the stream between requests the client object still existed but the
          underlying anyio MemoryObjectSendStream was closed, so the very next
          send raised ClosedResourceError.

        Fix: do a lightweight list_tools() probe (1 round-trip) before handing
        the client out. If it raises anything, evict and return None so the
        caller creates a fresh connection.
        """
        async with self._lock:
            cached = self._pool.get(key)
            if not cached:
                return None
            if cached.age() >= self.MAX_AGE:
                log("pool", f"{key} expired (age {cached.age():.0f}s) — evicting")
                try:
                    await cached.client.cleanup()
                except Exception:
                    pass
                del self._pool[key]
                return None
            # Snapshot the client reference before releasing the lock for the probe
            client = cached.client

        # ── Liveness probe outside the lock so other requests aren't blocked ──
        try:
            await client.list_tools()          # cheapest real MCP round-trip
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
            self._pool[key] = _CachedClient(client)
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
        """
        Ping each cached client outside the lock so live requests are never
        blocked. Old code held the lock for the entire HTTP round-trip.
        """
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
        await evict_stale_sessions()   # clean up idle tool cache entries


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), "AZURE_OPENAI_DEPLOYMENT_NAME missing in .env"
    log("boot", "FastAPI ready → http://localhost:8000")
    task = asyncio.create_task(_keepalive_loop())
    yield
    task.cancel()
    await _pool.invalidate_all()
    log("bye", "shutdown complete")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[STREAMLIT_URL, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Compact HTTP access log ──────────────────────────────────────────────────

@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
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
    """
    Return a live HubSpot MCP client.
    pool.get() now probes the connection before returning it, so if it returns
    None the connection was stale — we reconnect fresh automatically.
    """
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


async def _make_zoho_client() -> tuple[MCPClient | None, str]:
    """
    Return a live Zoho MCP client.
    pool.get() probes the connection; if stale it evicts and returns None,
    so we fall through to a fresh connect automatically.
    """
    mcp_url = get_mcp_url()
    if not mcp_url:
        log("warn", "Zoho MCP URL not set")
        return None, "mcp_url_missing"
    zo_ok, _ = zo_status()
    if not zo_ok:
        log("warn", "Zoho OAuth token missing")
        return None, "oauth_missing"

    cached = await _pool.get("zoho_crm")
    if cached:
        return cached, ""

    log("connect", "Zoho → connecting fresh…")
    client = MCPClient(url=mcp_url, headers={})
    ok, msg = await client.preflight()
    if not ok:
        log("error", f"Zoho preflight failed: {msg}")
        return None, "preflight_failed"
    try:
        await client.connect()
        await _pool.put("zoho_crm", client)
        log("ok", "Zoho connected ✓")
        return client, ""
    except ConnectionError as exc:
        log("error", f"Zoho connect error: {exc}")
        return None, "connect_failed"


# =============================================================================
# Agent runner
# =============================================================================

async def _run_agent_turn(message: str, history: list, agent: str, session_id: str = "default") -> dict:
    clients: dict[str, MCPClient] = {}
    scopes   = []
    is_admin = False

    if agent in ("hubspot", "both"):
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

    zo_err = ""
    if agent in ("zoho_crm", "both"):
        zo, zo_err = await _make_zoho_client()
        if zo:
            clients["zoho_crm"] = zo

    if agent == "zoho_crm" and "zoho_crm" not in clients:
        log("error", f"Zoho unavailable: {zo_err}")
        return {"ok": False, "error": zo_err, "detail": f"Zoho client unavailable: {zo_err}"}
    if agent == "hubspot" and "hubspot" not in clients:
        log("error", "HubSpot unavailable")
        return {"ok": False, "error": "hubspot_unavailable", "detail": "HubSpot client unavailable"}
    if agent == "both" and not clients:
        log("error", "No CRM clients connected")
        return {"ok": False, "error": "no_clients", "detail": "No CRM clients connected"}

    crms = ", ".join(clients.keys())
    log("ai", f"agent={agent} | crms=[{crms}] | running…")
    t0 = time.time()

    try:
        if agent == "both":
            text = await run_agent_both(
                message=message, history=history, clients=clients,
                granted_scopes=scopes, is_admin=is_admin,
                session_id=session_id,
            )
        else:
            text = await run_agent(
                message=message, history=history, clients=clients,
                agent=agent, granted_scopes=scopes, is_admin=is_admin,
                session_id=session_id,
            )
        elapsed = time.time() - t0
        log("ok", f"agent done → {len(text or '')} chars in {elapsed:.1f}s")
        return {"ok": True, "text": text}
    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        tb = traceback.format_exc().strip().splitlines()
        # Log the last 3 lines of traceback for compact but informative output
        for line in tb[-3:]:
            log("error", line.strip())
        log("error", f"agent error after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
        if agent in ("hubspot", "both"):
            await _pool.invalidate("hubspot")
        if agent in ("zoho_crm", "both"):
            await _pool.invalidate("zoho_crm")

        # Friendly error codes for common failures
        err_str = str(exc).lower()
        if "content_filter" in err_str or "responsibleaipolicyviolation" in err_str:
            return {"ok": False, "error": "content_filter",
                    "detail": "Your message could not be processed due to content safety filters. Please rephrase and try again."}
        if "rate_limit" in err_str or "429" in err_str:
            return {"ok": False, "error": "rate_limit",
                    "detail": "Too many requests. Please wait a moment and try again."}
        if "authentication" in err_str or "401" in err_str:
            return {"ok": False, "error": "auth_error",
                    "detail": "AI service authentication failed. Check your API key."}

        return {"ok": False, "error": "agent_error", "detail": str(exc)}


async def _stream_agent(message: str, history: list, agent: str, session_id: str = "default"):
    log("stream", f"start | agent={agent} | session={session_id[:8]} | '{message[:60]}'")
    result = await _run_agent_turn(message, history, agent, session_id=session_id)

    if not result["ok"]:
        log("error", f"stream failed: {result['error']} — {result.get('detail','')}")
        yield f"data: {json.dumps({'type': 'error', 'error': result['error'], 'detail': result['detail']})}\n\n"
        return

    text = result["text"] or ""
    chunks = 0
    for i in range(0, len(text), 160):
        yield f"data: {json.dumps({'type': 'chunk', 'text': text[i:i+160]})}\n\n"
        chunks += 1
        await asyncio.sleep(0.003)

    log("stream", f"done | {chunks} chunks | {len(text)} chars")
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# API routes
# =============================================================================

@app.get("/api/status")
async def api_status():
    hs_ok, hs_msg  = hs_status()
    zo_ok, zo_msg  = zo_status()
    mcp_ok, mcp_msg = get_mcp_status()
    log("info", f"status → HS={'✅' if hs_ok else '○'} Zo={'✅' if zo_ok else '○'}")
    return {
        "hubspot": {"connected": hs_ok, "message": hs_msg},
        "zoho":    {"connected": zo_ok, "message": zo_msg,
                    "mcp_ready": mcp_ok, "mcp_message": mcp_msg},
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
    if agent not in ("hubspot", "zoho_crm", "both"):
        return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)
    return StreamingResponse(
        _stream_agent(message, history, agent, session_id=session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/permissions")
async def api_permissions():
    scopes = get_token_scopes()
    if not scopes:
        log("info", "permissions → not connected")
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
    log("ok", f"permissions → {len(scopes)} scopes | admin={is_admin}")
    return {
        "connected": True,
        "is_admin":  is_admin,
        "scopes":    describe_scopes(scopes),
        "tools": [
            {"tool": name, "accessible": bool(sc), "requires": sc[:2]}
            for name, sc in tool_scope_map.items()
        ],
    }


@app.post("/api/session/end")
async def api_session_end(request: Request):
    """
    Call this when the user closes the chat / logs out.
    Deletes the tool cache for that session so memory is freed immediately.
    """
    body       = await request.json()
    session_id = body.get("session_id", "")
    if not session_id:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    await evict_session(session_id)
    log("bye", f"session ended: {session_id[:8]}")
    return {"evicted": True, "session_id": session_id}


@app.get("/api/cache-stats")
async def api_cache_stats():
    """Debug endpoint — shows what's in the tool cache right now."""
    return get_cache_stats()



@app.get("/api/debug-mcp")
async def api_debug_mcp(crm: str = "hubspot"):
    log("debug", f"MCP test → crm={crm}")
    if crm == "zoho":
        mcp_url = get_mcp_url()
        if not mcp_url:
            return JSONResponse({"error": "mcp_url_missing"}, status_code=400)
        client = MCPClient(url=mcp_url, headers={})
    else:
        try:
            token = hs_get_token()
        except RuntimeError as exc:
            log("error", f"debug-mcp no token: {exc}")
            return JSONResponse({"error": str(exc)}, status_code=401)
        client = MCPClient(url=HUBSPOT_MCP_URL, headers={"Authorization": f"Bearer {token}"})

    ok, msg = await client.preflight()
    if not ok:
        log("error", f"debug-mcp preflight failed: {msg}")
        return {"status": "preflight_failed", "detail": msg}
    try:
        await client.connect()
        tools = await client.list_tools()
        log("ok", f"debug-mcp → {len(tools)} tools")
        result = {
            "status":     "ok",
            "transport":  client.transport,
            "tool_count": len(tools),
            "tools":      [t.name for t in tools[:15]],
        }
        await client.cleanup()
        return result
    except ConnectionError as exc:
        log("error", f"debug-mcp connect failed: {exc}")
        return {"status": "connect_failed", "detail": str(exc)}


# =============================================================================
# HubSpot OAuth
# =============================================================================

@app.get("/oauth/connect")
async def oauth_connect():
    if not os.getenv("HUBSPOT_CLIENT_ID"):
        return JSONResponse({"error": "HUBSPOT_CLIENT_ID not set"}, status_code=500)
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)
    _hs_pkce[state] = verifier
    log("oauth", "HubSpot OAuth flow started")
    return RedirectResponse(hs_build_auth_url(code_challenge=challenge, state=state))


@app.get("/oauth/callback")
async def oauth_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        log("error", f"HubSpot OAuth: {error_description or error}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=hubspot")
    verifier = _hs_pkce.pop(state, None)
    if not verifier:
        log("error", "HubSpot OAuth state mismatch")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=hubspot")
    try:
        hs_exchange(code=code, code_verifier=verifier)
        log("ok", "HubSpot tokens exchanged ✓")
    except ValueError as exc:
        log("error", f"HubSpot token exchange failed: {exc}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")
    await _pool.invalidate("hubspot")
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    await _pool.invalidate("hubspot")
    await evict_all_sessions()   # flush tool cache so new account gets fresh scope guards
    log("bye", "HubSpot disconnected")
    return {"disconnected": True}


# =============================================================================
# Zoho OAuth
# =============================================================================

@app.get("/zoho/connect")
async def zoho_connect():
    if not ZO_CLIENT_ID:
        return JSONResponse({"error": "ZOHO_CLIENT_ID not set"}, status_code=500)
    verifier, challenge = zo_pkce()
    state = secrets.token_urlsafe(16)
    _zo_pkce[state] = verifier
    log("oauth", "Zoho OAuth flow started")
    return RedirectResponse(zo_build_auth_url(code_challenge=challenge, state=state))


@app.get("/zoho/callback")
async def zoho_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        log("error", f"Zoho OAuth: {error_description or error}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=zoho")
    verifier = _zo_pkce.pop(state, None)
    if not verifier:
        log("error", "Zoho OAuth state mismatch")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=zoho")
    try:
        zo_exchange(code=code, code_verifier=verifier)
        log("ok", "Zoho tokens exchanged ✓")
        # B9: clear disconnect sentinel on successful reconnect
        from zoho_auth import DISCONNECT_SENTINEL
        if DISCONNECT_SENTINEL.exists():
            DISCONNECT_SENTINEL.unlink()
    except ValueError as exc:
        log("error", f"Zoho token exchange failed: {exc}")
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=zoho")
    await _pool.invalidate("zoho_crm")
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=zoho")


@app.post("/zoho/save-mcp-url")
async def zoho_save_mcp_url(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    save_mcp_url(url)
    # B9: clear disconnect sentinel when user re-saves MCP URL
    from zoho_auth import DISCONNECT_SENTINEL
    if DISCONNECT_SENTINEL.exists():
        DISCONNECT_SENTINEL.unlink()
    await _pool.invalidate("zoho_crm")
    log("info", f"Zoho MCP URL saved → {url[:60]}")
    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    log("ok" if ok else "warn", f"Zoho MCP reachable={ok}: {msg}")
    return {"saved": True, "reachable": ok, "detail": msg}


@app.post("/zoho/disconnect")
async def zoho_disconnect():
    if ZO_TOKEN_FILE.exists():
        ZO_TOKEN_FILE.unlink()
    if MCP_URL_FILE.exists():
        MCP_URL_FILE.unlink()
    # B9: write sentinel so get_mcp_url() won't fall back to env var
    from zoho_auth import DISCONNECT_SENTINEL
    DISCONNECT_SENTINEL.write_text("disconnected")
    await _pool.invalidate("zoho_crm")
    await evict_all_sessions()   # flush tool cache so new account gets fresh scope guards
    log("bye", "Zoho disconnected")
    return {"disconnected": True}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)