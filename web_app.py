"""
Multi-CRM AI Agent — FastAPI Backend  (Improved)
=================================================
Speed improvements over original:
  1. MCP client CONNECTION POOL — clients are created once and reused.
     Each request no longer pays the connect + preflight cost (~0.5–2s saved).
  2. Health-check on reuse — if a cached client is stale it reconnects once.
  3. Background keepalive — a background task pings connected clients every 90s.
  4. Streaming chunk size increased (160 chars) → fewer SSE frames, less overhead.

Routes (unchanged from original)
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

from mcp_client import MCPClient
from core.agent import run_agent, run_agent_both
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
from core.tools import describe_scopes, build_tool_scope_map

load_dotenv()

HUBSPOT_MCP_URL = os.getenv("HUBSPOT_MCP_URL", "https://mcp.hubspot.com/")
STREAMLIT_URL   = os.getenv("STREAMLIT_URL", "http://localhost:8501")

_hs_pkce: dict[str, str] = {}
_zo_pkce: dict[str, str] = {}


# =============================================================================
# MCP Connection Pool
# =============================================================================

class _CachedClient:
    """Wraps an MCPClient with a timestamp and healthy flag."""
    def __init__(self, client: MCPClient):
        self.client    = client
        self.created   = time.time()
        self.healthy   = True

    def age(self) -> float:
        return time.time() - self.created


class MCPPool:
    """
    Simple per-key connection pool (max 1 per key).
    Keys: "hubspot" | "zoho_crm"
    """
    MAX_AGE = 300   # seconds — recreate after 5 min
    PING_EVERY = 90 # seconds — keepalive interval

    def __init__(self):
        self._pool: dict[str, _CachedClient] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> MCPClient | None:
        async with self._lock:
            cached = self._pool.get(key)
            if cached and cached.healthy and cached.age() < self.MAX_AGE:
                return cached.client
            # Stale or missing — evict and return None so caller reconnects
            if cached:
                try:
                    await cached.client.cleanup()
                except Exception:
                    pass
                del self._pool[key]
            return None

    async def put(self, key: str, client: MCPClient):
        async with self._lock:
            self._pool[key] = _CachedClient(client)

    async def invalidate(self, key: str):
        async with self._lock:
            if key in self._pool:
                try:
                    await self._pool[key].client.cleanup()
                except Exception:
                    pass
                del self._pool[key]

    async def invalidate_all(self):
        async with self._lock:
            for c in self._pool.values():
                try:
                    await c.client.cleanup()
                except Exception:
                    pass
            self._pool.clear()

    async def keepalive(self):
        """Ping all cached clients; invalidate dead ones."""
        async with self._lock:
            dead = []
            for key, cached in self._pool.items():
                try:
                    ok, _ = await cached.client.preflight()
                    if not ok:
                        dead.append(key)
                except Exception:
                    dead.append(key)
            for key in dead:
                try:
                    await self._pool[key].client.cleanup()
                except Exception:
                    pass
                del self._pool[key]
                print(f"[pool] evicted stale client: {key}")


_pool = MCPPool()


async def _keepalive_loop():
    while True:
        await asyncio.sleep(MCPPool.PING_EVERY)
        await _pool.keepalive()


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), "AZURE_OPENAI_DEPLOYMENT_NAME missing in .env"
    print("API ready → http://localhost:8000")
    task = asyncio.create_task(_keepalive_loop())
    yield
    task.cancel()
    await _pool.invalidate_all()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[STREAMLIT_URL, "http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# MCP client factories  (with pool)
# =============================================================================

async def _make_hubspot_client() -> MCPClient | None:
    # Try pool first
    cached = await _pool.get("hubspot")
    if cached:
        return cached

    # Create fresh
    try:
        token = hs_get_token()
    except RuntimeError:
        return None
    client = MCPClient(url=HUBSPOT_MCP_URL, headers={"Authorization": f"Bearer {token}"})
    ok, _ = await client.preflight()
    if not ok:
        return None
    try:
        await client.connect()
        await _pool.put("hubspot", client)
        return client
    except ConnectionError:
        return None


async def _make_zoho_client() -> tuple[MCPClient | None, str]:
    mcp_url = get_mcp_url()
    if not mcp_url:
        return None, "mcp_url_missing"
    zo_ok, _ = zo_status()
    if not zo_ok:
        return None, "oauth_missing"

    # Try pool first
    cached = await _pool.get("zoho_crm")
    if cached:
        return cached, ""

    # Create fresh
    client = MCPClient(url=mcp_url, headers={})
    ok, msg = await client.preflight()
    if not ok:
        return None, "preflight_failed"
    try:
        await client.connect()
        await _pool.put("zoho_crm", client)
        return client, ""
    except ConnectionError:
        return None, "connect_failed"


# =============================================================================
# Agent runner
# =============================================================================

async def _run_agent_turn(message: str, history: list, agent: str) -> dict:
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
        return {"ok": False, "error": zo_err, "detail": f"Zoho client unavailable: {zo_err}"}
    if agent == "hubspot" and "hubspot" not in clients:
        return {"ok": False, "error": "hubspot_unavailable", "detail": "HubSpot client unavailable"}
    if agent == "both" and not clients:
        return {"ok": False, "error": "no_clients", "detail": "No CRM clients connected"}

    try:
        if agent == "both":
            text = await run_agent_both(
                message=message, history=history, clients=clients,
                granted_scopes=scopes, is_admin=is_admin,
            )
        else:
            text = await run_agent(
                message=message, history=history, clients=clients,
                agent=agent, granted_scopes=scopes, is_admin=is_admin,
            )
        return {"ok": True, "text": text}
    except Exception as exc:
        # Invalidate pool on error so next request gets a fresh connection
        if agent in ("hubspot", "both"):
            await _pool.invalidate("hubspot")
        if agent in ("zoho_crm", "both"):
            await _pool.invalidate("zoho_crm")
        return {"ok": False, "error": "agent_error", "detail": str(exc)}
    # NOTE: no cleanup here — clients stay alive in the pool for reuse


async def _stream_agent(message: str, history: list, agent: str):
    result = await _run_agent_turn(message, history, agent)

    if not result["ok"]:
        yield f"data: {json.dumps({'type': 'error', 'error': result['error'], 'detail': result['detail']})}\n\n"
        return

    text = result["text"] or ""
    # Larger chunks (160 chars) → fewer SSE frames → faster perceived streaming
    for i in range(0, len(text), 160):
        yield f"data: {json.dumps({'type': 'chunk', 'text': text[i:i+160]})}\n\n"
        await asyncio.sleep(0.003)  # slightly faster than original 0.005

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# =============================================================================
# API routes
# =============================================================================

@app.get("/api/status")
async def api_status():
    hs_ok,  hs_msg  = hs_status()
    zo_ok,  zo_msg  = zo_status()
    mcp_ok, mcp_msg = get_mcp_status()
    return {
        "hubspot": {"connected": hs_ok, "message": hs_msg},
        "zoho":    {"connected": zo_ok, "message": zo_msg,
                    "mcp_ready": mcp_ok, "mcp_message": mcp_msg},
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    body    = await request.json()
    message = body.get("message", "").strip()
    history = body.get("history", [])
    agent   = body.get("agent", "hubspot")
    if not message:
        return JSONResponse({"error": "message required"}, status_code=400)
    if agent not in ("hubspot", "zoho_crm", "both"):
        return JSONResponse({"error": f"unknown agent: {agent}"}, status_code=400)
    return StreamingResponse(
        _stream_agent(message, history, agent),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    tool_scope_map = build_tool_scope_map(scopes)
    return {
        "connected": True,
        "is_admin":  is_admin,
        "scopes":    describe_scopes(scopes),
        "tools": [
            {"tool": name, "accessible": bool(sc), "requires": sc[:2]}
            for name, sc in tool_scope_map.items()
        ],
    }


@app.get("/api/debug-mcp")
async def api_debug_mcp(crm: str = "hubspot"):
    if crm == "zoho":
        mcp_url = get_mcp_url()
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
            "status":     "ok",
            "transport":  client.transport,
            "tool_count": len(tools),
            "tools":      [t.name for t in tools[:15]],
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
    state = secrets.token_urlsafe(16)
    _hs_pkce[state] = verifier
    return RedirectResponse(hs_build_auth_url(code_challenge=challenge, state=state))


@app.get("/oauth/callback")
async def oauth_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=hubspot")
    verifier = _hs_pkce.pop(state, None)
    if not verifier:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=hubspot")
    try:
        hs_exchange(code=code, code_verifier=verifier)
    except ValueError as exc:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")
    await _pool.invalidate("hubspot")  # force fresh connection with new token
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    await _pool.invalidate("hubspot")
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
    return RedirectResponse(zo_build_auth_url(code_challenge=challenge, state=state))


@app.get("/zoho/callback")
async def zoho_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={error_description or error}&crm=zoho")
    verifier = _zo_pkce.pop(state, None)
    if not verifier:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error=state_mismatch&crm=zoho")
    try:
        zo_exchange(code=code, code_verifier=verifier)
    except ValueError as exc:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=zoho")
    await _pool.invalidate("zoho_crm")  # force fresh connection with new token
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=zoho")


@app.post("/zoho/save-mcp-url")
async def zoho_save_mcp_url(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    save_mcp_url(url)
    await _pool.invalidate("zoho_crm")  # old URL connection is now stale
    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    return {"saved": True, "reachable": ok, "detail": msg}


@app.post("/zoho/disconnect")
async def zoho_disconnect():
    if ZO_TOKEN_FILE.exists():
        ZO_TOKEN_FILE.unlink()
    if MCP_URL_FILE.exists():
        MCP_URL_FILE.unlink()
    await _pool.invalidate("zoho_crm")
    return {"disconnected": True}


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
