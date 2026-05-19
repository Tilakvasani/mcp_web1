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
from openai import AsyncAzureOpenAI
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
# MCP Connection Pool
# =============================================================================

class _Cached:
    def __init__(self, client):
        self.client  = client
        self.created = time.time()

    def age(self):
        return time.time() - self.created


class MCPPool:
    MAX_AGE    = 600
    PING_EVERY = 600      # Zoho MCP drops idle streams fast — ping every 5s

    def __init__(self):
        self._pool = {}
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            cached = self._pool.get(key)
            if not cached:
                return None
            if cached.age() >= self.MAX_AGE:
                log("pool", f"{key} expired -- reconnecting")
                try:
                    await cached.client.reconnect()
                    cached.created = time.time()
                    log("pool", f"{key} reconnected (was expired)")
                    return cached.client
                except Exception as exc:
                    log("warn", f"{key} reconnect failed ({type(exc).__name__}) -- evicting")
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
            # Transport stale -- try to reconnect in-place
            log("warn", f"{key} stale ({type(exc).__name__}) -- reconnecting")
            try:
                await client.reconnect()
                async with self._lock:
                    c = self._pool.get(key)
                    if c:
                        c.created = time.time()
                log("pool", f"{key} reconnected after stale")
                return client
            except Exception as exc2:
                log("warn", f"{key} reconnect failed ({type(exc2).__name__}) -- evicting")
                async with self._lock:
                    self._pool.pop(key, None)
                try:
                    await client.cleanup()
                except Exception:
                    pass
                return None

    async def put(self, key, client):
        async with self._lock:
            self._pool[key] = _Cached(client)
        log("connect", f"{key} cached in pool")

    async def invalidate(self, key):
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
        for key, cached in snapshot.items():
            try:
                ok, _ = await cached.client.preflight()
                if ok:
                    log("ping", f"{key} alive")
                else:
                    # Try reconnect before evicting
                    log("warn", f"{key} preflight failed -- reconnecting")
                    try:
                        await cached.client.reconnect()
                        async with self._lock:
                            c = self._pool.get(key)
                            if c:
                                c.created = time.time()
                        log("pool", f"{key} reconnected via keepalive")
                    except Exception:
                        async with self._lock:
                            self._pool.pop(key, None)
                        try:
                            await cached.client.cleanup()
                        except Exception:
                            pass
                        log("warn", f"evicted stale: {key}")
            except Exception:
                async with self._lock:
                    self._pool.pop(key, None)
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
# Intent Detector  (NEW)
# =============================================================================

_INTENT_SYSTEM_PROMPT = """You are an intent classifier for a business AI assistant.

Given a user message, decide which backend agent should handle it.

Agents:
- "hubspot"     -> HubSpot CRM: deals, contacts, companies, tickets, tasks, meetings, emails, pipelines, workflows
- "zoho_people" -> Zoho People HRMS: employees, leave, attendance, departments, payroll, shifts, appraisals
- "cross"       -> Both systems: queries that span CRM + HR data (e.g. "which sales reps are on leave?")

Rules:
- CRM/sales/deal/contact/ticket topics only  -> hubspot
- HR/employee/leave/attendance topics only   -> zoho_people
- Both topics, or comparison/join queries    -> cross
- Greetings or off-topic                     -> hubspot (pre_router will catch them)

Respond with ONLY one word: hubspot, zoho_people, or cross."""

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncAzureOpenAI(
            api_key        = os.getenv("AZURE_OPENAI_API_KEY"),
            api_version    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    return _openai_client


async def detect_agent_intent(message: str, history: list) -> str:
    """Classify message as: hubspot | zoho_people | cross"""

    # Fast rule-based shortcuts (zero LLM cost)
    msg_lower = message.lower()

    crm_kw  = {"deal","deals","contact","contacts","company","companies","ticket","tickets",
                "pipeline","hubspot","task","tasks","meeting","meetings","email","crm","sales","stage"}
    hr_kw   = {"employee","employees","leave","attendance","department","departments","payroll",
                "shift","shifts","appraisal","zoho","hrms","hr","people","absent","on leave"}
    cross_kw = ["sales rep","account manager","both","compare","availability","assigned to"]

    has_crm  = any(k in msg_lower for k in crm_kw)
    has_hr   = any(k in msg_lower for k in hr_kw)
    has_both = any(k in msg_lower for k in cross_kw)

    if has_crm and has_hr:
        return "cross"
    if has_both and (has_crm or has_hr):
        return "cross"
    if has_hr and not has_crm:
        return "zoho_people"
    if has_crm and not has_hr:
        return "hubspot"

    # Ambiguous → LLM fallback
    try:
        history_str = ""
        if history:
            recent = history[-3:]
            history_str = "\n".join(
                f"{m['role'].upper()}: {m['content'][:120]}" for m in recent
            )
        user_content = (
            f"History:\n{history_str}\n\nMessage: {message}"
            if history_str else
            f"Message: {message}"
        )
        client = _get_openai_client()
        resp = await client.chat.completions.create(
            model    = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages = [
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens  = 5,
            temperature = 0,
        )
        result = resp.choices[0].message.content.strip().lower()
        if result in ("hubspot", "zoho_people", "cross"):
            log("intent", f"LLM detected -> {result} for '{message[:50]}'")
            return result
    except Exception as exc:
        log("warn", f"intent detection failed: {exc} — defaulting to hubspot")

    return "hubspot"


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    assert os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), "AZURE_OPENAI_DEPLOYMENT_NAME missing"
    assert os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"), "AZURE_OPENAI_EMBEDDING_DEPLOYMENT missing"
    log("boot", "FastAPI ready -> http://localhost:8000")
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

async def _make_hubspot_client():
    cached = await _pool.get("hubspot")
    if cached:
        return cached
    log("connect", "HubSpot -> connecting fresh...")
    try:
        token = hs_get_token()
    except RuntimeError:
        log("error", "HubSpot -> no valid token")
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


async def _make_zoho_people_client():
    mcp_url = zp_get_mcp_url()
    if not mcp_url:
        return None, "mcp_url_missing"
    cached = await _pool.get("zoho_people")
    if cached:
        return cached, ""
    log("connect", "Zoho People -> connecting fresh...")
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

_VALID_AGENTS = ("hubspot", "zoho_people", "cross", "auto")

async def _run_agent_turn(message, history, agent, session_id="default"):

    # Auto intent detection
    detected_agent = agent
    if agent == "auto":
        detected_agent = await detect_agent_intent(message, history)
        log("intent", f"auto-detected agent: {detected_agent}")

    clients  = {}
    scopes   = []
    is_admin = False

    if detected_agent in ("hubspot", "cross"):
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

    zp_err = ""
    if detected_agent in ("zoho_people", "cross"):
        zp, zp_err = await _make_zoho_people_client()
        if zp:
            clients["zoho_people"] = zp

    # Guard failures — with graceful cross fallback
    if detected_agent == "hubspot" and "hubspot" not in clients:
        return {"ok": False, "error": "hubspot_unavailable",
                "detail": "HubSpot not connected or token expired.", "agent": detected_agent}
    if detected_agent == "zoho_people" and "zoho_people" not in clients:
        return {"ok": False, "error": f"zoho_people_{zp_err}",
                "detail": f"Zoho People unavailable: {zp_err}", "agent": detected_agent}
    if detected_agent == "cross" and not clients:
        hs = await _make_hubspot_client()
        zp, _ = await _make_zoho_people_client()
        if hs:
            clients["hubspot"]  = hs
            detected_agent = "hubspot"
        elif zp:
            clients["zoho_people"] = zp
            detected_agent = "zoho_people"
        else:
            return {"ok": False, "error": "no_clients",
                    "detail": "Neither HubSpot nor Zoho People is connected.", "agent": detected_agent}

    t0 = time.time()
    try:
        text = await run_agent(
            message       = message,
            history       = history,
            clients       = clients,
            agent         = detected_agent,
            granted_scopes= scopes,
            is_admin      = is_admin,
            session_id    = session_id,
        )
        log("ok", f"agent done -> {len(text or '')} chars in {time.time()-t0:.1f}s")
        return {"ok": True, "text": text, "agent": detected_agent}

    except Exception as exc:
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-3:]:
            log("error", line.strip())

        # Only invalidate pool on auth errors, not transport errors.
        # Transport errors are handled by MCPPool auto-reconnect.
        err = str(exc).lower()
        if "content_filter" in err or "responsibleaipolicyviolation" in err:
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
    log("stream", f"start | agent={agent} | session={session_id[:8]} | '{message[:60]}'")
    result   = await _run_agent_turn(message, history, agent, session_id=session_id)
    detected = result.get("agent", agent)

    # Always emit the detected agent so the UI can show a badge
    yield f"data: {json.dumps({'type': 'agent', 'agent': detected})}\n\n"

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
        log("ok", "HubSpot tokens exchanged")
    except ValueError as exc:
        return RedirectResponse(f"{STREAMLIT_URL}?oauth_error={exc}&crm=hubspot")
    await _pool.invalidate("hubspot")
    deindex_agent_tools("hubspot")  # scopes may have changed -> re-index on next connect
    return RedirectResponse(f"{STREAMLIT_URL}?oauth_ok=hubspot")


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    await _pool.invalidate("hubspot")
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
    await _pool.invalidate("zoho_people")
    deindex_agent_tools("zoho_people")  # URL changed -> force re-index on next connect
    log("info", f"Zoho People MCP URL saved -> {url[:60]}")
    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    log("ok" if ok else "warn", f"Zoho People reachable={ok}: {msg}")
    return {"saved": True, "reachable": ok, "detail": msg}


@app.post("/zoho-people/disconnect")
async def zoho_people_disconnect():
    zp_disconnect_auth()
    zp_disconnect_oauth()
    await _pool.invalidate("zoho_people")
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


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)