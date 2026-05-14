"""
Multi-CRM AI Agent — Web App
==============================
FastAPI backend with Claude-style UI.
Two CRMs can be connected simultaneously; in "Both" mode the agent
queries HubSpot AND Zoho in parallel and returns labelled results.

Routes:
  GET  /                       → Chat UI
  GET  /api/status             → Both CRM connection statuses
  POST /api/chat               → SSE-streamed agent response
  GET  /oauth/connect          → HubSpot PKCE flow
  GET  /oauth/callback         → HubSpot token exchange
  POST /api/disconnect         → Disconnect HubSpot
  GET  /zoho/connect           → Zoho OAuth flow
  GET  /zoho/callback          → Zoho token exchange
  POST /zoho/save-mcp-url      → Save Zoho MCP server URL
  POST /zoho/disconnect        → Disconnect Zoho
  GET  /api/debug-mcp          → MCP connectivity test
"""

import os
import json
import asyncio
import secrets
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
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
    get_valid_token as zo_get_token,
    get_connection_status as zo_status,
    get_mcp_url,
    get_mcp_status,
    save_mcp_url,
    build_auth_url as zo_build_auth_url,
    exchange_code_for_tokens as zo_exchange,
    generate_pkce_pair as zo_pkce,
    TOKEN_FILE as ZO_TOKEN_FILE,
    ZOHO_CLIENT_ID as ZO_CLIENT_ID,
    get_zoho_org_info,
    MCP_URL_FILE,
)
from core.tools import describe_scopes, build_tool_scope_map

load_dotenv()

HUBSPOT_MCP_URL = os.getenv("HUBSPOT_MCP_URL", "https://mcp.hubspot.com/")

_hs_pkce: dict[str, str] = {}
_zo_pkce: dict[str, str] = {}


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    assert deployment, "AZURE_OPENAI_DEPLOYMENT_NAME missing in .env"
    print("🚀 Multi-CRM AI Agent ready at http://localhost:8000")
    yield


app = FastAPI(lifespan=lifespan)


# =============================================================================
# MCP client factories
# =============================================================================

async def _make_hubspot_client() -> MCPClient | None:
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
    client = MCPClient(url=mcp_url, headers={})
    ok, msg = await client.preflight()
    if not ok:
        print(f"[Zoho MCP] Preflight failed: {msg}")
        return None, "preflight_failed"
    try:
        await client.connect()
        return client, ""
    except ConnectionError as exc:
        print(f"[Zoho MCP] Connection failed: {exc}")
        return None, "connect_failed"


# =============================================================================
# Agent runner
# =============================================================================

async def _run_agent_turn(message: str, history: list, agent: str) -> str:
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

    try:
        if agent == "both":
            return await run_agent_both(
                message=message,
                history=history,
                clients=clients,
                granted_scopes=scopes,
                is_admin=is_admin,
            )
        elif agent == "zoho_crm":
            if "zoho_crm" not in clients:
                if zo_err == "oauth_missing":
                    return "⚠️ **Zoho CRM not authenticated.**\n\nClick the **+** button and complete Step 1 (OAuth login)."
                elif zo_err == "mcp_url_missing":
                    return "⚠️ **Zoho MCP URL not configured.**\n\nClick the **+** button and save your MCP Server URL from [mcp.zoho.in](https://mcp.zoho.in)."
                else:
                    return "⚠️ **Zoho MCP connection failed.**\n\nYour MCP URL may have expired. Regenerate it at [mcp.zoho.in](https://mcp.zoho.in)."
            return await run_agent(
                message=message, history=history, clients=clients,
                agent="zoho_crm", granted_scopes=scopes, is_admin=is_admin,
            )
        else:
            return await run_agent(
                message=message, history=history, clients=clients,
                agent="hubspot", granted_scopes=scopes, is_admin=is_admin,
            )
    finally:
        # Cleanup all clients without interrupting the response stream
        for c in clients.values():
            try:
                await c.cleanup()
            except Exception as e:
                # Log but don't raise — cleanup errors shouldn't interrupt the stream
                print(f"[Cleanup] Warning: {type(e).__name__} during client cleanup: {e}")


async def _stream_response(message: str, history: list, agent: str):
    """Stream agent response as SSE with robust error handling."""
    try:
        result = await _run_agent_turn(message, history, agent)
        if not result or not isinstance(result, str):
            result = str(result or "")
        
        words = result.split(" ")
        for i, word in enumerate(words):
            if not word:
                continue
            chunk = word + (" " if i < len(words) - 1 else "")
            try:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
                await asyncio.sleep(0.01)
            except Exception:
                # If yield fails, stream was probably closed by client
                return
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        try:
            yield f"data: {json.dumps({'type': 'error', 'text': error_msg})}\n\n"
        except Exception:
            # Stream already closed
            pass


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
    return StreamingResponse(
        _stream_response(message, history, agent),
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
    tool_access = [
        {"tool": name, "accessible": bool(sc), "requires": sc[:2]}
        for name, sc in tool_scope_map.items()
    ]
    return {
        "connected": True,
        "is_admin":  is_admin,
        "scopes":    describe_scopes(scopes),
        "tools":     tool_access,
    }


@app.get("/api/debug-mcp")
async def api_debug_mcp(crm: str = "hubspot"):
    if crm == "zoho":
        mcp_url = get_mcp_url()
        if not mcp_url:
            return {"error": "Zoho MCP URL not configured"}
        client = MCPClient(url=mcp_url, headers={})
    else:
        try:
            token = hs_get_token()
        except RuntimeError as exc:
            return {"error": str(exc)}
        client = MCPClient(url=HUBSPOT_MCP_URL, headers={"Authorization": f"Bearer {token}"})

    ok, msg = await client.preflight()
    result  = {"preflight": msg}
    if not ok:
        result["status"] = "preflight failed"
        return result
    try:
        await client.connect()
        tools = await client.list_tools()
        result["status"]     = "SUCCESS"
        result["transport"]  = client.transport
        result["tool_count"] = len(tools)
        result["tools"]      = [t.name for t in tools[:15]]
        await client.cleanup()
    except ConnectionError as exc:
        result["status"] = f"connection failed: {exc}"
    return result


# =============================================================================
# HubSpot OAuth routes
# =============================================================================

@app.get("/oauth/connect")
async def oauth_connect():
    if not os.getenv("HUBSPOT_CLIENT_ID"):
        return HTMLResponse(_simple_error("HUBSPOT_CLIENT_ID missing in .env"))
    verifier, challenge = generate_pkce_pair()
    state = secrets.token_urlsafe(16)
    _hs_pkce[state] = verifier
    return RedirectResponse(hs_build_auth_url(code_challenge=challenge, state=state))


@app.get("/oauth/callback")
async def oauth_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return HTMLResponse(_simple_error(error_description))
    verifier = _hs_pkce.pop(state, None)
    if not verifier:
        return HTMLResponse(_simple_error("PKCE state mismatch — please try again."), status_code=400)
    try:
        tokens = hs_exchange(code=code, code_verifier=verifier)
    except ValueError as exc:
        return HTMLResponse(_simple_error(str(exc)))
    portal_id = ""
    try:
        info = httpx.get(
            "https://api.hubapi.com/account-info/v3/details",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=10,
        ).json()
        portal_id = str(info.get("portalId", ""))
    except Exception:
        pass
    return HTMLResponse(_success_page("HubSpot", "🟠", portal_id or "Portal connected"))


@app.post("/api/disconnect")
async def api_disconnect():
    if HS_TOKEN_FILE.exists():
        HS_TOKEN_FILE.unlink()
    return {"disconnected": True}


# =============================================================================
# Zoho OAuth routes
# =============================================================================

@app.get("/zoho/connect")
async def zoho_connect():
    if not ZO_CLIENT_ID:
        return HTMLResponse(_simple_error("ZOHO_CLIENT_ID missing in .env"))
    verifier, challenge = zo_pkce()
    state = secrets.token_urlsafe(16)
    _zo_pkce[state] = verifier
    return RedirectResponse(zo_build_auth_url(code_challenge=challenge, state=state))


@app.get("/zoho/callback")
async def zoho_callback(code: str = "", state: str = "", error: str = "", error_description: str = ""):
    if error:
        return HTMLResponse(_simple_error(error_description))
    verifier = _zo_pkce.pop(state, None)
    if not verifier:
        return HTMLResponse(_simple_error("PKCE state mismatch — please try again."), status_code=400)
    try:
        tokens = zo_exchange(code=code, code_verifier=verifier)
    except ValueError as exc:
        return HTMLResponse(_simple_error(str(exc)))
    org    = get_zoho_org_info()
    label  = org.get("company_name", "Zoho CRM") if org else "Zoho CRM"
    return HTMLResponse(_success_page("Zoho CRM", "🔵", label))


@app.post("/zoho/save-mcp-url")
async def zoho_save_mcp_url(request: Request):
    body = await request.json()
    url  = body.get("url", "").strip()
    if not url:
        return JSONResponse({"error": "url required"}, status_code=400)
    save_mcp_url(url)
    client = MCPClient(url=url, headers={})
    ok, msg = await client.preflight()
    return {"saved": True, "reachable": ok, "message": msg}


@app.post("/zoho/disconnect")
async def zoho_disconnect():
    if ZO_TOKEN_FILE.exists():
        ZO_TOKEN_FILE.unlink()
    if MCP_URL_FILE.exists():
        MCP_URL_FILE.unlink()
    return {"disconnected": True}


# =============================================================================
# HTML helpers
# =============================================================================

def _simple_error(msg: str) -> str:
    return f"""<html><body style="font-family:sans-serif;padding:40px">
    <h2>❌ Error</h2><p>{msg}</p><a href="/">← Back</a></body></html>"""


def _success_page(crm: str, icon: str, detail: str) -> str:
    return f"""<!DOCTYPE html><html>
<head><title>Connected!</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f1117;display:flex;align-items:center;justify-content:center;
     min-height:100vh;margin:0;color:#e8eaf6}}
.card{{background:#1a1d27;border:1px solid #2e3250;border-radius:20px;
       padding:48px 56px;text-align:center;max-width:400px;
       box-shadow:0 20px 60px rgba(0,0,0,.4)}}
h1{{font-size:22px;margin-bottom:8px}}
p{{color:#8b92b8;font-size:14px;line-height:1.6}}
.badge{{display:inline-block;margin-top:16px;background:rgba(76,175,125,.15);
        color:#4caf7d;padding:6px 18px;border-radius:20px;font-size:13px;
        border:1px solid rgba(76,175,125,.3)}}
a{{display:inline-block;margin-top:20px;background:#5b8dee;color:white;
   padding:10px 24px;border-radius:10px;text-decoration:none;font-weight:500}}
</style></head>
<body><div class="card">
<div style="font-size:52px;margin-bottom:12px">{icon}</div>
<h1>{crm} Connected!</h1>
<p>{detail}</p>
<div class="badge">✅ Connected Successfully</div><br>
<a href="/">Open Agent →</a>
</div>
<script>setTimeout(()=>window.location.href='/',2500)</script>
</body></html>"""


# =============================================================================
# Frontend — Claude-style UI
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(CHAT_UI)


CHAT_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CRM AI Agent</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.0.6/purify.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:    #212121;
  --bg2:   #2f2f2f;
  --bg3:   #3a3a3a;
  --bg4:   #424242;
  --bd:    #4a4a4a;
  --tx:    #ececec;
  --tx2:   #a0a0a0;
  --tx3:   #666;
  --hs:    #ff7a59;
  --zo:    #4da3ff;
  --both:  #a78bfa;
  --green: #3dba7a;
  --red:   #f06060;
  --r:     12px;
  --sw:    260px;
}

html, body { height:100%; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:var(--bg); color:var(--tx); overflow:hidden; }

.app { display:flex; height:100vh; }

/* ── SIDEBAR ── */
.sidebar { width:var(--sw); background:var(--bg2); border-right:1px solid var(--bd); display:flex; flex-direction:column; flex-shrink:0; overflow:hidden; }

.sb-top { padding:14px 12px 10px; border-bottom:1px solid var(--bd); display:flex; align-items:center; justify-content:space-between; }
.logo { display:flex; align-items:center; gap:9px; }
.logo-icon { width:30px; height:30px; background:linear-gradient(135deg,#a78bfa,#4da3ff); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:14px; }
.logo-text { font-size:13px; font-weight:700; }
.new-btn { width:28px; height:28px; background:transparent; border:1px solid var(--bd); border-radius:7px; color:var(--tx2); cursor:pointer; display:flex; align-items:center; justify-content:center; transition:all .15s; }
.new-btn:hover { background:var(--bg3); color:var(--tx); }

.crm-sel { padding:10px 12px; border-bottom:1px solid var(--bd); }
.crm-sel-lbl { font-size:10px; font-weight:700; color:var(--tx3); text-transform:uppercase; letter-spacing:.7px; margin-bottom:7px; }
.crm-btns { display:flex; gap:4px; }
.crm-btn { flex:1; padding:6px 4px; background:transparent; border:1px solid var(--bd); border-radius:8px; color:var(--tx2); font-size:11px; font-weight:600; cursor:pointer; transition:all .15s; display:flex; align-items:center; justify-content:center; gap:4px; }
.crm-btn:hover { background:var(--bg3); color:var(--tx); }
.crm-btn.a-hs   { background:rgba(255,122,89,.15); color:var(--hs); border-color:rgba(255,122,89,.4); }
.crm-btn.a-zo   { background:rgba(77,163,255,.15); color:var(--zo); border-color:rgba(77,163,255,.4); }
.crm-btn.a-both { background:rgba(167,139,250,.15); color:var(--both); border-color:rgba(167,139,250,.4); }

.conn-area { padding:8px 12px; border-bottom:1px solid var(--bd); display:flex; flex-direction:column; gap:2px; }
.conn-row { display:flex; align-items:center; gap:7px; font-size:11.5px; color:var(--tx2); padding:5px 8px; border-radius:7px; cursor:pointer; transition:background .15s; }
.conn-row:hover { background:var(--bg3); }
.conn-dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; }
.conn-dot.on  { background:var(--green); box-shadow:0 0 5px var(--green); }
.conn-dot.off { background:var(--tx3); }
.conn-name { flex:1; font-weight:500; }
.conn-tag { font-size:10px; color:var(--tx3); background:var(--bg4); padding:2px 7px; border-radius:10px; }

.sb-nav { flex:1; overflow-y:auto; padding:8px; }
.nav-sec { font-size:10px; font-weight:700; color:var(--tx3); text-transform:uppercase; letter-spacing:.7px; padding:10px 8px 5px; }
.nav-btn { width:100%; display:flex; align-items:center; gap:9px; padding:7px 9px; border-radius:8px; background:transparent; border:none; color:var(--tx2); font-size:12.5px; cursor:pointer; text-align:left; transition:all .15s; }
.nav-btn:hover { background:var(--bg3); color:var(--tx); }

.add-conn { padding:10px 12px; border-top:1px solid var(--bd); }
.add-conn-btn { width:100%; display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:9px; background:var(--bg3); border:1px solid var(--bd); color:var(--tx2); font-size:12px; font-weight:600; cursor:pointer; transition:all .15s; }
.add-conn-btn:hover { background:var(--bg4); color:var(--tx); }
.plus-icon { width:22px; height:22px; background:var(--bg4); border-radius:6px; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:300; line-height:1; flex-shrink:0; }

/* ── MAIN ── */
.main { flex:1; display:flex; flex-direction:column; overflow:hidden; }

#messages { flex:1; overflow-y:auto; padding:32px 0; display:flex; flex-direction:column; align-items:center; }

.welcome { width:100%; max-width:680px; padding:60px 24px 20px; text-align:center; }
.w-logo { font-size:44px; margin-bottom:16px; filter:drop-shadow(0 0 24px rgba(167,139,250,.4)); }
.welcome h1 { font-size:26px; font-weight:700; letter-spacing:-.5px; margin-bottom:8px; }
.welcome p { font-size:14px; color:var(--tx2); line-height:1.7; }
.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:24px; }
.chip { background:var(--bg2); border:1px solid var(--bd); border-radius:20px; padding:7px 14px; font-size:12.5px; font-weight:500; color:var(--tx2); cursor:pointer; transition:all .15s; }
.chip:hover { border-color:var(--both); color:var(--both); transform:translateY(-1px); }

.msg-row { width:100%; max-width:680px; padding:2px 24px; animation:fi .18s ease; }
@keyframes fi { from { opacity:0; transform:translateY(5px); } }

.msg-row.user { display:flex; justify-content:flex-end; }
.ubub { background:var(--bg2); border:1px solid var(--bd); border-radius:18px 18px 4px 18px; padding:10px 16px; font-size:14px; line-height:1.6; max-width:85%; white-space:pre-wrap; }

.msg-row.assistant { display:flex; gap:12px; align-items:flex-start; }
.av { width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:13px; flex-shrink:0; margin-top:3px; }
.av.hs   { background:linear-gradient(135deg,var(--hs),#e05a3a); }
.av.zo   { background:linear-gradient(135deg,var(--zo),#0b76ef); }
.av.both { background:linear-gradient(135deg,var(--both),var(--zo)); }
.ac { flex:1; min-width:0; }
.an { font-size:12px; font-weight:700; margin-bottom:4px; }
.an.hs   { color:var(--hs); }
.an.zo   { color:var(--zo); }
.an.both { color:var(--both); }

.prose { font-size:14px; line-height:1.75; color:var(--tx); }
.prose p { margin-bottom:10px; }
.prose p:last-child { margin-bottom:0; }
.prose h1,.prose h2 { font-size:15px; font-weight:700; margin:14px 0 5px; color:var(--tx); }
.prose h3 { font-size:13.5px; font-weight:700; margin:12px 0 4px; color:var(--tx2); }
.prose ul,.prose ol { padding-left:20px; margin-bottom:10px; }
.prose li { margin-bottom:4px; }
.prose code { background:var(--bg3); padding:2px 6px; border-radius:5px; font-family:'SF Mono','Fira Code',monospace; font-size:12px; color:var(--zo); border:1px solid var(--bd); }
.prose pre { background:var(--bg3); border:1px solid var(--bd); border-radius:10px; padding:14px; overflow-x:auto; margin:10px 0; }
.prose pre code { background:none; border:none; color:var(--tx); }
.prose table { width:100%; border-collapse:collapse; margin:12px 0; font-size:13px; }
.prose th { background:var(--bg3); padding:7px 12px; text-align:left; color:var(--tx2); border:1px solid var(--bd); font-size:11px; text-transform:uppercase; letter-spacing:.4px; }
.prose td { padding:7px 12px; border:1px solid var(--bd); }
.prose tr:nth-child(even) td { background:rgba(255,255,255,.02); }
.prose strong { color:var(--tx); font-weight:600; }
.prose a { color:var(--zo); text-decoration:none; }
.prose a:hover { text-decoration:underline; }
.prose blockquote { border-left:3px solid var(--both); padding-left:14px; color:var(--tx2); margin:8px 0; }
.prose hr { border:none; border-top:1px solid var(--bd); margin:16px 0; }

.typing { display:flex; gap:5px; padding:4px 0; align-items:center; }
.typing span { width:6px; height:6px; border-radius:50%; background:var(--tx3); animation:td .9s infinite; }
.typing span:nth-child(2) { animation-delay:.15s; }
.typing span:nth-child(3) { animation-delay:.3s; }
@keyframes td { 0%,60%,100%{transform:translateY(0);opacity:.4;} 30%{transform:translateY(-5px);opacity:1;} }

/* ── INPUT ── */
.inp-area { padding:12px 24px 16px; display:flex; justify-content:center; }
.inp-wrap { width:100%; max-width:680px; background:var(--bg2); border:1px solid var(--bd); border-radius:14px; transition:border-color .15s; overflow:hidden; }
.inp-wrap:focus-within { border-color:var(--tx3); }
.inp-inner { display:flex; align-items:flex-end; gap:8px; padding:10px 12px 10px 16px; }
#msg-input { flex:1; background:none; border:none; outline:none; color:var(--tx); font-size:14px; resize:none; max-height:160px; line-height:1.6; font-family:inherit; padding:2px 0; }
#msg-input::placeholder { color:var(--tx3); }
.send-btn { width:32px; height:32px; background:var(--tx2); border:none; border-radius:8px; cursor:pointer; display:flex; align-items:center; justify-content:center; color:var(--bg); font-size:14px; flex-shrink:0; transition:all .15s; align-self:flex-end; }
.send-btn:hover:not(:disabled) { background:var(--tx); }
.send-btn:disabled { background:var(--bg4); cursor:not-allowed; color:var(--tx3); }
.inp-footer { padding:5px 16px 8px; font-size:11px; color:var(--tx3); display:flex; gap:6px; align-items:center; }
.badge { display:inline-flex; align-items:center; gap:4px; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
.badge.hs   { background:rgba(255,122,89,.12); color:var(--hs); }
.badge.zo   { background:rgba(77,163,255,.12);  color:var(--zo); }
.badge.both { background:rgba(167,139,250,.12); color:var(--both); }

/* ── MODAL ── */
.mbg { position:fixed; inset:0; background:rgba(0,0,0,.7); z-index:9999; display:flex; align-items:center; justify-content:center; padding:20px; backdrop-filter:blur(3px); }
.modal { background:var(--bg2); border:1px solid var(--bd); border-radius:16px; width:100%; max-width:520px; max-height:85vh; display:flex; flex-direction:column; overflow:hidden; box-shadow:0 24px 80px rgba(0,0,0,.7); animation:mp .2s cubic-bezier(.34,1.56,.64,1); }
@keyframes mp { from{opacity:0;transform:scale(.94) translateY(8px);} }
.mh { padding:16px 20px; border-bottom:1px solid var(--bd); display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
.mt { font-size:15px; font-weight:700; }
.ms { font-size:11.5px; color:var(--tx2); margin-top:2px; }
.mc { background:var(--bg3); border:1px solid var(--bd); color:var(--tx2); border-radius:7px; padding:4px 11px; cursor:pointer; font-size:12px; }
.mc:hover { color:var(--tx); }
.mb { padding:16px 20px; overflow-y:auto; flex:1; display:flex; flex-direction:column; gap:14px; }

.ctabs { display:flex; gap:6px; }
.ctab { flex:1; padding:9px; background:var(--bg3); border:1px solid var(--bd); border-radius:10px; cursor:pointer; text-align:center; transition:all .15s; display:flex; flex-direction:column; align-items:center; gap:4px; }
.ctab:hover { border-color:var(--tx3); }
.ctab.a-hs { border-color:var(--hs); background:rgba(255,122,89,.1); }
.ctab.a-zo { border-color:var(--zo); background:rgba(77,163,255,.1); }
.ctab .ci { font-size:22px; }
.ctab .cn { font-size:11px; font-weight:700; color:var(--tx2); }
.ctab .cs { font-size:10px; color:var(--tx3); }
.ctab.a-hs .cn { color:var(--hs); }
.ctab.a-zo .cn { color:var(--zo); }

.sc { background:var(--bg3); border:1px solid var(--bd); border-radius:11px; padding:14px; }
.sc-t { font-size:12.5px; font-weight:700; color:var(--tx); margin-bottom:5px; }
.sc-d { font-size:11.5px; color:var(--tx2); line-height:1.6; margin-bottom:10px; }
.btn-p { padding:8px 16px; border:none; border-radius:8px; font-size:12.5px; font-weight:600; cursor:pointer; transition:all .15s; }
.btn-hs { background:var(--hs); color:#fff; }
.btn-hs:hover { opacity:.88; }
.btn-zo { background:var(--zo); color:#fff; }
.btn-zo:hover { opacity:.88; }
.url-row { display:flex; gap:8px; margin-top:8px; }
.url-inp { flex:1; background:var(--bg2); border:1px solid var(--bd); border-radius:8px; padding:8px 11px; color:var(--tx); font-size:12px; outline:none; font-family:monospace; }
.url-inp:focus { border-color:var(--zo); }
.url-res { font-size:12px; color:var(--tx2); margin-top:7px; min-height:18px; }
.dbtn { background:none; border:none; color:var(--red); font-size:12px; font-weight:600; cursor:pointer; padding:0; }
.dbtn:hover { text-decoration:underline; }

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--bd); border-radius:3px; }
</style>
</head>
<body>
<div class="app">

  <!-- SIDEBAR -->
  <div class="sidebar">
    <div class="sb-top">
      <div class="logo">
        <div class="logo-icon">🤖</div>
        <div class="logo-text">CRM Agent</div>
      </div>
      <button class="new-btn" onclick="clearChat()" title="New chat">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 5v14M5 12h14"/></svg>
      </button>
    </div>

    <div class="crm-sel">
      <div class="crm-sel-lbl">Active CRM</div>
      <div class="crm-btns">
        <button class="crm-btn a-hs" id="btn-hs"   onclick="setAgent('hubspot')">🟠 HubSpot</button>
        <button class="crm-btn"      id="btn-zo"   onclick="setAgent('zoho_crm')">🔵 Zoho</button>
        <button class="crm-btn"      id="btn-both" onclick="setAgent('both')">⚡ Both</button>
      </div>
    </div>

    <div class="conn-area">
      <div class="conn-row" onclick="openModal('hubspot')">
        <div class="conn-dot off" id="hs-dot"></div>
        <span class="conn-name">HubSpot</span>
        <span class="conn-tag" id="hs-tag">Connect</span>
      </div>
      <div class="conn-row" onclick="openModal('zoho')">
        <div class="conn-dot off" id="zo-dot"></div>
        <span class="conn-name">Zoho CRM</span>
        <span class="conn-tag" id="zo-tag">Connect</span>
      </div>
    </div>

    <div class="sb-nav" id="sb-nav"></div>

    <div class="add-conn">
      <button class="add-conn-btn" onclick="openModal('hubspot')">
        <div class="plus-icon">+</div>
        <span>Add connector</span>
      </button>
    </div>
  </div>

  <!-- MAIN -->
  <div class="main">
    <div id="messages">
      <div class="welcome" id="welcome">
        <div class="w-logo">🤖</div>
        <h1>How can I help?</h1>
        <p id="wdesc">Connect HubSpot or Zoho CRM using the <strong>+</strong> button.<br>Use <strong>⚡ Both</strong> to query both CRMs at once.</p>
        <div class="chips" id="wchips"></div>
      </div>
    </div>

    <div class="inp-area">
      <div class="inp-wrap">
        <div class="inp-inner">
          <textarea id="msg-input" placeholder="Ask anything about your CRM data..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
          <button class="send-btn" id="send-btn" onclick="sendFromInput()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
          </button>
        </div>
        <div class="inp-footer">
          <span id="abadge" class="badge hs">🟠 HubSpot</span>
          <span>· Enter to send · Shift+Enter for new line</span>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- CONNECTOR MODAL -->
<div class="mbg" id="mbg" style="display:none" onclick="closeMbg(event)">
  <div class="modal">
    <div class="mh">
      <div><div class="mt">Connect a CRM</div><div class="ms">Connect HubSpot, Zoho CRM, or both</div></div>
      <button class="mc" onclick="closeModal()">✕</button>
    </div>
    <div class="mb">
      <div class="ctabs">
        <div class="ctab a-hs" id="ctab-hs" onclick="switchTab('hubspot')">
          <div class="ci">🟠</div><div class="cn">HubSpot</div><div class="cs" id="chs-st">Not connected</div>
        </div>
        <div class="ctab" id="ctab-zo" onclick="switchTab('zoho')">
          <div class="ci">🔵</div><div class="cn">Zoho CRM</div><div class="cs" id="czo-st">Not connected</div>
        </div>
      </div>

      <!-- HubSpot panel -->
      <div id="p-hs">
        <div class="sc">
          <div class="sc-t">🔑 OAuth Authentication</div>
          <div class="sc-d">Authenticate with your HubSpot account. Requires <code>HUBSPOT_CLIENT_ID</code> in <code>.env</code>.</div>
          <button class="btn-p btn-hs" onclick="window.location.href='/oauth/connect'">Connect via OAuth →</button>
        </div>
        <div class="sc" style="background:rgba(240,96,96,.06);border-color:rgba(240,96,96,.2)">
          <div class="sc-t" style="color:var(--red)">Disconnect</div>
          <button class="dbtn" onclick="disconnectHs()">Disconnect HubSpot</button>
        </div>
      </div>

      <!-- Zoho panel -->
      <div id="p-zo" style="display:none">
        <div class="sc">
          <div class="sc-t">① OAuth Authentication</div>
          <div class="sc-d">Authenticate with your Zoho account to grant CRM access.</div>
          <div style="display:flex;align-items:center;gap:10px">
            <button class="btn-p btn-zo" onclick="window.location.href='/zoho/connect'">Connect via OAuth →</button>
            <span style="font-size:11px;color:var(--tx3)" id="zo-ol"></span>
          </div>
        </div>
        <div class="sc">
          <div class="sc-t">② MCP Server URL</div>
          <div class="sc-d">Copy your URL from <a href="https://mcp.zoho.in" target="_blank" style="color:var(--zo)">mcp.zoho.in</a> → Connect → Copy URL.</div>
          <div class="url-row">
            <input type="text" class="url-inp" id="mcp-inp" placeholder="https://crm-data-metadata-XXXXX.zohomcp.in/mcp/APIKEY/message">
            <button class="btn-p btn-zo" onclick="saveMcp()">Save & Test</button>
          </div>
          <div class="url-res" id="mcp-res"></div>
        </div>
        <div class="sc" style="background:rgba(240,96,96,.06);border-color:rgba(240,96,96,.2)">
          <div class="sc-t" style="color:var(--red)">Disconnect</div>
          <button class="dbtn" onclick="disconnectZo()">Disconnect Zoho CRM</button>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
marked.setOptions({ breaks:true, gfm:true });

let history=[], streaming=false, activeAgent='hubspot';
let hsOk=false, zoOk=false;

const AG = {
  hubspot:  { label:'HubSpot',   icon:'🟠', cls:'hs',
    chips:[{t:'Open deals',m:'Show me all open deals'},{t:'All contacts',m:'Show me all contacts'},{t:'Pipeline overview',m:'Give me a pipeline overview'},{t:'Open tickets',m:'Show all open support tickets'}],
    nav:[{s:'Quick Actions'},{i:'💰',l:'Open Deals',m:'Show me all open deals'},{i:'👤',l:'All Contacts',m:'Show me all contacts'},{i:'🎫',l:'Open Tickets',m:'Show all open tickets'},{i:'🏢',l:'Companies',m:'Show me all companies'},{s:'Tools'},{i:'🔍',l:'Find contact',p:'/find_contact '},{i:'📊',l:'Deal report',m:'/deal_report'},{s:'Settings'},{i:'🔑',l:'My Permissions',a:'showHsPerms'}]},
  zoho_crm: { label:'Zoho CRM',  icon:'🔵', cls:'zo',
    chips:[{t:'All leads',m:'Show me all leads in Zoho CRM'},{t:'All contacts',m:'List all contacts with email'},{t:'Open deals',m:'Show all deals in the pipeline'},{t:'Search records',m:'Search for a contact by email'}],
    nav:[{s:'Quick Actions'},{i:'🏆',l:'All Leads',m:'Show me all leads'},{i:'👤',l:'Contacts',m:'Show me all contacts'},{i:'💰',l:'Deals',m:'Show all deals grouped by stage'},{i:'🏢',l:'Accounts',m:'Show me all accounts'},{s:'Operations'},{i:'🔄',l:'Convert Lead',m:'Help me convert a lead'},{i:'➕',l:'Create Lead',m:'Create a new lead'},{i:'📊',l:'All Modules',m:'List all available Zoho CRM modules'},{s:'Settings'},{i:'🔌',l:'Manage Connection',a:"openModal('zoho')"}]},
  both:     { label:'Both CRMs', icon:'⚡', cls:'both',
    chips:[{t:'All deals (both)',m:'Show me all open deals'},{t:'All contacts (both)',m:'Show me all contacts'},{t:'Compare pipelines',m:'Show pipeline overview from both CRMs'},{t:'All leads (both)',m:'Show me all leads'}],
    nav:[{s:'Both CRMs'},{i:'💰',l:'All Deals',m:'Show all open deals from both CRMs'},{i:'👤',l:'All Contacts',m:'Show all contacts from both CRMs'},{i:'📊',l:'Pipeline Compare',m:'Compare pipelines from HubSpot and Zoho'},{i:'🏆',l:'All Leads',m:'Show all leads from both CRMs'}]},
};

window.addEventListener('load', async () => { await refresh(); renderSidebar(); renderWelcome(); });

async function refresh() {
  try {
    const d = await fetch('/api/status').then(r=>r.json());
    hsOk = d.hubspot.connected;
    zoOk = d.zoho.connected && d.zoho.mcp_ready;
    setDot('hs', hsOk); setDot('zo', zoOk);
    document.getElementById('hs-tag').textContent = hsOk ? '✓ Connected' : 'Connect';
    document.getElementById('zo-tag').textContent = zoOk ? '✓ Connected' : 'Connect';
    document.getElementById('chs-st').textContent = hsOk ? '✅ Connected' : 'Not connected';
    document.getElementById('czo-st').textContent = zoOk ? '✅ Connected' : 'Not connected';
    const ol = document.getElementById('zo-ol');
    if (ol) ol.textContent = d.zoho.connected ? '✅ OAuth connected' : '';
    const saved = localStorage.getItem('zoho_mcp_url');
    if (saved) { const e=document.getElementById('mcp-inp'); if(e&&!e.value) e.value=saved; }
  } catch(e) {}
}

function setDot(p, on) {
  const d = document.getElementById(p+'-dot');
  if (d) d.className = 'conn-dot '+(on?'on':'off');
}

function setAgent(a) {
  activeAgent = a;
  const map = {hubspot:'hs', zoho_crm:'zo', both:'both'};
  const cls = {hubspot:'a-hs', zoho_crm:'a-zo', both:'a-both'};
  ['hubspot','zoho_crm','both'].forEach(x => {
    const b = document.getElementById('btn-'+(x==='zoho_crm'?'zo':x==='hubspot'?'hs':'both'));
    if (b) b.className = 'crm-btn'+(x===a?' '+cls[x]:'');
  });
  const cfg = AG[a];
  const badge = document.getElementById('abadge');
  badge.className = 'badge '+cfg.cls;
  badge.textContent = cfg.icon+' '+cfg.label;
  document.getElementById('msg-input').placeholder = `Ask anything about ${cfg.label} data...`;
  renderSidebar(); renderWelcome();
}

function renderSidebar() {
  const cfg = AG[activeAgent];
  document.getElementById('sb-nav').innerHTML = cfg.nav.map(item => {
    if (item.s) return `<div class="nav-sec">${item.s}</div>`;
    let c = '';
    if (item.m) c = `sendMsg(${JSON.stringify(item.m)})`;
    if (item.p) c = `setInput(${JSON.stringify(item.p)})`;
    if (item.a) c = item.a+(item.a.includes('(')?'':'()');
    return `<button class="nav-btn" onclick="${c}"><span style="font-size:13px">${item.i}</span><span>${item.l}</span></button>`;
  }).join('');
}

function renderWelcome() {
  const cfg = AG[activeAgent];
  const el = document.getElementById('welcome'); if (!el) return;
  document.getElementById('wdesc').innerHTML = activeAgent==='both'
    ? 'Queries <strong>both HubSpot and Zoho CRM</strong> simultaneously.<br>Results shown side-by-side with clear labels.'
    : `Connect your <strong>${cfg.label}</strong> using the <strong>+</strong> button below.`;
  document.getElementById('wchips').innerHTML = cfg.chips.map(c =>
    `<div class="chip" onclick="sendMsg(${JSON.stringify(c.m)})">${c.t}</div>`).join('');
}

// Modal
function openModal(tab) { document.getElementById('mbg').style.display='flex'; switchTab(tab||'hubspot'); refresh(); }
function closeModal()    { document.getElementById('mbg').style.display='none'; }
function closeMbg(e)     { if(e.target===document.getElementById('mbg')) closeModal(); }

function switchTab(t) {
  document.getElementById('p-hs').style.display = t==='hubspot'?'':'none';
  document.getElementById('p-zo').style.display = t==='zoho'   ?'':'none';
  document.getElementById('ctab-hs').className  = 'ctab'+(t==='hubspot'?' a-hs':'');
  document.getElementById('ctab-zo').className  = 'ctab'+(t==='zoho'   ?' a-zo':'');
}

async function disconnectHs() {
  if (!confirm('Disconnect HubSpot?')) return;
  await fetch('/api/disconnect',{method:'POST'}); await refresh();
}
async function disconnectZo() {
  if (!confirm('Disconnect Zoho CRM?')) return;
  await fetch('/zoho/disconnect',{method:'POST'}); await refresh();
}
async function saveMcp() {
  const url = document.getElementById('mcp-inp').value.trim();
  if (!url) return;
  const res = document.getElementById('mcp-res');
  res.textContent = '⏳ Testing...';
  try {
    const r = await fetch('/zoho/save-mcp-url',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
    const d = await r.json();
    localStorage.setItem('zoho_mcp_url', url);
    res.style.color = d.reachable?'var(--green)':'var(--hs)';
    res.textContent = d.reachable?'✅ Connected! '+d.message:'⚠️ Saved but: '+d.message;
    await refresh();
  } catch(e) { res.style.color='var(--red)'; res.textContent='❌ Error: '+e.message; }
}

// Chat
function handleKey(e) { if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendFromInput();} }
function sendFromInput() {
  const inp=document.getElementById('msg-input'), txt=inp.value.trim();
  if (!txt||streaming) return;
  inp.value=''; autoResize(inp); sendMsg(txt);
}
function setInput(t) { const e=document.getElementById('msg-input'); e.value=t; e.focus(); autoResize(e); }
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,160)+'px'; }

async function sendMsg(text) {
  if (streaming) return;
  streaming = true;
  const w = document.getElementById('welcome'); if (w) w.remove();
  appendUser(text);
  history.push({role:'user', content:text});
  const cfg = AG[activeAgent];
  const row = appendAssistant('', cfg, true);
  document.getElementById('send-btn').disabled = true;
  let full = '';
  try {
    const resp = await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text, history:history.slice(0,-1), agent:activeAgent})});
    const reader=resp.body.getReader(), dec=new TextDecoder();
    while(true) {
      const {done,value}=await reader.read(); if(done) break;
      for (const line of dec.decode(value).split(/\r?\n/)) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d=JSON.parse(line.slice(6));
          if (d.type==='chunk') { full+=d.text; updA(row,full,true); }
          else if (d.type==='done') { updA(row,full,false); history.push({role:'assistant',content:full}); }
          else if (d.type==='error') { updA(row,'❌ '+d.text,false); }
        } catch(e) {}
      }
    }
  } catch(e) { updA(row,'❌ Error: '+e.message,false); }
  streaming=false;
  document.getElementById('send-btn').disabled=false;
  document.getElementById('msg-input').focus();
}

function appendUser(text) {
  const msgs=document.getElementById('messages'), row=document.createElement('div');
  row.className='msg-row user';
  row.innerHTML=`<div class="ubub">${esc(text)}</div>`;
  msgs.appendChild(row); msgs.scrollTop=msgs.scrollHeight;
}

function appendAssistant(text, cfg, typing) {
  const msgs=document.getElementById('messages'), row=document.createElement('div');
  row.className='msg-row assistant';
  const av=document.createElement('div'); av.className=`av ${cfg.cls}`; av.textContent=cfg.icon;
  const ac=document.createElement('div'); ac.className='ac';
  const an=document.createElement('div'); an.className=`an ${cfg.cls}`; an.textContent=cfg.label;
  const pr=document.createElement('div'); pr.className='prose';
  pr.innerHTML = typing ? '<div class="typing"><span></span><span></span><span></span></div>'
                        : DOMPurify.sanitize(marked.parse(text||''));
  ac.appendChild(an); ac.appendChild(pr); row.appendChild(av); row.appendChild(ac);
  msgs.appendChild(row); msgs.scrollTop=msgs.scrollHeight;
  return row;
}

function updA(row, text, streaming) {
  const pr=row.querySelector('.prose');
  if (streaming) pr.textContent=text+'▌';
  else pr.innerHTML=DOMPurify.sanitize(marked.parse(text));
  document.getElementById('messages').scrollTop=99999;
}

function clearChat() {
  history=[]; const msgs=document.getElementById('messages'); msgs.innerHTML='';
  const w=document.createElement('div'); w.id='welcome'; w.className='welcome';
  w.innerHTML='<div class="w-logo">🤖</div><h1>How can I help?</h1><p id="wdesc"></p><div class="chips" id="wchips"></div>';
  msgs.appendChild(w); renderWelcome();
}

function esc(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function showHsPerms() {
  const bg=document.createElement('div'); bg.className='mbg';
  bg.innerHTML=`<div class="modal"><div class="mh"><div><div class="mt">🔑 HubSpot Permissions</div><div class="ms" id="pr-role">Loading…</div></div><button class="mc" onclick="this.closest('.mbg').remove()">✕</button></div><div class="mb" id="pr-body" style="gap:0"></div></div>`;
  document.body.appendChild(bg);
  bg.addEventListener('click', e=>{ if(e.target===bg) bg.remove(); });
  try {
    const d=await fetch('/api/permissions').then(r=>r.json());
    if (!d.connected) { document.getElementById('pr-role').textContent='⚠️ Not connected'; return; }
    document.getElementById('pr-role').innerHTML = d.is_admin
      ? '<span style="color:var(--green)">🔑 Super Admin</span>'
      : '<span style="color:var(--zo)">👤 Standard User</span>';
    document.getElementById('pr-body').innerHTML =
      '<div><div style="font-size:10px;font-weight:700;color:var(--tx3);text-transform:uppercase;letter-spacing:.7px;margin-bottom:8px">Tool Access</div>'
      +'<div style="display:flex;flex-wrap:wrap;gap:5px">'
      +d.tools.map(t=>`<span style="padding:4px 10px;border-radius:16px;font-size:11px;font-weight:600;background:${t.accessible?'rgba(61,186,122,.12)':'rgba(240,96,96,.1)'};color:${t.accessible?'var(--green)':'var(--red)'};border:1px solid ${t.accessible?'rgba(61,186,122,.25)':'rgba(240,96,96,.25)'}">${t.accessible?'✅':'🚫'} ${t.tool}</span>`).join('')
      +'</div></div>';
  } catch(e) { document.getElementById('pr-role').textContent='Error'; }
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)