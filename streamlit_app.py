"""
Multi-CRM AI Agent — Streamlit Frontend (Improved)
====================================================
Clean UI using Streamlit-native components:
  - st.chat_message + st.chat_input  → sticky input, proper bubbles, fast
  - No 200-line CSS blob              → plain & fast rendering
  - Status cached 10s                → responsive to connect/disconnect changes
  - st.write_stream                  → true live streaming with no placeholder hacks

Run both servers:
    uvicorn web_app:app --port 8000
    streamlit run streamlit_app.py
"""

import json
import uuid
import requests
import streamlit as st
from crm_logger import log, suppress_noisy_libs
suppress_noisy_libs()

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="Multi-CRM AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS — only what Streamlit cannot do natively
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding-top: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":     [],
    "active_agent": "hubspot",
    "active_tab":   "chat",
    "prefill":      "",
    "session_id":   str(uuid.uuid4()),   # B5: unique per-browser-tab session
    # B10: per-agent message history
    "messages_hubspot":  [],
    "messages_zoho_crm": [],
    "messages_both":     [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Agent configs ──────────────────────────────────────────────────────────
# B12: unified "prompts" list — used for both empty-state chips and sidebar quick commands
AGENTS = {
    "hubspot": {
        "label": "HubSpot", "icon": "🟠",
        "placeholder": "Ask anything about your HubSpot CRM…",
        "prompts": [
            ("📊 Pipeline overview",  "Summarise my HubSpot deal pipeline by stage"),
            ("💼 Open deals",         "Show all open deals sorted by amount"),
            ("🔍 Find a contact",     "Find contact john@acme.com"),
            ("➕ Create a deal",      "Create a new deal for Acme Corp worth $50,000"),
            ("📋 Recent activity",    "Show CRM activities from the last 7 days"),
            ("✅ Closed won",         "List all deals closed won this quarter"),
        ],
    },
    "zoho_crm": {
        "label": "Zoho CRM", "icon": "🔵",
        "placeholder": "Ask anything about your Zoho CRM…",
        "prompts": [
            ("🏆 All leads",     "Show me all leads in Zoho CRM"),
            ("👤 All contacts",  "List all contacts with email and phone"),
            ("💰 Open deals",    "Show all deals in the pipeline"),
            ("🏢 All accounts",  "Show me all accounts"),
            ("🔄 Convert lead",  "Help me convert a lead to a contact"),
            ("📊 All modules",   "List all available Zoho CRM modules"),
        ],
    },
    "both": {
        "label": "Both CRMs", "icon": "⚡",
        "placeholder": "Query both HubSpot and Zoho CRM simultaneously…",
        "prompts": [
            ("💰 All deals (both)",    "Show me all open deals from both CRMs"),
            ("👤 All contacts (both)", "Show all contacts from both CRMs"),
            ("📊 Compare pipelines",   "Compare pipelines from HubSpot and Zoho"),
            ("🏆 All leads (both)",    "Show all leads from both CRMs"),
        ],
    },
}

# ── API helpers ─────────────────────────────────────────────────────────────

def api(path: str, method: str = "GET", **kwargs) -> dict:
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=10, **kwargs)
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def api_stream_chat(message: str, history: list, agent: str):
    """Yield text chunks from SSE stream — compatible with st.write_stream."""
    session_id = st.session_state.get("session_id", "default")   # B5
    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json={"message": message, "history": history, "agent": agent, "session_id": session_id},
            stream=True, timeout=120,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            # Handle non-streaming error responses (429, 400, 500)
            if resp.status_code == 429:
                yield "\n\n⏳ **Rate limit reached** — you're sending messages too quickly. Please wait a moment and try again."
                return
            if resp.status_code >= 400:
                yield f"\n\n❌ **Backend error {resp.status_code}** — please try again."
                return
            buffer = ""
            for raw in resp.iter_content(chunk_size=None):
                buffer += raw.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for line in block.splitlines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                t = data.get("type")
                                if t == "chunk":
                                    yield data.get("text", "")
                                elif t == "error":
                                    err    = data.get("error", "")
                                    detail = data.get("detail", "")
                                    log("error", f"stream error from backend: {err} — {detail}", source="frontend")
                                    yield _fmt_error(err, detail, agent)
                                    return
                                elif t == "done":
                                    return
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}"


def _fmt_error(error: str, detail: str, agent: str) -> str:
    msgs = {
        "mcp_url_missing":    "🔗 Zoho MCP URL not set — add it in the sidebar.",
        "oauth_missing":      "🔑 Zoho OAuth not connected — click **① Connect Zoho OAuth**.",
        "preflight_failed":   f"🌐 Cannot reach MCP server. Detail: `{detail}`",
        "hubspot_unavailable":"🔑 HubSpot not connected — click **Connect HubSpot**.",
        "no_clients":         "⚠️ No CRM connected. Connect at least one from the sidebar.",
        "content_filter":     "🛡️ Your request was flagged by content safety filters. Please try rephrasing your message.",
        "rate_limit":         "⏳ Too many requests — please wait a moment and try again.",
        "auth_error":         "🔑 AI service authentication failed. Please check your API key configuration.",
    }
    return f"\n\n{msgs.get(error, f'❌ {detail or error}')}"


@st.cache_data(ttl=10)   # B18: reduced from 30s to 10s
def get_status() -> dict:
    """Cache status 10s — responsive to connect/disconnect changes."""
    return api("/api/status")

# ── OAuth callback ──────────────────────────────────────────────────────────

def _handle_oauth_callback():
    p = st.query_params
    if ok := p.get("oauth_ok", ""):
        st.session_state["_ok"] = ok
        st.query_params.clear()
    elif err := p.get("oauth_error", ""):
        st.session_state["_err"] = err
        st.query_params.clear()


# ── Agent history helpers (B10) ──────────────────────────────────────────────

def _get_messages_key(agent: str) -> str:
    return f"messages_{agent}"


def _get_messages() -> list:
    """Return the message list for the active agent."""
    key = _get_messages_key(st.session_state.active_agent)
    return st.session_state.get(key, [])


def _set_messages(msgs: list):
    """Set the message list for the active agent."""
    key = _get_messages_key(st.session_state.active_agent)
    st.session_state[key] = msgs


# ── Sidebar ─────────────────────────────────────────────────────────────────

def _sidebar(status: dict):
    with st.sidebar:
        st.markdown("## 🤖 CRM Agent")
        st.caption("FastAPI · Streamlit · MCP")
        st.divider()

        st.markdown("**Active CRM**")
        cols = st.columns(3)
        for col, (ag, lbl) in zip(cols, [("hubspot","🟠 HS"),("zoho_crm","🔵 Zo"),("both","⚡")]):
            with col:
                active = st.session_state.active_agent == ag
                if st.button(lbl, key=f"ag_{ag}", use_container_width=True,
                              type="primary" if active else "secondary"):
                    if ag != st.session_state.active_agent:
                        # B10: switch agent — preserve current history, load target history
                        log("info", f"agent switched → {ag}", source="frontend")
                        st.session_state.active_agent = ag
                        st.rerun()

        st.divider()

        # HubSpot
        hs_data = status.get("hubspot", {})
        hs_ok   = hs_data.get("connected", False)
        st.markdown("**HubSpot**")
        if hs_ok:
            st.success(f"✅ {hs_data.get('message','Connected')}")
            if st.button("🔌 Disconnect HubSpot", use_container_width=True):
                api("/api/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Not connected")
            st.link_button("🔗 Connect HubSpot", f"{BACKEND}/oauth/connect",
                           use_container_width=True)

        st.divider()

        # Zoho
        zo_data  = status.get("zoho", {})
        zo_oauth = zo_data.get("connected", False)
        zo_mcp   = zo_data.get("mcp_ready", False)
        zo_ok    = zo_oauth and zo_mcp

        st.markdown("**Zoho CRM**")
        if zo_ok:
            st.success("✅ Zoho Connected")
            if st.button("🔌 Disconnect Zoho", use_container_width=True):
                api("/zoho/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.info("✅ OAuth done" if zo_oauth else "① OAuth needed")
            st.link_button("① Connect Zoho OAuth", f"{BACKEND}/zoho/connect",
                           use_container_width=True)
            if not zo_mcp:
                with st.expander("② Set Zoho MCP URL"):
                    st.caption("From [mcp.zoho.in](https://mcp.zoho.in) → Connect → Copy URL")
                    mcp_inp = st.text_input(
                        "MCP URL", label_visibility="collapsed", key="mcp_url_input",
                        placeholder="https://crm-data-metadata-XXXXX.zohomcp.in/mcp/APIKEY/message",
                    )
                    if st.button("💾 Save & Test URL", use_container_width=True):
                        if mcp_inp.strip():
                            with st.spinner("Testing…"):
                                r = api("/zoho/save-mcp-url", method="POST",
                                        json={"url": mcp_inp.strip()})
                            if r.get("reachable"):
                                st.success("✅ Connected!")
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.warning("⚠️ Saved but: " + r.get("detail",""))
                        else:
                            st.error("Paste the MCP URL first.")
            if zo_oauth:
                if st.button("🔌 Disconnect Zoho", key="zo_disc2", use_container_width=True):
                    api("/zoho/disconnect", method="POST")
                    st.cache_data.clear()
                    st.rerun()

        st.divider()

        # Navigation
        st.markdown("**Navigation**")
        for key, label in [("chat","💬 Chat"),("permissions","🔑 Permissions"),("debug","🛠️ Debug MCP")]:
            t = "primary" if st.session_state.active_tab == key else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=t):
                st.session_state.active_tab = key
                st.rerun()

        st.divider()

        # Quick commands — B12: reuse the unified "prompts" list
        cfg = AGENTS[st.session_state.active_agent]
        st.markdown(f"**Quick · {cfg['icon']} {cfg['label']}**")
        for label, prompt in cfg["prompts"]:
            if st.button(label, key=f"qcmd_{label}", use_container_width=True):
                log("info", f"quick cmd → {label}", source="frontend")
                st.session_state.prefill    = prompt
                st.session_state.active_tab = "chat"
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            _set_messages([])
            st.session_state.prefill  = ""
            st.rerun()

        st.caption("Backend :8000 · Frontend :8501")

# ── Sticky header (above chat) ───────────────────────────────────────────────

def _header(status: dict):
    cfg   = AGENTS[st.session_state.active_agent]
    hs_ok = status.get("hubspot", {}).get("connected", False)
    zo_ok = status.get("zoho", {}).get("connected", False) and \
            status.get("zoho", {}).get("mcp_ready", False)

    c1, c2 = st.columns([5, 3])
    with c1:
        st.markdown(f"### {cfg['icon']} Multi-CRM AI Agent")
        st.caption(f"LangGraph · MCP · FastAPI :8000 · Mode: **{cfg['label']}**")
    with c2:
        parts = []
        parts.append("🟠 HubSpot ✅" if hs_ok else "🟠 HubSpot ○")
        parts.append("🔵 Zoho ✅"    if zo_ok else "🔵 Zoho ○")
        st.markdown("  ·  ".join(parts))
        st.caption(f"{cfg['icon']} {cfg['label']} Mode active")
    st.divider()

# ── Chat tab ─────────────────────────────────────────────────────────────────

def _chat():
    agent = st.session_state.active_agent
    cfg   = AGENTS[agent]
    msgs  = _get_messages()   # B10: per-agent history

    # Empty state — quick action chips (B12: reuse unified prompts)
    if not msgs:
        st.markdown(f"#### {cfg['icon']} {cfg['label']} Agent ready")
        st.markdown(
            f"Your AI assistant for **{cfg['label']}**. "
            "Ask about deals, contacts, companies, leads, and pipelines."
        )
        st.markdown("**Try a quick action:**")
        chip_cols = st.columns(3)
        for i, (label, prompt) in enumerate(cfg["prompts"]):
            with chip_cols[i % 3]:
                # B17: scope chip keys to active agent
                if st.button(label, key=f"chip_{agent}_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
        st.divider()

    # Render history
    for m in msgs:
        avatar = cfg["icon"] if m["role"] == "assistant" else "👤"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    # Consume prefill (set by sidebar quick commands or chips)
    prefill = st.session_state.get("prefill", "")
    if prefill:
        st.session_state.prefill = ""

    # st.chat_input is automatically sticky at the bottom of the page
    user_input = st.chat_input(cfg["placeholder"])
    if not user_input and prefill:
        user_input = prefill

    if user_input:
        _send(user_input)


def _send(text: str):
    # Guard: strip and enforce max length before hitting backend
    text = text.strip()
    if not text:
        return
    if len(text) > 4000:
        st.warning("⚠️ Message too long — please keep it under 4,000 characters.")
        return

    agent = st.session_state.active_agent
    cfg   = AGENTS[agent]

    log("user", f"[{agent}] '{text[:80]}'", source="frontend")
    msgs = _get_messages()
    msgs.append({"role": "user", "content": text, "agent": agent})
    _set_messages(msgs)

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in msgs[:-1]
    ]

    import time as _time
    _t0 = _time.time()
    with st.chat_message("assistant", avatar=cfg["icon"]):
        # st.write_stream handles live token-by-token display natively
        full = st.write_stream(api_stream_chat(text, history, agent))
    _elapsed = _time.time() - _t0
    log("ai", f"[{agent}] response → {len(full or '')} chars in {_elapsed:.1f}s", source="frontend")

    msgs.append({
        "role": "assistant",
        "content": full or "⚠️ No response received.",
        "agent": agent,
    })
    _set_messages(msgs)

# ── Permissions tab ──────────────────────────────────────────────────────────

def _permissions(status: dict):    # B14: accept cached status dict
    st.markdown("### 🔑 HubSpot Permissions")
    st.caption("Live data from `/api/permissions`")

    with st.spinner("Loading…"):
        data = api("/api/permissions")

    if "_error" in data or not data.get("connected"):
        st.warning("⚠️ Not connected to HubSpot. Connect from the sidebar.")
        return

    st.success("🔑 Super Admin" if data.get("is_admin") else "👤 Standard User")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tool Access**")
        for t in data.get("tools", []):
            st.markdown(f"{'✅' if t['accessible'] else '🚫'} `{t['tool']}`")
    with col2:
        # B14: use the cached status dict instead of making a new HTTP call
        st.markdown("**Connection Status**")
        ok = status.get("hubspot", {}).get("connected", False)
        msg = status.get("hubspot", {}).get("message", "")
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")

    st.divider()
    scopes = data.get("scopes", [])
    st.markdown(f"**Granted Scopes ({len(scopes)})**")
    cats = {
        "CRM Objects": lambda s: s.startswith("crm.objects."),
        "CRM Schemas": lambda s: s.startswith("crm.schemas."),
        "CMS":         lambda s: s.startswith("cms."),
        "Settings":    lambda s: s.startswith("settings.") or s == "mcp.users.read",
        "Analytics":   lambda s: s == "crm.hubsql.execute" or s.startswith("analytics."),
        "Other":       lambda s: True,
    }
    used = set()
    for cat, fn in cats.items():
        grp = [s for s in scopes if s["scope"] not in used and fn(s["scope"])]
        if not grp:
            continue
        for s in grp:
            used.add(s["scope"])
        with st.expander(f"📁 {cat} ({len(grp)})", expanded=(cat == "CRM Objects")):
            for s in grp:
                st.markdown(f"- **{s.get('label', s['scope'])}** — `{s['scope']}`")

# ── Debug tab ────────────────────────────────────────────────────────────────

def _debug():
    import time
    st.markdown("### 🛠️ Debug MCP Connection")
    st.caption("Verify backend reachability and MCP tool connectivity")

    try:
        t0 = time.time()
        r  = requests.get(f"{BACKEND}/api/status", timeout=5)
        ms = int((time.time() - t0) * 1000)
        st.success(f"✅ FastAPI reachable — {ms}ms — HTTP {r.status_code}")
    except Exception as e:
        st.error(f"❌ FastAPI not reachable\n\n{e}\n\nRun: `uvicorn web_app:app --port 8000`")

    crm_choice = st.radio("Test MCP for:", ["HubSpot", "Zoho CRM"], horizontal=True)
    crm_param  = "zoho" if crm_choice == "Zoho CRM" else "hubspot"

    if st.button("🔄 Test MCP Connection"):
        with st.spinner("Testing…"):
            result = api(f"/api/debug-mcp?crm={crm_param}")

        status = result.get("status", result.get("_error", "unknown"))
        (st.success if "ok" in str(status).lower() else st.error)(str(status))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Connection Info**")
            for label, key in [("Transport","transport"),("Preflight","preflight"),("Tool count","tool_count")]:
                st.markdown(f"- **{label}**: `{result.get(key,'—')}`")
        with col2:
            tools = result.get("tools", [])
            if tools:
                st.markdown(f"**Tools ({len(tools)})**")
                for t in tools:
                    st.code(t, language=None)
        if "error" in result:
            st.error(f"**Error:** {result['error']}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _handle_oauth_callback()

    if crm := st.session_state.pop("_ok", ""):
        st.toast(f"✅ {'HubSpot' if crm == 'hubspot' else 'Zoho CRM'} connected!", icon="🎉")
    if err := st.session_state.pop("_err", ""):
        st.toast(f"❌ OAuth error: {err}", icon="⚠️")

    status = get_status()

    # B16: show error banner when backend is unreachable
    if "_error" in status:
        st.error(f"⚠️ Backend unreachable — is uvicorn running on :8000?\n\n`{status['_error']}`")

    _sidebar(status)
    _header(status)

    tab = st.session_state.active_tab
    if tab == "chat":
        _chat()
    elif tab == "permissions":
        _permissions(status)    # B14: pass cached status
    elif tab == "debug":
        _debug()


if __name__ == "__main__":
    main()