"""
Streamlit Frontend — AI Business Assistant
==========================================
Single unified chat. Intent is auto-detected by the backend.
"""

import json
import uuid
import requests
import streamlit as st
from crm_logger import suppress_noisy_libs
suppress_noisy_libs()

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title="AI Business Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
.main .block-container { padding-top: 0 !important; }
.agent-badge {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 1px 8px;
    border-radius: 999px;
    margin-bottom: 4px;
}
.badge-hubspot     { background: #E85D3022; color: #E85D30; border: 1px solid #E85D3055; }
.badge-zoho_people { background: #1D9E7522; color: #1D9E75; border: 1px solid #1D9E7555; }
.badge-cross       { background: #534AB722; color: #534AB7; border: 1px solid #534AB755; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "session_id" : str(uuid.uuid4()),
    "active_tab" : "chat",
    "prefill"    : "",
    "messages"   : [],   # single unified list — each item has role/content/agent
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Quick prompts (all agents combined) ────────────────────────────────────
QUICK_PROMPTS = [
    # HubSpot
    ("🟠 Pipeline overview",   "Summarise my HubSpot deal pipeline by stage"),
    ("🟠 Open deals",          "Show all open deals sorted by close date"),
    ("🟠 Unresolved tickets",  "Show all unresolved support tickets"),
    ("🟠 Tasks due today",     "Show all tasks due today"),
    # Zoho People
    ("🟢 All employees",       "List all active employees with their department"),
    ("🟢 On leave today",      "Who is on leave today?"),
    ("🟢 Attendance",          "Show attendance records for this week"),
    ("🟢 Pending leave",       "Show all pending leave requests"),
    # Cross
    ("⚡ Sales reps on leave", "Which sales reps are on leave today?"),
    ("⚡ Ownership gaps",      "Show open deals where the owner is currently absent"),
    ("⚡ Absent + tickets",    "Show support tickets assigned to employees on leave"),
    ("⚡ Team availability",   "Which account managers are available this week?"),
]

_AGENT_META = {
    "hubspot"     : {"icon": "🟠", "label": "HubSpot CRM",        "badge": "badge-hubspot"},
    "zoho_people" : {"icon": "🟢", "label": "Zoho People HRMS",   "badge": "badge-zoho_people"},
    "cross"       : {"icon": "⚡", "label": "Cross-System",        "badge": "badge-cross"},
}

# ── API helpers ─────────────────────────────────────────────────────────────

def api(path: str, method: str = "GET", **kwargs) -> dict:
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=10, **kwargs)
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def api_stream_chat(message: str, history: list):
    """Yields (chunk_text | None, detected_agent | None) tuples."""
    session_id = st.session_state.get("session_id", "default")
    detected_agent = None
    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json    = {"message": message, "history": history,
                       "agent": "auto", "session_id": session_id},
            stream  = True,
            timeout = 120,
            headers = {"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code == 429:
                yield None, None
                yield "\n\n⏳ **Rate limit reached** — please wait a moment.", None
                return
            if resp.status_code >= 400:
                yield None, None
                yield f"\n\n❌ **Backend error {resp.status_code}** — please try again.", None
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
                                t    = data.get("type")
                                if t == "agent":
                                    detected_agent = data.get("agent")
                                    yield None, detected_agent      # signal which agent
                                elif t == "chunk":
                                    yield data.get("text", ""), None
                                elif t == "error":
                                    yield _fmt_error(data.get("error",""), data.get("detail",""), detected_agent or "auto"), None
                                    return
                                elif t == "done":
                                    return
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}", None


def _fmt_error(error: str, detail: str, agent: str) -> str:
    msgs = {
        "mcp_url_missing"            : "🔗 MCP URL not set — add it in the sidebar.",
        "preflight_failed"           : f"🌐 Cannot reach MCP server. Detail: `{detail}`",
        "hubspot_unavailable"        : "🔑 HubSpot not connected — click **Connect HubSpot**.",
        "zoho_people_mcp_url_missing": "🔗 Zoho People MCP URL not set — add it in the sidebar.",
        "no_clients"                 : "⚠️ No system connected. Connect from the sidebar first.",
        "content_filter"             : "🛡️ Content safety filter triggered. Please rephrase.",
        "rate_limit"                 : "⏳ Too many requests — please wait a moment.",
        "auth_error"                 : "🔑 AI service auth failed. Check your API key.",
    }
    return f"\n\n{msgs.get(error, f'❌ {detail or error}')}"


@st.cache_data(ttl=10)
def get_status() -> dict:
    return api("/api/status")


def _handle_oauth_callback():
    p = st.query_params
    if ok  := p.get("oauth_ok",    ""):
        st.session_state["_ok"]  = ok
        st.query_params.clear()
    elif err := p.get("oauth_error", ""):
        st.session_state["_err"] = err
        st.query_params.clear()


# ── Sidebar ─────────────────────────────────────────────────────────────────

def _sidebar(status: dict):
    with st.sidebar:
        st.markdown("## 🤖 AI Assistant")
        st.caption("Auto-routes to HubSpot · Zoho People · Cross-System")
        st.divider()

        # Connection legend
        hs_ok = status.get("hubspot",     {}).get("connected", False)
        zp_ok = status.get("zoho_people", {}).get("connected", False)
        st.markdown("**Connected Systems**")
        st.markdown(
            f"{'✅' if hs_ok else '○'} HubSpot CRM  &nbsp;&nbsp; "
            f"{'✅' if zp_ok else '○'} Zoho People"
        )
        st.divider()

        # ── HubSpot ──────────────────────────────────────────────────────────
        st.markdown("**🟠 HubSpot CRM**")
        if hs_ok:
            st.success(f"✅ {status['hubspot'].get('message','Connected')}")
            if st.button("🔌 Disconnect HubSpot", use_container_width=True):
                api("/api/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Not connected")
            st.link_button("🔗 Connect HubSpot", f"{BACKEND}/oauth/connect",
                           use_container_width=True)

        st.divider()

        # ── Zoho People ───────────────────────────────────────────────────────────
        st.markdown("**🟢 Zoho People HRMS**")
        zp_data = status.get("zoho_people", {})
        if zp_ok:
            st.success("✅ MCP Connected")
            api_ok  = zp_data.get("api_connected", False)
            api_msg = zp_data.get("api_message", "")
            if api_ok:
                st.success(f"✅ {api_msg}")
            else:
                st.link_button(
                    "🔗 Connect Zoho People API",
                    f"{BACKEND}/oauth/zoho-people/connect",
                    use_container_width=True,
                )
                st.caption("Enables listing employees, departments & more")

            if st.button("🔌 Disconnect Zoho People", use_container_width=True):
                api("/zoho-people/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Not connected")
            api_msg = zp_data.get("api_message", "")
            if api_msg and "not configured" not in api_msg.lower():
                st.link_button(
                    "🔗 Connect Zoho People API",
                    f"{BACKEND}/oauth/zoho-people/connect",
                    use_container_width=True,
                )
            
            st.caption("Get your MCP URL from [Zoho People MCP](https://mcp.zoho.com)")
            mcp_inp = st.text_input(
                "Zoho People MCP URL",
                label_visibility = "collapsed",
                key              = "zp_mcp_url",
                placeholder      = "https://mcp.zoho.com/people/YOUR_ID/sse",
            )
            if st.button("💾 Save & Connect", key="zp_save", use_container_width=True):
                if mcp_inp.strip():
                    with st.spinner("Testing connection…"):
                        r = api("/zoho-people/save-mcp-url", method="POST",
                                json={"url": mcp_inp.strip()})
                    if r.get("reachable"):
                        st.success("✅ Connected!")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.warning(f"⚠️ Saved but unreachable: {r.get('detail','')}")
                else:
                    st.error("Paste your Zoho People MCP URL first.")

        st.divider()

        # ── Navigation ────────────────────────────────────────────────────────
        st.markdown("**Navigation**")
        for key, label in [("chat","💬 Chat"),("permissions","🔑 Permissions"),("debug","🛠️ Debug")]:
            t = "primary" if st.session_state.active_tab == key else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=t):
                st.session_state.active_tab = key
                st.rerun()

        st.divider()

        # ── Quick prompts ─────────────────────────────────────────────────────
        st.markdown("**Quick Actions**")
        for label, prompt in QUICK_PROMPTS:
            if st.button(label, key=f"qp_{label}", use_container_width=True):
                st.session_state.prefill    = prompt
                st.session_state.active_tab = "chat"
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.prefill  = ""
            st.rerun()

        st.caption("Backend :8000  ·  Frontend :8501")


# ── Header ───────────────────────────────────────────────────────────────────

def _header(status: dict):
    hs_ok = status.get("hubspot",     {}).get("connected", False)
    zp_ok = status.get("zoho_people", {}).get("connected", False)

    c1, c2 = st.columns([5, 3])
    with c1:
        st.markdown("### 🤖 AI Business Assistant")
        st.caption("LangGraph · MCP · RAG · Intent auto-detected per message")
    with c2:
        status_parts = [
            "🟠 HubSpot ✅" if hs_ok else "🟠 HubSpot ○",
            "🟢 Zoho ✅"    if zp_ok else "🟢 Zoho ○",
        ]
        st.markdown("  ·  ".join(status_parts))
        st.caption("Auto-routes: 🟠 HubSpot · 🟢 Zoho People · ⚡ Cross")
    st.divider()


# ── Chat tab ─────────────────────────────────────────────────────────────────

def _chat():
    msgs = st.session_state.messages

    if not msgs and not st.session_state.get("prefill"):
        st.markdown("#### 🤖 AI Business Assistant — Ready")
        st.info(
            "Just type naturally — I'll automatically figure out whether to query "
            "**🟠 HubSpot CRM**, **🟢 Zoho People HRMS**, or **⚡ both** at once."
        )
        st.markdown("**Try a quick action:**")
        cols = st.columns(4)
        for i, (label, prompt) in enumerate(QUICK_PROMPTS[:8]):
            with cols[i % 4]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
        st.divider()

    for m in msgs:
        agent_key = m.get("agent", "hubspot")
        meta      = _AGENT_META.get(agent_key, _AGENT_META["hubspot"])

        if m["role"] == "assistant":
            with st.chat_message("assistant", avatar=meta["icon"]):
                # Small badge showing which agent answered
                st.markdown(
                    f'<span class="agent-badge {meta["badge"]}">'
                    f'{meta["icon"]} {meta["label"]}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(m["content"])
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(m["content"])

    prefill = st.session_state.get("prefill", "")
    if prefill:
        st.session_state.prefill = ""

    user_input = st.chat_input("Ask anything about your CRM or HR data…")
    if not user_input and prefill:
        user_input = prefill

    if user_input:
        _send(user_input)


def _send(text: str):
    text = text.strip()
    if not text:
        return
    if len(text) > 4000:
        st.warning("⚠️ Message too long — max 4,000 characters.")
        return

    msgs = st.session_state.messages
    msgs.append({"role": "user", "content": text, "agent": None})
    st.session_state.messages = msgs

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)

    history = [{"role": m["role"], "content": m["content"]} for m in msgs[:-1]]

    import time as _t
    t0             = _t.time()
    detected_agent = "hubspot"   # will be updated from stream
    full_text      = ""

    with st.chat_message("assistant", avatar="🤖"):
        badge_slot = st.empty()
        text_slot  = st.empty()
        stream_buf = ""

        for chunk, agent_signal in api_stream_chat(text, history):
            if agent_signal:
                detected_agent = agent_signal
                meta = _AGENT_META.get(detected_agent, _AGENT_META["hubspot"])
                badge_slot.markdown(
                    f'<span class="agent-badge {meta["badge"]}">'
                    f'{meta["icon"]} {meta["label"]}</span>',
                    unsafe_allow_html=True,
                )
            if chunk:
                stream_buf += chunk
                text_slot.markdown(stream_buf + "▌")

        full_text = stream_buf
        text_slot.markdown(full_text or "⚠️ No response received.")

    elapsed = _t.time() - t0

    # Update the user message agent tag and save assistant reply
    msgs[-1]["agent"] = detected_agent
    msgs.append({"role": "assistant", "content": full_text or "⚠️ No response received.",
                 "agent": detected_agent})
    st.session_state.messages = msgs


# ── Permissions tab ──────────────────────────────────────────────────────────

def _permissions(status: dict):
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
        st.markdown("**Connection Status**")
        ok  = status.get("hubspot", {}).get("connected", False)
        msg = status.get("hubspot", {}).get("message", "")
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {msg}")
    st.divider()
    scopes = data.get("scopes", [])
    st.markdown(f"**Granted Scopes ({len(scopes)})**")
    cats = {
        "CRM Objects": lambda s: s.startswith("crm.objects."),
        "CRM Schemas": lambda s: s.startswith("crm.schemas."),
        "CMS"        : lambda s: s.startswith("cms."),
        "Settings"   : lambda s: s.startswith("settings.") or s == "mcp.users.read",
        "Analytics"  : lambda s: s in ("crm.hubsql.execute",) or s.startswith("analytics."),
        "Other"      : lambda s: True,
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

    crm_choice = st.radio("Test MCP for:", ["HubSpot", "Zoho People"], horizontal=True)
    crm_param  = "zoho_people" if crm_choice == "Zoho People" else "hubspot"

    if st.button("🔄 Test MCP Connection"):
        with st.spinner("Testing…"):
            result = api(f"/api/debug-mcp?crm={crm_param}")
        status = result.get("status", result.get("_error", "unknown"))
        (st.success if "ok" in str(status).lower() else st.error)(str(status))
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Connection Info**")
            for label, key in [("Transport","transport"),("Tool count","tool_count")]:
                st.markdown(f"- **{label}**: `{result.get(key,'—')}`")
        with col2:
            tools = result.get("tools", [])
            if tools:
                st.markdown(f"**Sample Tools ({len(tools)})**")
                for t in tools:
                    st.code(t, language=None)

    st.divider()
    st.markdown("**📊 Cache Stats**")
    if st.button("Refresh Stats"):
        stats = api("/api/cache-stats")
        st.json(stats)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _handle_oauth_callback()

    if crm := st.session_state.pop("_ok",  ""):
        st.toast(f"✅ {'HubSpot' if crm == 'hubspot' else 'Zoho People'} connected!", icon="🎉")
    if err := st.session_state.pop("_err", ""):
        st.toast(f"❌ OAuth error: {err}", icon="⚠️")

    status = get_status()

    if "_error" in status:
        st.error(
            f"⚠️ Backend unreachable — is uvicorn running on :8000?\n\n"
            f"`{status['_error']}`\n\nRun: `uvicorn web_app:app --port 8000`"
        )

    _sidebar(status)
    _header(status)

    tab = st.session_state.active_tab
    if tab == "chat":
        _chat()
    elif tab == "permissions":
        _permissions(status)
    elif tab == "debug":
        _debug()


if __name__ == "__main__":
    main()