"""
Streamlit Frontend — AI Business Assistant
==========================================
Tabs: HubSpot CRM · Zoho People HRMS · Cross-System
"""

import json
import uuid
import requests
import streamlit as st
from crm_logger import log, suppress_noisy_libs
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
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "session_id"       : str(uuid.uuid4()),
    "active_agent"     : "hubspot",
    "active_tab"       : "chat",
    "prefill"          : "",
    "messages_hubspot" : [],
    "messages_zoho_people": [],
    "messages_cross"   : [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Agent configs ──────────────────────────────────────────────────────────
AGENTS = {
    "hubspot": {
        "label"      : "HubSpot CRM",
        "icon"       : "🟠",
        "placeholder": "Ask anything about your HubSpot CRM…",
        "color"      : "#E85D30",
        "prompts"    : [
            ("📊 Pipeline overview",   "Summarise my HubSpot deal pipeline by stage"),
            ("💼 Open deals",          "Show all open deals sorted by close date"),
            ("🔍 Find a contact",      "Find contact john@acme.com"),
            ("🎫 Unresolved tickets",  "Show all unresolved support tickets"),
            ("🏢 Top companies",       "List top 10 companies by deal value"),
            ("✅ Closed won",          "List all deals closed won this quarter"),
            ("📋 Tasks due today",     "Show all tasks due today"),
            ("📧 Recent emails",       "Show emails sent in the last 7 days"),
        ],
    },
    "zoho_people": {
        "label"      : "Zoho People HRMS",
        "icon"       : "🟢",
        "placeholder": "Ask anything about HR, employees, or attendance…",
        "color"      : "#1D9E75",
        "prompts"    : [
            ("👥 All employees",       "List all active employees with their department"),
            ("🏖️  On leave today",    "Who is on leave today?"),
            ("📅 Attendance",          "Show attendance records for this week"),
            ("🏢 Departments",         "List all departments with headcount"),
            ("📝 Pending leave",       "Show all pending leave requests"),
            ("🕐 Late arrivals",       "Who checked in late today?"),
            ("👤 Employee profile",    "Show profile for Rahul Mehta"),
            ("📊 Leave summary",       "Give me a leave summary for this month"),
        ],
    },
    "cross": {
        "label"      : "Cross-System",
        "icon"       : "⚡",
        "placeholder": "Query HubSpot + Zoho People together…",
        "color"      : "#534AB7",
        "prompts"    : [
            ("🔗 Sales reps on leave", "Which sales reps are on leave today?"),
            ("🎫 Absent + tickets",    "Show support tickets assigned to employees on leave"),
            ("📊 Deals vs headcount",  "Compare active deals per department with headcount"),
            ("👤 Contact → employee",  "Show the employee linked to client ABC Technologies"),
            ("📅 Team availability",   "Which account managers are available this week?"),
            ("🔄 Ownership gaps",      "Show open deals where the owner is currently absent"),
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
    session_id = st.session_state.get("session_id", "default")
    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json    = {"message": message, "history": history,
                       "agent": agent, "session_id": session_id},
            stream  = True,
            timeout = 120,
            headers = {"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code == 429:
                yield "\n\n⏳ **Rate limit reached** — please wait a moment."
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
                                t    = data.get("type")
                                if t == "chunk":
                                    yield data.get("text", "")
                                elif t == "error":
                                    yield _fmt_error(data.get("error",""), data.get("detail",""), agent)
                                    return
                                elif t == "done":
                                    return
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}"


def _fmt_error(error: str, detail: str, agent: str) -> str:
    msgs = {
        "mcp_url_missing"        : "🔗 MCP URL not set — add it in the sidebar.",
        "preflight_failed"       : f"🌐 Cannot reach MCP server. Detail: `{detail}`",
        "hubspot_unavailable"    : "🔑 HubSpot not connected — click **Connect HubSpot**.",
        "zoho_people_mcp_url_missing": "🔗 Zoho People MCP URL not set — add it in the sidebar.",
        "no_clients"             : "⚠️ No system connected. Connect from the sidebar first.",
        "content_filter"         : "🛡️ Content safety filter triggered. Please rephrase.",
        "rate_limit"             : "⏳ Too many requests — please wait a moment.",
        "auth_error"             : "🔑 AI service auth failed. Check your API key.",
    }
    return f"\n\n{msgs.get(error, f'❌ {detail or error}')}"


@st.cache_data(ttl=10)
def get_status() -> dict:
    return api("/api/status")


def _handle_oauth_callback():
    p = st.query_params
    if ok  := p.get("oauth_ok", ""):
        st.session_state["_ok"]  = ok
        st.query_params.clear()
    elif err := p.get("oauth_error", ""):
        st.session_state["_err"] = err
        st.query_params.clear()


def _msg_key(agent: str) -> str:
    return f"messages_{agent}"

def _get_msgs() -> list:
    return st.session_state.get(_msg_key(st.session_state.active_agent), [])

def _set_msgs(msgs: list):
    st.session_state[_msg_key(st.session_state.active_agent)] = msgs


# ── Sidebar ─────────────────────────────────────────────────────────────────

def _sidebar(status: dict):
    with st.sidebar:
        st.markdown("## 🤖 AI Assistant")
        st.caption("HubSpot · Zoho People · Cross-System")
        st.divider()

        # Agent selector
        st.markdown("**Active Agent**")
        agent_cols = st.columns(3)
        for col, (ag, cfg) in zip(agent_cols, AGENTS.items()):
            with col:
                active = st.session_state.active_agent == ag
                if st.button(
                    f"{cfg['icon']}",
                    key              = f"ag_{ag}",
                    use_container_width = True,
                    type             = "primary" if active else "secondary",
                    help             = cfg["label"],
                ):
                    if ag != st.session_state.active_agent:
                        st.session_state.active_agent = ag
                        st.rerun()

        agent_label = AGENTS[st.session_state.active_agent]["label"]
        st.caption(f"Active: **{agent_label}**")
        st.divider()

        # ── HubSpot ──────────────────────────────────────────────────────────
        hs = status.get("hubspot", {})
        st.markdown("**🟠 HubSpot CRM**")
        if hs.get("connected"):
            st.success(f"✅ {hs.get('message','Connected')}")
            if st.button("🔌 Disconnect HubSpot", use_container_width=True):
                api("/api/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Not connected")
            st.link_button("🔗 Connect HubSpot", f"{BACKEND}/oauth/connect",
                           use_container_width=True)

        st.divider()

        # ── Zoho People ───────────────────────────────────────────────────────
        zp = status.get("zoho_people", {})
        st.markdown("**🟢 Zoho People HRMS**")
        if zp.get("connected"):
            st.success("✅ Zoho People Connected")
            if st.button("🔌 Disconnect Zoho People", use_container_width=True):
                api("/zoho-people/disconnect", method="POST")
                st.cache_data.clear()
                st.rerun()
        else:
            st.warning("Not connected")
            st.caption("Get your MCP URL from [Zoho People MCP](https://mcp.zoho.com) → copy your URL")
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
        cfg = AGENTS[st.session_state.active_agent]
        st.markdown(f"**Quick · {cfg['icon']} {cfg['label']}**")
        for label, prompt in cfg["prompts"]:
            if st.button(label, key=f"qp_{label}", use_container_width=True):
                st.session_state.prefill    = prompt
                st.session_state.active_tab = "chat"
                st.rerun()

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            _set_msgs([])
            st.session_state.prefill = ""
            st.rerun()

        st.caption("Backend :8000  ·  Frontend :8501")


# ── Header ───────────────────────────────────────────────────────────────────

def _header(status: dict):
    cfg   = AGENTS[st.session_state.active_agent]
    hs_ok = status.get("hubspot",      {}).get("connected", False)
    zp_ok = status.get("zoho_people",  {}).get("connected", False)

    c1, c2 = st.columns([5, 3])
    with c1:
        st.markdown(f"### {cfg['icon']} AI Business Assistant")
        st.caption(f"LangGraph · MCP · RAG · Mode: **{cfg['label']}**")
    with c2:
        status_parts = [
            "🟠 HubSpot ✅" if hs_ok else "🟠 HubSpot ○",
            "🟢 Zoho People ✅" if zp_ok else "🟢 Zoho People ○",
        ]
        st.markdown("  ·  ".join(status_parts))
        st.caption(f"{cfg['icon']} {cfg['label']} active")
    st.divider()


# ── Chat tab ─────────────────────────────────────────────────────────────────

def _chat():
    agent = st.session_state.active_agent
    cfg   = AGENTS[agent]
    msgs  = _get_msgs()

    if not msgs:
        st.markdown(f"#### {cfg['icon']} {cfg['label']} — Ready")
        if agent == "cross":
            st.info("⚡ Cross-system mode queries both **HubSpot CRM** and **Zoho People HRMS** simultaneously.")
        st.markdown("**Try a quick action:**")
        cols = st.columns(4)
        for i, (label, prompt) in enumerate(cfg["prompts"][:8]):
            with cols[i % 4]:
                if st.button(label, key=f"chip_{agent}_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
        st.divider()

    for m in msgs:
        avatar = cfg["icon"] if m["role"] == "assistant" else "👤"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    prefill = st.session_state.get("prefill", "")
    if prefill:
        st.session_state.prefill = ""

    user_input = st.chat_input(cfg["placeholder"])
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

    agent = st.session_state.active_agent
    cfg   = AGENTS[agent]
    msgs  = _get_msgs()
    msgs.append({"role": "user", "content": text})
    _set_msgs(msgs)

    with st.chat_message("user", avatar="👤"):
        st.markdown(text)

    history = [{"role": m["role"], "content": m["content"]} for m in msgs[:-1]]

    import time as _t
    t0 = _t.time()
    with st.chat_message("assistant", avatar=cfg["icon"]):
        full = st.write_stream(api_stream_chat(text, history, agent))
    elapsed = _t.time() - t0

    msgs.append({"role": "assistant", "content": full or "⚠️ No response received."})
    _set_msgs(msgs)


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

    # RAG / cache stats
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
