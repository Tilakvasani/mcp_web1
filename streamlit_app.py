"""
Multi-CRM AI Agent — Streamlit Frontend
=========================================
Full-featured UI for HubSpot, Zoho CRM, and Both modes.
Calls the FastAPI backend (web_app.py) running on http://localhost:8000.

Run both servers:
    uvicorn web_app:app --port 8000          (terminal 1)
    streamlit run streamlit_app.py           (terminal 2)
"""

import time
import json
import requests
import streamlit as st

BACKEND = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-CRM AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:#0f1117; --bg2:#1a1d27; --bg3:#22263a;
  --border:#2e3250; --text:#e8eaf6; --text2:#8b92b8;
  --hs:#FF7A59; --zo:#4da3ff; --both:#a78bfa;
  --green:#4caf7d; --blue:#5b8dee; --red:#f87171;
}
.main .block-container { padding-top:0.5rem; padding-bottom:0; max-width:100% }
.stApp { background:var(--bg); }
section[data-testid="stSidebar"] > div:first-child { padding-top:0; }
section[data-testid="stSidebar"] { background:var(--bg2); border-right:1px solid var(--border); }
#MainMenu, footer, header { visibility:hidden; }

.hs-div { border:none; border-top:1px solid var(--border); margin:10px 0; }

/* Status badges */
.badge-ok  { background:rgba(76,175,125,.15); color:var(--green); border:1px solid rgba(76,175,125,.3);
             padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }
.badge-off { background:rgba(248,113,113,.1); color:var(--red); border:1px solid rgba(248,113,113,.25);
             padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }
.badge-hs  { background:rgba(255,122,89,.15); color:var(--hs); border:1px solid rgba(255,122,89,.3);
             padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }
.badge-zo  { background:rgba(77,163,255,.15); color:var(--zo); border:1px solid rgba(77,163,255,.3);
             padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }
.badge-both{ background:rgba(167,139,250,.15); color:var(--both); border:1px solid rgba(167,139,250,.3);
             padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }

.s-label { font-size:10px; font-weight:700; color:var(--text2); text-transform:uppercase;
           letter-spacing:.7px; margin:14px 0 6px; }

/* Chat bubbles */
.msg-row { display:flex; gap:12px; margin-bottom:16px; }
.msg-row.user { flex-direction:row-reverse; }
.av { width:36px; height:36px; border-radius:50%; flex-shrink:0;
      display:flex; align-items:center; justify-content:center; font-size:18px; }
.av-u { background:var(--blue); }
.av-hs { background:linear-gradient(135deg,var(--hs),#e05a3a); }
.av-zo { background:linear-gradient(135deg,var(--zo),#0b76ef); }
.av-both { background:linear-gradient(135deg,var(--both),var(--zo)); }
.bubble { max-width:720px; padding:12px 16px; border-radius:14px; line-height:1.65; font-size:14px; }
.bub-u { background:var(--blue); color:white; border-top-right-radius:4px; }
.bub-b { background:var(--bg2); color:var(--text); border:1px solid var(--border); border-top-left-radius:4px; }
.bub-b code { background:var(--bg3); color:#a78bfa; padding:2px 6px; border-radius:4px; font-size:12px; }
.bub-b pre  { background:var(--bg3); border:1px solid var(--border); border-radius:8px;
              padding:12px; overflow-x:auto; margin:8px 0; }

.a-ok   { background:rgba(76,175,125,.1);  border:1px solid rgba(76,175,125,.3);
          border-radius:10px; padding:12px 16px; color:var(--green); font-size:13px; margin:8px 0; }
.a-warn { background:rgba(255,154,128,.1); border:1px solid rgba(255,154,128,.3);
          border-radius:10px; padding:12px 16px; color:var(--hs);    font-size:13px; margin:8px 0; }
.a-err  { background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.25);
          border-radius:10px; padding:12px 16px; color:var(--red);   font-size:13px; margin:8px 0; }

.sc-row { display:flex; align-items:center; justify-content:space-between;
          padding:6px 0; border-bottom:1px solid var(--border); font-size:12px; }
.sc-lbl { color:var(--text); }
.sc-key { color:var(--text2); font-family:monospace; font-size:10px; }
.perm-ttl { font-size:10px; font-weight:700; color:var(--text2); text-transform:uppercase;
            letter-spacing:.6px; margin:14px 0 8px; }
.chip { display:inline-block; margin:3px; padding:4px 10px; border-radius:16px;
        font-size:11px; font-weight:500; }

/* Input */
.stTextInput > div > div > input {
  background:var(--bg3) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; color:var(--text) !important;
  font-size:14px !important; padding:12px 16px !important;
}
.stTextInput > div > div > input:focus {
  border-color:var(--blue) !important;
  box-shadow:0 0 0 2px rgba(91,141,238,.2) !important;
}
/* Buttons */
.stButton > button {
  background:var(--bg3) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; border-radius:8px !important;
  font-size:13px !important; font-weight:500 !important;
}
.stButton > button:hover { border-color:var(--blue) !important; color:var(--blue) !important; }
div[data-testid="column"]:last-child .stButton > button {
  background:var(--blue) !important; border-color:var(--blue) !important;
  color:white !important; font-weight:700 !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
  background:#4a7de0 !important; border-color:#4a7de0 !important; color:white !important;
}
/* MCP URL input in sidebar */
.stTextArea textarea {
  background:var(--bg3) !important; border:1px solid var(--border) !important;
  border-radius:8px !important; color:var(--text) !important; font-size:12px !important;
  font-family:monospace !important;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "messages":    [],
    "processing":  False,
    "active_tab":  "chat",
    "active_agent":"hubspot",  # hubspot | zoho_crm | both
    "prefill":     "",
    "mcp_url":     "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# Agent configs
# ─────────────────────────────────────────────────────────────────────────────
AGENTS = {
    "hubspot": {
        "label": "HubSpot", "icon": "🟠", "av_cls": "av-hs", "badge": "badge-hs",
        "placeholder": "Ask anything about your HubSpot CRM…",
        "chips": [
            ("📊 Pipeline overview",  "Summarise my HubSpot deal pipeline by stage"),
            ("💼 Open deals",         "Show me all open deals"),
            ("🔍 Find a contact",     "Find contact john@acme.com"),
            ("➕ Create a deal",      "Create a deal for Acme Corp"),
            ("📋 Recent activity",    "Show recent CRM activities"),
            ("✅ Closed won",         "List deals closed won this quarter"),
        ],
        "nav_cmds": [
            ("📊", "Pipeline overview",  "Summarise my HubSpot deal pipeline by stage"),
            ("🔍", "Search contacts",    "Find all contacts added this week"),
            ("💼", "Open deals",         "Show all open deals sorted by amount"),
            ("➕", "Create a deal",      "Create a new deal for Acme Corp worth $50,000"),
            ("📋", "Recent activity",    "Show CRM activities from the last 7 days"),
            ("✅", "Closed won",         "List all deals closed won this quarter"),
        ],
    },
    "zoho_crm": {
        "label": "Zoho CRM", "icon": "🔵", "av_cls": "av-zo", "badge": "badge-zo",
        "placeholder": "Ask anything about your Zoho CRM…",
        "chips": [
            ("🏆 All leads",     "Show me all leads in Zoho CRM"),
            ("👤 All contacts",  "List all contacts with email and phone"),
            ("💰 Open deals",    "Show all deals in the pipeline"),
            ("🏢 All accounts",  "Show me all accounts"),
            ("🔄 Convert lead",  "Help me convert a lead to a contact"),
            ("📊 All modules",   "List all available Zoho CRM modules"),
        ],
        "nav_cmds": [
            ("🏆", "All leads",     "Show me all leads"),
            ("👤", "Contacts",      "Show me all contacts with email"),
            ("💰", "Deals",         "Show all deals grouped by stage"),
            ("🏢", "Accounts",      "Show me all accounts"),
            ("🔄", "Convert Lead",  "Help me convert a lead"),
            ("➕", "Create Lead",   "Create a new lead"),
        ],
    },
    "both": {
        "label": "Both CRMs", "icon": "⚡", "av_cls": "av-both", "badge": "badge-both",
        "placeholder": "Query both HubSpot and Zoho CRM simultaneously…",
        "chips": [
            ("💰 All deals (both)",    "Show me all open deals from both CRMs"),
            ("👤 All contacts (both)", "Show all contacts from both CRMs"),
            ("📊 Compare pipelines",   "Compare pipelines from HubSpot and Zoho"),
            ("🏆 All leads (both)",    "Show all leads from both CRMs"),
        ],
        "nav_cmds": [
            ("💰", "All Deals",      "Show all open deals from both CRMs"),
            ("👤", "All Contacts",   "Show all contacts from both CRMs"),
            ("📊", "Compare",        "Compare pipelines from HubSpot and Zoho"),
            ("🏆", "All Leads",      "Show all leads from both CRMs"),
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api(path: str, method: str = "GET", **kwargs) -> dict:
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=10, **kwargs)
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def api_stream_chat(message: str, history: list, agent: str):
    """POST /api/chat → yield text chunks from SSE stream."""
    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json={"message": message, "history": history, "agent": agent},
            stream=True, timeout=120,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            buffer = ""
            for raw in resp.iter_content(chunk_size=None):
                buffer += raw.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    for line in block.splitlines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "chunk":
                                    yield data.get("text", "")
                                elif data.get("type") == "error":
                                    yield f"\n\n❌ Error: {data.get('text','')}"
                                    return
                                elif data.get("type") == "done":
                                    return
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}"


def get_status() -> dict:
    return api("/api/status")

# ─────────────────────────────────────────────────────────────────────────────
# OAuth callback handler
# ─────────────────────────────────────────────────────────────────────────────

def _handle_oauth_callback():
    """
    FastAPI handles the full OAuth token exchange and redirects back here with:
      ?oauth_ok=hubspot|zoho   on success
      ?oauth_error=<msg>       on failure
    We just read those params and show a toast.
    """
    p = st.query_params
    if ok := p.get("oauth_ok", ""):
        st.session_state["_ok"] = ok
        st.query_params.clear()
    elif err := p.get("oauth_error", ""):
        st.session_state["_err"] = err
        st.query_params.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Bubble renderer
# ─────────────────────────────────────────────────────────────────────────────

def _bubble(role: str, content: str, agent: str = "hubspot", streaming_cursor: bool = False):
    import html as h, re

    if role == "user":
        body = h.escape(content).replace("\n", "<br>")
        st.markdown(f"""
        <div class="msg-row user">
          <div class="av av-u">👤</div>
          <div class="bubble bub-u">{body}</div>
        </div>""", unsafe_allow_html=True)
    else:
        cfg  = AGENTS.get(agent, AGENTS["hubspot"])
        body = content
        body = re.sub(r'```(\w*)\n?(.*?)```', lambda m: f'<pre><code>{h.escape(m.group(2))}</code></pre>', body, flags=re.DOTALL)
        body = re.sub(r'`([^`]+)`', lambda m: f'<code>{h.escape(m.group(1))}</code>', body)
        body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body)
        body = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', body)
        body = re.sub(r'(?m)^[-•]\s+(.+)', r'<li>\1</li>', body)
        body = re.sub(r'(<li>.*?</li>)+', lambda m: f'<ul style="margin:6px 0;padding-left:18px">{m.group(0)}</ul>', body, flags=re.DOTALL)
        body = body.replace('\n\n', '</p><p style="margin:6px 0">')
        body = f'<p style="margin:0">{body}</p>'
        cursor = '<span style="color:#a78bfa;animation:blink 1s infinite">▌</span>' if streaming_cursor else ""
        st.markdown(f"""
        <div class="msg-row assistant">
          <div class="av {cfg['av_cls']}">{cfg['icon']}</div>
          <div class="bubble bub-b">{body}{cursor}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar(status: dict):
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:16px 4px 10px">
          <div style="width:38px;height:38px;background:linear-gradient(135deg,#a78bfa,#4da3ff);
               border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px">🤖</div>
          <div>
            <div style="font-size:15px;font-weight:700;color:#e8eaf6">CRM Agent</div>
            <div style="font-size:11px;color:#8b92b8">FastAPI · Streamlit · MCP</div>
          </div>
        </div>
        <hr class="hs-div"/>
        """, unsafe_allow_html=True)

        # ── Agent selector ──
        st.markdown('<div class="s-label">Active CRM</div>', unsafe_allow_html=True)
        agent_col1, agent_col2, agent_col3 = st.columns(3)
        agents_list = [("hubspot", "🟠 HS"), ("zoho_crm", "🔵 Zo"), ("both", "⚡")]
        for col, (ag, lbl) in zip([agent_col1, agent_col2, agent_col3], agents_list):
            with col:
                active = st.session_state.active_agent == ag
                if st.button(lbl, key=f"ag_{ag}", use_container_width=True,
                              type="primary" if active else "secondary"):
                    st.session_state.active_agent = ag
                    st.session_state.messages = []
                    st.rerun()

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # ── HubSpot connection ──
        hs_data = status.get("hubspot", {})
        hs_ok   = hs_data.get("connected", False)
        st.markdown('<div class="s-label">HubSpot</div>', unsafe_allow_html=True)
        if hs_ok:
            st.markdown(f'<div class="badge-ok">✅ {hs_data.get("message","Connected")}</div>',
                        unsafe_allow_html=True)
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            if st.button("🔌 Disconnect HubSpot", use_container_width=True):
                api("/api/disconnect", method="POST")
                st.rerun()
        else:
            st.markdown('<div class="badge-off">⚡ Not connected</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            # Link to FastAPI OAuth route — it handles the PKCE flow and redirects back
            st.link_button("🔗 Connect HubSpot", f"{BACKEND}/oauth/connect", use_container_width=True)

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # ── Zoho connection ──
        zo_data  = status.get("zoho", {})
        zo_oauth = zo_data.get("connected", False)
        zo_mcp   = zo_data.get("mcp_ready", False)
        zo_ok    = zo_oauth and zo_mcp

        st.markdown('<div class="s-label">Zoho CRM</div>', unsafe_allow_html=True)
        if zo_ok:
            st.markdown('<div class="badge-ok">✅ Zoho Connected</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:6px'/>", unsafe_allow_html=True)
            if st.button("🔌 Disconnect Zoho", use_container_width=True):
                api("/zoho/disconnect", method="POST")
                st.rerun()
        else:
            # Step 1: OAuth
            if zo_oauth:
                st.markdown('<div class="badge-zo">✅ OAuth done</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-off">① OAuth needed</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
            st.link_button("① Connect Zoho OAuth", f"{BACKEND}/zoho/connect", use_container_width=True)

            # Step 2: MCP URL
            st.markdown("<div style='height:4px'/>", unsafe_allow_html=True)
            if zo_mcp:
                st.markdown('<div class="badge-zo">✅ MCP URL saved</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-off">② MCP URL needed</div>', unsafe_allow_html=True)

            with st.expander("② Set Zoho MCP URL"):
                st.markdown(
                    '<div style="font-size:11px;color:#8b92b8;margin-bottom:6px">'
                    'Copy URL from <a href="https://mcp.zoho.in" target="_blank" style="color:#4da3ff">mcp.zoho.in</a> → Connect → Copy URL</div>',
                    unsafe_allow_html=True)
                mcp_inp = st.text_input(
                    "MCP Server URL", value=st.session_state.mcp_url,
                    placeholder="https://crm-data-metadata-XXXXX.zohomcp.in/mcp/APIKEY/message",
                    label_visibility="collapsed", key="mcp_url_input",
                )
                if st.button("💾 Save & Test URL", use_container_width=True):
                    if mcp_inp.strip():
                        with st.spinner("Testing…"):
                            r = api("/zoho/save-mcp-url", method="POST",
                                    json={"url": mcp_inp.strip()})
                        st.session_state.mcp_url = mcp_inp.strip()
                        if r.get("reachable"):
                            st.success("✅ " + r.get("message", "Connected!"))
                        else:
                            st.warning("⚠️ Saved but: " + r.get("message", ""))
                        st.rerun()
                    else:
                        st.error("Please paste the MCP URL first.")

            if zo_oauth:
                if st.button("🔌 Disconnect Zoho", key="zo_disc2", use_container_width=True):
                    api("/zoho/disconnect", method="POST")
                    st.rerun()

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown('<div class="s-label">Navigation</div>', unsafe_allow_html=True)
        for key, label in [("chat", "💬  Chat"), ("permissions", "🔑  Permissions"), ("debug", "🛠️  Debug MCP")]:
            btn_type = "primary" if st.session_state.active_tab == key else "secondary"
            if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
                st.session_state.active_tab = key
                st.rerun()

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # ── Quick commands ──
        cfg = AGENTS[st.session_state.active_agent]
        st.markdown(f'<div class="s-label">Quick Commands · {cfg["icon"]} {cfg["label"]}</div>',
                    unsafe_allow_html=True)
        for icon, label, prompt in cfg["nav_cmds"]:
            if st.button(f"{icon}  {label}", key=f"cmd_{label}", use_container_width=True):
                st.session_state.prefill    = prompt
                st.session_state.active_tab = "chat"
                st.rerun()

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.prefill  = ""
            st.rerun()

        st.markdown("""
        <div style="padding:16px 4px 8px;font-size:10px;color:#8b92b8;text-align:center">
          Backend :8000 · Frontend :8501
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat tab
# ─────────────────────────────────────────────────────────────────────────────

def _chat():
    msgs  = st.session_state.messages
    agent = st.session_state.active_agent
    cfg   = AGENTS[agent]

    if not msgs:
        st.markdown(f"""
        <div style="text-align:center;padding:40px 20px">
          <div style="font-size:52px;margin-bottom:14px">{cfg['icon']}</div>
          <div style="font-size:22px;font-weight:700;color:#e8eaf6;margin-bottom:8px">
            {cfg['label']} Agent
          </div>
          <div style="font-size:13px;color:#8b92b8;max-width:420px;margin:0 auto;line-height:1.7">
            Your AI assistant for {cfg['label']}.<br>
            Ask about deals, contacts, companies, leads, and pipelines.
          </div>
        </div>
        """, unsafe_allow_html=True)

        chips = cfg["chips"]
        cols  = st.columns(min(len(chips), 3))
        for i, (label, prompt) in enumerate(chips):
            with cols[i % 3]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
    else:
        for m in msgs:
            _bubble(m["role"], m["content"], agent=m.get("agent", agent))

    st.markdown("<div style='height:70px'/>", unsafe_allow_html=True)
    st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

    prefill = st.session_state.prefill
    if prefill:
        st.session_state.prefill = ""

    col_in, col_btn = st.columns([10, 1])
    with col_in:
        user_text = st.text_input(
            "Message", value=prefill,
            placeholder=cfg["placeholder"],
            label_visibility="collapsed", key="chat_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    # Active agent badge
    st.markdown(
        f'<div style="margin-top:6px;font-size:11px;color:#8b92b8">'
        f'Sending to: <span class="{cfg["badge"]}">{cfg["icon"]} {cfg["label"]}</span></div>',
        unsafe_allow_html=True,
    )

    if send and user_text and not st.session_state.processing:
        _send(user_text)


def _send(text: str):
    agent = st.session_state.active_agent
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": text, "agent": agent})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    _bubble("user", text)

    full = ""
    placeholder = st.empty()

    for chunk in api_stream_chat(text, history, agent):
        full += chunk
        with placeholder.container():
            _bubble("assistant", full, agent=agent, streaming_cursor=True)

    placeholder.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full or "⚠️ No response received.",
        "agent": agent,
    })
    st.session_state.processing = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Permissions tab
# ─────────────────────────────────────────────────────────────────────────────

def _permissions():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#e8eaf6;margin-bottom:4px">
      🔑 HubSpot Permissions
    </div>
    <div style="font-size:13px;color:#8b92b8;margin-bottom:20px">
      Live data from <code>/api/permissions</code>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading…"):
        data = api("/api/permissions")

    if "_error" in data or not data.get("connected"):
        err = data.get("_error", "")
        st.markdown(f"""
        <div class="a-warn">
          ⚠️ Not connected to HubSpot.
          {f"<br><code style='font-size:11px'>{err}</code>" if err else ""}
          Connect from the sidebar.
        </div>""", unsafe_allow_html=True)
        return

    role_html = (
        '<span style="background:rgba(76,175,125,.15);color:#4caf7d;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600">🔑 Super Admin</span>'
        if data.get("is_admin") else
        '<span style="background:rgba(91,141,238,.15);color:#5b8dee;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600">👤 Standard User</span>'
    )
    st.markdown(f"<div style='margin-bottom:16px'>{role_html}</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="perm-ttl">Tool Access</div>', unsafe_allow_html=True)
        chips = ""
        for t in data.get("tools", []):
            acc    = t["accessible"]
            color  = "#4caf7d" if acc else "#f87171"
            bg     = "rgba(76,175,125,.12)" if acc else "rgba(248,113,113,.1)"
            border = "rgba(76,175,125,.25)" if acc else "rgba(248,113,113,.25)"
            chips += f'<span class="chip" style="background:{bg};color:{color};border:1px solid {border}">{"✅" if acc else "🚫"} {t["tool"]}</span>'
        st.markdown(f'<div style="line-height:2.5">{chips}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="perm-ttl">Connection Status</div>', unsafe_allow_html=True)
        s = api("/api/status")
        ok_ = s.get("hubspot", {}).get("connected", False)
        st.markdown(
            f'<div class="{"a-ok" if ok_ else "a-err"}">{"✅" if ok_ else "❌"} {s.get("hubspot",{}).get("message","")}</div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

    scopes = data.get("scopes", [])
    cats = {
        "CRM Objects":  lambda s: s.startswith("crm.objects."),
        "CRM Schemas":  lambda s: s.startswith("crm.schemas."),
        "CMS":          lambda s: s.startswith("cms."),
        "Settings":     lambda s: s.startswith("settings.") or s == "mcp.users.read",
        "Analytics":    lambda s: s == "crm.hubsql.execute" or s.startswith("analytics."),
        "Other":        lambda s: True,
    }
    st.markdown(f'<div class="perm-ttl">Granted Scopes ({len(scopes)})</div>', unsafe_allow_html=True)
    used = set()
    for cat, fn in cats.items():
        grp = [s for s in scopes if s["scope"] not in used and fn(s["scope"])]
        if not grp:
            continue
        for s in grp:
            used.add(s["scope"])
        with st.expander(f"📁 {cat}  ({len(grp)})", expanded=(cat == "CRM Objects")):
            st.markdown("".join(
                f'<div class="sc-row"><span class="sc-lbl">{s.get("label", s["scope"])}</span>'
                f'<span class="sc-key">{s["scope"]}</span></div>'
                for s in grp
            ), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Debug tab
# ─────────────────────────────────────────────────────────────────────────────

def _debug():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#e8eaf6;margin-bottom:4px">
      🛠️ Debug MCP Connection
    </div>
    <div style="font-size:13px;color:#8b92b8;margin-bottom:20px">
      Verify backend reachability and MCP tool connectivity
    </div>
    """, unsafe_allow_html=True)

    try:
        t0 = time.time()
        r  = requests.get(f"{BACKEND}/api/status", timeout=5)
        ms = int((time.time() - t0) * 1000)
        st.markdown(
            f'<div class="a-ok">✅ FastAPI reachable at <code>{BACKEND}</code> — {ms}ms — HTTP {r.status_code}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.markdown(
            f'<div class="a-err">❌ FastAPI not reachable at <code>{BACKEND}</code><br>{e}<br>'
            f'Run: <code>uvicorn web_app:app --port 8000</code></div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)

    crm_choice = st.radio("Test MCP for:", ["HubSpot", "Zoho CRM"], horizontal=True)
    crm_param  = "zoho" if crm_choice == "Zoho CRM" else "hubspot"

    if st.button("🔄  Test MCP Connection"):
        with st.spinner("Testing…"):
            result = api(f"/api/debug-mcp?crm={crm_param}")

        status = result.get("status", result.get("_error", "unknown"))
        is_ok  = "SUCCESS" in str(status)
        st.markdown(
            f'<div class="{"a-ok" if is_ok else "a-err"}">{"✅" if is_ok else "❌"} {status}</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="perm-ttl">Connection Info</div>', unsafe_allow_html=True)
            for label, key in [("Transport", "transport"), ("Preflight", "preflight"), ("Tool count", "tool_count")]:
                st.markdown(
                    f'<div class="sc-row"><span class="sc-lbl" style="font-weight:600">{label}</span>'
                    f'<span class="sc-key">{result.get(key, "—")}</span></div>',
                    unsafe_allow_html=True
                )
        with col2:
            tools = result.get("tools", [])
            if tools:
                st.markdown(f'<div class="perm-ttl">Available Tools ({len(tools)})</div>',
                            unsafe_allow_html=True)
                st.markdown("".join(
                    f'<span style="display:inline-block;margin:3px;padding:4px 10px;'
                    f'background:rgba(91,141,238,.12);color:#5b8dee;border:1px solid rgba(91,141,238,.25);'
                    f'border-radius:14px;font-size:11px;font-family:monospace">{t}</span>'
                    for t in tools
                ), unsafe_allow_html=True)

        if "error" in result:
            st.markdown(
                f'<div class="a-err"><strong>Error:</strong> {result["error"]}</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _handle_oauth_callback()

    if crm := st.session_state.pop("_ok", ""):
        icon = "🟠" if crm == "hubspot" else "🔵"
        label = "HubSpot" if crm == "hubspot" else "Zoho CRM"
        st.toast(f"✅ {label} connected!", icon=icon)
    if err := st.session_state.pop("_err", ""):
        st.toast(f"❌ OAuth error: {err}", icon="⚠️")

    # Fetch status once per render
    status = get_status()

    _sidebar(status)

    # Header
    cfg = AGENTS[st.session_state.active_agent]
    hs_ok = status.get("hubspot", {}).get("connected", False)
    zo_ok = status.get("zoho", {}).get("connected", False) and status.get("zoho", {}).get("mcp_ready", False)

    hs_badge = f'<span class="badge-ok">🟠 HubSpot ✅</span>' if hs_ok else '<span class="badge-off">🟠 HubSpot</span>'
    zo_badge = f'<span class="badge-ok">🔵 Zoho ✅</span>'    if zo_ok else '<span class="badge-off">🔵 Zoho</span>'

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;padding:14px 20px;
         background:#1a1d27;border-bottom:1px solid #2e3250;
         margin:-0.5rem -1rem 1rem -1rem;flex-wrap:wrap">
      <div style="width:38px;height:38px;background:linear-gradient(135deg,#a78bfa,#4da3ff);
           border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px">🤖</div>
      <div>
        <div style="font-size:16px;font-weight:700;color:#e8eaf6">Multi-CRM AI Agent</div>
        <div style="font-size:11px;color:#8b92b8">LangGraph · MCP · FastAPI :8000 + Streamlit :8501</div>
      </div>
      <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
        {hs_badge} {zo_badge}
        <span class="{cfg['badge']}">{cfg['icon']} {cfg['label']} Mode</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab = st.session_state.active_tab
    if tab == "chat":
        _chat()
    elif tab == "permissions":
        _permissions()
    elif tab == "debug":
        _debug()


if __name__ == "__main__":
    main()