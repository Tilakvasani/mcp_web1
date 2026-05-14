"""
HubSpot CRM Agent — Streamlit Frontend
========================================
Calls the FastAPI backend (web_app.py) running on http://localhost:8000

Run both:
    uvicorn web_app:app --port 8000    (terminal 1 — keep web_app.py as-is)
    streamlit run streamlit_app.py     (terminal 2)

Install extra dep:
    pip install sseclient-rs
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
    page_title="HubSpot CRM Agent",
    page_icon="🟠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:      #0f1117;  --bg2:    #1a1d27;  --bg3:    #22263a;
  --border:  #2e3250;  --text:   #e8eaf6;  --text2:  #8b92b8;
  --accent:  #FF7A59;  --accent2:#ff9a80;
  --green:   #4caf7d;  --blue:   #5b8dee;  --red:    #f87171;
}
.main .block-container { padding-top: 0.5rem; padding-bottom: 0; max-width: 100% }
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
section[data-testid="stSidebar"] { background: var(--bg2); border-right: 1px solid var(--border); }
#MainMenu, footer, header { visibility: hidden; }

/* Divider */
.hs-div { border:none; border-top:1px solid var(--border); margin:10px 0; }

/* Badges */
.badge-ok  { background:rgba(76,175,125,.15); color:var(--green);  border:1px solid rgba(76,175,125,.3);  padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }
.badge-off { background:rgba(248,113,113,.1); color:var(--red);    border:1px solid rgba(248,113,113,.25);padding:5px 14px; border-radius:20px; font-size:12px; font-weight:500; display:inline-block; }

/* Sidebar label */
.s-label { font-size:10px; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:.7px; margin:14px 0 6px; }

/* Header */
.hs-header {
  display:flex; align-items:center; gap:12px;
  padding:14px 20px; background:var(--bg2);
  border-bottom:1px solid var(--border);
  margin:-0.5rem -1rem 1rem -1rem;
}
.hs-logo { width:38px; height:38px; background:var(--accent); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }
.hs-title { font-size:16px; font-weight:700; color:var(--text); }
.hs-sub   { font-size:11px; color:var(--text2); margin-top:2px; }

/* Chat bubbles */
.msg-row { display:flex; gap:12px; margin-bottom:16px; }
.msg-row.user { flex-direction:row-reverse; }
.av { width:36px; height:36px; border-radius:50%; flex-shrink:0; display:flex; align-items:center; justify-content:center; font-size:18px; }
.av-u { background:var(--blue); }
.av-b { background:var(--accent); }
.bubble { max-width:680px; padding:12px 16px; border-radius:14px; line-height:1.65; font-size:14px; }
.bub-u { background:var(--blue); color:white; border-top-right-radius:4px; }
.bub-b { background:var(--bg2); color:var(--text); border:1px solid var(--border); border-top-left-radius:4px; }
.bub-b code { background:var(--bg3); color:var(--accent2); padding:2px 6px; border-radius:4px; font-size:12px; }
.bub-b pre  { background:var(--bg3); border:1px solid var(--border); border-radius:8px; padding:12px; overflow-x:auto; margin:8px 0; }

/* Alerts */
.a-ok   { background:rgba(76,175,125,.1);  border:1px solid rgba(76,175,125,.3);  border-radius:10px; padding:12px 16px; color:var(--green);  font-size:13px; margin:8px 0; }
.a-warn { background:rgba(255,154,128,.1); border:1px solid rgba(255,154,128,.3); border-radius:10px; padding:12px 16px; color:var(--accent2);font-size:13px; margin:8px 0; }
.a-err  { background:rgba(248,113,113,.1); border:1px solid rgba(248,113,113,.25);border-radius:10px; padding:12px 16px; color:var(--red);    font-size:13px; margin:8px 0; }

/* Scope rows */
.sc-row { display:flex; align-items:center; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border); font-size:12px; }
.sc-lbl { color:var(--text); }
.sc-key { color:var(--text2); font-family:monospace; font-size:10px; }
.perm-ttl { font-size:10px; font-weight:700; color:var(--text2); text-transform:uppercase; letter-spacing:.6px; margin:14px 0 8px; }
.chip { display:inline-block; margin:3px; padding:4px 10px; border-radius:16px; font-size:11px; font-weight:500; }

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
/* All buttons base */
.stButton > button {
  background:var(--bg3) !important; border:1px solid var(--border) !important;
  color:var(--text) !important; border-radius:8px !important;
  font-size:13px !important; font-weight:500 !important;
}
.stButton > button:hover { border-color:var(--blue) !important; color:var(--blue) !important; }
/* Send button (last column) */
div[data-testid="column"]:last-child .stButton > button {
  background:var(--accent) !important; border-color:var(--accent) !important;
  color:white !important; font-weight:700 !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
  background:var(--accent2) !important; border-color:var(--accent2) !important;
  color:white !important;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "messages":   [],
    "processing": False,
    "active_tab": "chat",
    "prefill":    "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
# API calls → FastAPI backend
# ─────────────────────────────────────────────────────────────────────────────
def api(path, method="GET", **kwargs) -> dict:
    try:
        r = requests.request(method, f"{BACKEND}{path}", timeout=10, **kwargs)
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def api_stream_chat(message: str, history: list):
    """
    POST /api/chat  →  yield text chunks from SSE stream.
    Uses plain requests + manual SSE parsing (no extra library needed).
    """
    try:
        with requests.post(
            f"{BACKEND}/api/chat",
            json={"message": message, "history": history},
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
                                elif data.get("type") == "done":
                                    return
                            except json.JSONDecodeError:
                                pass
    except Exception as e:
        yield f"\n\n❌ Stream error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# OAuth callback
# ─────────────────────────────────────────────────────────────────────────────
def _oauth_callback():
    """
    HubSpot redirects → Streamlit with ?code=&state=
    Forward to FastAPI /oauth/callback which does the token exchange.
    """
    p = st.query_params
    code, state, error = p.get("code",""), p.get("state",""), p.get("error","")
    if error:
        st.session_state["_err"] = p.get("error_description", error)
        st.query_params.clear(); return
    if code and state:
        try:
            r = requests.get(
                f"{BACKEND}/oauth/callback",
                params={"code": code, "state": state},
                timeout=15, allow_redirects=False,
            )
            if r.status_code in (200, 302):
                st.session_state["_ok"] = True
            else:
                st.session_state["_err"] = f"HTTP {r.status_code}"
        except Exception as e:
            st.session_state["_err"] = str(e)
        st.query_params.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Bubble renderer
# ─────────────────────────────────────────────────────────────────────────────
def _bubble(role: str, content: str, streaming_cursor: bool = False):
    import html as h, re

    if role == "user":
        body = h.escape(content).replace("\n", "<br>")
        st.markdown(f"""
        <div class="msg-row user">
          <div class="av av-u">👤</div>
          <div class="bubble bub-u">{body}</div>
        </div>""", unsafe_allow_html=True)
    else:
        body = content
        body = re.sub(r'```(\w*)\n?(.*?)```', lambda m: f'<pre><code>{h.escape(m.group(2))}</code></pre>', body, flags=re.DOTALL)
        body = re.sub(r'`([^`]+)`', lambda m: f'<code>{h.escape(m.group(1))}</code>', body)
        body = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', body)
        body = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', body)
        body = re.sub(r'(?m)^[-•]\s+(.+)', r'<li>\1</li>', body)
        body = re.sub(r'(<li>.*?</li>)+', lambda m: f'<ul style="margin:6px 0;padding-left:18px">{m.group(0)}</ul>', body, flags=re.DOTALL)
        body = body.replace('\n\n', '</p><p style="margin:6px 0">')
        body = f'<p style="margin:0">{body}</p>'
        cursor = '<span style="color:#FF7A59;animation:blink 1s infinite">▌</span>' if streaming_cursor else ""
        st.markdown(f"""
        <div class="msg-row assistant">
          <div class="av av-b">🟠</div>
          <div class="bubble bub-b">{body}{cursor}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:16px 4px 10px">
          <div style="width:38px;height:38px;background:#FF7A59;border-radius:10px;
               display:flex;align-items:center;justify-content:center;font-size:20px">🟠</div>
          <div>
            <div style="font-size:15px;font-weight:700;color:#e8eaf6">HubSpot Agent</div>
            <div style="font-size:11px;color:#8b92b8">FastAPI · Streamlit · MCP</div>
          </div>
        </div>
        <hr class="hs-div"/>
        """, unsafe_allow_html=True)

        # Connection status
        status = api("/api/status")
        ok  = status.get("connected", False)
        msg = status.get("message", "Backend unreachable" if "_error" in status else "")

        if ok:
            st.markdown(f'<div class="badge-ok">✅ {msg}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
            if st.button("🔌 Disconnect", use_container_width=True):
                api("/api/disconnect", method="POST")
                st.rerun()
        else:
            st.markdown('<div class="badge-off">⚡ Not connected</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'/>", unsafe_allow_html=True)
            # OAuth lives in FastAPI — link opens the /oauth/connect route
            st.link_button("🔗 Connect HubSpot", f"{BACKEND}/oauth/connect", use_container_width=True)

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # Nav
        st.markdown('<div class="s-label">Navigation</div>', unsafe_allow_html=True)
        for key, label in [("chat","💬  Chat"), ("permissions","🔑  Permissions"), ("debug","🛠️  Debug MCP")]:
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()

        st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

        # Quick commands
        st.markdown('<div class="s-label">Quick Commands</div>', unsafe_allow_html=True)
        cmds = [
            ("📊","Pipeline overview",  "Summarise my HubSpot deal pipeline by stage"),
            ("🔍","Search contacts",    "Find all contacts added this week"),
            ("💼","Open deals",         "Show all open deals sorted by amount"),
            ("➕","Create a deal",      "Create a new deal for Acme Corp worth $50,000"),
            ("📋","Recent activity",    "Show CRM activities from the last 7 days"),
            ("✅","Closed won",         "List all deals closed won this quarter"),
            ("📈","Conversion report",  "What is my lead-to-deal conversion rate this month?"),
            ("🏷️","Company search",     "Find all contacts at TechCorp"),
        ]
        for icon, label, prompt in cmds:
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
        <div style="padding:20px 4px 8px;font-size:10px;color:#8b92b8;text-align:center">
          Backend :8000 · Frontend :8501
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat tab
# ─────────────────────────────────────────────────────────────────────────────
def _chat():
    msgs = st.session_state.messages

    if not msgs:
        st.markdown("""
        <div style="text-align:center;padding:40px 20px">
          <div style="font-size:52px;margin-bottom:14px">🟠</div>
          <div style="font-size:22px;font-weight:700;color:#e8eaf6;margin-bottom:8px">HubSpot CRM Agent</div>
          <div style="font-size:13px;color:#8b92b8;max-width:380px;margin:0 auto;line-height:1.7">
            Your AI assistant for HubSpot CRM.<br>
            Ask about deals, contacts, companies, and pipelines.
          </div>
        </div>
        """, unsafe_allow_html=True)

        chips = [
            ("📊 Pipeline overview",  "Summarise my HubSpot deal pipeline by stage"),
            ("💼 Open deals",         "Show me all open deals"),
            ("🔍 Find a contact",     "Find contact john@acme.com"),
            ("➕ Create a deal",      "Create a deal for Acme Corp"),
            ("📋 Recent activity",    "Show recent CRM activities"),
            ("✅ Closed won",         "List deals closed won this quarter"),
        ]
        c1, c2, c3 = st.columns(3)
        for i, (label, prompt) in enumerate(chips):
            with [c1, c2, c3][i % 3]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    st.session_state.prefill = prompt
                    st.rerun()
    else:
        for m in msgs:
            _bubble(m["role"], m["content"])

    st.markdown("<div style='height:70px'/>", unsafe_allow_html=True)
    st.markdown('<hr class="hs-div"/>', unsafe_allow_html=True)

    # Grab and clear prefill
    prefill = st.session_state.prefill
    if prefill:
        st.session_state.prefill = ""

    col_in, col_btn = st.columns([10, 1])
    with col_in:
        user_text = st.text_input(
            "Message", value=prefill,
            placeholder="Ask anything about your HubSpot CRM…",
            label_visibility="collapsed", key="chat_input",
        )
    with col_btn:
        send = st.button("Send ➤", use_container_width=True)

    if send and user_text and not st.session_state.processing:
        _send(user_text)


def _send(text: str):
    st.session_state.processing = True
    st.session_state.messages.append({"role": "user", "content": text})

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Render user bubble instantly
    _bubble("user", text)

    # Stream bot response
    full = ""
    placeholder = st.empty()

    for chunk in api_stream_chat(text, history):
        full += chunk
        with placeholder.container():
            _bubble("assistant", full, streaming_cursor=True)

    placeholder.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": full or "⚠️ No response received.",
    })
    st.session_state.processing = False
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Permissions tab
# ─────────────────────────────────────────────────────────────────────────────
def _permissions():
    st.markdown("""
    <div style="font-size:20px;font-weight:700;color:#e8eaf6;margin-bottom:4px">
      🔑 My HubSpot Permissions
    </div>
    <div style="font-size:13px;color:#8b92b8;margin-bottom:20px">
      Live data from FastAPI <code>/api/permissions</code>
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

    # Role
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
        ok_ = s.get("connected", False)
        st.markdown(
            f'<div class="{"a-ok" if ok_ else "a-err"}">{"✅" if ok_ else "❌"} {s.get("message","")}</div>',
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
        if not grp: continue
        for s in grp: used.add(s["scope"])
        with st.expander(f"📁 {cat}  ({len(grp)})", expanded=(cat == "CRM Objects")):
            st.markdown("".join(
                f'<div class="sc-row"><span class="sc-lbl">{s.get("label",s["scope"])}</span><span class="sc-key">{s["scope"]}</span></div>'
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
      Calls FastAPI <code>/api/debug-mcp</code> · verifies HubSpot MCP endpoint
    </div>
    """, unsafe_allow_html=True)

    # Backend reachability
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

    if st.button("🔄  Test MCP Connection"):
        with st.spinner("Testing…"):
            result = api("/api/debug-mcp")

        status = result.get("status", result.get("_error", "unknown"))
        is_ok  = "SUCCESS" in str(status)
        st.markdown(
            f'<div class="{"a-ok" if is_ok else "a-err"}">{"✅" if is_ok else "❌"} {status}</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="perm-ttl">Connection Info</div>', unsafe_allow_html=True)
            for label, key in [("MCP URL","url"),("Token","token"),("Transport","transport"),("Preflight","preflight")]:
                st.markdown(
                    f'<div class="sc-row"><span class="sc-lbl" style="font-weight:600">{label}</span><span class="sc-key">{result.get(key,"—")}</span></div>',
                    unsafe_allow_html=True
                )
        with col2:
            tools = result.get("tools", [])
            if tools:
                st.markdown(f'<div class="perm-ttl">Available Tools ({len(tools)})</div>', unsafe_allow_html=True)
                st.markdown("".join(
                    f'<span style="display:inline-block;margin:3px;padding:4px 10px;background:rgba(91,141,238,.12);color:#5b8dee;border:1px solid rgba(91,141,238,.25);border-radius:14px;font-size:11px;font-family:monospace">{t}</span>'
                    for t in tools
                ), unsafe_allow_html=True)

        if "error" in result:
            st.markdown(
                f'<div class="a-err"><strong>Error:</strong> {result["error"]}<br>'
                f'<strong>Fix:</strong> {result.get("fix","")}</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _oauth_callback()

    if st.session_state.pop("_ok", False):
        st.toast("✅ HubSpot connected!", icon="🟠")
    if err := st.session_state.pop("_err", ""):
        st.toast(f"❌ OAuth error: {err}", icon="⚠️")

    _sidebar()

    st.markdown("""
    <div class="hs-header">
      <div class="hs-logo">🟠</div>
      <div>
        <div class="hs-title">HubSpot CRM Agent</div>
        <div class="hs-sub">LangChain · LangGraph · MCP · OAuth 2.1 PKCE · FastAPI :8000 + Streamlit :8501</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab = st.session_state.active_tab
    if tab == "chat":         _chat()
    elif tab == "permissions": _permissions()
    elif tab == "debug":       _debug()


if __name__ == "__main__":
    main()