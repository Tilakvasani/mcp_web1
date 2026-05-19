"""
ui/app.py
=========
Clean chat UI — single window, sidebar for connections.
"""

import requests
import streamlit as st

BACKEND = "http://localhost:8000"

st.set_page_config(
    page_title = "Multi-CRM Agent",
    page_icon  = "🤖",
    layout     = "wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────

if "messages"   not in st.session_state: st.session_state.messages   = []
if "hs_connected" not in st.session_state: st.session_state.hs_connected = False
if "zo_connected" not in st.session_state: st.session_state.zo_connected = False
if "zo_url_input" not in st.session_state: st.session_state.zo_url_input = ""


def _get(path: str) -> dict:
    try:
        return requests.get(f"{BACKEND}{path}", timeout=5).json()
    except Exception:
        return {}


def _post(path: str, body: dict = {}) -> dict:
    try:
        return requests.post(f"{BACKEND}{path}", json=body, timeout=60).json()
    except Exception as e:
        return {"error": str(e)}


def _refresh_status():
    s = _get("/api/status")
    st.session_state.hs_connected = s.get("hubspot_connected", False)
    st.session_state.zo_connected = s.get("zoho_connected", False)


_refresh_status()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔌 Connections")

    # ── HubSpot ──────────────────────────────────────────────────────────────
    st.subheader("HubSpot")
    if st.session_state.hs_connected:
        st.success("Connected ✅")
        if st.button("Disconnect HubSpot", use_container_width=True):
            _post("/oauth/disconnect")
            _refresh_status()
            st.rerun()
    else:
        st.warning("Not connected")
        if st.button("Connect HubSpot", use_container_width=True, type="primary"):
            st.markdown(
                f'<meta http-equiv="refresh" content="0; url={BACKEND}/oauth/connect">',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Zoho HRMS ────────────────────────────────────────────────────────────
    st.subheader("Zoho HRMS")
    if st.session_state.zo_connected:
        st.success("Connected ✅")
        if st.button("Disconnect Zoho", use_container_width=True):
            _post("/zoho/disconnect")
            _refresh_status()
            st.rerun()
    else:
        st.warning("Not connected")
        st.caption("Get your URL from mcp.zoho.in → Connect → Copy URL")
        zo_url = st.text_input(
            "Zoho MCP URL",
            placeholder = "https://mcpweb-XXXXX.zohomcp.in/mcp/.../message",
            label_visibility = "collapsed",
        )
        if st.button("Save & Connect", use_container_width=True, type="primary"):
            if zo_url.strip():
                resp = _post("/zoho/save-url", {"url": zo_url.strip()})
                if resp.get("saved"):
                    _refresh_status()
                    st.rerun()
                else:
                    st.error(resp.get("error", "Failed to connect"))
            else:
                st.error("Please paste your Zoho MCP URL")

    st.divider()

    # ── What each app handles ─────────────────────────────────────────────
    st.caption("**HubSpot** → deals, contacts, companies, tickets")
    st.caption("**Zoho HRMS** → employees, leave, HR, attendance")

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main chat
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 Multi-CRM AI Agent")

# Connection check
if not st.session_state.hs_connected and not st.session_state.zo_connected:
    st.info("👈 Connect at least one app from the sidebar to get started.")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask anything about your CRM or HR data..."):

    if not st.session_state.hs_connected and not st.session_state.zo_connected:
        st.warning("Please connect an app from the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = _post("/api/chat", {
                "message": prompt,
                "history": st.session_state.messages[:-1],
            })
            reply = resp.get("reply") or resp.get("error") or "Something went wrong."

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
