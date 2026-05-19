"""
auth/zoho.py
============
Zoho HRMS connects via MCP URL only (no OAuth needed).
Each user pastes their own URL from mcp.zoho.in — stored per session.
"""

import json
from pathlib import Path
from typing import Optional

from core.logger import log

# Per-session MCP URLs stored in memory
# session_id → mcp_url
_session_urls: dict[str, str] = {}

# Fallback: single URL file for backward compat / env var
_MCP_URL_FILE = Path(__file__).parent.parent / ".zoho_mcp_url"


def save_mcp_url(url: str, session_id: str = "default") -> None:
    """Save Zoho MCP URL for this session."""
    _session_urls[session_id] = url.strip()
    log("ok", f"Zoho MCP URL saved | session={session_id[:8]}")


def get_mcp_url(session_id: str = "default") -> Optional[str]:
    """Get Zoho MCP URL for this session."""
    # 1. Per-session (in memory)
    if session_id in _session_urls and _session_urls[session_id]:
        return _session_urls[session_id]
    # 2. Fallback: file or env
    if _MCP_URL_FILE.exists():
        url = _MCP_URL_FILE.read_text().strip()
        if url:
            return url
    from config.settings import ZOHO_MCP_URL
    return ZOHO_MCP_URL or None


def disconnect(session_id: str = "default") -> None:
    """Remove Zoho MCP URL for this session."""
    _session_urls.pop(session_id, None)
    log("bye", f"Zoho disconnected | session={session_id[:8]}")


def is_connected(session_id: str = "default") -> bool:
    return bool(get_mcp_url(session_id))
