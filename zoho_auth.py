"""
Zoho MCP URL Manager
====================
Manages the Zoho MCP Server URL only.
Auth is handled entirely by mcp.zoho.in — no client_id / client_secret needed.

MCP URL file: .zoho_mcp_url (single line)
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MCP_URL_FILE        = Path(__file__).parent / ".zoho_mcp_url"
DISCONNECT_SENTINEL = Path(__file__).parent / ".zoho_disconnected"


# ---------------------------------------------------------------------------
# MCP URL management
# ---------------------------------------------------------------------------

def save_mcp_url(url: str) -> None:
    MCP_URL_FILE.write_text(url.strip())


def get_mcp_url() -> Optional[str]:
    if DISCONNECT_SENTINEL.exists():
        return None
    if MCP_URL_FILE.exists():
        url = MCP_URL_FILE.read_text().strip()
        return url if url else None
    return os.getenv("ZOHO_MCP_URL", "") or None


def get_mcp_status() -> tuple[bool, str]:
    url = get_mcp_url()
    if url:
        return True, "Zoho MCP \u2705  Connected"
    return False, "Zoho MCP URL not set"


def get_connection_status() -> tuple[bool, str]:
    """Alias kept for compatibility — Zoho is 'connected' when MCP URL is set."""
    return get_mcp_status()
