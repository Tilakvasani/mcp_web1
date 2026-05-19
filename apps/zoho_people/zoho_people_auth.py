"""
apps/zoho_people/zoho_people_auth.py
=====================================
Zoho People HRMS — MCP URL manager.

Auth is handled entirely by the Zoho People MCP server itself.
We only store the MCP URL the user pastes from mcp.zoho.com.

Files:
  .zoho_people_mcp_url   — stores the MCP URL
  .zoho_people_disconnected — sentinel file when user disconnects
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_BASE            = Path(__file__).parent.parent.parent  # project root
MCP_URL_FILE     = _BASE / ".zoho_people_mcp_url"
DISCONNECT_FILE  = _BASE / ".zoho_people_disconnected"


# ─────────────────────────────────────────────────────────────────────────────
# URL management
# ─────────────────────────────────────────────────────────────────────────────

def save_mcp_url(url: str) -> None:
    """Save the Zoho People MCP URL to disk."""
    MCP_URL_FILE.write_text(url.strip())
    # Remove disconnect sentinel if present
    if DISCONNECT_FILE.exists():
        DISCONNECT_FILE.unlink()


def get_mcp_url() -> str | None:
    """Return the saved MCP URL, or None if disconnected / not set."""
    if DISCONNECT_FILE.exists():
        return None
    if MCP_URL_FILE.exists():
        url = MCP_URL_FILE.read_text().strip()
        return url or None
    return os.getenv("ZOHO_PEOPLE_MCP_URL", "") or None


def disconnect() -> None:
    """Mark Zoho People as disconnected."""
    if MCP_URL_FILE.exists():
        MCP_URL_FILE.unlink()
    DISCONNECT_FILE.write_text("disconnected")


def reconnect() -> None:
    """Remove the disconnect sentinel so the MCP URL from .env can load."""
    if DISCONNECT_FILE.exists():
        DISCONNECT_FILE.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Status helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_connection_status() -> tuple[bool, str]:
    """Returns (is_connected, display_message)."""
    url = get_mcp_url()
    if url:
        short = url[:50] + "…" if len(url) > 50 else url
        return True, f"Zoho People MCP ✅  {short}"
    return False, "Zoho People not connected"
