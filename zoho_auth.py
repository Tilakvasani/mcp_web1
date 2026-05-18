"""
Zoho CRM OAuth 2.0 + MCP URL Manager
======================================
Handles Zoho OAuth2 (authorization_code flow) for CRM access.
Also manages the Zoho MCP Server URL (API key embedded).

Token file: .zoho_tokens.json
MCP URL file: .zoho_mcp_url (single line)
"""

import os
import json
import time
import base64
import hashlib
import secrets
from pathlib import Path
from urllib.parse import urlencode
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKEN_FILE          = Path(__file__).parent / ".zoho_tokens.json"
MCP_URL_FILE        = Path(__file__).parent / ".zoho_mcp_url"
DISCONNECT_SENTINEL = Path(__file__).parent / ".zoho_disconnected"   # B9

ZOHO_CLIENT_ID     = os.getenv("ZOHO_CLIENT_ID", "")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET", "")
# B2: accept both URI and URL spellings; prefer URI, fall back to URL
ZOHO_REDIRECT_URI  = os.getenv("ZOHO_REDIRECT_URI") or os.getenv("ZOHO_REDIRECT_URL", "http://localhost:8000/zoho/callback")
ZOHO_ACCOUNTS_URL  = os.getenv("ZOHO_ACCOUNTS_URL", "https://accounts.zoho.in")

ZOHO_SCOPES = [
    "ZohoCRM.modules.ALL",
    "ZohoCRM.settings.ALL",
    "ZohoCRM.users.ALL",
    "ZohoCRM.org.ALL",
]


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def generate_pkce_pair() -> tuple[str, str]:
    verifier  = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    digest    = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(code_challenge: str, state: str) -> str:
    params = {
        "response_type":         "code",
        "client_id":             ZOHO_CLIENT_ID,
        "redirect_uri":          ZOHO_REDIRECT_URI,
        "scope":                 " ".join(ZOHO_SCOPES),
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
        "access_type":           "offline",
    }
    return f"{ZOHO_ACCOUNTS_URL}/oauth/v2/auth?" + urlencode(params)


def exchange_code_for_tokens(code: str, code_verifier: str) -> dict:
    r = httpx.post(
        f"{ZOHO_ACCOUNTS_URL}/oauth/v2/token",
        data={
            "grant_type":    "authorization_code",
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "redirect_uri":  ZOHO_REDIRECT_URI,
            "code":          code,
            "code_verifier": code_verifier,
        },
        timeout=15,
    )
    data = r.json()
    if "access_token" not in data:
        raise ValueError(f"Token exchange failed: {data}")
    payload = {
        "access_token":  data["access_token"],
        "refresh_token": data.get("refresh_token", ""),
        "expires_at":    time.time() + data.get("expires_in", 3600) - 60,
        "api_domain":    data.get("api_domain", "https://www.zohoapis.in"),
    }
    TOKEN_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def _refresh_tokens(refresh_token: str) -> dict:
    r = httpx.post(
        f"{ZOHO_ACCOUNTS_URL}/oauth/v2/token",
        data={
            "grant_type":    "refresh_token",
            "client_id":     ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    data = r.json()
    if "access_token" not in data:
        raise RuntimeError(f"Token refresh failed: {data}")
    payload = {
        "access_token":  data["access_token"],
        "refresh_token": refresh_token,
        "expires_at":    time.time() + data.get("expires_in", 3600) - 60,
        "api_domain":    data.get("api_domain", "https://www.zohoapis.in"),
    }
    TOKEN_FILE.write_text(json.dumps(payload, indent=2))
    return payload


def get_valid_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError("Zoho not connected — no token file found")
    tokens = json.loads(TOKEN_FILE.read_text())
    if time.time() < tokens.get("expires_at", 0):
        return tokens["access_token"]
    rt = tokens.get("refresh_token", "")
    if not rt:
        raise RuntimeError("Zoho token expired and no refresh_token available")
    refreshed = _refresh_tokens(rt)
    return refreshed["access_token"]


def get_connection_status() -> tuple[bool, str]:
    try:
        token = get_valid_token()
        return True, "Zoho CRM ✅  Connected"
    except RuntimeError as exc:
        return False, str(exc)


def get_zoho_org_info() -> Optional[dict]:
    """Fetch org info from Zoho CRM API."""
    try:
        token = get_valid_token()
        tokens = json.loads(TOKEN_FILE.read_text())
        api_domain = tokens.get("api_domain", "https://www.zohoapis.in")
        r = httpx.get(
            f"{api_domain}/crm/v2/org",
            headers={"Authorization": f"Zoho-oauthtoken {token}"},
            timeout=10,
        )
        data = r.json()
        org_list = data.get("org", [])
        return org_list[0] if org_list else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# MCP URL management
# ---------------------------------------------------------------------------

def save_mcp_url(url: str) -> None:
    MCP_URL_FILE.write_text(url.strip())


def get_mcp_url() -> Optional[str]:
    # B9: if user explicitly disconnected, don't fall back to env var
    if DISCONNECT_SENTINEL.exists():
        return None
    if MCP_URL_FILE.exists():
        url = MCP_URL_FILE.read_text().strip()
        return url if url else None
    return os.getenv("ZOHO_MCP_URL", "") or None


def get_mcp_status() -> tuple[bool, str]:
    url = get_mcp_url()
    if url:
        return True, "MCP URL configured ✅"
    return False, "Zoho MCP URL not set"
