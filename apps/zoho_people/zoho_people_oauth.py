"""
apps/zoho_people/zoho_people_oauth.py
=====================================
Zoho People OAuth 2.0 — Server-based Application flow.

Flow:
  1. User clicks "Connect Zoho People API" in the sidebar
  2. Redirect to Zoho authorization URL with scopes
  3. Zoho redirects back to /oauth/zoho-people/callback with ?code=...
  4. Exchange code for access + refresh tokens
  5. Persist tokens; auto-refresh when expired

Env vars (in .env):
  ZOHO_PEOPLE_CLIENT_ID
  ZOHO_PEOPLE_CLIENT_SECRET
  ZOHO_ACCOUNTS_DOMAIN    — e.g. "accounts.zoho.in" (default)
  ZOHO_PEOPLE_API_DOMAIN  — e.g. "people.zoho.in"   (default)
"""

from __future__ import annotations
import os
import json
import time
from pathlib import Path
from urllib.parse import urlencode
import httpx
from dotenv import load_dotenv
from crm_logger import log

load_dotenv()

_BASE       = Path(__file__).parent.parent.parent  # project root
TOKEN_FILE  = _BASE / ".zoho_people_tokens.json"
REDIRECT_URI = os.getenv(
    "ZOHO_PEOPLE_REDIRECT_URI",
    "http://localhost:8000/oauth/zoho-people/callback",
)

# Default scopes for Zoho People API access
SCOPES = "ZOHOPEOPLE.forms.ALL,ZOHOPEOPLE.dashboard.READ"


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------

def _accounts_domain() -> str:
    return os.getenv("ZOHO_ACCOUNTS_DOMAIN", "accounts.zoho.in")


def _api_domain() -> str:
    return os.getenv("ZOHO_PEOPLE_API_DOMAIN", "people.zoho.in")


# ---------------------------------------------------------------------------
# OAuth configuration check
# ---------------------------------------------------------------------------

def is_oauth_configured() -> bool:
    """True if we have valid saved tokens OR client credentials to start a flow."""
    if _load_tokens():
        return True
    return bool(
        os.getenv("ZOHO_PEOPLE_CLIENT_ID")
        and os.getenv("ZOHO_PEOPLE_CLIENT_SECRET")
    )


def is_oauth_connected() -> bool:
    """True if we have saved tokens (connected, regardless of expiry)."""
    return _load_tokens() is not None


# ---------------------------------------------------------------------------
# Authorization URL
# ---------------------------------------------------------------------------

def build_auth_url(state: str) -> str:
    """Build the Zoho People OAuth authorization URL."""
    params = {
        "scope":         SCOPES,
        "client_id":     os.getenv("ZOHO_PEOPLE_CLIENT_ID", ""),
        "response_type": "code",
        "access_type":   "offline",       # gives us a refresh_token
        "redirect_uri":  REDIRECT_URI,
        "state":         state,
        "prompt":        "consent",        # always show consent screen
    }
    return f"https://{_accounts_domain()}/oauth/v2/auth?" + urlencode(params)


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------

def exchange_code_for_tokens(code: str) -> dict:
    """Exchange authorization code for access + refresh tokens."""
    r = httpx.post(
        f"https://{_accounts_domain()}/oauth/v2/token",
        data={
            "grant_type":    "authorization_code",
            "client_id":     os.getenv("ZOHO_PEOPLE_CLIENT_ID", ""),
            "client_secret": os.getenv("ZOHO_PEOPLE_CLIENT_SECRET", ""),
            "redirect_uri":  REDIRECT_URI,
            "code":          code,
        },
        timeout=15,
    )

    if r.status_code != 200:
        raise ValueError(f"Token exchange failed: {r.status_code} — {r.text}")

    tokens = r.json()
    if "error" in tokens:
        raise ValueError(f"Token exchange error: {tokens.get('error')}")
    if "access_token" not in tokens:
        raise ValueError(f"No access_token in response: {tokens}")

    _save_tokens(tokens)
    log("oauth", "Zoho People tokens exchanged successfully")
    return tokens


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def _save_tokens(data: dict):
    data["saved_at"] = int(time.time())
    TOKEN_FILE.write_text(json.dumps(data, indent=2))


def _load_tokens() -> dict | None:
    if TOKEN_FILE.exists():
        try:
            return json.loads(TOKEN_FILE.read_text())
        except Exception:
            return None
    return None


def _is_expired(tokens: dict) -> bool:
    saved_at   = tokens.get("saved_at", 0)
    expires_in = tokens.get("expires_in", 3600)
    return (time.time() - saved_at) > (expires_in - 300)


def _refresh(tokens: dict) -> dict:
    """Refresh the access token using the refresh token."""
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("No refresh_token available. Re-connect Zoho People.")

    r = httpx.post(
        f"https://{_accounts_domain()}/oauth/v2/token",
        data={
            "grant_type":    "refresh_token",
            "client_id":     os.getenv("ZOHO_PEOPLE_CLIENT_ID", ""),
            "client_secret": os.getenv("ZOHO_PEOPLE_CLIENT_SECRET", ""),
            "refresh_token": refresh_token,
        },
        timeout=15,
    )
    r.raise_for_status()
    new = r.json()

    if "error" in new:
        raise RuntimeError(f"Zoho token refresh failed: {new.get('error')}")

    # Zoho doesn't return refresh_token on refresh — keep the original
    if "refresh_token" not in new:
        new["refresh_token"] = refresh_token

    _save_tokens(new)
    log("oauth", f"Zoho People access token refreshed (expires in {new.get('expires_in', '?')}s)")
    return new


# ---------------------------------------------------------------------------
# Public: get a valid access token
# ---------------------------------------------------------------------------

async def get_access_token() -> str:
    """
    Return a valid Zoho People OAuth access token.
    Refreshes automatically when expired.
    """
    tokens = _load_tokens()
    if not tokens:
        raise RuntimeError(
            "Zoho People not connected. Click 'Connect Zoho People API' in the sidebar."
        )

    if _is_expired(tokens):
        tokens = _refresh(tokens)

    return tokens["access_token"]


def get_api_base() -> str:
    """Return the Zoho People REST API base URL."""
    return f"https://{_api_domain()}/people/api"


# ---------------------------------------------------------------------------
# Status + disconnect
# ---------------------------------------------------------------------------

def get_oauth_status() -> tuple[bool, str]:
    """Returns (is_connected, display_message) for the OAuth connection."""
    tokens = _load_tokens()
    if not tokens:
        has_creds = bool(
            os.getenv("ZOHO_PEOPLE_CLIENT_ID")
            and os.getenv("ZOHO_PEOPLE_CLIENT_SECRET")
        )
        if has_creds:
            return False, "API not connected — click Connect"
        return False, "API credentials not configured"

    if _is_expired(tokens):
        try:
            _refresh(tokens)
            return True, "API connected ✅ (refreshed)"
        except Exception:
            return False, "API token expired ⚠️ — re-connect"

    return True, "API connected ✅"


def disconnect_oauth():
    """Remove saved OAuth tokens."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
    log("bye", "Zoho People OAuth disconnected")
