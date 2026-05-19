"""
HubSpot OAuth 2.1 — PKCE Flow
==============================
Uses Authorization Code + PKCE (S256).
client_secret is still needed in the token exchange (HubSpot requirement).

Flow:
  1. Fetch authorize/token endpoints from OAuth metadata discovery
  2. Generate code_verifier + code_challenge (S256)
  3. Redirect user to the discovered authorize URL with code_challenge
     (NO scope parameter — scopes are determined by the MCP Auth App and
      the user's grant at install time)
  4. HubSpot redirects back to /oauth/callback with ?code=...
  5. Exchange code + code_verifier for access/refresh tokens via discovered token URL
  6. Persist tokens; auto-refresh when expired
"""

import os, json, time, base64, hashlib, secrets
from pathlib import Path
from urllib.parse import urlencode
import httpx
import httpx as _httpx_async  # used for async calls in check_is_admin
from dotenv import load_dotenv

load_dotenv()

TOKEN_FILE   = Path(__file__).parent / ".hubspot_tokens.json"
REDIRECT_URI = os.getenv("HUBSPOT_REDIRECT_URI", "http://localhost:8000/oauth/callback")

# OAuth 2.1 metadata discovery — avoids hardcoding regional endpoints
OAUTH_DISCOVERY_URL = "https://mcp.hubspot.com/.well-known/oauth-authorization-server"

# Token URL for exchanges/refreshes (standard HubSpot token endpoint)
TOKEN_URL = "https://api.hubapi.com/oauth/v1/token"

# Cache discovered endpoints for the process lifetime
_oauth_metadata: dict | None = None


def _get_oauth_metadata() -> dict:
    """Fetch (and cache) OAuth server metadata from the discovery endpoint."""
    global _oauth_metadata
    if _oauth_metadata is None:
        r = httpx.get(OAUTH_DISCOVERY_URL, timeout=10)
        r.raise_for_status()
        _oauth_metadata = r.json()
    return _oauth_metadata


def get_auth_url() -> str:
    """Return the authorization_endpoint from OAuth discovery metadata."""
    meta = _get_oauth_metadata()
    auth_url = meta.get("authorization_endpoint")
    if not auth_url:
        raise ValueError(
            f"No authorization_endpoint in OAuth metadata from {OAUTH_DISCOVERY_URL}"
        )
    return auth_url


# =============================================================================
# PKCE helpers
# =============================================================================

def generate_pkce_pair() -> tuple[str, str]:
    """
    Returns (code_verifier, code_challenge).
    verifier  = 64-byte URL-safe random string  (~86 chars)
    challenge = BASE64URL(SHA256(verifier))  — S256 method
    """
    verifier  = secrets.token_urlsafe(64)
    digest    = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_auth_url(code_challenge: str, state: str) -> str:
    """
    Build the HubSpot OAuth 2.1 authorization URL.

    Scopes are NOT sent — they are determined automatically by the MCP Auth App
    configuration and the user's grant at install time (per HubSpot MCP docs).
    """
    params = {
        "client_id":             os.getenv("HUBSPOT_CLIENT_ID", ""),
        "redirect_uri":          REDIRECT_URI,
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    return get_auth_url() + "?" + urlencode(params)


def exchange_code_for_tokens(code: str, code_verifier: str) -> dict:
    """
    Exchange authorization code + PKCE verifier → access/refresh tokens.
    HubSpot still requires client_id + client_secret here (hybrid PKCE).
    """
    r = httpx.post(TOKEN_URL, data={
        "grant_type":    "authorization_code",
        "client_id":     os.getenv("HUBSPOT_CLIENT_ID", ""),
        "client_secret": os.getenv("HUBSPOT_CLIENT_SECRET", ""),
        "redirect_uri":  REDIRECT_URI,
        "code":          code,
        "code_verifier": code_verifier,
    }, timeout=15)

    if r.status_code != 200:
        raise ValueError(f"Token exchange failed: {r.status_code} — {r.text}")

    tokens = r.json()
    _save_tokens(tokens)
    return tokens


# =============================================================================
# Token storage
# =============================================================================

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
    expires_in = tokens.get("expires_in", 1800)
    return (time.time() - saved_at) > (expires_in - 300)


def _refresh(tokens: dict) -> dict:
    r = httpx.post(TOKEN_URL, data={
        "grant_type":    "refresh_token",
        "client_id":     os.getenv("HUBSPOT_CLIENT_ID", ""),
        "client_secret": os.getenv("HUBSPOT_CLIENT_SECRET", ""),
        "refresh_token": tokens["refresh_token"],
    }, timeout=15)
    r.raise_for_status()
    new = r.json()
    if "refresh_token" not in new:
        new["refresh_token"] = tokens["refresh_token"]
    _save_tokens(new)
    return new


def get_valid_token() -> str:
    """Return a valid access token, refreshing if needed."""
    tokens = _load_tokens()
    if tokens:
        if _is_expired(tokens):
            tokens = _refresh(tokens)
        return tokens["access_token"]
    raise RuntimeError("No HubSpot token. Connect via /oauth/connect first.")


# =============================================================================
# Status helpers
# =============================================================================

def get_connection_status() -> tuple[bool, str]:
    """Returns (is_connected, display_string)."""
    tokens = _load_tokens()
    if not tokens:
        return False, "Not connected"

    if _is_expired(tokens):
        try:
            _refresh(tokens)
            return True, "OAuth 2.1 ✅ (refreshed)"
        except Exception:
            return False, "OAuth expired ⚠️"

    try:
        r = httpx.get(
            "https://api.hubapi.com/account-info/v3/details",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
            timeout=8,
        )
        if r.status_code == 200:
            portal = r.json().get("portalId", "")
            return True, f"OAuth 2.1 ✅  Portal: {portal}"
        return False, "Token invalid ⚠️"
    except Exception:
        return True, "OAuth 2.1 ✅"


def disconnect():
    """Remove saved tokens."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        print("✅ Disconnected from HubSpot.")
    else:
        print("Not connected.")


# =============================================================================
# Permission helpers
# =============================================================================

def get_token_scopes() -> list[str]:
    """Return the list of scopes granted in the saved token, or empty list."""
    tokens = _load_tokens()
    return tokens.get("scopes", []) if tokens else []


def get_token_user_id() -> int | None:
    """Return the user_id stored in the saved token, or None."""
    tokens = _load_tokens()
    return tokens.get("user_id") if tokens else None


async def check_is_admin(access_token: str, user_id: int) -> bool:
    """
    Return True if the HubSpot user is a super-admin.
    Calls GET /settings/v3/users/{userId} — requires the token to have
    crm.objects.owners.read or mcp.users.read scope.
    Falls back to False on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(
                f"https://api.hubapi.com/settings/v3/users/{user_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if r.status_code == 200:
                return r.json().get("superAdmin", False)
    except Exception:
        pass
    return False