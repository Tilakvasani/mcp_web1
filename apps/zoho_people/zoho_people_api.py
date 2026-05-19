"""
apps/zoho_people/zoho_people_api.py
====================================
Dynamic ``ZohoPeople_callAPI`` tool for calling ANY Zoho People REST API
endpoint directly.

The MCP server exposes only 15 fixed tools (all single-record-by-ID).
This tool lets the LLM call endpoints like:
  - GET /forms/employee/getRecords         (list all employees)
  - GET /forms/department/getRecords       (list departments)
  - GET /forms/leave/getRecords            (list leave records)
  - Any other Zoho People REST API endpoint

Auth is handled via OAuth refresh-token flow (zoho_people_oauth.py).
"""

from __future__ import annotations
import json
import httpx
from crm_logger import log
from apps.zoho_people.zoho_people_oauth import get_access_token, get_api_base, is_oauth_configured


async def zoho_people_call_api(
    method: str = "GET",
    path: str = "/forms/employee/getRecords",
    params: dict | None = None,
    data: dict | None = None,
) -> str:
    """
    Call any Zoho People REST API endpoint.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE)
        path:   API path after /people/api, e.g. "/forms/employee/getRecords"
        params: Query parameters dict
        data:   JSON body for POST/PUT

    Returns:
        Raw JSON response as string for the LLM to interpret.
    """
    if not is_oauth_configured():
        return (
            "Error: Zoho People OAuth is not configured. "
            "Please set ZOHO_PEOPLE_CLIENT_ID, ZOHO_PEOPLE_CLIENT_SECRET, "
            "and ZOHO_PEOPLE_REFRESH_TOKEN in the .env file."
        )

    try:
        token = await get_access_token()
    except Exception as e:
        return f"Error refreshing Zoho People token: {e}"

    api_base = get_api_base()
    # Normalize path
    clean_path = path.strip()
    if not clean_path.startswith("/"):
        clean_path = "/" + clean_path
    url = f"{api_base}{clean_path}"

    headers = {
        "Authorization": f"Zoho-oauthtoken {token}",
        "Content-Type":  "application/json",
    }

    log("api", f"ZohoPeople_callAPI {method} {clean_path} params={params}")

    try:
        async with httpx.AsyncClient(timeout=20) as hc:
            r = await hc.request(
                method  = method.upper(),
                url     = url,
                headers = headers,
                params  = params or {},
                json    = data if data else None,
            )
        log("api", f"ZohoPeople_callAPI -> {r.status_code}")

        # Try to parse as JSON for cleaner output
        try:
            result = r.json()
            return json.dumps(result, indent=2)
        except Exception:
            return r.text[:4000]

    except Exception as e:
        return f"API call failed: {e}"
