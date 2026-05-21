"""
MCP Client — Remote Streamable HTTP server (HubSpot's hosted MCP).

Transport:
  Streamable HTTP only (mcp >= 1.8), as documented by HubSpot.
  Connect directly to the URL as-is — no path rewriting, no SSE fallback.

Properly unwraps Python 3.11+ ExceptionGroup so the real error is visible.
"""
import asyncio
import json
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp.client.sse import sse_client
import httpx
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client


# =============================================================================
# Helpers
# =============================================================================

def _unwrap_exception(exc: BaseException) -> str:
    """Recursively unwrap ExceptionGroup (Python 3.11+) to get the real cause."""
    if hasattr(exc, "exceptions"):           # ExceptionGroup / BaseExceptionGroup
        parts = [_unwrap_exception(e) for e in exc.exceptions]
        return " | ".join(parts)
    return f"{type(exc).__name__}: {exc}"


# =============================================================================
# MCPClient
# =============================================================================

class MCPClient:
    """
    MCP client for HubSpot's remote MCP server.

    Uses Streamable HTTP transport exclusively, connecting to the URL as-is.
    Per HubSpot's official docs the base URL is https://mcp.hubspot.com/ (with
    trailing slash) and no path manipulation is needed.

    Usage:
        client = MCPClient(url="https://mcp.hubspot.com/",
                           headers={"Authorization": "Bearer <token>"})
        await client.connect()       # raises ConnectionError with real cause
        tools = await client.list_tools()
        await client.cleanup()

    Or as async context manager:
        async with MCPClient(url=..., headers=...) as client:
            tools = await client.list_tools()
    """

    def __init__(self, url: str, headers: Any = None):
        self._url        = url
        self._headers    = headers or {}
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._transport_used: str = "none"

    def get_headers(self) -> dict:
        """Evaluate headers dynamic callback if provided, otherwise return the static dict."""
        if callable(self._headers):
            return self._headers()
        return self._headers

    # ------------------------------------------------------------------
    # Pre-flight check
    # ------------------------------------------------------------------
    async def preflight(self) -> tuple[bool, str]:
        """
        Verify the server is reachable and the token is valid BEFORE opening
        a long-lived connection.

        Strategy:
          1. Try a minimal MCP JSON-RPC POST (ping) — works for Zoho MCP and
             any other Streamable HTTP MCP server that rejects plain GET.
          2. Fall back to GET for servers (e.g. HubSpot) that accept it.

        Returns (ok, message).
        """
        # --- attempt 1: minimal MCP POST ping ---
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                r = await hc.post(
                    self._url,
                    headers={
                        **self.get_headers(),
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={"jsonrpc": "2.0", "id": 0, "method": "ping"},
                    follow_redirects=True,
                )
            msg = f"HTTP {r.status_code} on {self._url}"
            if r.status_code == 401:
                return False, f"401 Unauthorized — token invalid or expired ({self._url})"
            if r.status_code == 403:
                return False, f"403 Forbidden — missing OAuth scopes ({self._url})"
            # Any 2xx, 4xx (except 401/403) means the server is up and auth passed
            if r.status_code < 500:
                return True, msg
        except Exception:
            pass  # fall through to GET attempt

        # --- attempt 2: plain GET fallback (HubSpot-style) ---
        try:
            async with httpx.AsyncClient(timeout=10) as hc:
                r = await hc.get(self._url, headers=self.get_headers(), follow_redirects=True)
            msg = f"HTTP {r.status_code} on {self._url}"
            if r.status_code == 401:
                return False, f"401 Unauthorized — token invalid or expired ({self._url})"
            if r.status_code == 403:
                return False, f"403 Forbidden — missing OAuth scopes ({self._url})"
            if r.status_code in (200, 400, 404, 405, 406):
                return True, msg
            return False, msg
        except Exception as e:
            return False, _unwrap_exception(e)

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------
    async def connect(self):
        """
        Open MCP connection using Streamable HTTP transport.
        This works for both HubSpot and Zoho MCP servers.
        """
        try:
            async def _do_connect():
                transport = await self._exit_stack.enter_async_context(
                    streamablehttp_client(
                        self._url,
                        headers={
                            **self.get_headers(),
                            "Accept": "application/json, text/event-stream",
                            "Content-Type": "application/json",
                        },
                        timeout=15.0,
                    )
                )
                self._transport_used = f"streamable_http ({self._url})"
                # Extract read/write streams
                read, write = transport[0], transport[1]
                
                self._session = await self._exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await self._session.initialize()

            # Guard connections with a 15-second timeout to prevent indefinite hangs
            await asyncio.wait_for(_do_connect(), timeout=15.0)
            
        except asyncio.TimeoutError as exc:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._session    = None
            self._exit_stack = AsyncExitStack()
            raise ConnectionError("MCP connection timed out after 15 seconds") from exc
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._session    = None
            self._exit_stack = AsyncExitStack()
            raise ConnectionError(
                f"MCP connection failed: {_unwrap_exception(exc)}"
            ) from exc
            
    # ------------------------------------------------------------------
    # Session accessor
    # ------------------------------------------------------------------
    def session(self) -> ClientSession:
        if self._session is None:
            raise ConnectionError("Not connected. Call connect() first.")
        return self._session

    @property
    def transport(self) -> str:
        return self._transport_used

    # ------------------------------------------------------------------
    # MCP operations
    # ------------------------------------------------------------------
    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        return result.tools

    async def call_tool(self, tool_name: str, tool_input: dict) -> types.CallToolResult | None:
        return await self.session().call_tool(tool_name, tool_input)

    async def list_prompts(self) -> list[types.Prompt]:
        result = await self.session().list_prompts()
        return result.prompts

    async def get_prompt(self, prompt_name: str, args: dict[str, str]):
        result = await self.session().get_prompt(prompt_name, args)
        return result.messages

    async def read_resource(self, uri: str) -> Any:
        result = await self.session().read_resource(uri)
        if not result.contents:
            return []
        first = result.contents[0]
        text  = getattr(first, "text", None)
        if text is None:
            return []
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    async def reconnect(self):
        """

        Close the old transport and re-establish the connection.

        The MCPClient object keeps its identity so any closures holding
        a reference to ``self`` (e.g. the LangChain tool wrappers in
        session_cache) keep working transparently.
        """
        try:
            await asyncio.wait_for(self._exit_stack.aclose(), timeout=2.0)
        except Exception:
            pass
        self._session    = None
        self._exit_stack = AsyncExitStack()
        await self.connect()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    async def cleanup(self):
        try:
            await asyncio.wait_for(self._exit_stack.aclose(), timeout=2.0)
        except Exception:
            pass
        self._session = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------
    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()