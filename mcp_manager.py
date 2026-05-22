"""
mcp_manager.py
==============
Stores MCP server configs and returns LangChain tools on demand.
Uses MultiServerMCPClient per call — no persistent connections.
Handles both sync and async token functions.
"""
import asyncio
import inspect
from langchain_mcp_adapters.client import MultiServerMCPClient


class MCPManager:

    def __init__(self):
        self._configs: dict = {}  # name -> internal config dict

    def set_hubspot(self, access_token_fn):
        """access_token_fn: sync callable that returns the current token."""
        self._configs["hubspot"] = {
            "url": "https://mcp.hubspot.com/",
            "transport": "streamable_http",
            "_token_fn": access_token_fn,
        }

    def set_zoho_people(self, mcp_url: str, access_token_fn):
        """access_token_fn: sync or async callable that returns the current token."""
        self._configs["zoho_people"] = {
            "url": mcp_url,
            "transport": "streamable_http",
            "_token_fn": access_token_fn,
        }

    def remove(self, name: str):
        self._configs.pop(name, None)

    def is_connected(self, name: str) -> bool:
        return name in self._configs

    def get_active_names(self) -> list[str]:
        return list(self._configs.keys())

    def get_active_clients(self) -> dict:
        return {name: self for name in self._configs}

    async def get_tools(self, agent_names: list[str] | None = None) -> list:
        names = agent_names or list(self._configs.keys())
        configs = {n: self._configs[n] for n in names if n in self._configs}
        if not configs:
            return []

        # Build resolved configs with plain-dict headers (await async token fns)
        resolved = {}
        for name, cfg in configs.items():
            token_fn = cfg.get("_token_fn")
            if token_fn:
                if inspect.iscoroutinefunction(token_fn):
                    token = await token_fn()
                else:
                    token = token_fn()
                headers = {"Authorization": f"Bearer {token}"}
            else:
                headers = {}

            resolved[name] = {
                "url": cfg["url"],
                "transport": cfg["transport"],
                "headers": headers,
            }

        client = MultiServerMCPClient(resolved)
        return await client.get_tools()