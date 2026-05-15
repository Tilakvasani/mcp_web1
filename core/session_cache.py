"""
Session-based Tool Cache
========================
Loads all MCP tools ONCE per session, then filters to relevant tools
per message using intent detection. Evicts automatically after TTL.

Usage
-----
  cache = SessionToolCache()

  # On each chat message:
  tools = await cache.get_or_load(session_id, agent, crm_clients, scopes)
  filtered = cache.filter_for_message(message, tools)

  # When session ends (user disconnects / logout):
  cache.evict(session_id)

  # Background: stale sessions auto-evict after SESSION_TTL seconds of inactivity.
"""

import time
import asyncio
import re
from dataclasses import dataclass, field

from crm_logger import log
from core.tools import get_langchain_tools


# =============================================================================
# Config
# =============================================================================

SESSION_TTL = 600   # seconds of inactivity before auto-evict (10 min)


# =============================================================================
# Intent → module keyword map
# =============================================================================
# Keys are words the user might write. Values are substrings that must appear
# in the tool name for the tool to be included.
# Order matters: more specific entries first.

_INTENT_MAP: list[tuple[list[str], list[str]]] = [
    # keywords in user message        → tool name substrings to keep
    (["lead"],                         ["Lead"]),
    (["deal", "opportunity"],          ["Deal"]),
    (["account", "company"],           ["Account"]),
    (["contact"],                      ["Contact"]),
    (["convert"],                      ["Lead", "Contact", "Account", "Deal", "convert"]),
    (["workflow"],                     ["Workflow", "workflow"]),
    (["email"],                        ["Email", "email"]),
    (["territory"],                    ["Territory"]),
    (["blueprint"],                    ["Blueprint"]),
    (["connection"],                   ["Connection"]),
    (["configuration", "config"],      ["Configuration"]),
]

# Tools that are always included regardless of intent (utility / meta tools)
_ALWAYS_INCLUDE: list[str] = [
    "getModules", "getFields", "getMetaFields",  # Zoho field discovery
    "get_properties", "get_object_types",         # HubSpot field discovery
]


def filter_tools_by_intent(message: str, tools: list) -> list:
    """
    Given a user message and the full tool list, return only the tools
    relevant to the detected intent.

    Falls back to ALL tools if no intent is matched (safe default).
    """
    msg_lower = message.lower()

    matched_modules: list[str] = []
    for keywords, modules in _INTENT_MAP:
        if any(kw in msg_lower for kw in keywords):
            matched_modules.extend(modules)

    # No match — return everything (safe fallback, happens on vague queries)
    if not matched_modules:
        log("cache", f"no intent match for '{message[:40]}' — using all {len(tools)} tools")
        return tools

    filtered = [
        t for t in tools
        if (
            any(mod in t.name for mod in matched_modules)
            or any(always in t.name for always in _ALWAYS_INCLUDE)
        )
    ]

    # Safety: if filter is too aggressive and leaves nothing, return all
    if not filtered:
        return tools

    log("cache", f"intent filter '{message[:40]}' → {len(filtered)}/{len(tools)} tools")
    return filtered


# =============================================================================
# Cache entry
# =============================================================================

@dataclass
class _CacheEntry:
    tools:      list
    agent:      str
    created_at: float = field(default_factory=time.time)
    last_used:  float = field(default_factory=time.time)

    def touch(self):
        self.last_used = time.time()

    def is_stale(self) -> bool:
        return (time.time() - self.last_used) > SESSION_TTL

    def age(self) -> float:
        return time.time() - self.created_at


# =============================================================================
# SessionToolCache
# =============================================================================

class SessionToolCache:
    """
    Per-session tool cache. Thread-safe via asyncio.Lock.

    - get_or_load()  : return cached tools or load fresh from MCP
    - filter_for_message() : intent-based subset for a single message
    - evict()        : explicit eviction (session end)
    - evict_stale()  : TTL-based eviction (call from background task)
    """

    def __init__(self):
        self._cache: dict[str, _CacheEntry] = {}
        self._lock  = asyncio.Lock()

    async def get_or_load(
        self,
        session_id: str,
        agent: str,
        crm_clients: dict,
        granted_scopes: list[str] | None = None,
    ) -> list:
        """
        Return cached tools for this session, or load them fresh if:
          - session not in cache
          - session cached a different agent
          - cache entry is stale
        """
        async with self._lock:
            entry = self._cache.get(session_id)

        # Cache HIT — same agent, not stale
        if entry and entry.agent == agent and not entry.is_stale():
            entry.touch()
            log("cache", f"HIT session={session_id[:8]} agent={agent} tools={len(entry.tools)}")
            return entry.tools

        # Cache MISS or stale — load fresh
        reason = "MISS" if not entry else ("STALE" if entry.is_stale() else "AGENT_CHANGE")
        log("cache", f"{reason} session={session_id[:8]} agent={agent} — loading tools…")

        tools = await get_langchain_tools(crm_clients, granted_scopes=granted_scopes or [])

        async with self._lock:
            self._cache[session_id] = _CacheEntry(tools=tools, agent=agent)

        log("cache", f"STORED session={session_id[:8]} — {len(tools)} tools cached")
        return tools

    def filter_for_message(self, message: str, tools: list) -> list:
        """Intent-filter a tool list for a single message. Stateless."""
        return filter_tools_by_intent(message, tools)

    async def evict(self, session_id: str):
        """Explicitly remove a session (call on disconnect / logout)."""
        async with self._lock:
            entry = self._cache.pop(session_id, None)
        if entry:
            log("cache", f"EVICT session={session_id[:8]} (age {entry.age():.0f}s)")

    async def evict_stale(self):
        """Remove all sessions that have exceeded SESSION_TTL. Call periodically."""
        async with self._lock:
            stale = [sid for sid, e in self._cache.items() if e.is_stale()]
            for sid in stale:
                del self._cache[sid]
        if stale:
            log("cache", f"TTL evicted {len(stale)} stale session(s)")

    def stats(self) -> dict:
        return {
            "sessions": len(self._cache),
            "entries": [
                {
                    "session": sid[:8],
                    "agent":   e.agent,
                    "tools":   len(e.tools),
                    "age_s":   round(e.age()),
                    "idle_s":  round(time.time() - e.last_used),
                }
                for sid, e in self._cache.items()
            ],
        }
