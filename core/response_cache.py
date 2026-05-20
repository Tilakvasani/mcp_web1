"""
core/response_cache.py
======================
In-memory TTL cache for LLM responses.

Rules:
  - Read queries cached for 5 minutes
  - Write / mutate queries NEVER cached (create, update, delete, etc.)
  - Cache is per (session_id + agent + normalised message)
  - Session clear called on disconnect / logout
"""

import time
import hashlib
from crm_logger import log

_CACHE: dict[str, tuple[str, float]] = {}
_TTL   = 300  # 5 minutes

import re

# Any message containing these words → treat as write → never cache
_WRITE_TOKENS = {
    "create", "add", "new", "make", "insert", "post",
    "update", "edit", "change", "modify", "patch", "upsert",
    "delete", "remove", "archive", "cancel", "close", "purge",
    "send", "schedule", "assign", "convert", "merge", "move",
    "yes", "ok", "sure", "confirm", "go ahead", "do it", "proceed",
    "approve", "reject", "submit",
}


def _is_write(message: str) -> bool:
    """
    Check if the user message indicates a write/mutation action.
    Cleans punctuation using regular expressions to ensure words like
    'create-deal' or 'update!' are correctly matched against write tokens.
    """
    # Replace non-alphanumeric characters with spaces to handle punctuation/hyphens cleanly
    clean_msg = re.sub(r'[^a-z0-9\s]', ' ', message.lower())
    words = set(clean_msg.split())
    return bool(words & _WRITE_TOKENS)


def _key(message: str, agent: str, session_id: str) -> str:
    raw = f"{session_id}::{agent}::{message.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


def get_cached(message: str, agent: str, session_id: str) -> str | None:
    """Return cached response or None if not found / expired / write query."""
    if _is_write(message):
        return None
    k     = _key(message, agent, session_id)
    entry = _CACHE.get(k)
    if not entry:
        return None
    response, ts = entry
    if time.time() - ts > _TTL:
        del _CACHE[k]
        return None
    log("cache", f"HIT  '{message[:50]}'")
    return response


def set_cached(message: str, agent: str, session_id: str, response: str):
    """Cache a response. Silently skips write queries."""
    if _is_write(message) or not response:
        return
    k = _key(message, agent, session_id)
    _CACHE[k] = (response, time.time())
    _evict()
    log("cache", f"SET  '{message[:50]}'")


def clear_session(session_id: str):
    """Remove all cached responses for a session (on disconnect / logout)."""
    keys = [k for k in list(_CACHE.keys())]
    # We can't filter by session_id from hash alone — clear all on disconnect
    # This is safe: single-user app, disconnect = fresh start
    _CACHE.clear()
    log("cache", f"cleared for session={session_id[:8]}")


def cache_stats() -> dict:
    now   = time.time()
    alive = {k: v for k, v in _CACHE.items() if now - v[1] <= _TTL}
    return {"entries": len(alive), "ttl_seconds": _TTL}


def _evict():
    now   = time.time()
    stale = [k for k, (_, ts) in _CACHE.items() if now - ts > _TTL]
    for k in stale:
        del _CACHE[k]
