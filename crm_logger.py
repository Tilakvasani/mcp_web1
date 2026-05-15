"""
crm_logger.py — Compact one-line emoji logger for Multi-CRM AI Agent
---------------------------------------------------------------------
Usage:
    from crm_logger import log, suppress_noisy_libs

    log("chat",    "user → 'Show all deals'")
    log("tool",    "search_crm_objects called")
    log("ok",      "HubSpot connected (pool hit)")
    log("error",   "preflight failed: 401")
    log("info",    "API ready on :8000")

suppress_noisy_libs()   # call once at startup to silence MCP / httpx DEBUG spam
"""

import logging
import sys
from datetime import datetime

# ── emoji map ───────────────────────────────────────────────────────────────
_ICONS = {
    "chat":    "💬",
    "user":    "👤",
    "ai":      "🤖",
    "tool":    "🛠️ ",
    "ok":      "✅",
    "error":   "❌",
    "warn":    "⚠️ ",
    "info":    "ℹ️ ",
    "connect": "🔌",
    "pool":    "♻️ ",
    "stream":  "📡",
    "oauth":   "🔑",
    "boot":    "🚀",
    "bye":     "👋",
    "ping":    "🏓",
    "schema":  "📋",
    "debug":   "🐛",
    "time":    "⏱️ ",
}

_LABELS = {
    "chat":    "CHAT",
    "user":    "USER",
    "ai":      "AI  ",
    "tool":    "TOOL",
    "ok":      "OK  ",
    "error":   "ERR ",
    "warn":    "WARN",
    "info":    "INFO",
    "connect": "MCP ",
    "pool":    "POOL",
    "stream":  "SSE ",
    "oauth":   "AUTH",
    "boot":    "BOOT",
    "bye":     "BYE ",
    "ping":    "PING",
    "schema":  "SCHM",
    "debug":   "DBG ",
    "time":    "TIME",
}


def log(kind: str, msg: str, source: str = "backend") -> None:
    """Print a single compact log line to stdout."""
    now   = datetime.now().strftime("%H:%M:%S")
    icon  = _ICONS.get(kind, "·")
    label = _LABELS.get(kind, kind.upper()[:4])
    src   = f"[{source.upper()[:2]}]" if source else ""
    print(f"{now} {icon} {label} {src} {msg}", flush=True)


def suppress_noisy_libs() -> None:
    """
    Silence the verbose DEBUG spam from mcp, httpx, httpcore, and uvicorn access logs.
    Keeps WARNING+ from those libs so real errors still surface.
    """
    for name in (
        "mcp",
        "mcp.client",
        "mcp.client.sse",
        "mcp.client.streamable_http",
        "httpx",
        "httpcore",
        "httpcore.connection",
        "httpcore.http11",
        "asyncio",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Replace uvicorn access log with our own (handled in web_app.py middleware)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Root logger: only WARNING and above from third-party libs
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
