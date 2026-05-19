"""
core/logger.py
==============
Simple coloured console logger used across the whole app.
"""

from datetime import datetime

_ICONS = {
    "ok":    "✅",
    "error": "❌",
    "warn":  "⚠️",
    "oauth": "🔑",
    "rag":   "🔍",
    "route": "🗺️",
    "agent": "🤖",
    "mcp":   "🔌",
    "bye":   "👋",
    "info":  "ℹ️",
    "tool":  "🔧",
    "cache": "📦",
}


def log(kind: str, msg: str) -> None:
    icon = _ICONS.get(kind, "•")
    ts   = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {icon}  {msg}")
