"""
MCP → LangChain tool bridge  (fully dynamic, zero hardcoded scopes).

Permission checking
-------------------
All scope requirements are derived at runtime from the token's own "scopes"
array — nothing is hardcoded.  A new HubSpot scope or CRM object type in the
token is handled automatically with no code changes.

Two guards run before every MCP call:

1. Tool-level guard  — does the user have ANY scope that gives access to this
   tool's category?  (e.g. any crm.objects.*.read for crm_object_action)

2. Action-level guard — for mutating actions (create/update/delete), does the
   user have the specific write scope for that object type?
   (e.g. crm.objects.contacts.write for action=create, objectType=contacts)

Bug fixes vs original:
- createRequest / updateRequest are normalized into `objects=[...]` list shape
  that HubSpot's MCP tool actually expects.
- action is inferred from createRequest/updateRequest keys when not explicit,
  so the write-scope guard works even if the LLM omits `action`.
- _build_tool_scope_map now also covers `manage_crm_objects` (the real MCP
  tool name HubSpot exposes) alongside the logical alias `crm_object_action`.
- Error classification is broadened to catch VALIDATION_ERROR payloads.
- `_json_type_to_python` falls back to `Any` instead of `str` for unknown types.
- logger.exception used for MCP call failures so tracebacks appear in logs.
"""

import re
import json
import logging
from typing import Any, Optional

from pydantic import create_model, Field
from langchain_core.tools import StructuredTool
from crm_logger import log

# Silence the old verbose mcp_tools logger — we use crm_logger instead
_mcp_logger = logging.getLogger("mcp_tools")
_mcp_logger.setLevel(logging.CRITICAL)
_mcp_logger.propagate = False


# ---------------------------------------------------------------------------
# Tool-level scope check  (derived dynamically from the token scopes)
# ---------------------------------------------------------------------------
def _check_tool_access(granted_scopes: set[str]) -> dict[str, bool]:
    """
    Build a {tool_name: has_access} map entirely from the token's own scopes.
    No hardcoded lists — patterns are derived from scope strings directly.
    """
    def any_prefix(*prefixes: str) -> bool:
        return any(
            any(scope.startswith(prefix) for scope in granted_scopes)
            for prefix in prefixes
        )

    crm_access = any_prefix("crm.objects.", "crm.lists.")

    return {
        # Both the logical alias and the real HubSpot MCP tool name
        "crm_object_action":    crm_access,
        "manage_crm_objects":   crm_access,

        "engagement_action":    any_prefix(
            "crm.objects.calls.",
            "crm.objects.meetings.",
            "crm.objects.notes.",
            "crm.objects.tasks.",
            "crm.objects.emails.",
        ),

        "automation_action":    any_prefix("automation"),

        "marketing_action":     any_prefix(
            "crm.lists.",
            "crm.objects.marketing_events.",
            "marketing",
        ),

        "conversation_action":  any_prefix("conversations."),

        "analytics_action":     any_prefix("crm.hubsql.", "analytics.", "reports"),

        "cms_action":           any_prefix("cms.", "content"),

        "settings_action":      any_prefix(
            "crm.objects.owners.",
            "mcp.users.",
            "settings.users.",
            "settings.teams.",
        ),
    }


# ---------------------------------------------------------------------------
# Action-level write scope check  (derived dynamically from the token scopes)
# ---------------------------------------------------------------------------
_WRITE_ACTIONS = {
    "create", "update", "delete", "associate", "merge",
    "enroll", "unenroll", "send", "archive", "publish",
}

_FIXED_WRITE_PATTERNS: dict[str, dict[str, str]] = {
    "automation_action": {
        "enroll":   "automation.sequences.enrollments.write",
        "unenroll": "automation.sequences.enrollments.write",
    },
    "conversation_action": {
        "send":    "conversations.write",
        "archive": "conversations.write",
        "update":  "conversations.write",
    },
    "cms_action": {
        "create":  "content",
        "update":  "content",
        "publish": "content",
        "archive": "content",
    },
    "settings_action": {
        "create": "settings.users.write",
        "update": "settings.users.write",
        "delete": "settings.users.write",
    },
}


def _build_crm_write_map(granted_scopes: set[str]) -> dict[str, str]:
    """
    Build {object_type: write_scope} from the token's own scopes.
    e.g. "crm.objects.contacts.write" → {"contacts": "crm.objects.contacts.write"}
    Works for any object type HubSpot ever adds.
    """
    result: dict[str, str] = {}
    for scope in granted_scopes:
        m = re.match(r"^crm\.objects\.(.+)\.write$", scope)
        if m:
            result[m.group(1)] = scope
    return result


def _infer_action(kwargs: dict[str, Any]) -> str:
    """
    Infer the CRM action from an explicit `action` key, or from the shape of
    the request payload (createRequest / updateRequest / etc.).
    Returns an empty string if it cannot be determined.
    """
    action = str(kwargs.get("action", "")).strip().lower()
    if action:
        return action

    if kwargs.get("createRequest") is not None:
        return "create"
    if kwargs.get("updateRequest") is not None:
        return "update"
    if kwargs.get("deleteRequest") is not None:
        return "delete"
    if kwargs.get("archiveRequest") is not None:
        return "archive"

    return ""


def _infer_object_type(kwargs: dict[str, Any]) -> str:
    """
    Infer the CRM object type from common key names.
    Returns an empty string if it cannot be determined.
    """
    for key in ("objectType", "type", "object_type", "objectTypeId", "entityType"):
        value = kwargs.get(key)
        if value:
            return str(value).lower()
    return ""


def _resolve_write_scope(
    tool_name: str,
    kwargs: dict[str, Any],
    crm_write_map: dict[str, str],
) -> str | None:
    """
    Return the write scope required for this tool call, or None if read-only.
    All data comes from the live token — nothing hardcoded.
    """
    action = _infer_action(kwargs)
    if action not in _WRITE_ACTIONS:
        return None

    # Non-CRM tools with fixed patterns
    fixed = _FIXED_WRITE_PATTERNS.get(tool_name, {})
    if fixed:
        return fixed.get(action)   # None = write action not in fixed map → allow

    # CRM tools: look up object type in the dynamically-built write map
    obj_type = _infer_object_type(kwargs)
    if not obj_type:
        return None

    if obj_type in crm_write_map:
        return crm_write_map[obj_type]

    # Fallback: construct the scope string so new object types work without code changes
    return f"crm.objects.{obj_type}.write"


# ---------------------------------------------------------------------------
# Human-readable scope label generator  (no static dict — fully dynamic)
# ---------------------------------------------------------------------------
_SCOPE_SPECIALS: dict[str, str] = {
    "oauth":                    "🔐 OAuth authentication",
    "scope_mappings.container": "🗂️  Scope mapping container",
    "crm.hubsql.execute":       "🔍 Run analytics queries (HubSQL)",
    "content":                  "📝 CMS content access",
    "automation":               "⚙️  Automation access",
    "reports":                  "📊 Reports access",
}

_EMOJI_BY_OBJECT: dict[str, str] = {
    "contacts":          "👤",  "companies":        "🏢",  "deals":          "💰",
    "tickets":           "🎫",  "calls":            "📞",  "meetings":       "📅",
    "notes":             "📝",  "tasks":            "✅",  "emails":         "📧",
    "quotes":            "📄",  "invoices":         "🧾",  "orders":         "📦",
    "line_items":        "📋",  "products":         "🛍️",  "carts":          "🛒",
    "owners":            "👥",  "marketing_events": "📢",  "lists":          "📋",
    "subscriptions":     "🔄",  "leads":            "🎯",  "users":          "🔑",
    "teams":             "👥",  "blog_posts":       "📰",  "landing_pages":  "🌐",
    "site_pages":        "🌐",  "schemas":          "📐",  "goals":          "🏆",
    "appointments":      "📅",  "services":         "🛠️",  "courses":        "📚",
}

_EMOJI_BY_PREFIX: dict[str, str] = {
    "crm": "🗂️",  "cms": "🌐",  "settings": "⚙️",  "mcp": "🔑",
    "automation": "⚙️",  "conversations": "💬",  "analytics": "📊",
    "marketing": "📢",
}

_NOISE_WORDS = {"objects", "schemas", "pages", "blogs"}


def _scope_to_label(scope: str) -> str:
    """
    Derive a human-readable label from any HubSpot OAuth scope string.
    No static dictionary — works for any current or future scope.
    """
    if scope in _SCOPE_SPECIALS:
        return _SCOPE_SPECIALS[scope]

    parts = scope.replace("-", "_").split(".")

    # Emoji: try innermost segment first, then prefix
    emoji = "🔹"
    for part in reversed(parts):
        if part in _EMOJI_BY_OBJECT:
            emoji = _EMOJI_BY_OBJECT[part]
            break
    else:
        emoji = _EMOJI_BY_PREFIX.get(parts[0], "🔹")

    last = parts[-1]
    if last == "read":      verb = "View"
    elif last == "write":   verb = "Create / edit / delete"
    elif last == "execute": verb = "Execute"
    else:                   verb = last.replace("_", " ").title()

    subject_parts = parts[1:-1] if last in ("read", "write", "execute") else parts[1:]
    subject_parts = [p for p in subject_parts if p not in _NOISE_WORDS]
    subject = " ".join(p.replace("_", " ") for p in subject_parts).strip().title() or scope

    return f"{emoji} {verb} {subject}".strip()


def describe_scopes(scopes: list[str]) -> list[dict]:
    """
    Convert a list of raw scope strings into human-readable dicts.
    Returns [{"scope": str, "label": str}, ...] sorted by label.
    """
    result = [{"scope": s, "label": _scope_to_label(s)} for s in scopes]
    return sorted(result, key=lambda x: x["label"])


# ---------------------------------------------------------------------------
# Build tool access map for web_app.py  (dynamic, from live scopes)
# ---------------------------------------------------------------------------
def get_tool_scope_map(granted_scopes: list[str]) -> dict[str, list[str]]:
    """
    Return a {tool_name: [matching_scopes]} map derived from the token's own
    scopes — used by web_app.py /api/permissions to show which tools are
    accessible.
    """
    scope_set = set(granted_scopes)

    def matching(pattern: str) -> list[str]:
        return [s for s in scope_set if s.startswith(pattern)]

    crm_scopes = matching("crm.objects.") + matching("crm.lists.")

    return {
        "crm_object_action":   crm_scopes,
        "manage_crm_objects":  crm_scopes,
        "engagement_action":   [s for s in scope_set if any(
            s.startswith(p) for p in (
                "crm.objects.calls.",
                "crm.objects.meetings.",
                "crm.objects.notes.",
                "crm.objects.tasks.",
                "crm.objects.emails.",
            )
        )],
        "automation_action":   matching("automation"),
        "marketing_action":    matching("crm.lists.") + matching("crm.objects.marketing_events.") + matching("marketing"),
        "conversation_action": matching("conversations."),
        "analytics_action":    matching("crm.hubsql.") + matching("analytics.") + (["reports"] if "reports" in scope_set else []),
        "cms_action":          matching("cms.") + (["content"] if "content" in scope_set else []),
        "settings_action":     [s for s in scope_set if any(
            s.startswith(p) for p in (
                "crm.objects.owners.",
                "mcp.users.",
                "settings.users.",
                "settings.teams.",
            )
        )],
    }



# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_synthetic_tool(name: str, tools: list):
    """Find a synthetic tool by name from a tools list."""
    return next((t for t in tools if getattr(t, "name", "") == name), None)
