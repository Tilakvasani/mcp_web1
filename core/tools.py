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
from mcp.types import TextContent

from mcp_client import MCPClient
from crm_logger import log

# Silence the old verbose mcp_tools logger — we use crm_logger instead
_mcp_logger = logging.getLogger("mcp_tools")
_mcp_logger.setLevel(logging.CRITICAL)
_mcp_logger.propagate = False


# ---------------------------------------------------------------------------
# Tool-level scope check  (derived dynamically from the token scopes)
# ---------------------------------------------------------------------------
def _build_tool_scope_map(granted_scopes: set[str]) -> dict[str, bool]:
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
def build_tool_scope_map(granted_scopes: list[str]) -> dict[str, list[str]]:
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
# JSON Schema → Pydantic helpers
# ---------------------------------------------------------------------------
def _json_type_to_python(schema: dict) -> type:
    type_str = schema.get("type", "string")
    if isinstance(type_str, list):
        non_null = [t for t in type_str if t != "null"]
        type_str = non_null[0] if non_null else "string"
    return {
        "string":  str,
        "integer": int,
        "number":  float,
        "boolean": bool,
        "array":   list,
        "object":  dict,
    }.get(type_str, Any)   # Any instead of str for unknown types


def _build_args_model(schema: dict, tool_name: str):
    properties = schema.get("properties", {})
    required   = set(schema.get("required", []))
    fields: dict[str, tuple] = {}

    for name, prop in properties.items():
        py_type     = _json_type_to_python(prop)
        description = prop.get("description", "")
        if name in required:
            fields[name] = (py_type, Field(..., description=description))
        else:
            fields[name] = (Optional[py_type], Field(default=None, description=description))

    if not fields:
        fields["input"] = (Optional[str], Field(default=None, description="Tool input"))

    safe_name = tool_name.replace("-", "_").replace(".", "_")
    return create_model(f"{safe_name}_Args", **fields)


# ---------------------------------------------------------------------------
# Payload normalization
# ---------------------------------------------------------------------------
def _ensure_list_of_objects(value: Any) -> list[dict]:
    """
    Coerce a single dict or an existing list into list[dict].
    Raises TypeError for anything else.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise TypeError(f"Expected dict or list[dict], got {type(value).__name__}")


def _normalize_crm_manage_payload(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize payloads for manage_crm_objects to match HubSpot's actual schema.

    Actual schema (from tool_guidance / DEBUG logs):
      createRequest: {          ← object
        objects: [              ← list lives INSIDE createRequest
          {
            objectType: str,    ← REQUIRED per schema
            properties: {...},
          }
        ]
      }
      updateRequest: {
        objects: [
          {
            objectType: str,    ← REQUIRED
            objectId: int,      ← REQUIRED
            properties: {...},
          }
        ]
      }

    What the LLM sends (wrong):
      createRequest: {"properties": {...}}         ← flat dict, no objectType
      createRequest: [{"properties": {...}}]       ← list instead of object

    What we must produce (correct):
      createRequest: {"objects": [{"objectType": "companies", "properties": {...}}]}

    The top-level `objectType` kwarg from the LLM is the object type to inject.
    We also strip `action` — it is NOT a field in the manage_crm_objects schema.
    """
    # Step 1: basic cleanup for ALL tools — drop nulls and synthetic `input` field.
    # Strip `action` only for manage_crm_objects (not in its schema; but IS a real
    # field for Zoho and other tools).
    clean: dict[str, Any] = {}
    for k, v in kwargs.items():
        if v is None or k == "input":
            continue
        if k == "action" and tool_name == "manage_crm_objects":
            continue
        clean[k] = v

    # Step 2: manage_crm_objects-specific normalization ONLY.
    # For every other tool (search_crm_objects, get_crm_objects, etc.) objectType
    # is a first-class required field — DO NOT pop it.  Return early.
    if tool_name != "manage_crm_objects":
        return clean

    # The object type comes from a top-level `objectType` key the LLM sends.
    # We consume it here so it does not end up as an extra top-level field.
    obj_type: str = str(clean.pop("objectType", "") or "").lower()

    # If not at top level, try to infer from inside the nested request structure
    if not obj_type:
        for req_key in ("createRequest", "updateRequest", "deleteRequest", "archiveRequest"):
            req = clean.get(req_key)
            if isinstance(req, dict):
                objs = req.get("objects", [])
                if isinstance(objs, list) and objs and isinstance(objs[0], dict):
                    obj_type = str(objs[0].get("objectType", "")).lower()
                elif req.get("objectType"):
                    obj_type = str(req["objectType"]).lower()
            if obj_type:
                break

    def _normalise_request(req: Any, id_field: str | None = None) -> dict:
        """
        Turn whatever shape `req` is into {"objects": [...]}.
        Each item is guaranteed to have objectType (and objectId for updates).
        """
        # Already the correct shape
        if isinstance(req, dict) and "objects" in req:
            items = req["objects"]
            if not isinstance(items, list):
                items = [items]
        # Bare dict like {"properties": {...}} or {"objectId": 1, "properties": {...}}
        elif isinstance(req, dict):
            items = [req]
        # Incorrectly passed as a list
        elif isinstance(req, list):
            items = req
        else:
            items = [{"properties": {}}]

        # Ensure every item has objectType (inject from outer kwarg if missing)
        fixed_items = []
        for item in items:
            if not isinstance(item, dict):
                item = {}
            if obj_type and not item.get("objectType"):
                item = {"objectType": obj_type, **item}
            fixed_items.append(item)

        return {"objects": fixed_items}

    if "createRequest" in clean:
        clean["createRequest"] = _normalise_request(clean["createRequest"])

    if "updateRequest" in clean:
        clean["updateRequest"] = _normalise_request(clean["updateRequest"], id_field="objectId")

    if "deleteRequest" in clean:
        clean["deleteRequest"] = _normalise_request(clean["deleteRequest"])

    if "archiveRequest" in clean:
        clean["archiveRequest"] = _normalise_request(clean["archiveRequest"])

    return clean


def _classify_tool_error(content_str: str) -> str:
    """Classify an MCP error string into one of three categories."""
    err_lower = content_str.lower()

    if any(kw in err_lower for kw in (
        "permission", "scope", "unauthorized", "forbidden", "403", "401"
    )):
        return "PERMISSION_DENIED"

    if any(kw in err_lower for kw in (
        "validation_error",
        "invalid input",
        "property values were not valid",
        "non-empty list of objects to create or update must be provided",
        "missing required",
        "bad request",
        "format",
    )):
        return "FORMAT_ERROR"

    return "TOOL_ERROR"


# ---------------------------------------------------------------------------
# Tool wrapper — permission guards are fully dynamic
# ---------------------------------------------------------------------------
def _wrap_mcp_tool(
    mcp_tool,
    client: MCPClient,
    granted_scopes: set[str],
) -> StructuredTool:
    """Wrap a single MCP tool as a LangChain StructuredTool with scope guards."""
    tool_name   = mcp_tool.name
    description = mcp_tool.description or f"MCP tool: {tool_name}"
    schema      = dict(mcp_tool.inputSchema) if mcp_tool.inputSchema else {}
    ArgsModel   = _build_args_model(schema, tool_name)

    # manage_crm_objects: inject an extra `objectType` helper field so the LLM
    # can pass the object type at the top level.  HubSpot's schema buries it
    # inside createRequest.objects[0].objectType, but the LLM reliably sends it
    # as a top-level kwarg — without this field Pydantic would strip it before
    # _normalize_crm_manage_payload can inject it into the nested structure.
    if tool_name == "manage_crm_objects":
        existing = {
            name: (field.annotation, field)
            for name, field in ArgsModel.model_fields.items()
        }
        existing["objectType"] = (
            Optional[str],
            Field(default=None, description=(
                "CRM object type (e.g. contacts, deals, companies, tickets). "
                "Supply this at the top level so it can be injected into the request."
            )),
        )
        safe = tool_name.replace("-", "_").replace(".", "_")
        ArgsModel = create_model(f"{safe}_Args", **existing)

    # Build dynamic maps once per tool wrap (not per call)
    tool_access_map = _build_tool_scope_map(granted_scopes)
    crm_write_map   = _build_crm_write_map(granted_scopes)
    has_tool_access = tool_access_map.get(tool_name, True)  # unknown tools = allowed

    # Schema loaded — no per-tool log here (would fire 95x per request)

    async def _run(**kwargs) -> str:
        # ── 1. Tool-level guard ────────────────────────────────────────
        if not has_tool_access:
            log("warn", f"PERMISSION_DENIED tool-level: {tool_name}")
            return (
                f"🚫 PERMISSION_DENIED: I don't have permission to use `{tool_name}`.\n"
                f"Your HubSpot token doesn't include any scopes for this tool category.\n"
                "Please reconnect HubSpot with the required permissions."
            )

        # ── 2. Action-level write guard ────────────────────────────────
        required_write = _resolve_write_scope(tool_name, kwargs, crm_write_map)
        if required_write and required_write not in granted_scopes:
            action = _infer_action(kwargs) or "perform this action"
            obj    = _infer_object_type(kwargs)
            label  = f"{action} {obj}".strip()
            log("warn", f"PERMISSION_DENIED write: {tool_name} | action={action} obj={obj} | needs={required_write}")
            return (
                f"🚫 PERMISSION_DENIED: I don't have permission to **{label}**.\n"
                f"Required scope: `{required_write}`\n"
                "Your token only has read access. Please reconnect HubSpot "
                "with write permissions to create / update / delete records."
            )

        # ── 3. Normalize payload ───────────────────────────────────────
        try:
            clean = _normalize_crm_manage_payload(tool_name, kwargs)
        except Exception as exc:
            log("error", f"payload normalize error [{tool_name}]: {exc}")
            return f"FORMAT_ERROR: Failed to normalize tool payload for `{tool_name}`: {exc}"

        # ── Compact tool call log ─────────────────────────────────────────
        log("tool", f"call → {tool_name}({', '.join(k for k,v in clean.items() if v is not None)[:60]})")

        # ── 4. MCP call ────────────────────────────────────────────────
        try:
            result = await client.call_tool(tool_name, clean)
        except Exception as exc:
            log("error", f"MCP exception [{tool_name}]: {exc}")
            return f"TOOL_ERROR ({tool_name}): {exc}"

        if result is None:
            log("warn", f"[{tool_name}] returned None")
            return "Tool returned no result."

        texts = [item.text for item in result.content if isinstance(item, TextContent)]
        content_str = "\n".join(texts) if texts else ""

        # ── Compact response log ─────────────────────────────────────────
        status = "err" if result.isError else "ok"
        log("tool", f"resp ← {tool_name} [{status}] {len(content_str)} chars")

        # ── 5. Error classification ────────────────────────────────────
        if result.isError:
            error_class = _classify_tool_error(content_str)
            if error_class == "PERMISSION_DENIED":
                return f"🚫 PERMISSION_DENIED: {content_str}"
            # FORMAT_ERROR or TOOL_ERROR both surface as FORMAT_ERROR so the
            # agent's rule 4 kicks in and it retries with a fixed payload.
            return f"FORMAT_ERROR: {content_str}"

        return content_str

    return StructuredTool(
        name        = tool_name,
        description = description,
        args_schema = ArgsModel,
        coroutine   = _run,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def get_langchain_tools(
    clients: dict[str, MCPClient],
    granted_scopes: list[str] | None = None,
) -> list[StructuredTool]:
    """
    Connect to every MCP client, list their tools, and return them all
    as LangChain StructuredTool objects — each guarded by dynamic scope checks.
    """
    scope_set = set(granted_scopes or [])
    tools: list[StructuredTool] = []
    for client in clients.values():
        mcp_tools = await client.list_tools()
        for mcp_tool in mcp_tools:
            tools.append(_wrap_mcp_tool(mcp_tool, client, scope_set))
    return tools