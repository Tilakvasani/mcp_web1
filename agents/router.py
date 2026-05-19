"""
agents/router.py
================
Intent router — classifies user messages to the right app/agent.

Returns one of:
  APP_HUBSPOT_CRM      → deals, contacts, companies, pipelines
  APP_HUBSPOT_TICKETS  → support tickets
  APP_ZOHO_HRMS        → employees, leave, attendance, HR
  APP_BOTH             → cross-app queries
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# App constants
# ─────────────────────────────────────────────────────────────────────────────

APP_HUBSPOT_CRM     = "hubspot_crm"
APP_HUBSPOT_TICKETS = "hubspot_tickets"
APP_ZOHO_HRMS       = "zoho_hrms"
APP_BOTH            = "both"


# ─────────────────────────────────────────────────────────────────────────────
# Keyword patterns  (compiled once)
# ─────────────────────────────────────────────────────────────────────────────

_ZOHO_KEYWORDS = re.compile(
    r"\b("
    r"employee|employees|staff|workforce|team member|headcount|"
    r"leave|leaves|vacation|pto|time.?off|sick.?leave|casual.?leave|"
    r"attendance|check.?in|check.?out|clock.?in|clock.?out|"
    r"hr|hrms|human.?resource|payroll|salary|appraisal|"
    r"department|designation|shift|roster|"
    r"zoho|zoho.?people|zoho.?hrms"
    r")\b",
    re.IGNORECASE,
)

_TICKET_KEYWORDS = re.compile(
    r"\b("
    r"ticket|tickets|support.?ticket|help.?desk|"
    r"issue|issues|bug|bugs|"
    r"support|support.?request|case|cases|"
    r"escalat|priorit|sla|resolution|"
    r"open.?ticket|close.?ticket|assign.?ticket"
    r")\b",
    re.IGNORECASE,
)

_HUBSPOT_KEYWORDS = re.compile(
    r"\b("
    r"deal|deals|opportunity|opportunities|pipeline|pipelines|stage|"
    r"contact|contacts|lead|leads|prospect|"
    r"company|companies|account|accounts|"
    r"crm|hubspot|"
    r"note|notes|call|calls|meeting|meetings|task|tasks|email|emails|"
    r"engagement|activity|activities|"
    r"property|properties|owner|owners|"
    r"quote|quotes|invoice|invoices|product|products|line.?item"
    r")\b",
    re.IGNORECASE,
)

_BOTH_KEYWORDS = re.compile(
    r"\b("
    r"both|all.?apps|everything|across|combined|"
    r"hubspot.+zoho|zoho.+hubspot|"
    r"crm.+hr|hr.+crm"
    r")\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

async def detect_intent(message: str) -> str:
    """
    Classify a user message into one of the app constants.

    Priority:
      1. Explicit "both" keywords → APP_BOTH
      2. Zoho HR keywords → APP_ZOHO_HRMS
      3. Ticket keywords → APP_HUBSPOT_TICKETS
      4. HubSpot CRM keywords → APP_HUBSPOT_CRM
      5. Default → APP_HUBSPOT_CRM  (most common use-case)

    If both Zoho and HubSpot keywords are detected → APP_BOTH.
    """
    msg = message.strip()
    if not msg:
        return APP_HUBSPOT_CRM

    # Explicit cross-app request
    if _BOTH_KEYWORDS.search(msg):
        return APP_BOTH

    has_zoho    = bool(_ZOHO_KEYWORDS.search(msg))
    has_ticket  = bool(_TICKET_KEYWORDS.search(msg))
    has_hubspot = bool(_HUBSPOT_KEYWORDS.search(msg))

    # Mixed signals → both
    if has_zoho and (has_hubspot or has_ticket):
        return APP_BOTH

    # Single app
    if has_zoho:
        return APP_ZOHO_HRMS
    if has_ticket:
        return APP_HUBSPOT_TICKETS
    if has_hubspot:
        return APP_HUBSPOT_CRM

    # Default
    return APP_HUBSPOT_CRM