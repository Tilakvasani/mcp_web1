"""
core/pre_router.py
==================
Fast rule-based filter that answers simple messages BEFORE hitting the LLM.

Handles:
  - Greetings     → welcome message (zero LLM cost)
  - Help queries  → capability overview (zero LLM cost)
  - Off-topic     → polite redirect (zero LLM cost)

Returns None for everything else → message goes to LLM as normal.
"""

import re

# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

_GREETING = re.compile(
    r"^(hi+|hello+|hey+|howdy|hiya|yo|sup|"
    r"good\s+(morning|afternoon|evening|day)|"
    r"what'?s\s+up|how\s+are\s+you)[!?.,\s]*$",
    re.IGNORECASE,
)

_HELP = re.compile(
    r"\b(what can you do|help me|how do (i|you)|what do you support|"
    r"capabilities|features|commands|show me what|what (are|is) your|"
    r"what can i ask|guide me|get started)\b",
    re.IGNORECASE,
)

_OFF_TOPIC = re.compile(
    r"\b(weather|news today|joke|poem|recipe|sports score|"
    r"movie|music|song lyrics|stock price|bitcoin|crypto price|"
    r"write me a story|tell me a story|capital of|population of|"
    r"translate|what is [0-9]+ \+)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Replies
# ─────────────────────────────────────────────────────────────────────────────

_GREETING_REPLY = """\
👋 Hey! I'm your business AI assistant.

I can help you with:

**🟠 HubSpot CRM**
Deals · Contacts · Companies · Tickets · Tasks · Meetings · Emails · Pipelines

**🟢 Zoho People HRMS**
Employees · Leave · Attendance · Departments · Payroll · Shifts

**⚡ Cross-system queries**
*"Which sales reps are on leave today?"*
*"Show tickets assigned to absent employees"*

What would you like to do?
"""

_HELP_REPLY = """\
Here's everything I can do:

**🟠 HubSpot CRM**
- Show, search, create, update, delete: Deals · Contacts · Companies · Tickets
- View pipeline stages, tasks, meetings, emails, notes, owners
- Run HubSQL analytics queries
- Manage workflows, lists, properties

**🟢 Zoho People HRMS**
- Employee profiles and department listings
- Leave requests — who's on leave, approve/reject
- Attendance records and shift schedules
- Payroll and appraisal data

**⚡ Cross-system**
- *"Which sales reps are on leave today?"* → HubSpot + Zoho People
- *"Show support tickets assigned to absent employees"*
- *"Compare active deals with employee headcount per department"*

Just ask in plain English — I'll figure out which system to use automatically.
"""

_OFF_TOPIC_REPLY = (
    "I'm focused on HubSpot CRM and Zoho People HRMS queries — "
    "that topic is outside my scope. "
    "Ask me about your deals, contacts, employees, or HR data and I'll get right on it! 🚀"
)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def pre_route(message: str) -> str | None:
    """
    Returns an instant reply string if the message doesn't need an LLM call.
    Returns None if the message should proceed to the LLM normally.
    """
    s = message.strip()
    if _GREETING.match(s):
        return _GREETING_REPLY
    if _HELP.search(s):
        return _HELP_REPLY
    if _OFF_TOPIC.search(s):
        return _OFF_TOPIC_REPLY
    return None
