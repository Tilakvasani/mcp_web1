"""
core/memory.py
==============
SQLite-backed local memory store for user preferences and context.
Allows persistent storage of key-value pairs (e.g. user name, department, timezone)
across sessions.
"""

import sqlite3
from pathlib import Path
from crm_logger import log
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

DB_PATH = Path("memory.db")

class SavePreferenceInput(BaseModel):
    key: str = Field(description="The key of the preference or fact to remember (e.g., 'user_name', 'department', 'timezone', 'preferred_view')")
    value: str = Field(description="The value or content of the preference/fact (e.g., 'Tilak', 'Sales', 'EST', 'deals closing this month')")

def init_db():
    """Initialises the SQLite memory database and creates the preferences table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                session_id TEXT,
                key TEXT,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, key)
            )
            """
        )
        conn.commit()
        log("memory", "SQLite memory database initialised.")
    except Exception as e:
        log("error", f"Failed to initialise memory database: {e}")
    finally:
        conn.close()

def save_preference(session_id: str, key: str, value: str):
    """
    Saves or updates a key-value preference for a specific session.
    
    Args:
        session_id: The active session identifier.
        key: The key of the preference to save.
        value: The value of the preference.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_preferences (session_id, key, value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, key, value)
        )
        conn.commit()
        log("memory", f"Saved preference: session={session_id[:8]} | {key} = {value}")
    except Exception as e:
        log("error", f"Failed to save preference to database: {e}")
    finally:
        conn.close()

def get_preferences(session_id: str) -> dict[str, str]:
    """
    Retrieves all preferences and context saved for a specific session.
    
    Args:
        session_id: The active session identifier.
        
    Returns:
        A dictionary containing all key-value preferences for the session.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT key, value FROM user_preferences WHERE session_id = ?",
            (session_id,)
        )
        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        log("error", f"Failed to retrieve preferences from database: {e}")
        return {}
    finally:
        conn.close()

def create_save_preference_tool(session_id: str) -> StructuredTool:
    """
    Creates a StructuredTool that saves a user preference for a given session.
    """
    async def _run(key: str, value: str) -> str:
        save_preference(session_id, key, value)
        return f"Successfully remembered user preference: {key} = {value}"

    return StructuredTool(
        name="save_user_preference",
        description="Saves a user preference, business context, or fact (e.g., name, department, timezone, view preferences) to persist across sessions.",
        args_schema=SavePreferenceInput,
        coroutine=_run,
    )
