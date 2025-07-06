import sqlite3
import json
from typing import Dict

DB_PATH = 'ecofood.db'

def get_profile(user_id: int) -> Dict:
    """Fetch the JSON blob `profile_data` for this user (or return empty dict)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT profile_data FROM users WHERE id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}
    return {}

def update_profile(user_id: int, profile: Dict) -> None:
    """
    Persist the given `profile` dict as JSON in the users.profile_data column.
    """
    profile_json = json.dumps(profile)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE users SET profile_data = ? WHERE id = ?",
        (profile_json, user_id)
    )
    conn.commit()
    conn.close()
