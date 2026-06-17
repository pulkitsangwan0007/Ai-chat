"""
database.py
-------------
Database layer for the AI-Powered Customer Support Chatbot.

Responsible for:
    - Creating/initializing the SQLite database and schema.
    - Inserting new chat log entries (user message + bot response + timestamp).
    - Fetching chat history (all or paginated).
    - Searching previous conversations by keyword.
    - Clearing chat history.

The module is implemented using a single class (`ChatDatabase`) following
OOP principles so that it can be imported and reused across the FastAPI
backend (api.py) and the Streamlit frontend (app.py).
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_NAME = "chatbot.db"
DB_PATH = os.path.join(DATA_DIR, DB_NAME)

# Ensure the data directory exists before any DB connection is attempted.
os.makedirs(DATA_DIR, exist_ok=True)


class ChatDatabase:
    """
    Encapsulates all SQLite database operations for the chatbot.

    Table Schema: chat_logs
    ------------------------
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    user_message  TEXT NOT NULL
    bot_response  TEXT NOT NULL
    timestamp     TEXT NOT NULL
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._initialize_database()

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------
    @contextmanager
    def _get_connection(self):
        """
        Context manager that yields a SQLite connection and guarantees
        that it is properly closed (and committed) afterwards, even if
        an exception occurs mid-operation.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema initialization
    # ------------------------------------------------------------------
    def _initialize_database(self) -> None:
        """Create the chat_logs table if it does not already exist."""
        create_table_query = """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """
        with self._get_connection() as conn:
            conn.execute(create_table_query)

    # ------------------------------------------------------------------
    # Insert operation
    # ------------------------------------------------------------------
    def log_chat(self, user_message: str, bot_response: str) -> int:
        """
        Insert a new chat interaction into the database.

        Args:
            user_message: The message sent by the user.
            bot_response: The chatbot's generated response.

        Returns:
            The auto-generated row id of the inserted record.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_query = """
            INSERT INTO chat_logs (user_message, bot_response, timestamp)
            VALUES (?, ?, ?);
        """
        with self._get_connection() as conn:
            cursor = conn.execute(insert_query, (user_message, bot_response, timestamp))
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_all_chats(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve all chat logs ordered from oldest to newest.

        Args:
            limit: Optional cap on number of most recent records returned.

        Returns:
            A list of dictionaries, each representing a chat_logs row.
        """
        if limit:
            query = """
                SELECT * FROM (
                    SELECT * FROM chat_logs ORDER BY id DESC LIMIT ?
                ) sub ORDER BY id ASC;
            """
            params = (limit,)
        else:
            query = "SELECT * FROM chat_logs ORDER BY id ASC;"
            params = ()

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def search_chats(self, keyword: str) -> List[Dict]:
        """
        Search previous conversations where either the user message or
        the bot response contains the given keyword (case-insensitive).

        Args:
            keyword: The search term provided by the user.

        Returns:
            A list of matching chat_logs rows as dictionaries.
        """
        query = """
            SELECT * FROM chat_logs
            WHERE user_message LIKE ? OR bot_response LIKE ?
            ORDER BY id DESC;
        """
        like_pattern = f"%{keyword}%"
        with self._get_connection() as conn:
            rows = conn.execute(query, (like_pattern, like_pattern)).fetchall()
            return [dict(row) for row in rows]

    def get_chat_count(self) -> int:
        """Return the total number of chat log entries stored."""
        query = "SELECT COUNT(*) as total FROM chat_logs;"
        with self._get_connection() as conn:
            result = conn.execute(query).fetchone()
            return result["total"] if result else 0

    # ------------------------------------------------------------------
    # Delete operation
    # ------------------------------------------------------------------
    def clear_all_chats(self) -> None:
        """Delete every record from the chat_logs table."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_logs;")


# ----------------------------------------------------------------------
# Singleton instance for easy import across modules
# ----------------------------------------------------------------------
db = ChatDatabase()


if __name__ == "__main__":
    # Simple manual test when running this file directly.
    test_db = ChatDatabase()
    new_id = test_db.log_chat("Hello, how do I reset my password?",
                               "Click on 'Forgot Password' on the login page.")
    print(f"Inserted chat log with id: {new_id}")
    print("All chats:", test_db.get_all_chats())
    print("Search results for 'password':", test_db.search_chats("password"))
    print("Total chat count:", test_db.get_chat_count())
