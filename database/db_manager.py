"""
SQLite Database Manager for Human Activity Recognition (HAR) System.
Handles logging of predictions, fall alerts, performance logs, and history queries.
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "activity_history.db")


def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the SQLite database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            activity TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_fall INTEGER NOT NULL DEFAULT 0,
            duration_seconds REAL DEFAULT 0.0,
            details TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


def log_activity(source: str, activity: str, confidence: float, is_fall: bool = False, duration_seconds: float = 0.0, details: str = ""):
    """
    Inserts a new activity prediction record into the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO activity_logs (timestamp, source, activity, confidence, is_fall, duration_seconds, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp_str, source, activity, round(float(confidence), 2), 1 if is_fall else 0, round(float(duration_seconds), 2), details))
    conn.commit()
    conn.close()


def get_recent_logs(limit: int = 100) -> pd.DataFrame:
    """Retrieves recent prediction logs as a Pandas DataFrame."""
    conn = get_connection()
    query = f"SELECT * FROM activity_logs ORDER BY id DESC LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_fall_alerts(limit: int = 50) -> pd.DataFrame:
    """Retrieves fall alerts specifically."""
    conn = get_connection()
    query = f"SELECT * FROM activity_logs WHERE is_fall = 1 ORDER BY id DESC LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_activity_stats() -> pd.DataFrame:
    """Retrieves aggregated activity counts and average confidence."""
    conn = get_connection()
    query = """
        SELECT activity, COUNT(*) as count, AVG(confidence) as avg_confidence, SUM(is_fall) as total_falls
        FROM activity_logs
        GROUP BY activity
        ORDER BY count DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def clear_history():
    """Clears all records from activity_logs table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM activity_logs")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at {DB_PATH}")
