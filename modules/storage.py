"""
SQLite storage module for stock market data.

Provides a unified interface for storing and retrieving session data
that was previously stored as JSON files in the sources/ directory.

Usage:
    from modules.storage import SourcesDB, DEFAULT_DB_PATH

    with SourcesDB() as db:
        sessions = db.get_sessions("BBCA")
        for session in sessions:
            print(session["date"], session["analyzed"]["price_feed"]["close"])
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple, TypeAlias

from .common import BASE_DIR

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DB_PATH: Final[Path] = BASE_DIR / "sources.db"

JsonDict: TypeAlias = Dict[str, Any]

# =============================================================================
# Database Schema
# =============================================================================

SCHEMA_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    session INTEGER NOT NULL,
    date TEXT NOT NULL,
    analyzed JSON NOT NULL,
    running_trade JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE INDEX IF NOT EXISTS idx_sessions_symbol ON sessions(symbol);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
CREATE INDEX IF NOT EXISTS idx_sessions_symbol_session ON sessions(symbol, session);
"""


# =============================================================================
# SourcesDB Class
# =============================================================================

class SourcesDB:
    """
    SQLite database wrapper for stock market session data.

    Provides methods for storing and retrieving analyzed session data
    that was previously stored as JSON files.

    Usage as context manager (recommended):
        with SourcesDB() as db:
            db.insert_session(...)
            sessions = db.get_sessions("BBCA")

    Usage without context manager:
        db = SourcesDB()
        try:
            db.insert_session(...)
        finally:
            db.close()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file. Defaults to sources.db in project root.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with proper settings."""
        with self._lock:
            conn = sqlite3.connect(
                self.db_path,
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield conn
            finally:
                conn.close()

    def _get_persistent_connection(self) -> sqlite3.Connection:
        """Get or create a persistent connection for batch operations."""
        with self._lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self.db_path,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    timeout=30.0,
                    check_same_thread=False,
                )
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
            return self._conn

    def close(self) -> None:
        """Close the persistent database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "SourcesDB":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close connection."""
        self.close()

    # =========================================================================
    # Write Operations
    # =========================================================================

    def insert_session(
        self,
        symbol: str,
        session: int,
        date: str,
        analyzed: JsonDict,
        running_trade: Optional[JsonDict] = None,
    ) -> int:
        """
        Insert or replace a session record.

        Args:
            symbol: Stock symbol (e.g., "BBCA")
            session: Session number (1, 2, 3, ...)
            date: Trading date in YYYY-MM-DD format
            analyzed: Analyzed data dictionary
            running_trade: Running trade data dictionary (optional)

        Returns:
            Row ID of the inserted record
        """
        symbol = symbol.upper()
        analyzed_json = json.dumps(analyzed, ensure_ascii=False)
        running_trade_json = json.dumps(running_trade, ensure_ascii=False) if running_trade else None

        with self._lock:
            conn = self._get_persistent_connection()
            cursor = conn.execute(
                """
                INSERT OR REPLACE INTO sessions (symbol, session, date, analyzed, running_trade)
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, session, date, analyzed_json, running_trade_json),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def insert_session_batch(
        self,
        records: List[Tuple[str, int, str, JsonDict, Optional[JsonDict]]],
    ) -> int:
        """
        Insert multiple session records in a single transaction.

        Args:
            records: List of tuples (symbol, session, date, analyzed, running_trade)

        Returns:
            Number of records inserted
        """
        conn = self._get_persistent_connection()
        cursor = conn.cursor()

        prepared = [
            (
                symbol.upper(),
                session,
                date,
                json.dumps(analyzed, ensure_ascii=False),
                json.dumps(running_trade, ensure_ascii=False) if running_trade else None,
            )
            for symbol, session, date, analyzed, running_trade in records
        ]

        cursor.executemany(
            """
            INSERT OR REPLACE INTO sessions (symbol, session, date, analyzed, running_trade)
            VALUES (?, ?, ?, ?, ?)
            """,
            prepared,
        )
        conn.commit()
        return cursor.rowcount

    # =========================================================================
    # Read Operations
    # =========================================================================

    def get_symbols(self) -> List[str]:
        """
        Get list of all symbols in the database.

        Returns:
            Sorted list of unique symbols
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT symbol FROM sessions ORDER BY symbol"
            )
            return [row["symbol"] for row in cursor.fetchall()]

    def get_sessions(self, symbol: str) -> List[JsonDict]:
        """
        Get all sessions for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of session dictionaries with keys:
            - session: int
            - date: str
            - analyzed: dict
            - running_trade: dict or None
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session, date, analyzed, running_trade
                FROM sessions
                WHERE symbol = ?
                ORDER BY session ASC
                """,
                (symbol,),
            )
            return [
                {
                    "session": row["session"],
                    "date": row["date"],
                    "analyzed": json.loads(row["analyzed"]),
                    "running_trade": json.loads(row["running_trade"]) if row["running_trade"] else None,
                }
                for row in cursor.fetchall()
            ]

    def get_session(self, symbol: str, session: int) -> Optional[JsonDict]:
        """
        Get a specific session for a symbol.

        Args:
            symbol: Stock symbol
            session: Session number

        Returns:
            Session dictionary or None if not found
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session, date, analyzed, running_trade
                FROM sessions
                WHERE symbol = ? AND session = ?
                """,
                (symbol, session),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "session": row["session"],
                "date": row["date"],
                "analyzed": json.loads(row["analyzed"]),
                "running_trade": json.loads(row["running_trade"]) if row["running_trade"] else None,
            }

    def get_session_by_date(self, symbol: str, date: str) -> Optional[JsonDict]:
        """
        Get session for a symbol on a specific date.

        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format

        Returns:
            Session dictionary or None if not found
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session, date, analyzed, running_trade
                FROM sessions
                WHERE symbol = ? AND date = ?
                """,
                (symbol, date),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "session": row["session"],
                "date": row["date"],
                "analyzed": json.loads(row["analyzed"]),
                "running_trade": json.loads(row["running_trade"]) if row["running_trade"] else None,
            }

    def get_latest_session(self, symbol: str) -> Optional[JsonDict]:
        """
        Get the most recent session for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Session dictionary or None if no sessions exist
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT session, date, analyzed, running_trade
                FROM sessions
                WHERE symbol = ?
                ORDER BY session DESC
                LIMIT 1
                """,
                (symbol,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "session": row["session"],
                "date": row["date"],
                "analyzed": json.loads(row["analyzed"]),
                "running_trade": json.loads(row["running_trade"]) if row["running_trade"] else None,
            }

    def get_next_session_number(self, symbol: str) -> int:
        """
        Get the next session number for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Next session number (1 if no sessions exist)
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(session) as max_session FROM sessions WHERE symbol = ?",
                (symbol,),
            )
            row = cursor.fetchone()
            max_session = row["max_session"] if row and row["max_session"] else 0
            return max_session + 1

    def get_session_numbers(self, symbol: str) -> List[int]:
        """
        Get list of all session numbers for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Sorted list of session numbers
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT session FROM sessions WHERE symbol = ? ORDER BY session ASC",
                (symbol,),
            )
            return [row["session"] for row in cursor.fetchall()]

    def has_data_for_date(self, symbol: str, date: str) -> bool:
        """
        Check if data exists for a symbol on a specific date.

        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format

        Returns:
            True if data exists, False otherwise
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE symbol = ? AND date = ? LIMIT 1",
                (symbol, date),
            )
            return cursor.fetchone() is not None

    def get_session_count(self, symbol: Optional[str] = None) -> int:
        """
        Get count of sessions.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            Number of sessions
        """
        with self._get_connection() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM sessions WHERE symbol = ?",
                    (symbol.upper(),),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM sessions")
            row = cursor.fetchone()
            return row["cnt"] if row else 0

    def get_date_range(self, symbol: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the date range of sessions.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            Tuple of (min_date, max_date) or (None, None) if no data
        """
        with self._get_connection() as conn:
            if symbol:
                cursor = conn.execute(
                    "SELECT MIN(date) as min_date, MAX(date) as max_date FROM sessions WHERE symbol = ?",
                    (symbol.upper(),),
                )
            else:
                cursor = conn.execute(
                    "SELECT MIN(date) as min_date, MAX(date) as max_date FROM sessions"
                )
            row = cursor.fetchone()
            if row and row["min_date"]:
                return row["min_date"], row["max_date"]
            return None, None

    # =========================================================================
    # Delete Operations
    # =========================================================================

    def delete_symbol(self, symbol: str) -> int:
        """
        Delete all sessions for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Number of deleted records
        """
        symbol = symbol.upper()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE symbol = ?",
                (symbol,),
            )
            conn.commit()
            return cursor.rowcount

    def delete_before_date(self, date: str) -> int:
        """
        Delete all sessions before a specific date.

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Number of deleted records
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE date < ?",
                (date,),
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def vacuum(self) -> None:
        """Reclaim unused space in the database."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")

    def get_db_size(self) -> int:
        """
        Get the database file size in bytes.

        Returns:
            File size in bytes, 0 if file doesn't exist
        """
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    def get_stats(self) -> JsonDict:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT
                    COUNT(DISTINCT symbol) as symbol_count,
                    COUNT(*) as session_count,
                    MIN(date) as min_date,
                    MAX(date) as max_date
                FROM sessions
                """
            )
            row = cursor.fetchone()

            return {
                "db_path": str(self.db_path),
                "db_size_bytes": self.get_db_size(),
                "db_size_mb": round(self.get_db_size() / (1024 * 1024), 2),
                "symbol_count": row["symbol_count"] if row else 0,
                "session_count": row["session_count"] if row else 0,
                "min_date": row["min_date"] if row else None,
                "max_date": row["max_date"] if row else None,
            }
