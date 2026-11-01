"""SQLite-backed L1 sliding-window conversation session store."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any, cast

from qq_agent.types import ChatMessage, MessageRole

_VALID_ROLES: frozenset[str] = frozenset({"system", "user", "assistant"})


class SessionStore:
    """L1 conversation sliding window isolated per `session_key` (e.g. `group:88888`, `private:12345`)."""

    def __init__(self, db_path: Path | str = ":memory:", *, max_window: int = 40) -> None:
        if max_window < 1:
            raise ValueError(f"max_window must be >= 1, got {max_window}")
        self._max_window = int(max_window)

        if isinstance(db_path, Path) or str(db_path) != ":memory:":
            resolved = Path(db_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            target = str(resolved)
        else:
            target = ":memory:"

        self._conn: sqlite3.Connection | None = sqlite3.connect(target)
        self._init_schema()

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SessionStore is already closed")
        return self._conn

    def _init_schema(self) -> None:
        conn = self._require_conn()
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_session_messages_key_id
            ON session_messages (session_key, id)
            """
        )
        conn.commit()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def append(
        self,
        session_key: str,
        role: MessageRole,
        content: str,
        *,
        created_at: float | None = None,
    ) -> list[ChatMessage]:
        """Append a message turn, prune oldest entries exceeding `max_window`, and return evicted messages."""
        if role not in _VALID_ROLES:
            raise ValueError(f"Invalid message role: {role!r}")
        ts = time.time() if created_at is None else float(created_at)
        conn = self._require_conn()
        with conn:
            conn.execute(
                """
                INSERT INTO session_messages (session_key, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_key, role, content, ts),
            )
            evicted_rows = conn.execute(
                """
                SELECT id, role, content, created_at
                FROM session_messages
                WHERE session_key = ?
                  AND id NOT IN (
                      SELECT id FROM session_messages
                      WHERE session_key = ?
                      ORDER BY created_at DESC, id DESC
                      LIMIT ?
                  )
                ORDER BY created_at ASC, id ASC
                """,
                (session_key, session_key, self._max_window),
            ).fetchall()

            if evicted_rows:
                placeholders = ",".join("?" for _ in evicted_rows)
                conn.execute(
                    f"DELETE FROM session_messages WHERE id IN ({placeholders})",
                    tuple(int(row[0]) for row in evicted_rows),
                )

        return [
            ChatMessage(
                role=cast(MessageRole, str(row[1])),
                content=str(row[2]),
                created_at=float(row[3]),
            )
            for row in evicted_rows
        ]

    def get_window(self, session_key: str, *, limit: int | None = None) -> list[ChatMessage]:
        """Return the most recent window messages for `session_key` in chronological order."""
        effective_limit = (
            self._max_window if limit is None else min(self._max_window, max(0, int(limit)))
        )
        if effective_limit == 0:
            return []

        conn = self._require_conn()
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM (
                SELECT id, role, content, created_at
                FROM session_messages
                WHERE session_key = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            ORDER BY created_at ASC, id ASC
            """,
            (session_key, effective_limit),
        ).fetchall()

        return [
            ChatMessage(
                role=cast(MessageRole, str(row[0])),
                content=str(row[1]),
                created_at=float(row[2]),
            )
            for row in rows
        ]

    def clear(self, session_key: str) -> int:
        """Clear all stored messages for `session_key` and return the number of deleted rows."""
        conn = self._require_conn()
        with conn:
            cur = conn.execute(
                "DELETE FROM session_messages WHERE session_key = ?",
                (session_key,),
            )
        return int(cur.rowcount)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
