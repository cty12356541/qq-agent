"""Rate-limiting and SQLite-backed daily token/cost budget guardrails (§4.6)."""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import TracebackType

from qq_agent.types import GuardDecision, InboundMessage

_HARD_LIMIT_REFUSAL_COOLDOWN_S = 60.0
_GROUP_RATE_WINDOW_S = 60.0
_GROUP_RATE_REFUSAL_COOLDOWN_S = 30.0


@dataclass(slots=True, frozen=True)
class QuotaSnapshot:
    """Point-in-time snapshot of daily LLM call, token, and micro-USD consumption."""

    day: str
    calls_used: int
    max_daily_calls: int
    tokens_used: int
    max_daily_tokens: int
    cost_micros_used: int
    soft_budget_micros: int
    hard_budget_micros: int


class GuardrailManager:
    """Enforces per-user cooldowns, per-group sliding rate limits, and SQLite-persisted daily budgets."""

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        *,
        user_cooldown_s: float = 3.0,
        group_max_per_minute: int = 12,
        max_daily_calls: int = 300,
        max_daily_tokens: int = 500_000,
        soft_daily_budget_micros: int = 1_500_000,
        hard_daily_budget_micros: int = 3_000_000,
        cheap_model: str | None = None,
        rate_refusal: str = "说话太快啦，让我喘口气~",
        budget_refusal: str = "今日脑细胞额度已耗尽，明天再聊叭~",
    ) -> None:
        self.db_path = db_path
        self.user_cooldown_s = user_cooldown_s
        self.group_max_per_minute = group_max_per_minute
        self.max_daily_calls = max_daily_calls
        self.max_daily_tokens = max_daily_tokens
        self.soft_daily_budget_micros = soft_daily_budget_micros
        self.hard_daily_budget_micros = hard_daily_budget_micros
        self.cheap_model = cheap_model
        self.rate_refusal = rate_refusal
        self.budget_refusal = budget_refusal

        db_str = str(db_path)
        if db_str != ":memory:" and not db_str.startswith("file:"):
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        if db_str != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_usage (
                day TEXT PRIMARY KEY,
                calls_used INTEGER NOT NULL DEFAULT 0,
                tokens_used INTEGER NOT NULL DEFAULT 0,
                cost_micros_used INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.commit()

        # In-memory rate-limiting & anti-spam refusal windows
        self._user_last_ts: dict[tuple[str, int], float] = {}
        self._group_windows: dict[str, deque[float]] = {}
        self._group_rate_notice_ts: dict[str, float] = {}
        self._hard_limit_notice_ts: dict[str, float] = {}

    def _read_daily_row(self, day: str) -> tuple[int, int, int]:
        row = self._conn.execute(
            "SELECT calls_used, tokens_used, cost_micros_used FROM daily_usage WHERE day = ?",
            (day,),
        ).fetchone()
        if row is None:
            return (0, 0, 0)
        return (int(row[0]), int(row[1]), int(row[2]))

    def check(
        self,
        msg: InboundMessage,
        *,
        now: float | None = None,
        today: str | None = None,
    ) -> GuardDecision:
        ts = time.monotonic() if now is None else float(now)
        day = date.today().isoformat() if today is None else today

        with self._lock:
            calls_used, tokens_used, cost_micros_used = self._read_daily_row(day)

            # 1) Daily hard limits (calls / tokens / hard micro-USD budget)
            if (
                calls_used >= self.max_daily_calls
                or tokens_used >= self.max_daily_tokens
                or cost_micros_used >= self.hard_daily_budget_micros
            ):
                last_hard_notice = self._hard_limit_notice_ts.get(msg.session_key)
                if (
                    last_hard_notice is not None
                    and (ts - last_hard_notice) < _HARD_LIMIT_REFUSAL_COOLDOWN_S
                ):
                    refusal: str | None = None
                else:
                    self._hard_limit_notice_ts[msg.session_key] = ts
                    refusal = self.budget_refusal
                return GuardDecision(False, "daily_hard_limit", refusal_reply=refusal)

            # 2) Per-(session, user) cooldown (silent drop on rapid double-send)
            user_key = (msg.session_key, msg.user_id)
            if self.user_cooldown_s > 0:
                last_user = self._user_last_ts.get(user_key)
                if last_user is not None and (ts - last_user) < self.user_cooldown_s:
                    return GuardDecision(False, "user_cooldown", refusal_reply=None)

            # 3) Per-group 60s sliding window limit
            if msg.chat_type == "group":
                window = self._group_windows.setdefault(msg.session_key, deque())
                while window and (ts - window[0]) >= _GROUP_RATE_WINDOW_S:
                    window.popleft()
                if len(window) >= self.group_max_per_minute:
                    last_rate_notice = self._group_rate_notice_ts.get(msg.session_key)
                    if (
                        last_rate_notice is not None
                        and (ts - last_rate_notice) < _GROUP_RATE_REFUSAL_COOLDOWN_S
                    ):
                        rate_reply: str | None = None
                    else:
                        self._group_rate_notice_ts[msg.session_key] = ts
                        rate_reply = self.rate_refusal
                    return GuardDecision(False, "group_rate_limit", refusal_reply=rate_reply)

            # 4) Passed all rate limits: record trigger timestamp & evaluate soft budget downgrade
            self._user_last_ts[user_key] = ts
            if msg.chat_type == "group":
                self._group_windows.setdefault(msg.session_key, deque()).append(ts)

            if (
                cost_micros_used >= self.soft_daily_budget_micros
                and self.cheap_model is not None
                and self.cheap_model.strip() != ""
            ):
                return GuardDecision(
                    True,
                    "daily_budget_soft_downgrade",
                    override_model=self.cheap_model,
                )

            return GuardDecision(True, "ok")

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_micros_usd: int = 0,
        today: str | None = None,
    ) -> QuotaSnapshot:
        day = date.today().isoformat() if today is None else today
        tokens_delta = (prompt_tokens + completion_tokens) if total_tokens == 0 else total_tokens

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO daily_usage (day, calls_used, tokens_used, cost_micros_used)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    calls_used = calls_used + 1,
                    tokens_used = tokens_used + excluded.tokens_used,
                    cost_micros_used = cost_micros_used + excluded.cost_micros_used
                """,
                (day, tokens_delta, cost_micros_usd),
            )
            self._conn.commit()
            return self.snapshot(today=day)

    def snapshot(self, *, today: str | None = None) -> QuotaSnapshot:
        day = date.today().isoformat() if today is None else today
        with self._lock:
            calls_used, tokens_used, cost_micros_used = self._read_daily_row(day)
        return QuotaSnapshot(
            day=day,
            calls_used=calls_used,
            max_daily_calls=self.max_daily_calls,
            tokens_used=tokens_used,
            max_daily_tokens=self.max_daily_tokens,
            cost_micros_used=cost_micros_used,
            soft_budget_micros=self.soft_daily_budget_micros,
            hard_budget_micros=self.hard_daily_budget_micros,
        )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def __enter__(self) -> GuardrailManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
