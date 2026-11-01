"""W5 Shared Cortex 3-layer cross-group cognition mixin over thread-safe SQLite."""

from __future__ import annotations

from collections.abc import Sequence
import sqlite3
import threading
import time
from typing import TYPE_CHECKING

from qq_agent.cortex_cues import (
    SharedFact,
    UserProfile,
    _LONG_DIGITS_RE,
    _SHARED_FACT_ERROR_MARKERS,
    _SHARED_FACT_MAX_BRIEF_CHARS,
    _extract_query_features,
    _is_safe_core_containment,
    _normalize_topic_key,
    _sanitize_memory_text,
    _sanitize_public_self_topic,
)


class SharedCortexMixin:
    """Mixin implementing W5 3-layer cross-group Shared Cortex over `self._lock` and `self._require_conn()`."""

    _lock: threading.RLock
    if TYPE_CHECKING:

        def _require_conn(self) -> sqlite3.Connection: ...

    # ------------------------------------------------------------------
    # W5 Shared Cortex Layer 1: Cross-Group Objective Research Fact Cache
    # ------------------------------------------------------------------

    def record_shared_fact(
        self,
        query: str,
        brief: str,
        *,
        ttl_s: float = 1800.0,
        now: float | None = None,
    ) -> SharedFact | None:
        """Cache a sanitized objective fact brief for `ttl_s` seconds; prunes expired entries on write."""
        ttl = float(ttl_s)
        if ttl <= 0.0:
            return None

        cleaned_query = " ".join(_sanitize_memory_text(query).split())
        cleaned_brief = _sanitize_memory_text(brief)
        if not cleaned_query or not cleaned_brief:
            return None

        lower_brief = cleaned_brief.lower()
        if any(marker in lower_brief for marker in _SHARED_FACT_ERROR_MARKERS):
            return None

        topic_key = _normalize_topic_key(cleaned_query)
        if not topic_key:
            return None

        if len(cleaned_brief) > _SHARED_FACT_MAX_BRIEF_CHARS:
            cleaned_brief = cleaned_brief[: _SHARED_FACT_MAX_BRIEF_CHARS - 3].rstrip() + "..."

        ts = float(now) if now is not None else time.time()
        expires_at = ts + ttl

        with self._lock:
            conn = self._require_conn()
            with conn:
                conn.execute("DELETE FROM shared_facts WHERE expires_at <= ?", (ts,))
                conn.execute(
                    """
                    INSERT INTO shared_facts (topic_key, query_text, brief_text, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(topic_key) DO UPDATE SET
                        query_text = excluded.query_text,
                        brief_text = excluded.brief_text,
                        updated_at = excluded.updated_at,
                        expires_at = excluded.expires_at
                    """,
                    (topic_key, cleaned_query, cleaned_brief, ts, expires_at),
                )

        return SharedFact(
            topic_key=topic_key,
            query_text=cleaned_query,
            brief_text=cleaned_brief,
            updated_at=ts,
            expires_at=expires_at,
        )

    def list_shared_facts(self, *, now: float | None = None) -> list[SharedFact]:
        """Return all currently unexpired `SharedFact` entries ordered by `updated_at` descending."""
        ts = float(now) if now is not None else time.time()
        with self._lock:
            conn = self._require_conn()
            rows = conn.execute(
                """
                SELECT topic_key, query_text, brief_text, updated_at, expires_at
                FROM shared_facts
                WHERE expires_at > ?
                ORDER BY updated_at DESC
                """,
                (ts,),
            ).fetchall()

        return [
            SharedFact(
                topic_key=str(row[0]),
                query_text=str(row[1]),
                brief_text=str(row[2]),
                updated_at=float(row[3]),
                expires_at=float(row[4]),
            )
            for row in rows
        ]

    def find_relevant_shared_facts(
        self,
        query_text: str,
        *,
        limit: int = 2,
        now: float | None = None,
    ) -> list[SharedFact]:
        """Match unexpired `shared_facts` via zero-LLM alphanumeric token + CJK 2-gram overlap."""
        max_items = int(limit)
        if max_items <= 0 or not query_text or not query_text.strip():
            return []

        q_feat = _extract_query_features(query_text, is_fact_query=True)
        if not q_feat.alnum_tokens and not q_feat.cjk_bigrams and len(q_feat.norm_core) < 2:
            return []

        active_facts = self.list_shared_facts(now=now)
        if not active_facts:
            return []

        scored: list[tuple[float, SharedFact]] = []
        for fact in active_facts:
            fq_feat = _extract_query_features(fact.query_text, is_fact_query=True)
            fb_feat = _extract_query_features(fact.brief_text, is_fact_query=False)

            fact_all_alnum = fq_feat.alnum_tokens | fb_feat.alnum_tokens
            fact_high_alnum = fq_feat.high_alnum_tokens | fb_feat.high_alnum_tokens
            fact_all_bigrams = fq_feat.cjk_bigrams | fb_feat.cjk_bigrams
            fact_specific_bigrams = fq_feat.specific_cjk_bigrams | fb_feat.specific_cjk_bigrams

            matched_high_alnum = q_feat.alnum_tokens & fact_high_alnum
            matched_all_alnum = q_feat.alnum_tokens & fact_all_alnum
            matched_bigrams = q_feat.cjk_bigrams & fact_all_bigrams
            matched_specific_bigrams = q_feat.specific_cjk_bigrams & fact_specific_bigrams

            has_core_substring = _is_safe_core_containment(
                q_feat.norm_core,
                fq_feat.norm_core,
            ) and bool(
                q_feat.specific_cjk_bigrams
                or q_feat.high_alnum_tokens
                or matched_all_alnum
            )

            qualifies = (
                len(matched_high_alnum) >= 1
                or (len(matched_bigrams) >= 2 and len(matched_specific_bigrams) >= 1)
                or (
                    len(matched_all_alnum) >= 1
                    and len(matched_all_alnum | matched_specific_bigrams) >= 2
                )
                or has_core_substring
            )
            if not qualifies:
                continue

            query_high_hits = len(q_feat.alnum_tokens & fq_feat.high_alnum_tokens)
            query_bigram_hits = len(q_feat.cjk_bigrams & fq_feat.cjk_bigrams)
            score = (
                3.5 * query_high_hits
                + 2.5 * (len(matched_high_alnum) - query_high_hits)
                + 1.5 * len(matched_all_alnum - matched_high_alnum)
                + 2.0 * query_bigram_hits
                + 1.0 * (len(matched_bigrams) - query_bigram_hits)
                + 1.0 * len(matched_specific_bigrams)
                + (2.0 if has_core_substring else 0.0)
            )
            scored.append((score, fact))

        scored.sort(key=lambda item: (item[0], item[1].updated_at), reverse=True)
        return [fact for _, fact in scored[:max_items]]

    # ------------------------------------------------------------------
    # W5 Shared Cortex Layer 2: Non-Destructive Public Trait Observation
    # ------------------------------------------------------------------

    def observe_public_traits(
        self,
        user_id: int,
        sender_name: str,
        cues: Sequence[str],
        *,
        max_chars: int = 120,
    ) -> UserProfile | None:
        """Non-destructively append deduplicated public interest cues (`；`-joined) without overwriting manual `/profile` traits."""
        uid = int(user_id)
        limit = max(1, int(max_chars))
        cleaned_name = _sanitize_memory_text(sender_name.strip())

        valid_cues: list[str] = []
        for raw_cue in cues:
            cue = _sanitize_memory_text(" ".join(str(raw_cue).split()))
            if not cue or _LONG_DIGITS_RE.search(cue):
                continue
            if cue not in valid_cues:
                valid_cues.append(cue)

        with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT user_id, display_name, traits_text, updated_at FROM user_profiles WHERE user_id = ?",
                (uid,),
            ).fetchone()

            if not valid_cues:
                if row is None:
                    return None
                if cleaned_name and cleaned_name != str(row[1]):
                    now = time.time()
                    with conn:
                        conn.execute(
                            "UPDATE user_profiles SET display_name = ?, updated_at = ? WHERE user_id = ?",
                            (cleaned_name, now, uid),
                        )
                    return UserProfile(
                        user_id=uid,
                        display_name=cleaned_name,
                        traits_text=str(row[2]),
                        updated_at=now,
                    )
                return UserProfile(
                    user_id=int(row[0]),
                    display_name=str(row[1]),
                    traits_text=str(row[2]),
                    updated_at=float(row[3]),
                )

            existing_name = str(row[1]) if row is not None else ""
            existing_traits = str(row[2]).strip() if row is not None else ""
            merged = existing_traits
            existing_segments = {seg.strip() for seg in merged.split("；") if seg.strip()}

            for cue in valid_cues:
                if cue in existing_segments or cue in merged:
                    continue
                candidate = f"{merged}；{cue}" if merged else cue
                if len(candidate) <= limit:
                    merged = candidate
                    existing_segments.add(cue)

            effective_name = cleaned_name if cleaned_name else existing_name
            if row is None and not merged and not effective_name:
                return None

            now = time.time()
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_profiles (user_id, display_name, traits_text, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        traits_text = excluded.traits_text,
                        updated_at = excluded.updated_at
                    """,
                    (uid, effective_name, merged, now),
                )
            return UserProfile(
                user_id=uid,
                display_name=effective_name,
                traits_text=merged,
                updated_at=now,
            )

    # ------------------------------------------------------------------
    # W5 Shared Cortex Layer 3: Sanitized Bot Self-State & Privacy Firewall
    # ------------------------------------------------------------------

    def update_public_self_state(
        self,
        topic_hint: str,
        *,
        chat_type: str = "group",
        now: float | None = None,
    ) -> str:
        """Update single-row `bot_self_state` ONLY from `group` chats after strict PII/ID/@ scrubbing."""
        if chat_type != "group":
            return self.get_public_self_state(now=now)

        sanitized = _sanitize_public_self_topic(topic_hint)
        if not sanitized:
            return self.get_public_self_state(now=now)

        ts = float(now) if now is not None else time.time()
        with self._lock:
            conn = self._require_conn()
            with conn:
                conn.execute(
                    """
                    INSERT INTO bot_self_state (id, state_text, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        state_text = excluded.state_text,
                        updated_at = excluded.updated_at
                    """,
                    (sanitized, ts),
                )
        return sanitized

    def get_public_self_state(
        self,
        *,
        max_age_s: float = 1800.0,
        now: float | None = None,
    ) -> str:
        """Return sanitized `bot_self_state` if within `max_age_s`, else `""`."""
        ttl = float(max_age_s)
        if ttl <= 0.0:
            return ""
        ts = float(now) if now is not None else time.time()

        with self._lock:
            conn = self._require_conn()
            row = conn.execute(
                "SELECT state_text, updated_at FROM bot_self_state WHERE id = 1"
            ).fetchone()
            if row is None:
                return ""
            state_text = _sanitize_memory_text(str(row[0]))
            updated_at = float(row[1])
            if not state_text or (ts - updated_at) >= ttl:
                return ""
            return state_text


__all__ = ["SharedCortexMixin"]
