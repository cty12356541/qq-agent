"""Tier-1 zero-cost rule trigger filter before LLM invocation (§4.2, M2 high-activity group trigger)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import random
import re
import time

from qq_agent.types import InboundMessage, TriggerDecision

DEFAULT_TOPIC_KEYWORDS: tuple[str, ...] = (
    "窝窝屎",
    "战舰",
    "打船",
    "开黑",
    "语音",
    "帝国",
    "steam",
    "能代",
    "专升本",
    "小丑",
    "路边一条",
    "二氧化碳",
    "睡不着",
    "头疼",
    "头痛",
    "难绷",
    "吃啥",
    "吃饭",
    "[动画表情]",
)


def _compile_wake_matcher(keyword: str) -> tuple[str, re.Pattern[str] | None]:
    """Build case-insensitive ASCII boundary regex for pure ASCII alnum words, else substring."""
    lowered = keyword.lower()
    if keyword.isascii() and keyword.isalnum():
        pattern = re.compile(
            rf"(?<![a-z0-9]){re.escape(lowered)}(?![a-z0-9])",
            flags=re.IGNORECASE | re.ASCII,
        )
        return lowered, pattern
    return lowered, None


def _dedupe_cleaned_keywords(items: Sequence[str]) -> tuple[str, ...]:
    seen_lower: set[str] = set()
    result: list[str] = []
    for raw in items:
        if not raw:
            continue
        cleaned = raw.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in seen_lower:
            seen_lower.add(key)
            result.append(cleaned)
    return tuple(result)


class RuleTrigger:
    """Evaluates whether an inbound message should wake the Agent without spending LLM tokens."""

    def __init__(
        self,
        *,
        bot_qq: int | None = None,
        keywords: Sequence[str] = (),
        topic_keywords: Sequence[str] = DEFAULT_TOPIC_KEYWORDS,
        allow_private: bool = True,
        allowed_group_ids: set[int] | None = None,
        followup_window_s: float = 60.0,
        max_followup_turns: int = 4,
        topic_prob: float = 0.0,
        proactive_prob: float = 0.0,
        proactive_every_n: int = 0,
        proactive_cooldown_s: float = 12.0,
        clock: Callable[[], float] | None = None,
        rng: Callable[[], float] | None = None,
    ) -> None:
        self.bot_qq = bot_qq
        self.allow_private = allow_private
        self.allowed_group_ids: set[int] | None = (
            set(allowed_group_ids) if allowed_group_ids is not None else None
        )
        self.followup_window_s = float(followup_window_s)
        self.max_followup_turns = int(max_followup_turns)
        self.topic_prob = float(topic_prob)
        self.proactive_prob = float(proactive_prob)
        self.proactive_every_n = int(proactive_every_n)
        self.proactive_cooldown_s = float(proactive_cooldown_s)

        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._rng: Callable[[], float] = rng if rng is not None else random.random

        self._keywords: tuple[str, ...] = ()
        self._wake_matchers: tuple[tuple[str, re.Pattern[str] | None], ...] = ()
        self._topic_keywords: tuple[str, ...] = ()
        self._topic_keywords_lower: tuple[str, ...] = ()

        # Per-group & per-(group, user) active conversation state
        self._untriggered_counts: dict[int | None, int] = {}
        self._last_proactive_ts: dict[int | None, float] = {}
        self._followup_windows: dict[tuple[int | None, int], tuple[float, int]] = {}

        self.update_keywords(keywords)
        self._set_topic_keywords(topic_keywords)

    @property
    def keywords(self) -> tuple[str, ...]:
        return self._keywords

    @property
    def topic_keywords(self) -> tuple[str, ...]:
        return self._topic_keywords

    def update_keywords(self, keywords: Sequence[str]) -> None:
        """Hot-update wake keywords (e.g. after persona nickname/alias reload)."""
        cleaned = tuple(kw.strip() for kw in keywords if kw and kw.strip())
        self._keywords = cleaned

        seen_lower: set[str] = set()
        matchers: list[tuple[str, re.Pattern[str] | None]] = []
        for kw in cleaned:
            key = kw.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)
            matchers.append(_compile_wake_matcher(kw))
        self._wake_matchers = tuple(matchers)

    def _set_topic_keywords(self, topic_keywords: Sequence[str]) -> None:
        deduped = _dedupe_cleaned_keywords(topic_keywords)
        self._topic_keywords = deduped
        self._topic_keywords_lower = tuple(kw.lower() for kw in deduped)

    def update_topic_keywords(
        self,
        topic_keywords: Sequence[str],
        *,
        include_defaults: bool = True,
    ) -> None:
        """Hot-update topic interest keywords, merging custom entries with DEFAULT_TOPIC_KEYWORDS."""
        if include_defaults:
            combined: Sequence[str] = (*DEFAULT_TOPIC_KEYWORDS, *topic_keywords)
        else:
            combined = topic_keywords
        self._set_topic_keywords(combined)

    def record_bot_reply(
        self,
        group_id: int | None,
        user_id: int | None = None,
        *,
        now: float | None = None,
    ) -> None:
        """Record a completed bot reply in a group, resetting untriggered count and opening/refreshing follow-up window."""
        if group_id is None:
            return

        ts = self._clock() if now is None else float(now)
        self._untriggered_counts[group_id] = 0

        if user_id is None or self.followup_window_s <= 0 or self.max_followup_turns <= 0:
            return

        key = (group_id, user_id)
        existing = self._followup_windows.get(key)
        if existing is not None:
            win_ts, turns_used = existing
            if (ts - win_ts) <= self.followup_window_s:
                self._followup_windows[key] = (ts, turns_used)
                return

        self._followup_windows[key] = (ts, 0)

    def _matches_wake_keyword(self, clean_text: str) -> bool:
        if not clean_text or not self._wake_matchers:
            return False
        lowered = clean_text.lower()
        for kw_lower, pattern in self._wake_matchers:
            if pattern is not None:
                if pattern.search(clean_text) is not None:
                    return True
            elif kw_lower in lowered:
                return True
        return False

    def _matches_topic_keyword(self, clean_text: str) -> bool:
        if not clean_text or not self._topic_keywords_lower:
            return False
        lowered = clean_text.lower()
        return any(tk in lowered for tk in self._topic_keywords_lower)

    def _is_proactive_cooldown_ready(self, group_id: int | None, now: float) -> bool:
        last_ts = self._last_proactive_ts.get(group_id)
        if last_ts is None:
            return True
        return (now - last_ts) >= self.proactive_cooldown_s

    def evaluate(self, msg: InboundMessage) -> TriggerDecision:
        # 1) Anti-self-loop guard
        if (self.bot_qq is not None and msg.user_id == self.bot_qq) or (
            msg.self_id is not None and msg.user_id == msg.self_id
        ):
            return TriggerDecision(False, "ignored_self")

        effective_bot_qq = self.bot_qq if self.bot_qq is not None else msg.self_id

        # 2) Group whitelist guard (only active when non-empty)
        if (
            msg.chat_type == "group"
            and self.allowed_group_ids
            and msg.group_id not in self.allowed_group_ids
        ):
            return TriggerDecision(False, "group_not_whitelisted")

        clean_text = msg.text.strip()

        # 3) Private chat rules
        if msg.chat_type == "private":
            if not self.allow_private:
                return TriggerDecision(False, "private_disabled")
            if clean_text:
                return TriggerDecision(True, "private_direct", clean_text)
            if msg.image_urls:
                return TriggerDecision(True, "private_direct", "[图片]")
            return TriggerDecision(False, "ignored_empty")

        # 4) Group chat multi-tier priority chain
        group_id = msg.group_id
        followup_key = (group_id, msg.user_id)
        now = self._clock()

        # 4.1) group_at (100% reply)
        if msg.mentioned_bot:
            self._untriggered_counts[group_id] = 0
            self._followup_windows.pop(followup_key, None)
            return TriggerDecision(True, "group_at", clean_text or "（戳了戳你）")

        # 4.2) group_reply_to_bot (quote-reply to bot without @, 100% reply)
        if effective_bot_qq is not None and msg.reply_to_user_id == effective_bot_qq:
            self._untriggered_counts[group_id] = 0
            self._followup_windows.pop(followup_key, None)
            return TriggerDecision(True, "group_reply_to_bot", clean_text or "（回复了你）")

        # 4.3) group_keyword (wake keyword match with ASCII boundary protection, 100% reply)
        if clean_text and self._matches_wake_keyword(clean_text):
            self._untriggered_counts[group_id] = 0
            self._followup_windows.pop(followup_key, None)
            return TriggerDecision(True, "group_keyword", clean_text)

        # 4.4) Empty message & polite yield when explicitly addressing another member
        if not clean_text and not msg.image_urls:
            return TriggerDecision(False, "ignored_empty")

        if msg.mentioned_other or (
            msg.reply_to_user_id is not None
            and effective_bot_qq is not None
            and msg.reply_to_user_id != effective_bot_qq
        ):
            self._untriggered_counts[group_id] = self._untriggered_counts.get(group_id, 0) + 1
            return TriggerDecision(False, "not_triggered")

        # 4.5) group_followup_window (60s same-user active follow-up window, up to max_followup_turns)
        existing_win = self._followup_windows.get(followup_key)
        if existing_win is not None:
            win_ts, turns_used = existing_win
            if (
                self.followup_window_s > 0
                and (now - win_ts) <= self.followup_window_s
                and turns_used < self.max_followup_turns
            ):
                self._followup_windows[followup_key] = (win_ts, turns_used + 1)
                self._untriggered_counts[group_id] = 0
                return TriggerDecision(
                    True,
                    "group_followup_window",
                    clean_text or "[图片]",
                )
            self._followup_windows.pop(followup_key, None)

        proactive_ready = self._is_proactive_cooldown_ready(group_id, now)

        # 4.6) group_topic_interest (topic keywords / animated stickers high-sensitivity chime-in)
        if (
            proactive_ready
            and self.topic_prob > 0.0
            and self._matches_topic_keyword(clean_text)
            and self._rng() < self.topic_prob
        ):
            self._last_proactive_ts[group_id] = now
            self._untriggered_counts[group_id] = 0
            return TriggerDecision(
                True,
                "group_topic_interest",
                clean_text or "[动画表情]",
            )

        # 4.7) group_proactive_pity / group_proactive_random (lurking pity every N + random prob)
        count = self._untriggered_counts.get(group_id, 0) + 1
        self._untriggered_counts[group_id] = count

        if proactive_ready:
            if self.proactive_every_n > 0 and count >= self.proactive_every_n:
                self._last_proactive_ts[group_id] = now
                self._untriggered_counts[group_id] = 0
                return TriggerDecision(
                    True,
                    "group_proactive_pity",
                    clean_text or "[图片]",
                )
            if self.proactive_prob > 0.0 and self._rng() < self.proactive_prob:
                self._last_proactive_ts[group_id] = now
                self._untriggered_counts[group_id] = 0
                return TriggerDecision(
                    True,
                    "group_proactive_random",
                    clean_text or "[图片]",
                )

        # 4.8) Default silent
        return TriggerDecision(False, "not_triggered")


__all__ = [
    "DEFAULT_TOPIC_KEYWORDS",
    "RuleTrigger",
]
