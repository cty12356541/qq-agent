"""Tier-1 zero-cost rule trigger filter before LLM invocation (§4.2)."""

from __future__ import annotations

from collections.abc import Sequence

from qq_agent.types import InboundMessage, TriggerDecision


class RuleTrigger:
    """Evaluates whether an inbound message should wake the Agent without spending LLM tokens."""

    def __init__(
        self,
        *,
        bot_qq: int | None = None,
        keywords: Sequence[str] = (),
        allow_private: bool = True,
        allowed_group_ids: set[int] | None = None,
    ) -> None:
        self.bot_qq = bot_qq
        self.allow_private = allow_private
        self.allowed_group_ids: set[int] | None = (
            set(allowed_group_ids) if allowed_group_ids is not None else None
        )
        self._keywords: tuple[str, ...] = ()
        self._keywords_lower: tuple[str, ...] = ()
        self.update_keywords(keywords)

    @property
    def keywords(self) -> tuple[str, ...]:
        return self._keywords

    def update_keywords(self, keywords: Sequence[str]) -> None:
        """Hot-update wake keywords (e.g. after persona nickname/alias reload)."""
        cleaned = tuple(kw.strip() for kw in keywords if kw and kw.strip())
        self._keywords = cleaned
        self._keywords_lower = tuple(kw.lower() for kw in cleaned)

    def evaluate(self, msg: InboundMessage) -> TriggerDecision:
        # 1) Anti-self-loop guard
        if (self.bot_qq is not None and msg.user_id == self.bot_qq) or (
            msg.self_id is not None and msg.user_id == msg.self_id
        ):
            return TriggerDecision(False, "ignored_self")

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
            if not clean_text:
                return TriggerDecision(False, "ignored_empty")
            return TriggerDecision(True, "private_direct", clean_text)

        # 4) Group chat rules
        if msg.mentioned_bot:
            effective_text = clean_text if clean_text else "（戳了戳你）"
            return TriggerDecision(True, "group_at", effective_text)

        if not clean_text:
            return TriggerDecision(False, "ignored_empty")

        lowered = clean_text.lower()
        if any(kw in lowered for kw in self._keywords_lower):
            return TriggerDecision(True, "group_keyword", clean_text)

        return TriggerDecision(False, "not_triggered")
