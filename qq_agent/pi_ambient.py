"""Passive group chat AmbientBuffer ring buffer and inbound burst coalescing helpers."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, NamedTuple

from qq_agent.types import InboundMessage

_SYNTHETIC_PLACEHOLDERS: frozenset[str] = frozenset({"（戳了戳你）", "（回复了你）", "[图片]"})


class AmbientRecord(NamedTuple):
    """4-tuple `(sender_name, user_id, text, timestamp)` stored inside `AmbientBuffer`."""

    sender_name: str
    user_id: int
    text: str
    timestamp: float


class AmbientBuffer:
    """Bounded ring buffer (`deque(maxlen=25)`) tracking passive group chat lines."""

    def __init__(self, maxlen: int = 25) -> None:
        self._maxlen = max(1, int(maxlen))
        self.entries: deque[AmbientRecord] = deque(maxlen=self._maxlen)
        self._unseen_count: int = 0

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def unseen_count(self) -> int:
        return self._unseen_count

    def __len__(self) -> int:
        return len(self.entries)

    def record(
        self,
        sender_name: str,
        user_id: int,
        text: str,
        timestamp: float = 0.0,
    ) -> bool:
        compact = " ".join(text.strip().split())
        if not compact:
            return False
        if len(compact) > 220:
            compact = f"{compact[:220]}..."
        self.entries.append(
            AmbientRecord(
                sender_name=str(sender_name),
                user_id=int(user_id),
                text=compact,
                timestamp=float(timestamp),
            )
        )
        self._unseen_count = min(self._maxlen, self._unseen_count + 1)
        return True

    def record_message(self, msg: InboundMessage) -> bool:
        body = msg.text.strip() or ("[图片]" if msg.image_urls else "")
        return self.record(msg.sender_name, msg.user_id, body, float(msg.timestamp))

    def consume_recent_lines(self, limit: int = 15) -> list[str]:
        if self._unseen_count <= 0 or not self.entries:
            self._unseen_count = 0
            return []
        take_n = min(self._unseen_count, max(1, int(limit)), len(self.entries))
        recent = list(self.entries)[-take_n:]
        self._unseen_count = 0
        return [f"[{r.sender_name}]: {r.text}" for r in recent]

    def snapshot(self) -> list[AmbientRecord]:
        return list(self.entries)

    def clear(self) -> None:
        self.entries.clear()
        self._unseen_count = 0


def _extract_item_user_id(item: Any, explicit_user_id: int | None = None) -> int | None:
    if explicit_user_id is not None:
        return int(explicit_user_id)
    if isinstance(item, InboundMessage):
        return item.user_id
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], InboundMessage)
    ):
        return item[0].user_id
    if isinstance(item, dict) and item.get("user_id") is not None:
        return int(item["user_id"])
    uid_attr = getattr(item, "user_id", None)
    return int(uid_attr) if isinstance(uid_attr, int) else None


def _extract_item_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, InboundMessage):
        return item.text
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], InboundMessage)
        and isinstance(item[1], str)
    ):
        return item[1]
    if isinstance(item, dict) and "text" in item:
        return str(item["text"])
    text_attr = getattr(item, "text", None)
    return str(text_attr) if isinstance(text_attr, str) else str(item)


def _coalesce_inbound_pairs(
    pairs: Sequence[tuple[InboundMessage, str]],
) -> tuple[InboundMessage, str]:
    assert pairs
    if len(pairs) == 1:
        return pairs[0]

    last_msg = pairs[-1][0]
    real_lines: list[str] = []
    fallback_placeholders: list[str] = []
    merged_imgs: list[str] = []
    seen_imgs: set[str] = set()
    quote_source: InboundMessage | None = None
    any_mentioned_bot = False

    for msg, clean_text in pairs:
        if msg.mentioned_bot:
            any_mentioned_bot = True
        if quote_source is None and (
            msg.reply_to_id is not None or msg.reply_to_text.strip()
        ):
            quote_source = msg
        stripped = clean_text.strip()
        if stripped:
            if stripped in _SYNTHETIC_PLACEHOLDERS:
                fallback_placeholders.append(stripped)
            else:
                real_lines.append(stripped)
        for url in msg.image_urls:
            if url not in seen_imgs and len(merged_imgs) < 4:
                seen_imgs.add(url)
                merged_imgs.append(url)

    if real_lines:
        merged_clean = "\n".join(real_lines)
    elif fallback_placeholders:
        merged_clean = fallback_placeholders[0]
    else:
        merged_clean = ""

    merged_msg = replace(
        last_msg,
        text=merged_clean,
        mentioned_bot=any_mentioned_bot or last_msg.mentioned_bot,
        image_urls=tuple(merged_imgs),
        reply_to_id=(
            quote_source.reply_to_id if quote_source is not None else last_msg.reply_to_id
        ),
        reply_to_user_id=(
            quote_source.reply_to_user_id
            if quote_source is not None
            else last_msg.reply_to_user_id
        ),
        reply_to_sender_name=(
            quote_source.reply_to_sender_name
            if quote_source is not None
            else last_msg.reply_to_sender_name
        ),
        reply_to_text=(
            quote_source.reply_to_text
            if quote_source is not None
            else last_msg.reply_to_text
        ),
    )
    return merged_msg, merged_clean


def default_coalesce_items(items: list[Any]) -> Any:
    """Coalesce a same-speaker burst of inbound items into a single turn input."""
    assert items
    if len(items) == 1:
        return items[0]
    if all(isinstance(x, InboundMessage) for x in items):
        merged_msg, _ = _coalesce_inbound_pairs([(m, m.text) for m in items])
        return merged_msg
    if all(
        isinstance(x, tuple)
        and len(x) == 2
        and isinstance(x[0], InboundMessage)
        and isinstance(x[1], str)
        for x in items
    ):
        return _coalesce_inbound_pairs(items)
    if all(isinstance(x, str) for x in items):
        return "\n".join(s for s in items if s)
    if all(isinstance(x, dict) and "text" in x for x in items):
        merged_text = "\n".join(str(d["text"]) for d in items if str(d.get("text", "")))
        return {**items[-1], "text": merged_text}
    return list(items)


__all__ = [
    "AmbientBuffer",
    "AmbientRecord",
    "default_coalesce_items",
]
