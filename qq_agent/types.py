"""Shared domain types across qq-agent adapter, trigger, guardrails, memory, and LLM provider."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

ChatType = Literal["group", "private"]
MessageRole = Literal["system", "user", "assistant"]


@dataclass(slots=True, frozen=True)
class InboundMessage:
    """Platform-decoupled inbound chat message carried across llmos Channel or direct queue."""

    event_id: str
    chat_type: ChatType
    user_id: int
    group_id: int | None
    sender_name: str
    text: str
    raw_message: str
    mentioned_bot: bool = False
    reply_to_id: int | None = None
    self_id: int | None = None
    timestamp: int = 0

    @property
    def session_key(self) -> str:
        if self.chat_type == "group" and self.group_id is not None:
            return f"group:{self.group_id}"
        return f"private:{self.user_id}"

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> InboundMessage:
        data = json.loads(raw.decode("utf-8"))
        chat_type: ChatType = "group" if data.get("chat_type") == "group" else "private"
        group_raw = data.get("group_id")
        reply_raw = data.get("reply_to_id")
        self_raw = data.get("self_id")
        return cls(
            event_id=str(data["event_id"]),
            chat_type=chat_type,
            user_id=int(data["user_id"]),
            group_id=int(group_raw) if group_raw is not None else None,
            sender_name=str(data.get("sender_name", "")),
            text=str(data.get("text", "")),
            raw_message=str(data.get("raw_message", "")),
            mentioned_bot=bool(data.get("mentioned_bot", False)),
            reply_to_id=int(reply_raw) if reply_raw is not None else None,
            self_id=int(self_raw) if self_raw is not None else None,
            timestamp=int(data.get("timestamp", 0)),
        )


@dataclass(slots=True, frozen=True)
class OutboundReply:
    """Platform-decoupled outbound reply queued from Agent kernel to channel adapter."""

    chat_type: ChatType
    target_id: int
    text: str
    reply_to_message_id: int | None = None
    at_user_id: int | None = None
    correlation_id: str = ""

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> OutboundReply:
        data = json.loads(raw.decode("utf-8"))
        chat_type: ChatType = "group" if data.get("chat_type") == "group" else "private"
        reply_raw = data.get("reply_to_message_id")
        at_raw = data.get("at_user_id")
        return cls(
            chat_type=chat_type,
            target_id=int(data["target_id"]),
            text=str(data["text"]),
            reply_to_message_id=int(reply_raw) if reply_raw is not None else None,
            at_user_id=int(at_raw) if at_raw is not None else None,
            correlation_id=str(data.get("correlation_id", "")),
        )


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """Single message turn inside an LLM conversation window."""

    role: MessageRole
    content: str
    created_at: float = 0.0


@dataclass(slots=True, frozen=True)
class ChatResult:
    """Normalized LLM provider completion response with token & micro-USD accounting."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_micros_usd: int = 0
    latency_ms: float = 0.0


@dataclass(slots=True, frozen=True)
class TriggerDecision:
    """Outcome of the pre-LLM message trigger filter."""

    should_reply: bool
    reason: str
    clean_text: str = ""


@dataclass(slots=True, frozen=True)
class GuardDecision:
    """Outcome of rate-limit and daily token/cost budget guardrails."""

    allowed: bool
    reason: str
    refusal_reply: str | None = None
    override_model: str | None = None


@dataclass(slots=True, frozen=True)
class CommandResult:
    """Outcome of the pre-LLM admin whitelist command router."""

    handled: bool
    reply_text: str | None = None
    command_name: str = ""
