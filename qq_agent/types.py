"""Shared domain types across qq-agent adapter, trigger, guardrails, memory, tools, and LLM provider."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ChatType = Literal["group", "private"]
MessageRole = Literal["system", "user", "assistant", "tool"]
RuntimeRole = Literal["supervisor", "adapter", "brain", "all-in-one"]


@dataclass(slots=True, frozen=True)
class ToolDef:
    """Normalized client-side tool definition exposed to Anthropic/OpenAI-compatible providers."""

    name: str
    description: str
    input_schema: dict[str, Any]
    is_chat_terminal: bool = False
    requires_followup: bool = True


@dataclass(slots=True, frozen=True)
class ToolCall:
    """Single model-emitted tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolOutcome:
    """Execution result of a single ToolCall fed back to the next agent_loop step."""

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True, frozen=True)
class SubAgentRequest:
    """Depth=1 non-blocking background research sub-agent task specification."""

    task_id: str
    session_key: str
    goal: str
    context_hint: str = ""
    origin_user_id: int = 0
    origin_sender_name: str = ""
    reply_to_message_id: int | None = None
    inbound_seq: int | None = None


@dataclass(slots=True, frozen=True)
class SubAgentResult:
    """Clean distilled brief returned from a depth=1 background sub-agent to SessionActor."""

    task_id: str
    session_key: str
    goal: str
    brief: str
    is_error: bool = False
    cancelled: bool = False
    origin_user_id: int = 0
    origin_sender_name: str = ""
    reply_to_message_id: int | None = None
    inbound_seq: int | None = None
    steps_used: int = 0
    total_tokens: int = 0


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
    image_urls: tuple[str, ...] = ()
    reply_to_user_id: int | None = None
    reply_to_sender_name: str = ""
    reply_to_text: str = ""
    mentioned_other: bool = False

    @property
    def session_key(self) -> str:
        if self.chat_type == "group" and self.group_id is not None:
            return f"group:{self.group_id}"
        return f"private:{self.user_id}"

    def to_json_bytes(self) -> bytes:
        payload = asdict(self)
        payload["image_urls"] = list(self.image_urls)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> InboundMessage:
        data = json.loads(raw.decode("utf-8"))
        chat_type: ChatType = "group" if data.get("chat_type") == "group" else "private"
        group_raw = data.get("group_id")
        reply_raw = data.get("reply_to_id")
        reply_uid_raw = data.get("reply_to_user_id")
        self_raw = data.get("self_id")
        raw_imgs = data.get("image_urls")
        image_urls = (
            tuple(str(u) for u in raw_imgs if isinstance(u, str) and u.strip())
            if isinstance(raw_imgs, list)
            else ()
        )
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
            image_urls=image_urls,
            reply_to_user_id=int(reply_uid_raw) if reply_uid_raw is not None else None,
            reply_to_sender_name=str(data.get("reply_to_sender_name", "")),
            reply_to_text=str(data.get("reply_to_text", "")),
            mentioned_other=bool(data.get("mentioned_other", False)),
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
    sticker_id: str | None = None

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> OutboundReply:
        data = json.loads(raw.decode("utf-8"))
        chat_type: ChatType = "group" if data.get("chat_type") == "group" else "private"
        reply_raw = data.get("reply_to_message_id")
        at_raw = data.get("at_user_id")
        sticker_raw = data.get("sticker_id")
        sticker_id = (
            str(sticker_raw).strip()
            if isinstance(sticker_raw, str) and sticker_raw.strip()
            else None
        )
        return cls(
            chat_type=chat_type,
            target_id=int(data["target_id"]),
            text=str(data["text"]),
            reply_to_message_id=int(reply_raw) if reply_raw is not None else None,
            at_user_id=int(at_raw) if at_raw is not None else None,
            correlation_id=str(data.get("correlation_id", "")),
            sticker_id=sticker_id,
        )


@dataclass(slots=True, frozen=True)
class ChatMessage:
    """Single message turn inside an LLM conversation window (with optional images & tool traces)."""

    role: MessageRole
    content: str
    created_at: float = 0.0
    images: tuple[tuple[str, str], ...] = ()  # tuple of (media_type, base64_data)
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolOutcome, ...] = ()


@dataclass(slots=True, frozen=True)
class ChatResult:
    """Normalized LLM provider completion response with token, micro-USD & tool-call accounting."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_micros_usd: int = 0
    latency_ms: float = 0.0
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = "end_turn"


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
