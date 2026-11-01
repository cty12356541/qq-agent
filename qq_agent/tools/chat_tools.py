"""`Use Chat as Tool` anthropomorphic multi-bubble chat tools (`send_message`, `stay_silent`) and text fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from qq_agent.channel_bridge import MessageBus
import qq_agent.stickers as _stickers_mod
from qq_agent.stickers import StickerRegistry, _STICKER_TAG_RE
from qq_agent.tools import ToolArgumentError, ToolBinding
from qq_agent.types import ChatType, InboundMessage, OutboundReply, ToolDef

MAX_BUBBLES_PER_SEND = 3
DEFAULT_TYPING_DELAY_S = 0.45

_SILENT_FALLBACK_TOKENS: frozenset[str] = frozenset(
    {
        "[silent]",
        "[[silent]]",
        "<silent>",
        "silent",
        "[no_reply]",
        "[[no_reply]]",
        "no_reply",
    }
)


@dataclass(slots=True)
class ChatTurnState:
    """Mutable per-turn tracking of outbound bubbles, sticker attachment, and intentional silence."""

    sent_bubbles: list[str] = field(default_factory=list)
    last_sticker_id: str | None = None
    stayed_silent: bool = False
    silence_reason: str = ""


def extract_sticker_tag(raw_text: str) -> tuple[str, str | None]:
    """Strip any inline `[[sticker:...]]` / `[表情包:...]` markers and return `(clean_text, first_tag_token)`."""
    upstream_fn = getattr(_stickers_mod, "extract_sticker_tag", None)
    if callable(upstream_fn):
        clean, sid = upstream_fn(raw_text)
        return str(clean), (str(sid) if sid is not None else None)

    first_token: str | None = None
    for match in _STICKER_TAG_RE.finditer(raw_text):
        candidate = match.group(1).strip()
        if candidate and first_token is None:
            first_token = candidate
    cleaned = _STICKER_TAG_RE.sub("", raw_text).strip()
    return cleaned, first_token


def _resolve_target_route(
    *,
    msg: InboundMessage | None,
    chat_type: ChatType | None,
    target_id: int | None,
    correlation_id: str,
) -> tuple[ChatType, int, str]:
    if msg is not None:
        eff_chat_type: ChatType = msg.chat_type
        if eff_chat_type == "group":
            if msg.group_id is None:
                raise ValueError("Group InboundMessage must have a non-None group_id")
            eff_target_id = int(msg.group_id)
        else:
            eff_target_id = int(msg.user_id)
        eff_corr = correlation_id or msg.event_id
        return eff_chat_type, eff_target_id, eff_corr

    if chat_type not in {"group", "private"} or target_id is None:
        raise ValueError(
            "Either `msg` or both `chat_type` ('group'|'private') and `target_id` must be provided"
        )
    return chat_type, int(target_id), correlation_id


def _build_allowed_sticker_map(
    allowed_sticker_ids: Collection[str] | None,
    stickers: StickerRegistry | None,
) -> dict[str, str]:
    """Build case-insensitive `lower_id -> canonical_id` lookup map for strict sticker whitelisting."""
    if allowed_sticker_ids is not None:
        return {
            str(sid).strip().lower(): str(sid).strip()
            for sid in allowed_sticker_ids
            if isinstance(sid, str) and str(sid).strip()
        }
    if stickers is not None:
        return {
            entry.id.strip().lower(): entry.id.strip()
            for entry in stickers.available_entries()
            if entry.id.strip()
        }
    return {}


def _validate_sticker_id(
    raw_sid: Any,
    allowed_map: dict[str, str],
) -> str | None:
    """Validate optional `sticker_id` against `allowed_map`; return canonical ID or `None` without raising."""
    if not isinstance(raw_sid, str) or not raw_sid.strip():
        return None
    token = raw_sid.strip()
    _, unwrapped = extract_sticker_tag(token)
    if unwrapped:
        token = unwrapped.strip()
    return allowed_map.get(token.lower())


def _parse_optional_message_id(raw_val: Any) -> int | None:
    if raw_val is None or isinstance(raw_val, bool):
        return None
    if isinstance(raw_val, int):
        return raw_val if raw_val != 0 else None
    if isinstance(raw_val, str):
        stripped = raw_val.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
            return parsed if parsed != 0 else None
        except ValueError:
            return None
    return None


def build_send_message_tool(
    bus: MessageBus,
    turn_state: ChatTurnState,
    *,
    msg: InboundMessage | None = None,
    chat_type: ChatType | None = None,
    target_id: int | None = None,
    correlation_id: str = "",
    allowed_sticker_ids: Collection[str] | None = None,
    stickers: StickerRegistry | None = None,
    typing_delay_s: float = DEFAULT_TYPING_DELAY_S,
    sleep_fn: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    compact_schema: bool = False,
) -> ToolBinding:
    """Build the `send_message` terminal chat tool streaming 1~3 anthropomorphic bubbles into `MessageBus`."""
    route_chat_type, route_target_id, base_corr_id = _resolve_target_route(
        msg=msg,
        chat_type=chat_type,
        target_id=target_id,
        correlation_id=correlation_id,
    )
    delay_s = max(0.0, float(typing_delay_s))

    if compact_schema:
        tool_desc = "向当前会话发送1~3条短句气泡回复。"
        bubbles_desc = "1~3条短句气泡。"
        sticker_desc = "可选，末条附带的合法表情包ID。"
        reply_desc = "可选，首条引用的目标消息ID。"
    else:
        initial_allowed = _build_allowed_sticker_map(allowed_sticker_ids, stickers)
        if initial_allowed:
            sample_ids = ", ".join(list(initial_allowed.values())[:16])
            sticker_desc = f"可选，附加在最后一条气泡后的表情包 ID（仅限合法 ID：{sample_ids}）。"
        else:
            sticker_desc = "可选，附加在最后一条气泡后的表情包 ID（非法 ID 会被自动忽略）。"
        tool_desc = (
            "向当前 QQ 会话发送回复消息。请模仿真实群友将发言自然拆成 1~3 条短句气泡（bubbles），"
            "需要引用某条消息时填 reply_to_message_id（仅首条生效），需要配表情包时填 sticker_id（仅末条生效）。"
        )
        bubbles_desc = "1~3 条短句聊天气泡文本列表。"
        reply_desc = "可选，引用回复的目标消息 ID。"

    definition = ToolDef(
        name="send_message",
        description=tool_desc,
        input_schema={
            "type": "object",
            "properties": {
                "bubbles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": MAX_BUBBLES_PER_SEND,
                    "description": bubbles_desc,
                },
                "sticker_id": {
                    "type": "string",
                    "description": sticker_desc,
                },
                "reply_to_message_id": {
                    "type": "integer",
                    "description": reply_desc,
                },
            },
            "required": ["bubbles"],
            "additionalProperties": False,
        },
        is_chat_terminal=True,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_bubbles = args.get("bubbles")
        if isinstance(raw_bubbles, str):
            bubble_candidates = [raw_bubbles]
        elif isinstance(raw_bubbles, (list, tuple)):
            bubble_candidates = [str(item) for item in raw_bubbles if item is not None]
        elif raw_bubbles is None:
            bubble_candidates = []
        else:
            raise ToolArgumentError("'bubbles' must be a list of strings (or a single string)")

        allowed_map = _build_allowed_sticker_map(allowed_sticker_ids, stickers)

        clean_bubbles: list[str] = []
        inline_tag_token: str | None = None
        for raw_bubble in bubble_candidates:
            stripped_text, extracted_tag = extract_sticker_tag(raw_bubble)
            if extracted_tag and inline_tag_token is None:
                inline_tag_token = extracted_tag
            normalized = stripped_text.strip()
            if normalized:
                clean_bubbles.append(normalized)

        clean_bubbles = clean_bubbles[:MAX_BUBBLES_PER_SEND]

        raw_sticker_arg = args.get("sticker_id")
        has_explicit_sticker_arg = isinstance(raw_sticker_arg, str) and bool(raw_sticker_arg.strip())
        if has_explicit_sticker_arg:
            effective_sticker_id = _validate_sticker_id(raw_sticker_arg, allowed_map)
        elif inline_tag_token is not None:
            effective_sticker_id = _validate_sticker_id(inline_tag_token, allowed_map)
        else:
            effective_sticker_id = None

        reply_to_id = _parse_optional_message_id(args.get("reply_to_message_id"))

        if not clean_bubbles and effective_sticker_id is None:
            raise ToolArgumentError(
                "send_message requires at least one non-empty bubble or a valid sticker_id"
            )

        turn_state.stayed_silent = False
        turn_state.silence_reason = ""

        # Pure sticker without text bubbles
        if not clean_bubbles and effective_sticker_id is not None:
            pure_reply = OutboundReply(
                chat_type=route_chat_type,
                target_id=route_target_id,
                text="",
                reply_to_message_id=None,
                at_user_id=None,
                correlation_id=base_corr_id,
                sticker_id=effective_sticker_id,
            )
            await bus.publish_outbound(pure_reply)
            turn_state.last_sticker_id = effective_sticker_id
            return f"Sent pure sticker ({effective_sticker_id})"

        last_idx = len(clean_bubbles) - 1
        for idx, bubble_text in enumerate(clean_bubbles):
            if idx > 0 and delay_s > 0.0:
                await sleep_fn(delay_s)

            bubble_reply_to = reply_to_id if idx == 0 else None
            bubble_sticker = effective_sticker_id if idx == last_idx else None
            if not base_corr_id:
                corr_id = ""
            elif idx == 0:
                corr_id = base_corr_id
            else:
                corr_id = f"{base_corr_id}-{idx + 1}"

            outbound = OutboundReply(
                chat_type=route_chat_type,
                target_id=route_target_id,
                text=bubble_text,
                reply_to_message_id=bubble_reply_to,
                at_user_id=None,
                correlation_id=corr_id,
                sticker_id=bubble_sticker,
            )
            await bus.publish_outbound(outbound)
            turn_state.sent_bubbles.append(bubble_text)

        if effective_sticker_id is not None:
            turn_state.last_sticker_id = effective_sticker_id
            return f"Sent {len(clean_bubbles)} bubble(s) + sticker ({effective_sticker_id})"
        return f"Sent {len(clean_bubbles)} bubble(s)"

    return ToolBinding(definition=definition, execute=_execute)


def build_stay_silent_tool(turn_state: ChatTurnState) -> ToolBinding:
    """Build the `stay_silent` terminal chat tool allowing natural non-intrusive silence in group chats."""
    definition = ToolDef(
        name="stay_silent",
        description=(
            "当群聊话题与你无关、群友之间正在私下对话、或接话会显得生硬打扰时，"
            "调用此工具保持安静不发言。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "简述本轮选择保持安静不接话的原因。",
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
        is_chat_terminal=True,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_reason = args.get("reason")
        reason = str(raw_reason).strip() if raw_reason is not None else ""
        if not reason:
            reason = "no_reply_needed"
        turn_state.stayed_silent = True
        turn_state.silence_reason = reason
        return f"Stayed silent ({reason})"

    return ToolBinding(definition=definition, execute=_execute)


async def flush_fallback_text_reply(
    bus: MessageBus,
    turn_state: ChatTurnState,
    final_text: str,
    *,
    msg: InboundMessage | None = None,
    chat_type: ChatType | None = None,
    target_id: int | None = None,
    correlation_id: str = "",
    reply_to_message_id: int | None = None,
    allowed_sticker_ids: Collection[str] | None = None,
    stickers: StickerRegistry | None = None,
) -> OutboundReply | None:
    """Safety net when fast models (e.g. `glm-5.3-flash`) emit plain text without calling `send_message`."""
    if turn_state.sent_bubbles or turn_state.last_sticker_id is not None or turn_state.stayed_silent:
        return None

    if not isinstance(final_text, str):
        return None
    stripped_raw = final_text.strip()
    if not stripped_raw:
        return None

    if stripped_raw.lower() in _SILENT_FALLBACK_TOKENS:
        turn_state.stayed_silent = True
        turn_state.silence_reason = "fallback_silent_token"
        return None

    route_chat_type, route_target_id, base_corr_id = _resolve_target_route(
        msg=msg,
        chat_type=chat_type,
        target_id=target_id,
        correlation_id=correlation_id,
    )

    sticker_id: str | None = None
    if stickers is not None:
        clean_text, resolved_sid = stickers.extract_from_reply(stripped_raw)
        if allowed_sticker_ids is not None:
            allowed_map = _build_allowed_sticker_map(allowed_sticker_ids, None)
            sticker_id = _validate_sticker_id(resolved_sid, allowed_map)
        else:
            sticker_id = resolved_sid
    else:
        clean_text, raw_tag_sid = extract_sticker_tag(stripped_raw)
        if allowed_sticker_ids is not None:
            allowed_map = _build_allowed_sticker_map(allowed_sticker_ids, None)
            sticker_id = _validate_sticker_id(raw_tag_sid, allowed_map)
        else:
            sticker_id = raw_tag_sid

    clean_text = clean_text.strip()
    if not clean_text and sticker_id is None:
        return None

    outbound = OutboundReply(
        chat_type=route_chat_type,
        target_id=route_target_id,
        text=clean_text,
        reply_to_message_id=reply_to_message_id,
        at_user_id=None,
        correlation_id=base_corr_id,
        sticker_id=sticker_id,
    )
    await bus.publish_outbound(outbound)

    if clean_text:
        turn_state.sent_bubbles.append(clean_text)
    if sticker_id is not None:
        turn_state.last_sticker_id = sticker_id
    return outbound


__all__ = [
    "ChatTurnState",
    "DEFAULT_TYPING_DELAY_S",
    "MAX_BUBBLES_PER_SEND",
    "build_send_message_tool",
    "build_stay_silent_tool",
    "extract_sticker_tag",
    "flush_fallback_text_reply",
]
