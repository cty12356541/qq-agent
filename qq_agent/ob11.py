# qq_agent/ob11.py
"""OneBot v11 事件解析与消息段构建（兼容 M0 探针、OB11 数组段、CQ 码、图片与原生动画表情包）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .types import ChatType, InboundMessage, OutboundReply

_CQ_TAG_RE = re.compile(r"\[CQ:([^,\]]+)(?:,([^\]]*))?\]")


@dataclass(slots=True)
class Ob11Event:
    post_type: str
    message_type: str  # "group" | "private"
    user_id: int
    group_id: int | None
    raw_message: str
    payload: dict[str, Any]


def parse_event(data: dict[str, Any]) -> Ob11Event | None:
    """非 message 事件返回 None；字段缺失或为 None 时安全容错（100% 兼容 M0 探针）。"""
    if not isinstance(data, dict) or data.get("post_type") != "message":
        return None
    raw_msg = data.get("raw_message")
    return Ob11Event(
        post_type=str(data["post_type"]),
        message_type=str(data.get("message_type") or "private"),
        user_id=int(data.get("user_id") or 0),
        group_id=int(data["group_id"]) if data.get("group_id") is not None else None,
        raw_message=str(raw_msg) if raw_msg is not None else "",
        payload=data,
    )


def _unescape_cq(text: str) -> str:
    """反转义 OneBot v11 CQ 实体（`&amp;` 必须在最后一步替换以避免二次反转义）。"""
    return (
        text.replace("&#91;", "[")
        .replace("&#93;", "]")
        .replace("&#44;", ",")
        .replace("&amp;", "&")
    )


def _parse_cq_segments(cq_text: str) -> list[dict[str, Any]]:
    """将 CQ 码字符串拆解为与 OB11 数组格式等价的消息段列表。"""
    segments: list[dict[str, Any]] = []
    cursor = 0
    for match in _CQ_TAG_RE.finditer(cq_text):
        start, end = match.span()
        if start > cursor:
            segments.append(
                {"type": "text", "data": {"text": _unescape_cq(cq_text[cursor:start])}}
            )
        seg_type = match.group(1).strip()
        params_raw = match.group(2)
        seg_data: dict[str, str] = {}
        if params_raw:
            for part in params_raw.split(","):
                if not part:
                    continue
                key, sep, val = part.partition("=")
                seg_data[key.strip()] = _unescape_cq(val) if sep else ""
        segments.append({"type": seg_type, "data": seg_data})
        cursor = end
    if cursor < len(cq_text):
        segments.append(
            {"type": "text", "data": {"text": _unescape_cq(cq_text[cursor:])}}
        )
    return segments


def _classify_image_placeholder(seg_data: dict[str, Any]) -> str:
    summary = str(seg_data.get("summary") or "").strip()
    sub_type = seg_data.get("sub_type")
    try:
        is_sticker = int(sub_type) == 1 if sub_type is not None else False
    except (TypeError, ValueError):
        is_sticker = False
    if summary:
        return summary
    return "[动画表情]" if is_sticker else "[图片]"


def event_to_inbound(
    data: dict[str, Any],
    bot_qq: int | None = None,
) -> InboundMessage | None:
    """将 OB11 事件字典转换为平台无关的 InboundMessage；非 message 事件返回 None。"""
    if not isinstance(data, dict) or data.get("post_type") != "message":
        return None

    chat_type: ChatType = "group" if data.get("message_type") == "group" else "private"
    user_id = int(data.get("user_id") or 0)
    group_id = (
        int(data["group_id"])
        if chat_type == "group" and data.get("group_id") is not None
        else None
    )
    self_id = int(data["self_id"]) if data.get("self_id") is not None else None
    effective_bot_qq = bot_qq if bot_qq is not None else self_id

    msg_field = data.get("message")
    raw_msg_field = data.get("raw_message")
    if raw_msg_field is not None:
        raw_message = str(raw_msg_field)
    elif isinstance(msg_field, str):
        raw_message = msg_field
    else:
        raw_message = ""

    if isinstance(msg_field, list):
        segments: list[dict[str, Any]] = [
            seg for seg in msg_field if isinstance(seg, dict)
        ]
    else:
        cq_source = msg_field if isinstance(msg_field, str) else raw_message
        segments = _parse_cq_segments(cq_source)

    mentioned_bot = False
    reply_to_id: int | None = None
    text_parts: list[str] = []
    image_urls: list[str] = []
    has_qq_face = False

    for seg in segments:
        seg_type = str(seg.get("type") or "")
        raw_seg_data = seg.get("data")
        seg_data = raw_seg_data if isinstance(raw_seg_data, dict) else {}

        if seg_type == "text":
            txt = seg_data.get("text")
            if txt is not None:
                text_parts.append(str(txt))
        elif seg_type == "at":
            qq_raw = seg_data.get("qq")
            if qq_raw is not None and effective_bot_qq is not None:
                qq_str = str(qq_raw).strip()
                if qq_str.lower() != "all":
                    try:
                        if int(qq_str) == effective_bot_qq:
                            mentioned_bot = True
                    except ValueError:
                        pass
        elif seg_type == "reply" and reply_to_id is None:
            reply_raw = seg_data.get("id")
            if reply_raw is not None:
                try:
                    reply_to_id = int(str(reply_raw).strip())
                except ValueError:
                    pass
        elif seg_type == "image":
            placeholder = _classify_image_placeholder(seg_data)
            if text_parts and not text_parts[-1].endswith((" ", "\n")):
                text_parts.append(" ")
            text_parts.append(placeholder)
            url_val = str(seg_data.get("url") or "").strip()
            file_val = str(seg_data.get("file") or "").strip()
            if url_val.lower().startswith(("http://", "https://")):
                image_urls.append(url_val)
            elif file_val.lower().startswith(("http://", "https://")):
                image_urls.append(file_val)
        elif seg_type == "mface":
            m_summary = str(seg_data.get("summary") or "[商城表情]").strip()
            if text_parts and not text_parts[-1].endswith((" ", "\n")):
                text_parts.append(" ")
            text_parts.append(m_summary)
        elif seg_type == "face":
            has_qq_face = True

    text = "".join(text_parts).strip()
    if not text and has_qq_face:
        text = "[QQ表情]"

    raw_sender = data.get("sender")
    sender = raw_sender if isinstance(raw_sender, dict) else {}
    card = str(sender.get("card") or "").strip()
    nickname = str(sender.get("nickname") or "").strip()
    if card:
        sender_name = card
    elif nickname:
        sender_name = nickname
    else:
        sender_name = str(user_id)

    event_id = str(data.get("message_id") or "")
    timestamp = int(data.get("time") or 0)

    return InboundMessage(
        event_id=event_id,
        chat_type=chat_type,
        user_id=user_id,
        group_id=group_id,
        sender_name=sender_name,
        text=text,
        raw_message=raw_message,
        mentioned_bot=mentioned_bot,
        reply_to_id=reply_to_id,
        self_id=self_id,
        timestamp=timestamp,
        image_urls=tuple(image_urls),
    )


def _wrap_target_frame(
    reply: OutboundReply,
    segments: list[dict[str, Any]],
    echo: str | None = None,
) -> dict[str, Any]:
    if reply.chat_type == "group":
        frame: dict[str, Any] = {
            "action": "send_group_msg",
            "params": {
                "group_id": reply.target_id,
                "message": segments,
            },
        }
    else:
        frame = {
            "action": "send_private_msg",
            "params": {
                "user_id": reply.target_id,
                "message": segments,
            },
        }
    if echo:
        frame["echo"] = echo
    return frame


def build_send_action(
    reply: OutboundReply,
    echo: str | None = None,
) -> dict[str, Any]:
    """根据 OutboundReply 构建单条 OneBot v11 文本发送消息动作帧。"""
    segments: list[dict[str, Any]] = []
    if reply.reply_to_message_id is not None:
        segments.append(
            {"type": "reply", "data": {"id": str(reply.reply_to_message_id)}}
        )

    text_body = reply.text
    if reply.chat_type == "group" and reply.at_user_id is not None:
        segments.append({"type": "at", "data": {"qq": str(reply.at_user_id)}})
        if not text_body.startswith(" "):
            text_body = f" {text_body}"

    segments.append({"type": "text", "data": {"text": text_body}})
    return _wrap_target_frame(reply, segments, echo=echo)


def build_sticker_action(
    reply: OutboundReply,
    sticker_file: str,
    echo: str | None = None,
) -> dict[str, Any]:
    """构建独立无边框原生 QQ `[动画表情]` (`sub_type=1`) 发送动作帧。"""
    segments: list[dict[str, Any]] = [
        {
            "type": "image",
            "data": {
                "file": sticker_file,
                "sub_type": 1,
                "summary": "[动画表情]",
            },
        }
    ]
    return _wrap_target_frame(reply, segments, echo=echo)


def build_send_actions(
    reply: OutboundReply,
    *,
    sticker_file: str | None = None,
    echo: str | None = None,
) -> list[dict[str, Any]]:
    """构建拟人化出站动作帧序列：纯文字 1 帧、纯表情包 1 帧、文字+表情包拆分为 2 帧。"""
    has_text = bool(reply.text.strip())
    if not sticker_file:
        if not has_text:
            return []
        return [build_send_action(reply, echo=echo)]

    if not has_text:
        return [build_sticker_action(reply, sticker_file, echo=echo)]

    sticker_echo = f"{echo}-sticker" if echo else None
    return [
        build_send_action(reply, echo=echo),
        build_sticker_action(reply, sticker_file, echo=sticker_echo),
    ]


__all__ = [
    "Ob11Event",
    "build_send_action",
    "build_send_actions",
    "build_sticker_action",
    "event_to_inbound",
    "parse_event",
]
