# qq_agent/ob11.py
"""OneBot v11 事件解析与消息段构建（兼容 M0 探针、OB11 数组段、CQ 码、合并转发、引用回复、图片与原生动画表情包）。"""

from __future__ import annotations

from collections.abc import Mapping
import re
from dataclasses import dataclass
from typing import Any

from .types import ChatType, InboundMessage, OutboundReply

_CQ_TAG_RE = re.compile(r"\[CQ:([^,\]]+)(?:,([^\]]*))?\]")
_FORWARD_HEADER = "[合并转发聊天记录]"
_FORWARD_TRUNC_SUFFIX = "...（合并转发已截断）"
_MAX_FORWARD_IMAGES = 4


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


def _extract_image_url(seg_data: dict[str, Any]) -> str | None:
    url_val = str(seg_data.get("url") or "").strip()
    file_val = str(seg_data.get("file") or "").strip()
    if url_val.lower().startswith(("http://", "https://")):
        return url_val
    if file_val.lower().startswith(("http://", "https://")):
        return file_val
    return None


def _extract_raw_and_segments(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
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
    return raw_message, segments


def _unwrap_forward_node(raw_node: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """提取单个合并转发节点的发言人（card > nickname > user_id）与消息段列表。"""
    if not isinstance(raw_node, dict):
        return None
    if raw_node.get("type") == "node" and isinstance(raw_node.get("data"), dict):
        inner: dict[str, Any] = raw_node["data"]
    else:
        inner = raw_node

    raw_sender = inner.get("sender")
    if not isinstance(raw_sender, dict):
        outer_sender = raw_node.get("sender")
        raw_sender = outer_sender if isinstance(outer_sender, dict) else {}

    card = str(raw_sender.get("card") or inner.get("card") or "").strip()
    nickname = str(
        raw_sender.get("nickname")
        or inner.get("nickname")
        or inner.get("name")
        or raw_sender.get("name")
        or ""
    ).strip()
    uid_val = (
        raw_sender.get("user_id")
        if raw_sender.get("user_id") is not None
        else (
            inner.get("user_id")
            if inner.get("user_id") is not None
            else (
                inner.get("uin")
                if inner.get("uin") is not None
                else raw_node.get("user_id")
            )
        )
    )
    uid_str = str(uid_val).strip() if uid_val is not None else ""

    if card:
        speaker = card
    elif nickname:
        speaker = nickname
    elif uid_str:
        speaker = uid_str
    else:
        speaker = "未知"

    msg_val = inner.get("message")
    cnt_val = inner.get("content")
    raw_val = inner.get("raw_message")
    if isinstance(msg_val, list) and msg_val:
        segs = [s for s in msg_val if isinstance(s, dict)]
    elif isinstance(cnt_val, list) and cnt_val:
        segs = [s for s in cnt_val if isinstance(s, dict)]
    elif isinstance(msg_val, list) or isinstance(cnt_val, list):
        segs = []
    elif isinstance(msg_val, dict):
        segs = [msg_val]
    elif isinstance(cnt_val, dict):
        segs = [cnt_val]
    elif isinstance(msg_val, str):
        segs = _parse_cq_segments(msg_val)
    elif isinstance(cnt_val, str):
        segs = _parse_cq_segments(cnt_val)
    elif isinstance(raw_val, str):
        segs = _parse_cq_segments(raw_val)
    else:
        segs = []

    return speaker, segs


def format_forward_nodes(
    nodes: list[Any],
    *,
    max_depth: int = 3,
    max_messages: int = 100,
    max_chars: int = 5000,
) -> tuple[str, list[str]]:
    """递归展开 NapCat OB11 合并转发节点列表，返回 `(formatted_text, extracted_image_urls)`。"""
    if not isinstance(nodes, list) or not nodes:
        return _FORWARD_HEADER, []

    effective_max_depth = max(1, int(max_depth))
    effective_max_messages = max(1, int(max_messages))
    effective_max_chars = max(1, int(max_chars))

    lines: list[str] = [_FORWARD_HEADER]
    current_chars = len(_FORWARD_HEADER)
    msg_count = 0
    truncated = False
    extracted_urls: list[str] = []
    seen_urls: set[str] = set()

    def _append_line(line: str) -> bool:
        nonlocal current_chars, truncated
        if truncated:
            return False
        projected = current_chars + 1 + len(line)
        if projected <= effective_max_chars:
            lines.append(line)
            current_chars = projected
            return True
        lines.append(line)
        truncated = True
        return False

    def _walk(node_list: list[Any], depth: int) -> None:
        nonlocal msg_count, truncated
        indent = "  " * (depth - 1)
        for raw_node in node_list:
            if truncated:
                return
            unwrapped = _unwrap_forward_node(raw_node)
            if unwrapped is None:
                continue
            if msg_count >= effective_max_messages:
                truncated = True
                return
            msg_count += 1

            speaker, segs = unwrapped
            inline_parts: list[str] = []
            child_forwards: list[list[Any]] = []
            has_face = False

            for seg in segs:
                seg_type = str(seg.get("type") or "")
                raw_seg_data = seg.get("data")
                seg_data = raw_seg_data if isinstance(raw_seg_data, dict) else {}

                if seg_type == "text":
                    txt = seg_data.get("text")
                    if txt is not None:
                        inline_parts.append(str(txt))
                elif seg_type == "at":
                    name_raw = str(seg_data.get("name") or "").strip()
                    qq_raw = seg_data.get("qq")
                    if name_raw:
                        at_label = name_raw if name_raw.startswith("@") else f"@{name_raw}"
                    elif qq_raw is not None:
                        qq_s = str(qq_raw).strip()
                        at_label = "@全体成员" if qq_s.lower() == "all" else f"@{qq_s}"
                    else:
                        at_label = "@"
                    if inline_parts and not inline_parts[-1].endswith((" ", "\n")):
                        inline_parts.append(" ")
                    inline_parts.append(at_label)
                elif seg_type == "image":
                    placeholder = _classify_image_placeholder(seg_data)
                    if inline_parts and not inline_parts[-1].endswith((" ", "\n")):
                        inline_parts.append(" ")
                    inline_parts.append(placeholder)
                    img_url = _extract_image_url(seg_data)
                    if (
                        img_url is not None
                        and img_url not in seen_urls
                        and len(extracted_urls) < _MAX_FORWARD_IMAGES
                    ):
                        seen_urls.add(img_url)
                        extracted_urls.append(img_url)
                elif seg_type == "mface":
                    m_summary = str(seg_data.get("summary") or "[商城表情]").strip()
                    if inline_parts and not inline_parts[-1].endswith((" ", "\n")):
                        inline_parts.append(" ")
                    inline_parts.append(m_summary)
                elif seg_type == "face":
                    has_face = True
                elif seg_type == "forward":
                    sub_content = seg_data.get("content")
                    if not isinstance(sub_content, list):
                        alt = seg_data.get("messages")
                        sub_content = alt if isinstance(alt, list) else None
                    if inline_parts and not inline_parts[-1].endswith((" ", "\n")):
                        inline_parts.append(" ")
                    if (
                        isinstance(sub_content, list)
                        and sub_content
                        and depth < effective_max_depth
                    ):
                        inline_parts.append(_FORWARD_HEADER)
                        child_forwards.append(sub_content)
                    else:
                        inline_parts.append("[合并转发消息]")

            body = "".join(inline_parts).strip()
            if not body and has_face:
                body = "[QQ表情]"
            line = f"{indent}[{speaker}]: {body}" if body else f"{indent}[{speaker}]:"
            if not _append_line(line):
                return

            for sub_nodes in child_forwards:
                if truncated:
                    return
                _walk(sub_nodes, depth + 1)

    _walk(nodes, 1)

    full_text = "\n".join(lines)
    if truncated:
        suffix_with_nl = f"\n{_FORWARD_TRUNC_SUFFIX}"
        if effective_max_chars > len(_FORWARD_HEADER) + len(suffix_with_nl):
            budget = effective_max_chars - len(suffix_with_nl)
            if len(full_text) > budget:
                full_text = full_text[:budget].rstrip()
        elif len(full_text) > effective_max_chars:
            full_text = full_text[:effective_max_chars].rstrip()
        full_text = f"{full_text}{suffix_with_nl}"

    return full_text, extracted_urls


def extract_unresolved_forward_ids(data: dict[str, Any]) -> list[str]:
    """提取事件 `message` 段（或 CQ 码段）中缺少内联 `data.content` 的 `forward` 段 ID 列表。"""
    if not isinstance(data, dict):
        return []
    _, segments = _extract_raw_and_segments(data)
    unresolved: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        if str(seg.get("type") or "") != "forward":
            continue
        raw_seg_data = seg.get("data")
        seg_data = raw_seg_data if isinstance(raw_seg_data, dict) else {}
        content = seg_data.get("content")
        if isinstance(content, list) and len(content) > 0:
            continue
        raw_id = seg_data.get("id")
        if raw_id is None:
            continue
        fid = str(raw_id).strip()
        if fid and fid not in seen:
            seen.add(fid)
            unresolved.append(fid)
    return unresolved


def event_to_inbound(
    data: dict[str, Any],
    bot_qq: int | None = None,
    *,
    resolved_forwards: Mapping[str, list[Any]] | None = None,
) -> InboundMessage | None:
    """将 OB11 事件字典转换为平台无关的 InboundMessage；非 message 事件返回 None。"""
    if not isinstance(data, dict) or data.get("post_type") != "message":
        return None

    raw_sender = data.get("sender")
    sender = raw_sender if isinstance(raw_sender, dict) else {}

    chat_type: ChatType = "group" if data.get("message_type") == "group" else "private"
    user_id = int(data.get("user_id") or sender.get("user_id") or 0)
    group_id = (
        int(data["group_id"])
        if chat_type == "group" and data.get("group_id") is not None
        else None
    )
    self_id = int(data["self_id"]) if data.get("self_id") is not None else None
    effective_bot_qq = bot_qq if bot_qq is not None else self_id

    raw_message, segments = _extract_raw_and_segments(data)

    mentioned_bot = False
    mentioned_other = False
    reply_to_id: int | None = None
    text_parts: list[str] = []
    image_urls: list[str] = []
    seen_image_urls: set[str] = set()
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
            if qq_raw is not None:
                qq_str = str(qq_raw).strip()
                if qq_str.lower() == "all":
                    mentioned_other = True
                elif effective_bot_qq is not None:
                    try:
                        if int(qq_str) == effective_bot_qq:
                            mentioned_bot = True
                        else:
                            mentioned_other = True
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
            img_url = _extract_image_url(seg_data)
            if img_url is not None and img_url not in seen_image_urls:
                seen_image_urls.add(img_url)
                image_urls.append(img_url)
        elif seg_type == "mface":
            m_summary = str(seg_data.get("summary") or "[商城表情]").strip()
            if text_parts and not text_parts[-1].endswith((" ", "\n")):
                text_parts.append(" ")
            text_parts.append(m_summary)
        elif seg_type == "face":
            has_qq_face = True
        elif seg_type == "forward":
            fwd_content = seg_data.get("content")
            if not (isinstance(fwd_content, list) and fwd_content) and resolved_forwards:
                fid = str(seg_data.get("id") or "").strip()
                if fid and fid in resolved_forwards:
                    fwd_content = resolved_forwards[fid]
            if isinstance(fwd_content, list) and fwd_content:
                fwd_text, fwd_imgs = format_forward_nodes(fwd_content)
                if text_parts and not text_parts[-1].endswith((" ", "\n")):
                    text_parts.append("\n")
                text_parts.append(fwd_text)
                for u in fwd_imgs:
                    if u not in seen_image_urls:
                        seen_image_urls.add(u)
                        image_urls.append(u)
            else:
                if text_parts and not text_parts[-1].endswith((" ", "\n")):
                    text_parts.append(" ")
                text_parts.append("[合并转发消息]")

    text = "".join(text_parts).strip()
    if not text and has_qq_face:
        text = "[QQ表情]"

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
        mentioned_other=mentioned_other,
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


def build_voice_action(
    reply: OutboundReply,
    echo: str | None = None,
) -> dict[str, Any] | None:
    """构建 QQ 语音 (`record`) 或 NapCat 群 AI 声线 (`send_group_ai_record`) 发送动作帧。"""
    if reply.voice_uri:
        segments: list[dict[str, Any]] = [
            {
                "type": "record",
                "data": {
                    "file": reply.voice_uri,
                },
            }
        ]
        return _wrap_target_frame(reply, segments, echo=echo)

    if reply.ai_voice_character and reply.chat_type == "group" and reply.text.strip():
        frame: dict[str, Any] = {
            "action": "send_group_ai_record",
            "params": {
                "group_id": reply.target_id,
                "character": reply.ai_voice_character,
                "text": reply.text.strip(),
            },
        }
        if echo:
            frame["echo"] = echo
        return frame

    return None


def build_send_actions(
    reply: OutboundReply,
    *,
    sticker_file: str | None = None,
    echo: str | None = None,
) -> list[dict[str, Any]]:
    """构建拟人化出站动作帧序列：支持纯文字、纯语音、纯表情包，以及文字/语音+表情包双帧拆分。"""
    voice_frame = build_voice_action(reply, echo=echo)
    if voice_frame is not None:
        if not sticker_file:
            return [voice_frame]
        sticker_echo = f"{echo}-sticker" if echo else None
        return [
            voice_frame,
            build_sticker_action(reply, sticker_file, echo=sticker_echo),
        ]

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
    "build_voice_action",
    "event_to_inbound",
    "extract_unresolved_forward_ids",
    "format_forward_nodes",
    "parse_event",
]
