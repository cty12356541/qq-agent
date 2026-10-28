# qq_agent/ob11.py
"""OneBot v11 事件解析（仅 M0 需要的最小面）。"""
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Ob11Event:
    post_type: str
    message_type: str  # "group" | "private"
    user_id: int
    group_id: int | None
    raw_message: str
    payload: dict[str, Any]


def parse_event(data: dict[str, Any]) -> Ob11Event | None:
    """非 message 事件返回 None；字段缺失时尽量容忍（M0 只观察）。"""
    if data.get("post_type") != "message":
        return None
    return Ob11Event(
        post_type=data["post_type"],
        message_type=data.get("message_type", "private"),
        user_id=int(data.get("user_id", 0)),
        group_id=int(data["group_id"]) if data.get("group_id") is not None else None,
        raw_message=str(data.get("raw_message", "")),
        payload=data,
    )
