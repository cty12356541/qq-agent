# tests/test_ob11.py
from qq_agent.ob11 import parse_event

GROUP_MSG = {
    "post_type": "message",
    "message_type": "group",
    "sub_type": "normal",
    "message_id": 123,
    "user_id": 10000,
    "group_id": 70000,
    "raw_message": "你好，机器人",
    "sender": {"user_id": 10000, "nickname": "测试号"},
}

def test_parse_group_message():
    ev = parse_event(GROUP_MSG)
    assert ev is not None
    assert ev.message_type == "group"
    assert ev.user_id == 10000
    assert ev.group_id == 70000
    assert ev.raw_message == "你好，机器人"
    assert ev.payload["message_id"] == 123

def test_parse_private_message_has_no_group():
    data = dict(GROUP_MSG, message_type="private")
    data.pop("group_id")
    ev = parse_event(data)
    assert ev is not None
    assert ev.message_type == "private"
    assert ev.group_id is None

def test_non_message_event_returns_none():
    assert parse_event({"post_type": "notice", "notice_type": "group_increase"}) is None
    assert parse_event({}) is None
