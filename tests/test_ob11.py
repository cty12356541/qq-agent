# tests/test_ob11.py
from qq_agent.ob11 import Ob11Event, build_send_action, event_to_inbound, parse_event
from qq_agent.types import OutboundReply

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
    assert isinstance(ev, Ob11Event)
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


def test_parse_event_tolerates_none_fields():
    ev = parse_event(
        {
            "post_type": "message",
            "message_type": None,
            "user_id": None,
            "group_id": None,
            "raw_message": None,
        }
    )
    assert ev is not None
    assert ev.message_type == "private"
    assert ev.user_id == 0
    assert ev.group_id is None
    assert ev.raw_message == ""


def test_event_to_inbound_non_message_returns_none():
    assert event_to_inbound({"post_type": "meta_event", "meta_event_type": "heartbeat"}) is None
    assert event_to_inbound({"post_type": "notice", "notice_type": "group_increase"}) is None
    assert event_to_inbound({"status": "ok", "retcode": 0, "echo": "send-1"}) is None


def test_event_to_inbound_array_segments_at_bot_and_reply():
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 8801,
        "time": 1700001234,
        "self_id": 999,
        "user_id": 10000,
        "group_id": 70000,
        "message": [
            {"type": "at", "data": {"qq": "999"}},
            {"type": "reply", "data": {"id": "55"}},
            {"type": "text", "data": {"text": " 你好 "}},
            {"type": "face", "data": {"id": "14"}},
            {"type": "text", "data": {"text": "在吗？ "}},
        ],
        "raw_message": "[CQ:at,qq=999][CQ:reply,id=55] 你好 [CQ:face,id=14]在吗？ ",
        "sender": {"user_id": 10000, "card": "群名片A", "nickname": "昵称A"},
    }
    msg = event_to_inbound(payload)
    assert msg is not None
    assert msg.event_id == "8801"
    assert msg.chat_type == "group"
    assert msg.user_id == 10000
    assert msg.group_id == 70000
    assert msg.sender_name == "群名片A"
    assert msg.mentioned_bot is True
    assert msg.reply_to_id == 55
    assert msg.text == "你好 在吗？"
    assert msg.self_id == 999
    assert msg.timestamp == 1700001234


def test_event_to_inbound_at_all_and_other_qq_do_not_trigger_mentioned_bot():
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 8802,
        "time": 1700001235,
        "self_id": 999,
        "user_id": 10001,
        "group_id": 70000,
        "message": [
            {"type": "at", "data": {"qq": "all"}},
            {"type": "at", "data": {"qq": "888"}},
            {"type": "text", "data": {"text": " 全体集合"}},
        ],
        "raw_message": "[CQ:at,qq=all][CQ:at,qq=888] 全体集合",
        "sender": {"card": "", "nickname": "群主"},
    }
    msg = event_to_inbound(payload)
    assert msg is not None
    assert msg.mentioned_bot is False
    assert msg.text == "全体集合"

    # Explicit bot_qq parameter overrides self_id when matching @ segments
    msg_override = event_to_inbound(payload, bot_qq=888)
    assert msg_override is not None
    assert msg_override.mentioned_bot is True


def test_event_to_inbound_cq_string_unescape_and_reply():
    payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 8803,
        "time": 1700001236,
        "self_id": 999,
        "user_id": 10002,
        "group_id": 70000,
        "raw_message": (
            "[CQ:reply,id=55][CQ:at,qq=999,name=&#91;bot&#93;&#44;x]"
            "[CQ:reply,id=66] 你好 &amp; &#91;世界&#93;&#44;literal=&amp;#91; "
        ),
        "sender": {"nickname": "CQ测试员"},
    }
    msg = event_to_inbound(payload)
    assert msg is not None
    assert msg.mentioned_bot is True
    assert msg.reply_to_id == 55  # first reply segment wins
    assert msg.text == "你好 & [世界],literal=&#91;"

    # Also ensure CQ-formatted string in `data["message"]` (string post format) works
    cq_all_payload = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 8804,
        "self_id": 999,
        "user_id": 10002,
        "group_id": 70000,
        "message": "[CQ:at,qq=all] 通知",
    }
    msg_all = event_to_inbound(cq_all_payload)
    assert msg_all is not None
    assert msg_all.mentioned_bot is False
    assert msg_all.text == "通知"
    assert msg_all.raw_message == "[CQ:at,qq=all] 通知"


def test_event_to_inbound_sender_card_fallback_chain():
    base = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 9001,
        "user_id": 24680,
        "group_id": 70000,
        "raw_message": "ping",
    }

    with_card = event_to_inbound(
        {**base, "sender": {"card": "专属名片", "nickname": "普通昵称"}}
    )
    assert with_card is not None
    assert with_card.sender_name == "专属名片"

    empty_card = event_to_inbound(
        {**base, "sender": {"card": "   ", "nickname": "普通昵称"}}
    )
    assert empty_card is not None
    assert empty_card.sender_name == "普通昵称"

    no_names = event_to_inbound({**base, "sender": {"card": "", "nickname": ""}})
    assert no_names is not None
    assert no_names.sender_name == "24680"

    missing_sender = event_to_inbound(base)
    assert missing_sender is not None
    assert missing_sender.sender_name == "24680"


def test_build_send_action_group_reply_and_at():
    reply = OutboundReply(
        chat_type="group",
        target_id=70000,
        text="收到指令",
        reply_to_message_id=123,
        at_user_id=10000,
        correlation_id="corr-1",
    )
    action = build_send_action(reply, echo="echo-1")
    assert action == {
        "action": "send_group_msg",
        "params": {
            "group_id": 70000,
            "message": [
                {"type": "reply", "data": {"id": "123"}},
                {"type": "at", "data": {"qq": "10000"}},
                {"type": "text", "data": {"text": " 收到指令"}},
            ],
        },
        "echo": "echo-1",
    }

    plain_group = OutboundReply(chat_type="group", target_id=70000, text="普通群回复")
    assert build_send_action(plain_group) == {
        "action": "send_group_msg",
        "params": {
            "group_id": 70000,
            "message": [{"type": "text", "data": {"text": "普通群回复"}}],
        },
    }


def test_build_send_action_private():
    reply = OutboundReply(
        chat_type="private",
        target_id=10000,
        text="私聊回复",
        reply_to_message_id=456,
        at_user_id=10000,  # ignored in private chat
    )
    assert build_send_action(reply) == {
        "action": "send_private_msg",
        "params": {
            "user_id": 10000,
            "message": [
                {"type": "reply", "data": {"id": "456"}},
                {"type": "text", "data": {"text": "私聊回复"}},
            ],
        },
    }

    plain_private = OutboundReply(chat_type="private", target_id=10000, text="你好")
    assert build_send_action(plain_private, echo="p-1") == {
        "action": "send_private_msg",
        "params": {
            "user_id": 10000,
            "message": [{"type": "text", "data": {"text": "你好"}}],
        },
        "echo": "p-1",
    }
