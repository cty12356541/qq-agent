from qq_agent.types import (
    ChatMessage,
    ChatResult,
    CommandResult,
    GuardDecision,
    InboundMessage,
    OutboundReply,
    TriggerDecision,
)


def test_inbound_message_roundtrip_and_session_keys():
    group_msg = InboundMessage(
        event_id="9001",
        chat_type="group",
        user_id=12345,
        group_id=88888,
        sender_name="阿哲",
        text="今天打什么游戏？",
        raw_message="[CQ:at,qq=99999] 今天打什么游戏？",
        mentioned_bot=True,
        reply_to_id=9000,
        self_id=99999,
        timestamp=1730000000,
    )
    assert group_msg.session_key == "group:88888"
    encoded = group_msg.to_json_bytes()
    decoded = InboundMessage.from_json_bytes(encoded)
    assert decoded == group_msg

    priv_msg = InboundMessage(
        event_id="9002",
        chat_type="private",
        user_id=12345,
        group_id=None,
        sender_name="阿哲",
        text="在吗",
        raw_message="在吗",
    )
    assert priv_msg.session_key == "private:12345"
    assert InboundMessage.from_json_bytes(priv_msg.to_json_bytes()) == priv_msg


def test_outbound_reply_roundtrip():
    reply = OutboundReply(
        chat_type="group",
        target_id=88888,
        text="来一把战舰世界！",
        reply_to_message_id=9001,
        at_user_id=12345,
        correlation_id="corr-1",
    )
    assert OutboundReply.from_json_bytes(reply.to_json_bytes()) == reply


def test_value_objects_defaults():
    cm = ChatMessage(role="user", content="hello")
    assert cm.role == "user"
    cr = ChatResult(content="hi", model="deepseek-chat", total_tokens=42, cost_micros_usd=120)
    assert cr.total_tokens == 42
    td = TriggerDecision(should_reply=True, reason="group_at", clean_text="hello")
    assert td.should_reply
    gd = GuardDecision(allowed=False, reason="daily_call_limit", refusal_reply="今天额度用完啦")
    assert not gd.allowed
    cmd = CommandResult(handled=True, reply_text="pong", command_name="ping")
    assert cmd.handled
