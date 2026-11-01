from pathlib import Path

from qq_agent.config import AgentConfig
from qq_agent.llmos_sdk import (
    INBOUND_CHANNEL_CREATE_KEY,
    OUTBOUND_CHANNEL_CREATE_KEY,
    derive_deterministic_id16,
    fiber_id_for_session,
    fiber_id_for_subagent,
    load_nlos_sdk,
    process_id_for_role,
)
from qq_agent.types import (
    ChatMessage,
    ChatResult,
    CommandResult,
    GuardDecision,
    InboundMessage,
    OutboundReply,
    SubAgentRequest,
    SubAgentResult,
    ToolCall,
    ToolDef,
    ToolOutcome,
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
        reply_to_user_id=99999,
        reply_to_sender_name="UD",
        reply_to_text="刚上线",
        mentioned_other=False,
    )
    assert group_msg.session_key == "group:88888"
    encoded = group_msg.to_json_bytes()
    decoded = InboundMessage.from_json_bytes(encoded)
    assert decoded == group_msg

    # Backward compatibility with legacy serialized payloads missing M2 reply fields
    legacy_bytes = (
        b'{"event_id":"9002","chat_type":"private","user_id":12345,"group_id":null,'
        b'"sender_name":"A","text":"hi","raw_message":"hi"}'
    )
    legacy_decoded = InboundMessage.from_json_bytes(legacy_bytes)
    assert legacy_decoded.session_key == "private:12345"
    assert legacy_decoded.reply_to_user_id is None
    assert legacy_decoded.reply_to_text == ""
    assert legacy_decoded.mentioned_other is False


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


def test_value_objects_and_m2_config_defaults():
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

    cfg = AgentConfig.from_env(
        env_file=None,
        environ={
            "QQ_DATA_DIR": "data",
            "QQ_GROUP_PROACTIVE_PROB": "0.25",
            "QQ_GROUP_TOPIC_PROB": "0.70",
            "QQ_GROUP_PROACTIVE_EVERY_N": "5",
        },
    )
    assert cfg.group_proactive_prob == 0.25
    assert cfg.group_topic_prob == 0.70
    assert cfg.group_proactive_every_n == 5
    assert str(cfg.memory_db_path).endswith("memory.sqlite3")
    assert str(cfg.effective_persona_local_path).endswith("persona.local.toml")


def test_w4_tool_subagent_contracts_and_llmos_sdk_identities():
    tdef = ToolDef(
        name="send_message",
        description="Send bubbles",
        input_schema={"type": "object"},
        is_chat_terminal=True,
        requires_followup=False,
    )
    tcall = ToolCall(id="call_1", name="send_message", arguments={"bubbles": ["早）"]})
    tout = ToolOutcome(tool_call_id="call_1", name="send_message", content="sent 1 bubble")
    msg = ChatMessage(role="assistant", content="", tool_calls=(tcall,))
    res = ChatResult(
        content="",
        model="glm-5.3-flash",
        tool_calls=(tcall,),
        stop_reason="tool_use",
    )
    sub_req = SubAgentRequest(
        task_id="sub-1",
        session_key="group:1001",
        goal="查询莫斯科传奇插改动评价",
        reply_to_message_id=8801,
        inbound_seq=12,
    )
    sub_res = SubAgentResult(
        task_id="sub-1",
        session_key="group:1001",
        goal=sub_req.goal,
        brief="改动后射程与散布收益稳定。",
        reply_to_message_id=8801,
        inbound_seq=12,
        steps_used=2,
        total_tokens=900,
    )
    assert tdef.is_chat_terminal and not tdef.requires_followup
    assert msg.tool_calls[0].name == "send_message"
    assert tout.tool_call_id == "call_1"
    assert res.stop_reason == "tool_use"
    assert sub_res.brief.startswith("改动后")

    assert len(INBOUND_CHANNEL_CREATE_KEY) == 16
    assert len(OUTBOUND_CHANNEL_CREATE_KEY) == 16
    assert INBOUND_CHANNEL_CREATE_KEY != OUTBOUND_CHANNEL_CREATE_KEY
    assert INBOUND_CHANNEL_CREATE_KEY == derive_deterministic_id16("qq-agent/channel/inbound/v1")
    assert process_id_for_role("adapter") != process_id_for_role("brain")
    assert fiber_id_for_session("group:1") != fiber_id_for_session("group:2")
    assert fiber_id_for_subagent("group:1", "sub-1") != fiber_id_for_session("group:1")

    cfg = AgentConfig.from_env(
        env_file=None,
        environ={
            "LLM_MODEL": "glm-5.3-flash",
            "SUBAGENT_ENABLE_THINKING": "true",
            "QQ_RUNTIME_ROLE": "brain",
            "QQ_DATA_DIR": "data",
        },
    )
    assert cfg.runtime_role == "brain"
    assert cfg.effective_subagent_model == "glm-5.3-flash"
    assert cfg.subagent_enable_thinking is True
    assert cfg.llmos_daemon_root.is_absolute()

    if Path(r"D:\llmos\sdk\python").is_dir():
        bundle = load_nlos_sdk(r"D:\llmos")
        assert bundle.version == "0.1.0"
        assert bundle.ChannelClient is not None
