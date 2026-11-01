"""Unit tests for `Use Chat as Tool` (`send_message`, `stay_silent`, fallback), scope-locked `remember_note`, and `spawn_subagent`/`cancel_subagent`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from qq_agent.channel_bridge import DirectMemoryBus
from qq_agent.memory import MemoryStore
from qq_agent.tools import (
    ChatTurnState,
    ToolBinding,
    build_cancel_subagent_tool,
    build_remember_note_tool,
    build_send_message_tool,
    build_spawn_subagent_tool,
    build_stay_silent_tool,
    execute_tool_call,
    flush_fallback_text_reply,
    format_tool_error,
    format_tool_result,
)
from qq_agent.types import InboundMessage, ToolCall


def _make_group_msg(
    *,
    event_id: str = "8801",
    group_id: int = 700001,
    user_id: int = 10001,
    sender_name: str = "阿哲",
    text: str = "帮我看看这个问题",
) -> InboundMessage:
    return InboundMessage(
        event_id=event_id,
        chat_type="group",
        user_id=user_id,
        group_id=group_id,
        sender_name=sender_name,
        text=text,
        raw_message=text,
        mentioned_bot=True,
    )


def _make_private_msg(
    *,
    event_id: str = "9902",
    user_id: int = 10001,
    sender_name: str = "管理员",
    text: str = "记一下备忘",
) -> InboundMessage:
    return InboundMessage(
        event_id=event_id,
        chat_type="private",
        user_id=user_id,
        group_id=None,
        sender_name=sender_name,
        text=text,
        raw_message=text,
    )


def test_send_message_streaming_bubbles_first_reply_id_last_sticker_and_truncation() -> None:
    recorded_delays: list[float] = []

    async def _zero_sleep(delay: float) -> None:
        recorded_delays.append(delay)

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        msg = _make_group_msg(event_id="5100", group_id=88888, user_id=20001)
        turn_state = ChatTurnState()

        send_tool = build_send_message_tool(
            bus,
            turn_state,
            msg=msg,
            allowed_sticker_ids={"naiwa_point_laugh", "nailong_stop"},
            typing_delay_s=0.45,
            sleep_fn=_zero_sleep,
        )
        assert isinstance(send_tool, ToolBinding)
        assert send_tool.definition.name == "send_message"
        assert send_tool.definition.is_chat_terminal is True
        assert send_tool.definition.requires_followup is False

        # Pass 5 bubbles (including 1 blank & 1 residual [[sticker:...]] tag) -> filtered & capped to 3
        summary = await send_tool.execute(
            {
                "bubbles": [
                    "先说结论",
                    "   ",
                    "这波确实是你配装歪了 [[sticker:residual_tag]]",
                    "换暴击头就能满星",
                    "第四句超长废话应被截断",
                ],
                "reply_to_message_id": 5100,
                "sticker_id": "naiwa_point_laugh",
            }
        )
        assert "3 bubble(s)" in summary
        assert turn_state.sent_bubbles == [
            "先说结论",
            "这波确实是你配装歪了",
            "换暴击头就能满星",
        ]
        assert turn_state.last_sticker_id == "naiwa_point_laugh"
        assert turn_state.stayed_silent is False
        # Inter-bubble typing delay called only for idx > 0 (2 sleeps for 3 bubbles)
        assert recorded_delays == [0.45, 0.45]

        queued = await bus.poll_outbound(limit=10)
        assert len(queued) == 3
        r0, r1, r2 = [item[1] for item in queued]

        # reply_to_message_id ONLY on bubble 0; sticker_id ONLY on last bubble
        assert r0.chat_type == "group"
        assert r0.target_id == 88888
        assert r0.text == "先说结论"
        assert r0.reply_to_message_id == 5100
        assert r0.sticker_id is None

        assert r1.text == "这波确实是你配装歪了"
        assert r1.reply_to_message_id is None
        assert r1.sticker_id is None

        assert r2.text == "换暴击头就能满星"
        assert r2.reply_to_message_id is None
        assert r2.sticker_id == "naiwa_point_laugh"

    asyncio.run(_run())


def test_send_message_filters_illegal_sticker_id_and_tolerates_single_string_bubble() -> None:
    async def _noop_sleep(_: float) -> None:
        return None

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        msg = _make_group_msg(event_id="5200", group_id=88888)
        turn_state = ChatTurnState()

        send_tool = build_send_message_tool(
            bus,
            turn_state,
            msg=msg,
            allowed_sticker_ids={"naiwa_stare"},
            sleep_fn=_noop_sleep,
        )

        # Illegal sticker_id must NOT raise error and must be stripped to None
        await send_tool.execute(
            {
                "bubbles": "单字符串容错测试",
                "sticker_id": "hallucinated_nonexistent_sticker_999",
            }
        )

        queued = await bus.poll_outbound(limit=10)
        assert len(queued) == 1
        reply = queued[0][1]
        assert reply.text == "单字符串容错测试"
        assert reply.sticker_id is None
        assert turn_state.sent_bubbles == ["单字符串容错测试"]
        assert turn_state.last_sticker_id is None

    asyncio.run(_run())


def test_stay_silent_and_flush_fallback_text_reply_behavior() -> None:
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        msg = _make_group_msg(event_id="5300", group_id=88888)

        # 1) stay_silent marks stayed_silent=True and suppresses fallback text
        silent_state = ChatTurnState()
        silent_tool = build_stay_silent_tool(silent_state)
        assert silent_tool.definition.name == "stay_silent"
        assert silent_tool.definition.is_chat_terminal is True
        assert silent_tool.definition.requires_followup is False

        await silent_tool.execute({"reason": "群友A和群友B在聊私事，不插嘴"})
        assert silent_state.stayed_silent is True
        assert "聊私事" in silent_state.silence_reason

        suppressed = await flush_fallback_text_reply(
            bus,
            silent_state,
            "本来不该发出来的漏网文本",
            msg=msg,
        )
        assert suppressed is None
        assert await bus.poll_outbound() == []

        # 2) [SILENT] fallback token also suppresses outbound reply
        token_state = ChatTurnState()
        assert await flush_fallback_text_reply(bus, token_state, "[SILENT]", msg=msg) is None
        assert token_state.stayed_silent is True
        assert await bus.poll_outbound() == []

        # 3) Plain text fallback when fast model forgot to call send_message
        fallback_state = ChatTurnState()
        fb_reply = await flush_fallback_text_reply(
            bus,
            fallback_state,
            "笑死这操作太下饭了 [[sticker:naiwa_stare]]",
            msg=msg,
            allowed_sticker_ids={"naiwa_stare"},
        )
        assert fb_reply is not None
        assert fb_reply.text == "笑死这操作太下饭了"
        assert fb_reply.sticker_id == "naiwa_stare"
        assert fallback_state.sent_bubbles == ["笑死这操作太下饭了"]
        assert fallback_state.last_sticker_id == "naiwa_stare"

        polled = await bus.poll_outbound()
        assert len(polled) == 1
        assert polled[0][1] == fb_reply

    asyncio.run(_run())


def test_remember_note_hard_locks_group_scope_never_writing_admin_private(tmp_path: Path) -> None:
    admin_qq = 10001
    ordinary_qq = 20002

    async def _run() -> None:
        with MemoryStore(tmp_path / "mem_tool.sqlite3") as store:
            # Admin speaking INSIDE a group chat -> MUST lock to group:700001, NEVER admin_private!
            group_msg_from_admin = _make_group_msg(
                group_id=700001,
                user_id=admin_qq,
                sender_name="群主老王",
            )
            group_mem_tool = build_remember_note_tool(
                store,
                group_msg_from_admin,
                admin_user_ids={admin_qq},
            )
            assert group_mem_tool.definition.name == "remember_note"
            assert group_mem_tool.definition.is_chat_terminal is False
            assert group_mem_tool.definition.requires_followup is False
            assert "scope" not in group_mem_tool.definition.input_schema["properties"]

            # Simulate prompt-injected rogue `scope="admin_private"` argument -> must be ignored!
            res_group = await group_mem_tool.execute(
                {
                    "content": "周五晚八点固定内战文明6",
                    "kind": "fact",
                    "scope": "admin_private",
                }
            )
            assert "scope=group:700001" in res_group
            assert store.list_memories(scope="admin_private", include_admin_private=True) == []
            group_facts = store.list_memories(scope="group:700001", include_admin_private=True)
            assert len(group_facts) == 1
            assert group_facts[0].scope == "group:700001"
            assert group_facts[0].content == "周五晚八点固定内战文明6"

            # Profile note via remember_note
            await group_mem_tool.execute(
                {
                    "content": "文明6巴比伦速科技绝活哥",
                    "kind": "user_profile",
                    "target_user_id": ordinary_qq,
                    "display_name": "小李",
                }
            )
            prof = store.get_user_profile(ordinary_qq)
            assert prof is not None
            assert prof.display_name == "小李"
            assert "巴比伦" in prof.traits_text

            # Admin private chat -> locks to admin_private
            admin_priv_msg = _make_private_msg(user_id=admin_qq)
            admin_priv_tool = build_remember_note_tool(
                store,
                admin_priv_msg,
                admin_user_ids={admin_qq},
            )
            await admin_priv_tool.execute({"content": "下周三提醒续费服务器"})
            admin_facts = store.list_memories(scope="admin_private", include_admin_private=True)
            assert len(admin_facts) == 1
            assert admin_facts[0].content == "下周三提醒续费服务器"

            # Non-admin private chat -> locks to private:<uid>, never admin_private
            user_priv_msg = _make_private_msg(user_id=ordinary_qq, sender_name="小李")
            user_priv_tool = build_remember_note_tool(
                store,
                user_priv_msg,
                admin_user_ids={admin_qq},
            )
            await user_priv_tool.execute(
                {"content": "偏好暗黑模式", "scope": "admin_private"}
            )
            assert len(store.list_memories(scope="admin_private", include_admin_private=True)) == 1
            priv_facts = store.list_memories(scope=f"private:{ordinary_qq}")
            assert len(priv_facts) == 1
            assert priv_facts[0].scope == f"private:{ordinary_qq}"

    asyncio.run(_run())


def test_spawn_subagent_inherits_reply_id_and_cancel_subagent_invokes_callback() -> None:
    spawn_calls: list[tuple[str, str, int | None]] = []
    cancel_calls: list[str | None] = []

    def _fake_spawn(goal: str, context_hint: str, reply_to_message_id: int | None) -> str:
        spawn_calls.append((goal, context_hint, reply_to_message_id))
        return f"subtask-{len(spawn_calls)}"

    async def _fake_cancel(task_id: str | None) -> int:
        cancel_calls.append(task_id)
        return 1 if task_id else 2

    async def _run() -> None:
        trigger_msg = _make_group_msg(event_id="7749", group_id=88888)
        spawn_tool = build_spawn_subagent_tool(_fake_spawn, msg=trigger_msg)
        cancel_tool = build_cancel_subagent_tool(_fake_cancel)

        # Both tools have requires_followup=False for 1-turn non-blocking short-circuit
        assert spawn_tool.definition.is_chat_terminal is False
        assert spawn_tool.definition.requires_followup is False
        assert cancel_tool.definition.is_chat_terminal is False
        assert cancel_tool.definition.requires_followup is False

        out1 = await spawn_tool.execute(
            {
                "goal": "查一下鸣潮 2.x 最新深塔增益与角色梯度对比",
                "context_hint": "群友刚抽了0+1新主C",
            }
        )
        assert "subtask-1" in out1
        assert spawn_calls[0] == (
            "查一下鸣潮 2.x 最新深塔增益与角色梯度对比",
            "群友刚抽了0+1新主C",
            7749,  # Inherited automatically from trigger_msg.event_id
        )

        # Explicit reply_to_message_id override
        await spawn_tool.execute(
            {
                "goal": "顺便对比专武与四星下位差距",
                "reply_to_message_id": 7755,
            }
        )
        assert spawn_calls[1] == ("顺便对比专武与四星下位差距", "", 7755)

        # Cancel specific task_id and cancel all active tasks in session
        c1 = await cancel_tool.execute({"task_id": "subtask-1"})
        c_all = await cancel_tool.execute({})
        assert "1" in c1
        assert "2" in c_all
        assert cancel_calls == ["subtask-1", None]

        # Unified execute_tool_call & error/result formatting helper checks
        bad_outcome = await execute_tool_call(
            spawn_tool,
            ToolCall(id="call_bad", name="spawn_subagent", arguments={"goal": "   "}),
        )
        assert bad_outcome.is_error is True
        assert "[tool_validation_error:spawn_subagent]" in bad_outcome.content

        redacted_err = format_tool_error(
            RuntimeError("Upstream failed with Bearer secret-xyz and sk-abcdef123456"),
            tool_name="web_search",
        )
        assert "secret-xyz" not in redacted_err
        assert "sk-abcdef123456" not in redacted_err
        assert format_tool_result({"ok": True}) == '{"ok": true}'

    asyncio.run(_run())


def test_send_message_and_fallback_unwrap_nested_or_stringified_bubble_lists() -> None:
    """Prevent nested lists or stringified Python/JSON string arrays from leaking `['...']` brackets."""

    async def _noop_sleep(_: float) -> None:
        return None

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        # 1) Nested 2D arrays like `[["戳我干嘛", "我人还在床上瘫着）"], ["怎么 有事？"]]`
        state_nested = ChatTurnState()
        tool_nested = build_send_message_tool(
            bus,
            state_nested,
            chat_type="group",
            target_id=700001,
            sleep_fn=_noop_sleep,
        )
        await tool_nested.execute(
            {
                "bubbles": [["戳我干嘛", "我人还在床上瘫着）"], ["怎么 有事？"]],
            }
        )
        assert state_nested.sent_bubbles == ["戳我干嘛", "我人还在床上瘫着）", "怎么 有事？"]

        # 2) Stringified Python / JSON lists inside `bubbles`
        state_str = ChatTurnState()
        tool_str = build_send_message_tool(
            bus,
            state_str,
            chat_type="group",
            target_id=700001,
            sleep_fn=_noop_sleep,
        )
        await tool_str.execute(
            {
                "bubbles": ["['戳我干嘛', '我人还在床上瘫着）']", '["怎么 有事？"]'],
            }
        )
        assert state_str.sent_bubbles == ["戳我干嘛", "我人还在床上瘫着）", "怎么 有事？"]

        # 3) Legitimate bracketed text (`[图片]`, `[1, 2, 3]`) is NEVER stripped
        state_legit = ChatTurnState()
        tool_legit = build_send_message_tool(
            bus,
            state_legit,
            chat_type="group",
            target_id=700001,
            sleep_fn=_noop_sleep,
        )
        await tool_legit.execute({"bubbles": ["[图片]", "[1, 2, 3]"]})
        assert state_legit.sent_bubbles == ["[图片]", "[1, 2, 3]"]

        # 4) Fallback plain-text reply also unwraps stringified Python list lines
        fb_state = ChatTurnState()
        fb_reply = await flush_fallback_text_reply(
            bus,
            fb_state,
            "['戳我干嘛', '我人还在床上瘫着）']\n['怎么 有事？']",
            chat_type="group",
            target_id=700001,
        )
        assert fb_reply is not None
        assert fb_reply.text == "戳我干嘛\n我人还在床上瘫着）\n怎么 有事？"

    asyncio.run(_run())

