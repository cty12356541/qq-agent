from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from qq_agent.pi_core import (
    AmbientBuffer,
    BudgetBlockedError,
    CancelToken,
    LlmStepEvent,
    RegisteredTool,
    SessionActor,
    SteeringInjectedEvent,
    SubAgentRunner,
    ToolExecutionEvent,
    TurnEndEvent,
    TurnStartEvent,
    agent_loop,
)
from qq_agent.types import (
    ChatMessage,
    ChatResult,
    InboundMessage,
    SubAgentRequest,
    SubAgentResult,
    ToolCall,
    ToolDef,
    ToolOutcome,
)


def _send_message_tool(sent_log: list[str]) -> RegisteredTool:
    async def _handler(args: dict[str, Any], token: CancelToken) -> str:
        token.raise_if_cancelled()
        bubbles = args.get("bubbles") or []
        for b in bubbles:
            sent_log.append(str(b))
        return f"sent {len(bubbles)} bubbles"

    return RegisteredTool(
        definition=ToolDef(
            name="send_message",
            description="Send QQ chat bubbles",
            input_schema={"type": "object"},
            is_chat_terminal=True,
            requires_followup=False,
        ),
        handler=_handler,
    )


def _remember_note_tool(notes_log: list[str]) -> RegisteredTool:
    async def _handler(args: dict[str, Any], token: CancelToken) -> str:
        token.raise_if_cancelled()
        note = str(args.get("note", ""))
        notes_log.append(note)
        return "saved note"

    return RegisteredTool(
        definition=ToolDef(
            name="remember_note",
            description="Save long-term memory note",
            input_schema={"type": "object"},
            is_chat_terminal=False,
            requires_followup=False,
        ),
        handler=_handler,
    )


def test_single_hop_terminal_short_circuit_and_combo_with_spawn_subagent() -> None:
    async def run() -> None:
        sent_bubbles: list[str] = []
        notes_saved: list[str] = []
        spawned_goals: list[str] = []
        llm_calls = 0

        class ScriptedFastProvider:
            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                nonlocal llm_calls
                llm_calls += 1
                assert enable_thinking is False
                if llm_calls == 1:
                    # Pure casual chat -> 1 send_message tool call -> must short-circuit in 1 hop
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        prompt_tokens=100,
                        completion_tokens=20,
                        total_tokens=120,
                        cost_micros_usd=20,
                        tool_calls=(
                            ToolCall("c1", "send_message", {"bubbles": ["早呀，今晚打战舰吗）"]}),
                        ),
                        stop_reason="tool_use",
                    )
                # Turn 2: send_message + spawn_subagent + remember_note in a single step
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    prompt_tokens=140,
                    completion_tokens=35,
                    total_tokens=175,
                    cost_micros_usd=30,
                    tool_calls=(
                        ToolCall("c_spawn", "spawn_subagent", {"goal": "查莫斯科传奇插改动"}),
                        ToolCall("c_send_1", "send_message", {"bubbles": ["稍等，我去翻下帖子）"]}),
                        ToolCall("c_note", "remember_note", {"note": "阿哲关心莫斯科传奇插"}),
                        ToolCall("c_send_2", "send_message", {"bubbles": ["查完喊你"]}),
                    ),
                    stop_reason="tool_use",
                )

        async def _spawn_handler(args: dict[str, Any], token: CancelToken) -> ToolOutcome:
            token.raise_if_cancelled()
            # Ensure terminal bubbles finished sequentially BEFORE non-terminal parallel actions
            assert sent_bubbles[-2:] == ["稍等，我去翻下帖子）", "查完喊你"]
            spawned_goals.append(str(args["goal"]))
            return ToolOutcome(tool_call_id="", name="spawn_subagent", content="spawned sub-1")

        spawn_tool = RegisteredTool(
            definition=ToolDef(
                name="spawn_subagent",
                description="Spawn depth=1 background researcher",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=False,
            ),
            handler=_spawn_handler,
        )

        provider = ScriptedFastProvider()
        tools = [
            _send_message_tool(sent_bubbles),
            _remember_note_tool(notes_saved),
            spawn_tool,
        ]

        # 1) Pure send_message short-circuits in 1 LLM call
        events_1 = [
            ev
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content="早！")],
                tools=tools,
                enable_thinking=False,
                max_steps=3,
            )
        ]
        assert llm_calls == 1
        assert isinstance(events_1[0], TurnStartEvent)
        assert isinstance(events_1[1], LlmStepEvent)
        assert isinstance(events_1[2], ToolExecutionEvent)
        end_1 = events_1[-1]
        assert isinstance(end_1, TurnEndEvent)
        assert end_1.reason == "terminal_tool_short_circuit"
        assert end_1.steps_used == 1
        assert end_1.total_tokens == 120
        assert sent_bubbles == ["早呀，今晚打战舰吗）"]

        # 2) send_message + spawn_subagent + remember_note ALSO short-circuits in 1 LLM call
        events_2 = [
            ev
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content="帮我深扒一下莫斯科传奇插")],
                tools=tools,
                enable_thinking=False,
                max_steps=3,
            )
        ]
        assert llm_calls == 2  # Still only 1 LLM request for Turn 2!
        end_2 = events_2[-1]
        assert isinstance(end_2, TurnEndEvent)
        assert end_2.reason == "terminal_tool_short_circuit"
        assert end_2.steps_used == 1
        assert spawned_goals == ["查莫斯科传奇插改动"]
        assert notes_saved == ["阿哲关心莫斯科传奇插"]
        tool_events_2 = [e for e in events_2 if isinstance(e, ToolExecutionEvent)]
        assert [e.call.id for e in tool_events_2] == [
            "c_spawn",
            "c_send_1",
            "c_note",
            "c_send_2",
        ]

    asyncio.run(run())


def test_two_hop_web_search_parallel_gather_last_step_filter_and_budget_guard() -> None:
    async def run() -> None:
        sent_bubbles: list[str] = []
        entered_a = asyncio.Event()
        entered_b = asyncio.Event()
        seen_step_tools: list[list[str]] = []
        accounted_tokens: list[int] = []

        async def _search_handler(args: dict[str, Any], token: CancelToken) -> str:
            q = str(args["query"])
            if q == "FAIL_ME":
                raise RuntimeError("upstream search 502")
            # Rendezvous barrier: proves both web_search calls execute concurrently via asyncio.gather!
            if "A" in q:
                entered_a.set()
                await asyncio.wait_for(entered_b.wait(), timeout=1.0)
                return "结果A：射程+8%"
            entered_b.set()
            await asyncio.wait_for(entered_a.wait(), timeout=1.0)
            return "结果B：主炮散布-7%"

        search_tool = RegisteredTool(
            definition=ToolDef(
                name="web_search",
                description="Web search",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_search_handler,
        )

        class TwoHopProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                self.calls += 1
                seen_step_tools.append([t.name for t in (tools or [])])
                if self.calls == 1:
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        total_tokens=100,
                        cost_micros_usd=15,
                        tool_calls=(
                            ToolCall("s1", "web_search", {"query": "Query A"}),
                            ToolCall("s2", "web_search", {"query": "Query B"}),
                            ToolCall("s3", "web_search", {"query": "FAIL_ME"}),
                        ),
                        stop_reason="tool_use",
                    )
                # Step 2 (final step when max_steps=2): verify tool results arrived in order
                tool_msg = messages[-1]
                assert tool_msg.role == "tool"
                assert [r.tool_call_id for r in tool_msg.tool_results] == ["s1", "s2", "s3"]
                assert "结果A" in tool_msg.tool_results[0].content
                assert "结果B" in tool_msg.tool_results[1].content
                assert tool_msg.tool_results[2].is_error is True
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    total_tokens=150,
                    cost_micros_usd=25,
                    tool_calls=(
                        ToolCall(
                            "fin",
                            "send_message",
                            {"bubbles": ["查到啦：射程+8%，散布-7%）"]},
                        ),
                    ),
                    stop_reason="tool_use",
                )

        provider = TwoHopProvider()
        events = [
            ev
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content="莫斯科传奇插属性？")],
                tools=[_send_message_tool(sent_bubbles), search_tool],
                max_steps=2,
                after_step=lambda r: accounted_tokens.append(r.total_tokens),
            )
        ]
        end_ev = events[-1]
        assert isinstance(end_ev, TurnEndEvent)
        assert end_ev.reason == "terminal_tool_short_circuit"
        assert end_ev.steps_used == 2
        assert end_ev.total_tokens == 250
        assert end_ev.total_cost_micros == 40
        assert accounted_tokens == [100, 150]
        assert sent_bubbles == ["查到啦：射程+8%，散布-7%）"]
        # Step 0 exposed both tools; Step 1 (max_steps-1) automatically filtered out web_search!
        assert seen_step_tools[0] == ["send_message", "web_search"]
        assert seen_step_tools[1] == ["send_message"]

        # Verify BudgetBlockedError safe termination in before_step
        def _block_on_step(_step: int) -> str | None:
            raise BudgetBlockedError("今日额度已耗尽，明天再聊~")

        blocked_events = [
            ev
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content="还在吗")],
                before_step=_block_on_step,
            )
        ]
        blocked_end = blocked_events[-1]
        assert isinstance(blocked_end, TurnEndEvent)
        assert blocked_end.reason == "budget_blocked"
        assert blocked_end.final_text == "今日额度已耗尽，明天再聊~"
        assert blocked_end.steps_used == 0

    asyncio.run(run())


def test_mid_turn_steer_injection_at_tool_boundary_changes_second_hop() -> None:
    async def run() -> None:
        sent_bubbles: list[str] = []
        search_in_flight = asyncio.Event()
        allow_search_finish = asyncio.Event()
        captured_hop2_messages: list[ChatMessage] = []
        observed_events: list[Any] = []

        async def _slow_search(args: dict[str, Any], token: CancelToken) -> str:
            search_in_flight.set()
            await allow_search_finish.wait()
            return f"旧查询结果({args['query']})"

        search_tool = RegisteredTool(
            definition=ToolDef(
                name="web_search",
                description="Search web",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_slow_search,
        )

        class SteerableProvider:
            def __init__(self) -> None:
                self.hop = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                self.hop += 1
                if self.hop == 1:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(ToolCall("w1", "web_search", {"query": "战舰世界更新"}),),
                        stop_reason="tool_use",
                    )
                captured_hop2_messages.extend(messages)
                last_user = messages[-1]
                assert last_user.role == "user"
                assert "<steering_update>" in last_user.content
                assert "改查明天上海天气" in last_user.content
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            "s_final",
                            "send_message",
                            {"bubbles": ["收到！改看上海天气：明天晴转多云）"]},
                        ),
                    ),
                    stop_reason="tool_use",
                )

        provider = SteerableProvider()

        async def _turn_handler(
            msg: InboundMessage,
            token: CancelToken,
            drain_steering: Any,
        ) -> None:
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content=msg.text)],
                tools=[_send_message_tool(sent_bubbles), search_tool],
                cancel_token=token,
                drain_steering=drain_steering,
                max_steps=3,
            ):
                observed_events.append(ev)

        actor = SessionActor("group:9001", _turn_handler, debounce_s=0.0)
        try:
            first_msg = InboundMessage(
                event_id="1",
                chat_type="group",
                user_id=1001,
                group_id=9001,
                sender_name="阿哲",
                text="帮我查下战舰世界更新",
                raw_message="帮我查下战舰世界更新",
                mentioned_bot=True,
            )
            route_1 = actor.submit(first_msg, steer_if_same_user=True)
            assert route_1 == "debouncing"

            # Wait until Hop 1's web_search is running inside SessionActor
            await asyncio.wait_for(search_in_flight.wait(), timeout=2.0)
            assert actor.is_running

            # Same user sends a live steering correction mid-turn
            second_msg = InboundMessage(
                event_id="2",
                chat_type="group",
                user_id=1001,
                group_id=9001,
                sender_name="阿哲",
                text="不对，改查明天上海天气",
                raw_message="不对，改查明天上海天气",
                mentioned_bot=True,
            )
            route_2 = actor.submit(second_msg, steer_if_same_user=True)
            assert route_2 == "steered"

            allow_search_finish.set()
            await actor.wait_idle(timeout=2.0)
        finally:
            await actor.aclose()

        steer_events = [e for e in observed_events if isinstance(e, SteeringInjectedEvent)]
        assert len(steer_events) == 1
        assert steer_events[0].texts == ("不对，改查明天上海天气",)
        assert sent_bubbles == ["收到！改看上海天气：明天晴转多云）"]
        assert any("<steering_update>" in m.content for m in captured_hop2_messages)

    asyncio.run(run())


def test_subagent_depth1_guard_zero_blocking_casual_chat_and_callback_wakeup() -> None:
    async def run() -> None:
        # 1) Hard depth=1 anti-recursion & terminal-tool rejection at SubAgentRunner init
        dummy_spawn = RegisteredTool(
            definition=ToolDef(
                name="spawn_subagent",
                description="Forbidden recursive spawn",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=False,
            ),
            handler=lambda _a, _t: asyncio.sleep(0, result="no"),
        )

        class DualBrainProvider:
            def __init__(self) -> None:
                self.fast_calls = 0
                self.slow_calls = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                if enable_thinking:
                    self.slow_calls += 1
                    sys_prompt = messages[0].content
                    assert "depth=1" in sys_prompt
                    if self.slow_calls == 1:
                        return ChatResult(
                            content="",
                            model="glm-5.3-flash",
                            total_tokens=200,
                            tool_calls=(
                                ToolCall("sub_s1", "deep_research", {"topic": "莫斯科传奇插"}),
                            ),
                            stop_reason="tool_use",
                        )
                    return ChatResult(
                        content="客观简报：莫斯科传奇插提供+8%射程与-7%最大散布，社区评价为T1必换配件。",
                        model="glm-5.3-flash",
                        total_tokens=260,
                        stop_reason="end_turn",
                    )

                self.fast_calls += 1
                user_text = messages[-1].content
                if "深扒莫斯科" in user_text:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        total_tokens=90,
                        tool_calls=(
                            ToolCall("f_send", "send_message", {"bubbles": ["我去翻下泥潭帖子）"]}),
                            ToolCall("f_spawn", "spawn_subagent", {"goal": "深扒莫斯科传奇插评价"}),
                        ),
                        stop_reason="tool_use",
                    )
                # Casual chat from another group member while sub-agent is running!
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    total_tokens=70,
                    tool_calls=(
                        ToolCall("f_bob", "send_message", {"bubbles": ["来！十级房走起"]}),
                    ),
                    stop_reason="tool_use",
                )

        provider = DualBrainProvider()

        with pytest.raises(ValueError, match="SubAgent depth>1 recursion is forbidden"):
            SubAgentRunner(provider, [dummy_spawn])

        with pytest.raises(ValueError, match="chat-terminal"):
            SubAgentRunner(provider, [_send_message_tool([])])

        # 2) Slow research tool gated by asyncio.Event so we deterministically interleave Bob's chat
        sub_research_started = asyncio.Event()
        sub_research_release = asyncio.Event()

        async def _deep_research_handler(args: dict[str, Any], token: CancelToken) -> str:
            sub_research_started.set()
            await sub_research_release.wait()
            token.raise_if_cancelled()
            return f"论坛实测数据：{args['topic']} 散布收益显著"

        research_tool = RegisteredTool(
            definition=ToolDef(
                name="deep_research",
                description="Slow deep webpage research",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_deep_research_handler,
        )

        runner = SubAgentRunner(
            provider,
            [research_tool],
            enable_thinking=True,
            max_steps=4,
            timeout_s=10.0,
            max_concurrent=1,
        )

        timeline: list[str] = []
        callback_done = asyncio.Event()

        async def _on_subagent_result(res: SubAgentResult, token: CancelToken) -> None:
            assert not token.is_cancelled
            assert res.task_id == "sub-moskva"
            assert res.reply_to_message_id == 101
            assert "T1必换配件" in res.brief
            timeline.append(f"subagent_callback:{res.origin_sender_name}:{res.brief[:12]}")
            callback_done.set()

        actor_ref: dict[str, SessionActor] = {}

        async def _spawn_tool_handler(args: dict[str, Any], _token: CancelToken) -> str:
            req = SubAgentRequest(
                task_id="sub-moskva",
                session_key="group:88888",
                goal=str(args["goal"]),
                origin_user_id=1001,
                origin_sender_name="Alice",
                reply_to_message_id=101,
            )
            handle = runner.spawn(req, on_complete=actor_ref["actor"].enqueue_subagent_result)
            assert handle.task_id == "sub-moskva"
            # Concurrency limit (max_concurrent=1) enforced cleanly
            with pytest.raises(RuntimeError, match="subagent concurrency limit reached"):
                runner.spawn(
                    SubAgentRequest(
                        task_id="sub-extra",
                        session_key="group:88888",
                        goal="extra",
                    )
                )
            return f"spawned {handle.task_id}"

        spawn_tool = RegisteredTool(
            definition=ToolDef(
                name="spawn_subagent",
                description="Spawn background researcher",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=False,
            ),
            handler=_spawn_tool_handler,
        )

        async def _main_turn_handler(
            msg: InboundMessage,
            token: CancelToken,
            drain_steering: Any,
        ) -> None:
            bubbles: list[str] = []
            async for _ in agent_loop(
                provider,
                [ChatMessage(role="user", content=msg.text)],
                tools=[_send_message_tool(bubbles), spawn_tool],
                enable_thinking=False,
                cancel_token=token,
                drain_steering=drain_steering,
            ):
                pass
            for b in bubbles:
                timeline.append(f"reply_to_{msg.sender_name}:{b}")

        actor = SessionActor(
            "group:88888",
            _main_turn_handler,
            subagent_result_handler=_on_subagent_result,
            subagent_runner=runner,
            debounce_s=0.02,
        )
        actor_ref["actor"] = actor

        try:
            # Verify AmbientBuffer tuple contract (maxlen=25, 4-tuple)
            amb = AmbientBuffer(maxlen=25)
            for idx in range(30):
                actor.record_ambient("路人", 3000 + idx, f"闲聊第{idx}句", float(idx))
                amb.record("路人", 3000 + idx, f"闲聊第{idx}句", float(idx))
            assert len(actor.ambient) == 25
            sname, uid, txt, ts = actor.ambient.entries[0]
            assert (sname, uid, txt, ts) == ("路人", 3005, "闲聊第5句", 5.0)

            # Alice sends 2 rapid burst lines -> coalesced by SessionActor debounce!
            alice_1 = InboundMessage(
                event_id="100",
                chat_type="group",
                user_id=1001,
                group_id=88888,
                sender_name="Alice",
                text="UD在吗",
                raw_message="UD在吗",
                mentioned_bot=True,
            )
            alice_2 = InboundMessage(
                event_id="101",
                chat_type="group",
                user_id=1001,
                group_id=88888,
                sender_name="Alice",
                text="帮我深扒莫斯科传奇插",
                raw_message="帮我深扒莫斯科传奇插",
                mentioned_bot=True,
            )
            actor.submit(alice_1)
            actor.submit(alice_2)

            # Wait until Alice's fast turn finishes AND background sub-agent has entered deep_research
            await asyncio.wait_for(sub_research_started.wait(), timeout=2.0)
            await actor.wait_idle(timeout=2.0)
            assert timeline == ["reply_to_Alice:我去翻下泥潭帖子）"]
            assert len(runner.active_tasks()) == 1

            # While SubAgent is STILL blocked in background research, Bob chats in the SAME group!
            bob_msg = InboundMessage(
                event_id="102",
                chat_type="group",
                user_id=1002,
                group_id=88888,
                sender_name="Bob",
                text="今晚有人开黑吗",
                raw_message="今晚有人开黑吗",
                mentioned_bot=True,
            )
            actor.submit(bob_msg)
            await actor.wait_idle(timeout=2.0)

            # Bob got an instant reply BEFORE sub_research_release was even set!
            assert timeline == [
                "reply_to_Alice:我去翻下泥潭帖子）",
                "reply_to_Bob:来！十级房走起",
            ]
            assert len(runner.active_tasks()) == 1

            # Now release the slow-brain SubAgent -> triggers enqueue_subagent_result wakeup!
            sub_research_release.set()
            await asyncio.wait_for(callback_done.wait(), timeout=2.0)
            await actor.wait_idle(timeout=2.0)

            assert len(timeline) == 3
            assert timeline[2].startswith("subagent_callback:Alice:客观简报：莫斯科传奇插")
            assert runner.active_tasks() == []
            assert provider.fast_calls == 2
            assert provider.slow_calls == 2
        finally:
            await actor.aclose()
            await runner.aclose()

    asyncio.run(run())


def test_stop_and_cancel_token_millisecond_preemption_of_tools_and_subagents() -> None:
    async def run() -> None:
        main_tool_started = asyncio.Event()
        sub_tool_started = asyncio.Event()
        executed_turns: list[str] = []
        sub_callbacks_fired: list[SubAgentResult] = []

        async def _very_slow_tool(_args: dict[str, Any], _token: CancelToken) -> str:
            main_tool_started.set()
            await asyncio.sleep(15.0)
            return "should never finish"

        async def _very_slow_sub_tool(_args: dict[str, Any], _token: CancelToken) -> str:
            sub_tool_started.set()
            await asyncio.sleep(15.0)
            return "sub should never finish"

        slow_main_reg = RegisteredTool(
            definition=ToolDef(
                name="web_search",
                description="Slow search",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_very_slow_tool,
        )
        slow_sub_reg = RegisteredTool(
            definition=ToolDef(
                name="sub_search",
                description="Slow subagent search",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_very_slow_sub_tool,
        )

        class AlwaysToolProvider:
            def __init__(self, tool_name: str) -> None:
                self._tool_name = tool_name

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(ToolCall("slow_1", self._tool_name, {}),),
                    stop_reason="tool_use",
                )

        runner = SubAgentRunner(
            AlwaysToolProvider("sub_search"),
            [slow_sub_reg],
            timeout_s=20.0,
        )
        main_end_events: list[TurnEndEvent] = []

        async def _turn_handler(item: str, token: CancelToken, drain_steering: Any) -> None:
            executed_turns.append(item)
            async for ev in agent_loop(
                AlwaysToolProvider("web_search"),
                [ChatMessage(role="user", content=item)],
                tools=[slow_main_reg],
                cancel_token=token,
                drain_steering=drain_steering,
                timeout_s=20.0,
            ):
                if isinstance(ev, TurnEndEvent):
                    main_end_events.append(ev)

        async def _sub_cb(res: SubAgentResult) -> None:
            sub_callbacks_fired.append(res)

        actor = SessionActor(
            "group:77777",
            _turn_handler,
            subagent_result_handler=_sub_cb,
            subagent_runner=runner,
            debounce_s=0.0,
        )

        try:
            # Start slow main turn + queue a followUp + spawn a slow background subagent
            actor.submit("turn-1-slow")
            await asyncio.wait_for(main_tool_started.wait(), timeout=2.0)
            actor.enqueue_followup("turn-2-should-be-purged")

            sub_handle = runner.spawn(
                SubAgentRequest(
                    task_id="sub-slow-1",
                    session_key="group:77777",
                    goal="漫长查证",
                ),
                on_complete=actor.enqueue_subagent_result,
            )
            await asyncio.wait_for(sub_tool_started.wait(), timeout=2.0)

            # Call stop(): must cancel active main CancelToken, purge followup, and cancel subagent in milliseconds!
            t0 = time.perf_counter()
            cancelled_main, cancelled_subs = actor.stop(reason="user_stop")
            sub_res = await asyncio.wait_for(sub_handle.task, timeout=1.0)
            await actor.wait_idle(timeout=1.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            assert cancelled_main is True
            assert cancelled_subs == 1
            assert elapsed_ms < 250.0  # Interrupted 15s sleeps in a few milliseconds!
            assert sub_res.cancelled is True
            assert executed_turns == ["turn-1-slow"]  # Followup was purged!
            assert sub_callbacks_fired == []  # Cancelled SubAgentResult never triggers callback!
            assert len(main_end_events) == 1
            assert main_end_events[0].cancelled is True
            assert main_end_events[0].reason == "user_stop"

            # Also verify steer(..., cancel_active=True) cancels active turn immediately
            main_tool_started.clear()
            actor.submit("turn-3-steered-cancel")
            await asyncio.wait_for(main_tool_started.wait(), timeout=2.0)
            t1 = time.perf_counter()
            actor.steer("算了别查了", cancel_active=True)
            await actor.wait_idle(timeout=1.0)
            assert (time.perf_counter() - t1) * 1000.0 < 250.0
            assert main_end_events[-1].cancelled is True
            assert main_end_events[-1].reason == "steered_cancel"
        finally:
            await actor.aclose()
            await runner.aclose()

    asyncio.run(run())


def test_followup_continuation_terminal_steer_guard_and_subagent_timeout_partial_brief() -> None:
    async def run() -> None:
        # 1) FollowUp seamless continuation + steer blocking premature terminal short-circuit
        send_started = asyncio.Event()
        send_release = asyncio.Event()
        sent_bubbles: list[str] = []

        async def _gated_send(args: dict[str, Any], token: CancelToken) -> str:
            token.raise_if_cancelled()
            for b in args.get("bubbles") or []:
                sent_bubbles.append(str(b))
            if not send_started.is_set():
                send_started.set()
                await send_release.wait()
            return "sent"

        gated_send_tool = RegisteredTool(
            definition=ToolDef(
                name="send_message",
                description="Send bubbles",
                input_schema={"type": "object"},
                is_chat_terminal=True,
                requires_followup=False,
            ),
            handler=_gated_send,
        )

        class MultiTurnProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                self.calls += 1
                last_content = messages[-1].content
                if "<steering_update>" in last_content:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(
                            ToolCall("c_amend", "send_message", {"bubbles": ["补充收到：改约9点！"]}),
                        ),
                        stop_reason="tool_use",
                    )
                if "Bob追问" in last_content:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(
                            ToolCall("c_bob", "send_message", {"bubbles": ["Bob算你一个！"]}),
                        ),
                        stop_reason="tool_use",
                    )
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(
                        ToolCall("c_first", "send_message", {"bubbles": ["今晚8点发车"]}),
                    ),
                    stop_reason="tool_use",
                )

        provider = MultiTurnProvider()

        async def _turn_handler(
            msg: InboundMessage,
            token: CancelToken,
            drain_steering: Any,
        ) -> None:
            async for _ in agent_loop(
                provider,
                [ChatMessage(role="user", content=msg.text)],
                tools=[gated_send_tool],
                cancel_token=token,
                drain_steering=drain_steering,
                max_steps=3,
            ):
                pass

        actor = SessionActor("group:55555", _turn_handler, debounce_s=0.0)
        try:
            alice_msg = InboundMessage(
                event_id="1",
                chat_type="group",
                user_id=101,
                group_id=55555,
                sender_name="Alice",
                text="今晚几点发车",
                raw_message="今晚几点发车",
                mentioned_bot=True,
            )
            actor.submit(alice_msg, steer_if_same_user=True)
            await asyncio.wait_for(send_started.wait(), timeout=2.0)

            # While Alice's terminal send_message is in flight:
            # - Alice steers ("改约9点") -> must prevent short-circuit and run step 2!
            # - Bob submits ("Bob追问") -> different user -> must queue into followUp!
            alice_steer = InboundMessage(
                event_id="2",
                chat_type="group",
                user_id=101,
                group_id=55555,
                sender_name="Alice",
                text="改约9点",
                raw_message="改约9点",
                mentioned_bot=True,
            )
            bob_follow = InboundMessage(
                event_id="3",
                chat_type="group",
                user_id=202,
                group_id=55555,
                sender_name="Bob",
                text="Bob追问带我一个",
                raw_message="Bob追问带我一个",
                mentioned_bot=True,
            )
            assert actor.submit(alice_steer, steer_if_same_user=True) == "steered"
            assert actor.submit(bob_follow, steer_if_same_user=True) == "followup"

            send_release.set()
            await actor.wait_idle(timeout=2.0)
            assert sent_bubbles == [
                "今晚8点发车",
                "补充收到：改约9点！",
                "Bob算你一个！",
            ]
        finally:
            await actor.aclose()

        # 2) SubAgentRunner timeout returns partial gathered evidence instead of losing everything
        class TimeoutOnStep2Provider:
            def __init__(self) -> None:
                self.step = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                self.step += 1
                if self.step == 1:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(ToolCall("r1", "fast_clue", {}),),
                        stop_reason="tool_use",
                    )
                await asyncio.sleep(5.0)
                return ChatResult(content="unreachable", model="glm-5.3-flash")

        clue_tool = RegisteredTool(
            definition=ToolDef(
                name="fast_clue",
                description="Fast clue lookup",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=lambda _a, _t: asyncio.sleep(0, result="核心线索：0.13.9版本传奇插装填不变"),
        )
        results: list[SubAgentResult] = []
        sub_runner = SubAgentRunner(
            TimeoutOnStep2Provider(),
            [clue_tool],
            timeout_s=0.12,
        )
        try:
            handle = sub_runner.spawn(
                SubAgentRequest(
                    task_id="sub-timeout-partial",
                    session_key="group:55555",
                    goal="查询0.13.9改动",
                ),
                on_complete=lambda r: results.append(r),
            )
            final_res = await asyncio.wait_for(handle.task, timeout=2.0)
            assert final_res.cancelled is False
            assert final_res.is_error is False
            assert "核心线索：0.13.9版本传奇插装填不变" in final_res.brief
            assert len(results) == 1
        finally:
            await sub_runner.aclose()

    asyncio.run(run())
