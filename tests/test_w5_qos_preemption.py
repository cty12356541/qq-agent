"""Wave W5-03 tests: SessionActor hard/soft QoS preemption, SubAgentRunner per-session fair quota, and TurnEndEvent latency_ms."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any

import pytest

from qq_agent.pi_core import (
    BudgetBlockedError,
    CancelToken,
    RegisteredTool,
    SessionActor,
    SubAgentRunner,
    SubmitRoute,
    TurnEndEvent,
    TurnPriority,
    agent_loop,
)
from qq_agent.types import (
    ChatMessage,
    ChatResult,
    InboundMessage,
    SubAgentRequest,
    ToolCall,
    ToolDef,
)


@dataclass(frozen=True, slots=True)
class _SeqItem:
    seq: int
    msg: InboundMessage

    @property
    def user_id(self) -> int:
        return self.msg.user_id

    @property
    def text(self) -> str:
        return self.msg.text


def _make_group_msg(
    event_id: str,
    *,
    group_id: int = 90001,
    user_id: int = 1001,
    sender_name: str = "群友A",
    text: str = "闲聊一句",
    mentioned_bot: bool = False,
) -> InboundMessage:
    return InboundMessage(
        event_id=event_id,
        chat_type="group",
        user_id=user_id,
        group_id=group_id,
        sender_name=sender_name,
        text=text,
        raw_message=text,
        mentioned_bot=mentioned_bot,
    )


def _send_tool(bubbles_out: list[str]) -> RegisteredTool:
    async def _handler(args: dict[str, Any], token: CancelToken) -> str:
        token.raise_if_cancelled()
        for b in args.get("bubbles") or []:
            bubbles_out.append(str(b))
        return "sent"

    return RegisteredTool(
        definition=ToolDef(
            name="send_message",
            description="Send QQ bubbles",
            input_schema={"type": "object"},
            is_chat_terminal=True,
            requires_followup=False,
        ),
        handler=_handler,
    )


def test_hard_trigger_preempts_inflight_soft_turn_under_30ms() -> None:
    """① Soft turn in long inference (no bubble sent, can_preempt=True) is cancelled in <30ms by @AME hard trigger."""

    async def run() -> None:
        soft_llm_started = asyncio.Event()
        soft_cancelled_at: float = 0.0
        hard_started_at: float = 0.0
        sent_bubbles: list[str] = []
        turn_end_events: list[TurnEndEvent] = []
        observed_priorities: list[tuple[str, TurnPriority]] = []
        dropped_items: list[_SeqItem] = []
        active_turn_bubbles = 0

        class PreemptProvider:
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
                nonlocal soft_cancelled_at, hard_started_at
                prompt = messages[-1].content
                if "软插话慢推理" in prompt:
                    soft_llm_started.set()
                    try:
                        await asyncio.sleep(10.0)
                    except asyncio.CancelledError:
                        soft_cancelled_at = time.perf_counter()
                        raise
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(
                            ToolCall("s_never", "send_message", {"bubbles": ["不该发出的软插话"]}),
                        ),
                        stop_reason="tool_use",
                    )

                hard_started_at = time.perf_counter()
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    total_tokens=88,
                    tool_calls=(
                        ToolCall("h_1", "send_message", {"bubbles": ["在呢！马上帮你看配置）"]}),
                    ),
                    stop_reason="tool_use",
                )

        provider = PreemptProvider()
        actor_holder: dict[str, SessionActor] = {}

        async def _turn_handler(
            item: _SeqItem,
            token: CancelToken,
            drain_steering: Any,
        ) -> None:
            nonlocal active_turn_bubbles
            active_turn_bubbles = 0
            actor = actor_holder["actor"]
            observed_priorities.append((item.msg.event_id, actor.active_turn_priority))
            turn_bubbles: list[str] = []
            async for ev in agent_loop(
                provider,
                [ChatMessage(role="user", content=item.text)],
                tools=[_send_tool(turn_bubbles)],
                cancel_token=token,
                drain_steering=drain_steering,
            ):
                if isinstance(ev, TurnEndEvent):
                    turn_end_events.append(ev)
            active_turn_bubbles = len(turn_bubbles)
            sent_bubbles.extend(turn_bubbles)

        actor = SessionActor(
            "group:90001",
            _turn_handler,
            debounce_s=0.0,
            can_preempt_active_fn=lambda: active_turn_bubbles == 0,
            on_item_dropped=dropped_items.append,
        )
        actor_holder["actor"] = actor

        try:
            soft_item = _SeqItem(
                seq=1,
                msg=_make_group_msg("soft-1", user_id=1001, text="软插话慢推理"),
            )
            assert actor.submit(soft_item, priority="soft") == "debouncing"
            await asyncio.wait_for(soft_llm_started.wait(), timeout=2.0)
            assert actor.is_running
            assert actor.active_turn_priority == "soft"

            hard_item = _SeqItem(
                seq=2,
                msg=_make_group_msg(
                    "hard-2",
                    user_id=1002,
                    sender_name="阿哲",
                    text="@AME 帮我看下配置",
                    mentioned_bot=True,
                ),
            )
            t_submit = time.perf_counter()
            route: SubmitRoute = actor.submit(
                hard_item,
                steer_if_same_user=True,
                priority="hard",
            )
            assert route == "followup"
            assert actor.preempted_count == 1

            await actor.wait_idle(timeout=2.0)

            cancel_latency_ms = (soft_cancelled_at - t_submit) * 1000.0
            hard_exec_latency_ms = (hard_started_at - t_submit) * 1000.0
            assert 0.0 <= cancel_latency_ms < 30.0
            assert 0.0 <= hard_exec_latency_ms < 30.0
            assert sent_bubbles == ["在呢！马上帮你看配置）"]
            assert observed_priorities == [("soft-1", "soft"), ("hard-2", "hard")]
            assert actor.active_turn_priority == "hard"
            assert dropped_items == []

            assert len(turn_end_events) == 2
            assert turn_end_events[0].cancelled is True
            assert turn_end_events[0].reason == "preempted_by_hard_trigger"
            assert turn_end_events[0].latency_ms > 0.0
            assert turn_end_events[1].cancelled is False
            assert turn_end_events[1].reason == "terminal_tool_short_circuit"
            assert turn_end_events[1].latency_ms > 0.0
        finally:
            await actor.aclose()

    asyncio.run(run())


def test_non_preemptible_soft_turn_after_first_bubble_queues_hard_followup() -> None:
    """② Once a soft turn has emitted its first bubble (can_preempt=False), hard trigger queues into followup cleanly."""

    async def run() -> None:
        first_bubble_emitted = asyncio.Event()
        release_soft_finish = asyncio.Event()
        sent_bubbles: list[str] = []
        current_turn_bubble_count = 0

        async def _staged_send(args: dict[str, Any], token: CancelToken) -> str:
            nonlocal current_turn_bubble_count
            token.raise_if_cancelled()
            bubbles = [str(b) for b in (args.get("bubbles") or [])]
            if bubbles and "软回复首条" in bubbles[0]:
                sent_bubbles.append(bubbles[0])
                current_turn_bubble_count += 1
                first_bubble_emitted.set()
                await release_soft_finish.wait()
                token.raise_if_cancelled()
                for extra in bubbles[1:]:
                    sent_bubbles.append(extra)
                    current_turn_bubble_count += 1
                return "sent soft"
            for b in bubbles:
                sent_bubbles.append(b)
                current_turn_bubble_count += 1
            return "sent hard"

        staged_send_tool = RegisteredTool(
            definition=ToolDef(
                name="send_message",
                description="Staged send bubbles",
                input_schema={"type": "object"},
                is_chat_terminal=True,
                requires_followup=False,
            ),
            handler=_staged_send,
        )

        class TwoTurnProvider:
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
                if "软触发话题" in messages[-1].content:
                    return ChatResult(
                        content="",
                        model="glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                "s1",
                                "send_message",
                                {"bubbles": ["软回复首条", "软回复收尾不腰斩"]},
                            ),
                        ),
                        stop_reason="tool_use",
                    )
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(
                        ToolCall("h1", "send_message", {"bubbles": ["硬触发紧接回复！"]}),
                    ),
                    stop_reason="tool_use",
                )

        provider = TwoTurnProvider()

        async def _turn_handler(
            msg: InboundMessage,
            token: CancelToken,
            drain_steering: Any,
        ) -> None:
            nonlocal current_turn_bubble_count
            current_turn_bubble_count = 0
            async for _ in agent_loop(
                provider,
                [ChatMessage(role="user", content=msg.text)],
                tools=[staged_send_tool],
                cancel_token=token,
                drain_steering=drain_steering,
            ):
                pass

        actor = SessionActor(
            "group:90002",
            _turn_handler,
            debounce_s=0.0,
            can_preempt_active_fn=lambda: current_turn_bubble_count == 0,
        )
        try:
            soft_msg = _make_group_msg("s-1", user_id=1001, text="软触发话题")
            actor.submit(soft_msg, priority="soft")
            await asyncio.wait_for(first_bubble_emitted.wait(), timeout=2.0)
            assert sent_bubbles == ["软回复首条"]
            assert actor.active_turn_priority == "soft"

            hard_msg = _make_group_msg(
                "h-2",
                user_id=1002,
                text="@AME 轮到我提问",
                mentioned_bot=True,
            )
            route = actor.submit(hard_msg, steer_if_same_user=True, priority="hard")
            assert route == "followup"
            assert actor.preempted_count == 0

            release_soft_finish.set()
            await actor.wait_idle(timeout=2.0)

            assert sent_bubbles == [
                "软回复首条",
                "软回复收尾不腰斩",
                "硬触发紧接回复！",
            ]
            assert actor.preempted_count == 0
        finally:
            await actor.aclose()

    asyncio.run(run())


def test_busy_actor_and_hard_debounce_drop_passive_soft_and_release_seqs() -> None:
    """③ Passive soft items arriving during hard debounce, RUNNING, or pending hard followup drop immediately and release seqs."""

    async def run() -> None:
        turn_entered = asyncio.Event()
        release_turn = asyncio.Event()
        executed_seqs: list[int] = []
        dropped_seqs: list[int] = []

        async def _turn_handler(item: _SeqItem, token: CancelToken) -> None:
            token.raise_if_cancelled()
            executed_seqs.append(item.seq)
            if item.seq == 1:
                turn_entered.set()
                await release_turn.wait()

        actor = SessionActor(
            "group:90003",
            _turn_handler,
            debounce_s=0.06,
            on_item_dropped=lambda it: dropped_seqs.append(it.seq),
        )
        try:
            # seq=1 opens a hard debounce window
            r1 = actor.submit(
                _SeqItem(1, _make_group_msg("1", user_id=1001, text="@AME 第一问", mentioned_bot=True)),
                priority="hard",
            )
            assert r1 == "debouncing"

            # seq=2 passive soft from another speaker during hard debounce -> dropped_soft immediately
            r2 = actor.submit(
                _SeqItem(2, _make_group_msg("2", user_id=1002, text="路人插嘴A")),
                priority="soft",
            )
            assert r2 == "dropped_soft"
            assert dropped_seqs == [2]

            # Wait until seq=1 enters RUNNING
            await asyncio.wait_for(turn_entered.wait(), timeout=2.0)
            assert actor.is_running

            # seq=3 and seq=4 passive soft messages while RUNNING (even from same user_id=1001) -> dropped_soft
            r3 = actor.submit(
                _SeqItem(3, _make_group_msg("3", user_id=1003, text="路人插嘴B")),
                priority="soft",
            )
            r4 = actor.submit(
                _SeqItem(4, _make_group_msg("4", user_id=1001, text="同用户软插话")),
                steer_if_same_user=True,
                priority="soft",
            )
            assert r3 == "dropped_soft"
            assert r4 == "dropped_soft"
            assert dropped_seqs == [2, 3, 4]

            # seq=5 hard trigger from user 1004 queues into followup
            r5 = actor.submit(
                _SeqItem(5, _make_group_msg("5", user_id=1004, text="@AME 第二问", mentioned_bot=True)),
                priority="hard",
            )
            assert r5 == "followup"

            # seq=6 passive soft while RUNNING + hard followup pending -> dropped_soft
            r6 = actor.submit(
                _SeqItem(6, _make_group_msg("6", user_id=1005, text="路人插嘴C")),
                priority="soft",
            )
            assert r6 == "dropped_soft"
            assert dropped_seqs == [2, 3, 4, 6]

            release_turn.set()
            await actor.wait_idle(timeout=2.0)

            assert executed_seqs == [1, 5]
            # Every single seq 1..6 was accounted for (either executed or dropped via callback)
            assert sorted(executed_seqs + dropped_seqs) == [1, 2, 3, 4, 5, 6]
        finally:
            await actor.aclose()

    asyncio.run(run())


def test_soft_debounce_eviction_same_speaker_upgrade_and_stop_purge() -> None:
    """④ Soft debounce is evicted by another speaker's hard trigger (firing on_item_dropped), upgraded by same speaker, and purged on stop()."""

    async def run() -> None:
        executed_turns: list[tuple[tuple[int, ...], str, TurnPriority]] = []
        dropped_seqs: list[int] = []
        actor_ref: dict[str, SessionActor] = {}
        block_turn = asyncio.Event()
        turn_blocked = asyncio.Event()

        def _coalesce_seq_items(items: list[_SeqItem]) -> tuple[tuple[int, ...], InboundMessage]:
            seqs = tuple(it.seq for it in items)
            merged_text = "\n".join(it.text for it in items)
            last = items[-1].msg
            merged_msg = InboundMessage(
                event_id=last.event_id,
                chat_type=last.chat_type,
                user_id=last.user_id,
                group_id=last.group_id,
                sender_name=last.sender_name,
                text=merged_text,
                raw_message=merged_text,
                mentioned_bot=any(it.msg.mentioned_bot for it in items),
            )
            return seqs, merged_msg

        async def _turn_handler(
            payload: tuple[tuple[int, ...], InboundMessage] | _SeqItem,
            token: CancelToken,
        ) -> None:
            actor = actor_ref["actor"]
            if isinstance(payload, _SeqItem):
                seqs = (payload.seq,)
                txt = payload.text
            else:
                seqs, msg = payload
                txt = msg.text
            if "阻塞等待stop" in txt:
                turn_blocked.set()
                await block_turn.wait()
                token.raise_if_cancelled()
            executed_turns.append((seqs, txt, actor.active_turn_priority))

        actor = SessionActor(
            "group:90004",
            _turn_handler,
            debounce_s=0.08,
            coalesce_fn=_coalesce_seq_items,
            on_item_dropped=lambda it: dropped_seqs.append(it.seq),
        )
        actor_ref["actor"] = actor

        try:
            # 1) User 1001 sends 2 soft burst lines (seq=10, 11) + User 1003 queues soft followup (seq=12);
            #    then User 1002 sends hard @AME (seq=13) -> evicts seq=10,11 AND purges residual soft seq=12!
            assert (
                actor.submit(
                    _SeqItem(10, _make_group_msg("10", user_id=1001, text="软防抖第一句")),
                    priority="soft",
                )
                == "debouncing"
            )
            assert (
                actor.submit(
                    _SeqItem(11, _make_group_msg("11", user_id=1001, text="软防抖第二句")),
                    priority="soft",
                )
                == "debouncing"
            )
            assert (
                actor.submit(
                    _SeqItem(12, _make_group_msg("12", user_id=1003, text="另一路人软排队")),
                    priority="soft",
                )
                == "followup"
            )
            assert dropped_seqs == []

            route_hard = actor.submit(
                _SeqItem(
                    13,
                    _make_group_msg("13", user_id=1002, text="@AME 硬触发冲掉软防抖", mentioned_bot=True),
                ),
                priority="hard",
            )
            assert route_hard == "debouncing"
            assert dropped_seqs == [10, 11, 12]

            await actor.wait_idle(timeout=2.0)
            assert executed_turns == [((13,), "@AME 硬触发冲掉软防抖", "hard")]

            # 2) Same speaker upgrades soft debounce (seq=20) to hard (seq=21): coalesces & upgrades priority to hard!
            executed_turns.clear()
            dropped_seqs.clear()
            assert (
                actor.submit(
                    _SeqItem(20, _make_group_msg("20", user_id=1001, text="刚发一半")),
                    priority="soft",
                )
                == "debouncing"
            )
            assert (
                actor.submit(
                    _SeqItem(
                        21,
                        _make_group_msg("21", user_id=1001, text="@AME 补上艾特", mentioned_bot=True),
                    ),
                    priority="hard",
                )
                == "debouncing"
            )
            # Because debounce upgraded to hard, another speaker's soft (seq=22) is immediately dropped_soft
            assert (
                actor.submit(
                    _SeqItem(22, _make_group_msg("22", user_id=1009, text="路人软插话")),
                    priority="soft",
                )
                == "dropped_soft"
            )
            assert dropped_seqs == [22]
            await actor.wait_idle(timeout=2.0)
            assert executed_turns == [((20, 21), "刚发一半\n@AME 补上艾特", "hard")]

            # 3) Mid-sleep soft debounce eviction while _wait_and_pop_debounced_turn is actively awaiting _debounce_bump_event
            executed_turns.clear()
            dropped_seqs.clear()
            actor.debounce_s = 0.15
            assert (
                actor.submit(
                    _SeqItem(25, _make_group_msg("25", user_id=1001, text="睡眠中软防抖")),
                    priority="soft",
                )
                == "debouncing"
            )
            await asyncio.sleep(0.02)
            assert (
                actor.submit(
                    _SeqItem(
                        26,
                        _make_group_msg("26", user_id=1002, text="@AME 睡眠中硬打断", mentioned_bot=True),
                    ),
                    priority="hard",
                )
                == "debouncing"
            )
            assert dropped_seqs == [25]
            await actor.wait_idle(timeout=2.0)
            assert executed_turns == [((26,), "@AME 睡眠中硬打断", "hard")]

            # 4) stop() purges un-coalesced _debounce_items individually and _followup_queue items via on_item_dropped
            dropped_seqs.clear()
            assert (
                actor.submit(
                    _SeqItem(30, _make_group_msg("30", user_id=1001, text="待取消防抖1")),
                    priority="hard",
                )
                == "debouncing"
            )
            assert (
                actor.submit(
                    _SeqItem(31, _make_group_msg("31", user_id=1001, text="待取消防抖2")),
                    priority="hard",
                )
                == "debouncing"
            )
            assert (
                actor.submit(
                    _SeqItem(32, _make_group_msg("32", user_id=1002, text="待取消队列项")),
                    priority="hard",
                )
                == "followup"
            )
            actor.stop(reason="manual_stop")
            await actor.wait_idle(timeout=1.0)
            assert dropped_seqs == [30, 31, 32]
        finally:
            block_turn.set()
            await actor.aclose()

    asyncio.run(run())


def test_multigroup_actors_parallelism_and_subagent_per_session_fair_quota() -> None:
    """⑤ Multiple group SessionActors run truly in parallel; SubAgentRunner(max_concurrent=6, max_per_session=2) isolates per-group quota."""

    async def run() -> None:
        # Part 1: 3 SessionActors execute simultaneously (verified via Barrier(3) rendezvous)
        barrier = asyncio.Barrier(3)
        completed_groups: list[str] = []

        async def _parallel_group_turn(item: str, token: CancelToken) -> None:
            token.raise_if_cancelled()
            await asyncio.wait_for(barrier.wait(), timeout=1.5)
            completed_groups.append(item)

        actors = [
            SessionActor(f"group:{gid}", _parallel_group_turn, debounce_s=0.0)
            for gid in (10001, 10002, 10003)
        ]
        try:
            for idx, act in enumerate(actors, start=1):
                act.submit(f"g{idx}")
            await asyncio.gather(*(act.wait_idle(timeout=2.0) for act in actors))
            assert sorted(completed_groups) == ["g1", "g2", "g3"]
        finally:
            await asyncio.gather(*(act.aclose() for act in actors))

        # Part 2: SubAgentRunner(max_concurrent=6, max_per_session=2) fair per-session quota isolation
        release_research = asyncio.Event()

        async def _gated_research(args: dict[str, Any], token: CancelToken) -> str:
            await release_research.wait()
            token.raise_if_cancelled()
            return f"done:{args.get('q', '')}"

        research_tool = RegisteredTool(
            definition=ToolDef(
                name="web_search",
                description="Research search",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=True,
            ),
            handler=_gated_research,
        )

        class SubProvider:
            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.3,
                max_tokens: int = 1024,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                if messages[-1].role == "tool":
                    return ChatResult(content="查证简报完成", model="glm-5.3-flash")
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(ToolCall("t1", "web_search", {"q": "test"}),),
                    stop_reason="tool_use",
                )

        runner = SubAgentRunner(
            SubProvider(),
            [research_tool],
            max_concurrent=6,
            max_per_session=2,
            enable_thinking=False,
        )
        try:
            # Group A fills its 2-slot per-session quota
            h_a1 = runner.spawn(SubAgentRequest(task_id="a-1", session_key="group:A", goal="A1"))
            h_a2 = runner.spawn(SubAgentRequest(task_id="a-2", session_key="group:A", goal="A2"))
            assert runner.active_count() == 2
            assert runner.active_count(session_key="group:A") == 2
            assert runner.active_count(session_key="group:B") == 0

            # Group A's 3rd task is rejected by per-session limit
            with pytest.raises(
                RuntimeError,
                match=r"subagent per-session concurrency limit \(2\) reached for group:A",
            ):
                runner.spawn(SubAgentRequest(task_id="a-3", session_key="group:A", goal="A3"))

            # Groups B and C are completely unaffected and each spawn 2 tasks cleanly (total = 6)
            h_b1 = runner.spawn(SubAgentRequest(task_id="b-1", session_key="group:B", goal="B1"))
            h_b2 = runner.spawn(SubAgentRequest(task_id="b-2", session_key="group:B", goal="B2"))
            h_c1 = runner.spawn(SubAgentRequest(task_id="c-1", session_key="group:C", goal="C1"))
            h_c2 = runner.spawn(SubAgentRequest(task_id="c-2", session_key="group:C", goal="C2"))
            assert runner.active_count() == 6
            assert runner.active_count(session_key="group:B") == 2
            assert runner.active_count(session_key="group:C") == 2

            # Group D now hits the global max_concurrent=6 ceiling
            with pytest.raises(RuntimeError, match="^subagent concurrency limit reached$"):
                runner.spawn(SubAgentRequest(task_id="d-1", session_key="group:D", goal="D1"))

            # Cancelling one Group A task frees 1 slot for both Group A and global pool
            assert runner.cancel("a-1") == 1
            assert runner.active_count(session_key="group:A") == 1
            assert runner.active_count() == 5
            h_d1 = runner.spawn(SubAgentRequest(task_id="d-1", session_key="group:D", goal="D1"))
            assert runner.active_count(session_key="group:D") == 1
            assert runner.active_count() == 6

            release_research.set()
            results = await asyncio.wait_for(
                asyncio.gather(
                    h_a1.task,
                    h_a2.task,
                    h_b1.task,
                    h_b2.task,
                    h_c1.task,
                    h_c2.task,
                    h_d1.task,
                ),
                timeout=2.0,
            )
            assert results[0].cancelled is True
            assert all(not r.cancelled and not r.is_error for r in results[1:])
            assert runner.active_count() == 0
        finally:
            release_research.set()
            await runner.aclose()

    asyncio.run(run())


def test_turn_end_event_latency_ms_populated_across_exit_paths() -> None:
    """⑥ TurnEndEvent.latency_ms > 0 across end_turn, terminal_tool_short_circuit, cancelled, timeout, budget_blocked, and max_steps_reached."""

    async def run() -> None:
        bubbles: list[str] = []

        class EndTurnProvider:
            async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
                await asyncio.sleep(0.002)
                return ChatResult(content="纯文本结束", model="glm-5.3-flash")

        class TerminalToolProvider:
            async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
                await asyncio.sleep(0.002)
                return ChatResult(
                    content="",
                    model="glm-5.3-flash",
                    tool_calls=(ToolCall("c1", "send_message", {"bubbles": ["气泡短路"]}),),
                    stop_reason="tool_use",
                )

        class SlowProvider:
            async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
                await asyncio.sleep(5.0)
                return ChatResult(content="unreachable", model="glm-5.3-flash")

        class LoopToolProvider:
            async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
                return ChatResult(
                    content="阶段文本",
                    model="glm-5.3-flash",
                    tool_calls=(ToolCall("n1", "noop_tool", {}),),
                    stop_reason="tool_use",
                )

        noop_tool = RegisteredTool(
            definition=ToolDef(
                name="noop_tool",
                description="Non-terminal tool",
                input_schema={"type": "object"},
                is_chat_terminal=False,
                requires_followup=False,
            ),
            handler=lambda _a, _t: asyncio.sleep(0.001, result="ok"),
        )

        # 1) end_turn
        ev_end = [ev async for ev in agent_loop(EndTurnProvider(), [ChatMessage("user", "hi")])][-1]
        assert isinstance(ev_end, TurnEndEvent)
        assert ev_end.reason == "end_turn"
        assert ev_end.latency_ms > 0.0

        # 2) terminal_tool_short_circuit
        ev_short = [
            ev
            async for ev in agent_loop(
                TerminalToolProvider(),
                [ChatMessage("user", "hi")],
                tools=[_send_tool(bubbles)],
            )
        ][-1]
        assert isinstance(ev_short, TurnEndEvent)
        assert ev_short.reason == "terminal_tool_short_circuit"
        assert ev_short.latency_ms > 0.0

        # 3) timeout
        ev_timeout = [
            ev
            async for ev in agent_loop(
                SlowProvider(),
                [ChatMessage("user", "hi")],
                timeout_s=0.05,
            )
        ][-1]
        assert isinstance(ev_timeout, TurnEndEvent)
        assert ev_timeout.reason == "timeout"
        assert ev_timeout.latency_ms >= 20.0

        # 4) budget_blocked
        def _raise_budget(_step: int) -> str | None:
            raise BudgetBlockedError("配额用尽")

        ev_budget = [
            ev
            async for ev in agent_loop(
                EndTurnProvider(),
                [ChatMessage("user", "hi")],
                before_step=_raise_budget,
            )
        ][-1]
        assert isinstance(ev_budget, TurnEndEvent)
        assert ev_budget.reason == "budget_blocked"
        assert ev_budget.latency_ms > 0.0

        # 5) pre-cancelled token
        pre_tok = CancelToken()
        pre_tok.cancel("pre_cancelled")
        ev_cancel = [
            ev
            async for ev in agent_loop(
                EndTurnProvider(),
                [ChatMessage("user", "hi")],
                cancel_token=pre_tok,
            )
        ][-1]
        assert isinstance(ev_cancel, TurnEndEvent)
        assert ev_cancel.cancelled is True
        assert ev_cancel.reason == "pre_cancelled"
        assert ev_cancel.latency_ms > 0.0

        # 6) max_steps_reached
        ev_max = [
            ev
            async for ev in agent_loop(
                LoopToolProvider(),
                [ChatMessage("user", "hi")],
                tools=[noop_tool],
                max_steps=2,
            )
        ][-1]
        assert isinstance(ev_max, TurnEndEvent)
        assert ev_max.reason == "max_steps_reached"
        assert ev_max.latency_ms > 0.0

    asyncio.run(run())
