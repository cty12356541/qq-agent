"""Tests for maintenance-window delayed replies:
1. Hard triggers (`@`, quote-reply, keyword, DM) hitting transient upstream LLM/proxy maintenance (`ProviderError` 502/503/timeout) retry and deliver a delayed real reply.
2. Graceful `brain.reload` sentinel drains in-flight `@` turns without polling new inbound messages, so new `@` messages queued in `MessageBus` during reload are delay-replied upon restart.
3. Abrupt `AgentKernel.aclose()` preserves un-replied hard-trigger sequences un-ACKed (while ACKing soft triggers) so the respawned kernel replays and delay-replies them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from qq_agent.agent import AgentKernel
from qq_agent.channel_bridge import DirectMemoryBus
from qq_agent.cli import main as cli_main
from qq_agent.config import AgentConfig
from qq_agent.guardrails import GuardrailManager
from qq_agent.persona import PersonaManager
from qq_agent.provider import ProviderError
from qq_agent.session import SessionStore
from qq_agent.trigger import RuleTrigger
from qq_agent.types import ChatMessage, ChatResult, InboundMessage, OutboundReply, ToolCall, ToolDef

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONA_PATH = REPO_ROOT / "config" / "persona.default.toml"


class _MaintenanceWindowProvider:
    """Fake LLMProvider simulating transient upstream maintenance before recovering."""

    def __init__(self, fail_times: int, *, status_code: int | None = 503, reply_text: str = "维护刚恢复，来啦）") -> None:
        self.fail_times = fail_times
        self.status_code = status_code
        self.reply_text = reply_text
        self.calls = 0

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        tools: list[ToolDef] | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatResult:
        _ = messages, temperature, max_tokens, enable_thinking
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError("Upstream LLM gateway in maintenance window", status_code=self.status_code)
        if tools:
            return ChatResult(
                content="",
                model=model or "glm-5.3-flash",
                prompt_tokens=30,
                completion_tokens=12,
                total_tokens=42,
                cost_micros_usd=10,
                tool_calls=(
                    ToolCall(
                        id=f"call_ok_{self.calls}",
                        name="send_message",
                        arguments={"bubbles": [self.reply_text]},
                    ),
                ),
            )
        return ChatResult(
            content=self.reply_text,
            model=model or "glm-5.3-flash",
            prompt_tokens=30,
            completion_tokens=12,
            total_tokens=42,
            cost_micros_usd=10,
        )


def _build_kernel(
    tmp_path: Path,
    bus: DirectMemoryBus,
    provider: Any,
    *,
    retry_delays_s: tuple[float, ...] = (0.02, 0.04),
    reload_sentinel_path: Path | None = None,
) -> tuple[AgentKernel, SessionStore, GuardrailManager]:
    persona = PersonaManager(DEFAULT_PERSONA_PATH)
    sessions = SessionStore(tmp_path / "sessions.sqlite3")
    guardrails = GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0)
    trigger = RuleTrigger(
        bot_qq=999,
        allowed_group_ids={70001},
        topic_prob=1.0,
        proactive_prob=0.0,
        proactive_cooldown_s=0.0,
        rng=lambda: 0.0,
    )
    kernel = AgentKernel(
        bus=bus,
        provider=provider,  # type: ignore[arg-type]
        persona=persona,
        sessions=sessions,
        trigger=trigger,
        guardrails=guardrails,
        bot_qq=999,
        bubble_typing_delay_s=0.0,
        hard_trigger_retry_delays_s=retry_delays_s,
        reload_sentinel_path=reload_sentinel_path,
    )
    kernel.trigger.update_topic_keywords(("鸣潮",))
    return kernel, sessions, guardrails


def test_hard_trigger_delayed_reply_across_upstream_maintenance_window(tmp_path: Path) -> None:
    """Hard `@` trigger retries across transient 503 maintenance window and delivers real delayed reply."""

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        provider = _MaintenanceWindowProvider(fail_times=2, status_code=503, reply_text="刚维护完，收到你的@啦）")
        kernel, sessions, guardrails = _build_kernel(
            tmp_path,
            bus,
            provider,
            retry_delays_s=(0.02, 0.03),
        )
        try:
            at_msg = InboundMessage(
                event_id="101",
                chat_type="group",
                user_id=20001,
                group_id=70001,
                sender_name="群友甲",
                text="在不",
                raw_message="[CQ:at,qq=999] 在不",
                mentioned_bot=True,
                self_id=999,
            )
            reply = await kernel.handle_inbound(at_msg)
            assert reply is not None
            assert reply.text == "刚维护完，收到你的@啦）"
            assert provider.calls == 3  # 2 maintenance failures + 1 delayed success
            assert guardrails.snapshot().calls_used == 1

            # Soft trigger (`group_followup_window`) does NOT spam retries during maintenance
            soft_provider = _MaintenanceWindowProvider(fail_times=2, status_code=503)
            kernel.provider = soft_provider  # type: ignore[assignment]
            soft_msg = InboundMessage(
                event_id="102",
                chat_type="group",
                user_id=20001,
                group_id=70001,
                sender_name="群友甲",
                text="顺便问下今天深塔好打吗",
                raw_message="顺便问下今天深塔好打吗",
                mentioned_bot=False,
                self_id=999,
            )
            _ = await kernel.handle_inbound(soft_msg)
            assert soft_provider.calls == 1

            # Non-retryable 401 auth error fails fast without retry delay even on hard trigger
            auth_provider = _MaintenanceWindowProvider(fail_times=2, status_code=401)
            kernel.provider = auth_provider  # type: ignore[assignment]
            auth_msg = InboundMessage(
                event_id="103",
                chat_type="private",
                user_id=20003,
                group_id=None,
                sender_name="私聊用户",
                text="测试401",
                raw_message="测试401",
                self_id=999,
            )
            fallback = await kernel.handle_inbound(auth_msg)
            assert fallback is not None
            assert fallback.text == kernel.error_fallback_reply
            assert auth_provider.calls == 1
        finally:
            await kernel.aclose()
            sessions.close()
            guardrails.close()
            await bus.close()

    asyncio.run(_run())


def test_graceful_reload_sentinel_drains_inflight_and_delays_queued_at_messages(tmp_path: Path) -> None:
    """`--reload-brain` drains in-flight `@` turn and leaves new `@` messages in bus for delayed reply by respawned brain."""

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        sentinel_path = tmp_path / "brain.reload"

        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        class _GateProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 512,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                _ = messages, temperature, max_tokens, enable_thinking
                self.calls += 1
                idx = self.calls
                if idx == 1:
                    first_entered.set()
                    await release_first.wait()
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            id=f"c_{idx}",
                            name="send_message",
                            arguments={"bubbles": [f"延迟回复#{idx}"]},
                        ),
                    )
                    if tools
                    else (),
                )

        provider = _GateProvider()
        kernel1, sessions1, guardrails1 = _build_kernel(
            tmp_path,
            bus,
            provider,
            reload_sentinel_path=sentinel_path,
        )
        try:
            # 1) First @AME enters active turn before maintenance reload
            await bus.publish_inbound(
                InboundMessage(
                    event_id="201",
                    chat_type="group",
                    user_id=30001,
                    group_id=70001,
                    sender_name="群友A",
                    text="第一问",
                    raw_message="[CQ:at,qq=999] 第一问",
                    mentioned_bot=True,
                    self_id=999,
                )
            )
            run_task1 = asyncio.create_task(kernel1.run_forever(poll_interval_s=0.01))
            await asyncio.wait_for(first_entered.wait(), timeout=2.0)

            # 2) Trigger `--reload-brain` sentinel AND publish a second @AME during the maintenance drain window!
            rc = cli_main(["--env-file", str(tmp_path / "nonexistent.env"), "--reload-brain"])
            assert rc == 0
            sentinel_path.write_text("reload\n", encoding="utf-8")

            await bus.publish_inbound(
                InboundMessage(
                    event_id="202",
                    chat_type="group",
                    user_id=30002,
                    group_id=70001,
                    sender_name="群友B",
                    text="维护窗口里的第二问",
                    raw_message="[CQ:at,qq=999] 维护窗口里的第二问",
                    mentioned_bot=True,
                    self_id=999,
                )
            )

            # Allow first turn to finish; kernel1 drains turn #1, unlinks sentinel, and exits without touching turn #2!
            await asyncio.sleep(0.03)
            release_first.set()
            await asyncio.wait_for(run_task1, timeout=2.0)
            assert not sentinel_path.exists()

            out_after_drain = await bus.poll_outbound()
            assert len(out_after_drain) == 1
            assert out_after_drain[0][1].text == "延迟回复#1"
            await bus.ack_outbound(out_after_drain[0][0])

            # Turn #2 is still safely un-ACKed in bus!
            assert bus.inbound_high_water == 1
        finally:
            sessions1.close()
            guardrails1.close()

        # 3) Respawned brain (kernel2) attaches to the same bus and immediately delay-replies turn #2!
        kernel2, sessions2, guardrails2 = _build_kernel(
            tmp_path,
            bus,
            provider,
            reload_sentinel_path=sentinel_path,
        )
        stop2 = asyncio.Event()
        run_task2 = asyncio.create_task(kernel2.run_forever(stop_event=stop2, poll_interval_s=0.01))
        try:
            replies: list[OutboundReply] = []
            for _ in range(50):
                batch = await bus.poll_outbound()
                for seq, rep in batch:
                    replies.append(rep)
                    await bus.ack_outbound(seq)
                if replies:
                    break
                await asyncio.sleep(0.02)
            assert len(replies) == 1
            assert replies[0].text == "延迟回复#2"
            assert bus.inbound_high_water == 2
        finally:
            stop2.set()
            await asyncio.wait_for(run_task2, timeout=2.0)
            sessions2.close()
            guardrails2.close()
            await bus.close()

    asyncio.run(_run())


def test_abrupt_aclose_preserves_unreplied_hard_trigger_unacked_for_replay(tmp_path: Path) -> None:
    """If `brain` closes mid-turn before any bubble is sent, hard `@` seq stays un-ACKed and replays on restart."""

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        entered_first = asyncio.Event()
        hold_first = asyncio.Event()

        class _SlowThenFastProvider:
            def __init__(self) -> None:
                self.calls = 0

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 512,
                tools: list[ToolDef] | None = None,
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                _ = messages, temperature, max_tokens, enable_thinking
                self.calls += 1
                if self.calls == 1:
                    entered_first.set()
                    await hold_first.wait()
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            id=f"replay_{self.calls}",
                            name="send_message",
                            arguments={"bubbles": ["重启后补回你的@消息）"]},
                        ),
                    )
                    if tools
                    else (),
                )

        provider = _SlowThenFastProvider()
        kernel1, sessions1, guardrails1 = _build_kernel(tmp_path, bus, provider)
        stop1 = asyncio.Event()
        await bus.publish_inbound(
            InboundMessage(
                event_id="301",
                chat_type="group",
                user_id=40001,
                group_id=70001,
                sender_name="群友C",
                text="重启前被中断的@",
                raw_message="[CQ:at,qq=999] 重启前被中断的@",
                mentioned_bot=True,
                self_id=999,
            )
        )
        t1 = asyncio.create_task(kernel1.run_forever(stop_event=stop1, poll_interval_s=0.01))
        await asyncio.wait_for(entered_first.wait(), timeout=2.0)

        # Abruptly stop kernel1 while hard trigger is still waiting for LLM (0 bubbles sent!)
        stop1.set()
        await asyncio.wait_for(t1, timeout=2.0)
        sessions1.close()
        guardrails1.close()

        # Sequence 1 MUST remain un-ACKed (high_water == 0) so restarted brain replays it!
        assert bus.inbound_high_water == 0

        kernel2, sessions2, guardrails2 = _build_kernel(tmp_path, bus, provider)
        stop2 = asyncio.Event()
        t2 = asyncio.create_task(kernel2.run_forever(stop_event=stop2, poll_interval_s=0.01))
        try:
            for _ in range(50):
                if bus.inbound_high_water == 1:
                    break
                await asyncio.sleep(0.02)
            assert bus.inbound_high_water == 1
            out = await bus.poll_outbound()
            assert len(out) == 1
            assert out[0][1].text == "重启后补回你的@消息）"
        finally:
            stop2.set()
            await asyncio.wait_for(t2, timeout=2.0)
            sessions2.close()
            guardrails2.close()
            await bus.close()

    asyncio.run(_run())


def test_config_parses_hard_trigger_retry_delays() -> None:
    cfg = AgentConfig.from_env(
        env_file=None,
        environ={
            "LLM_API_KEY": "sk-test",
            "QQ_HARD_TRIGGER_RETRY_DELAYS": "1.5, 3.0, 6.0",
        },
    )
    assert cfg.hard_trigger_retry_delays_s == (1.5, 3.0, 6.0)

    cfg_off = AgentConfig.from_env(
        env_file=None,
        environ={
            "LLM_API_KEY": "sk-test",
            "QQ_HARD_TRIGGER_RETRY_DELAYS": "off",
        },
    )
    assert cfg_off.hard_trigger_retry_delays_s == ()

    with pytest.raises(ValueError, match="QQ_HARD_TRIGGER_RETRY_DELAYS"):
        AgentConfig.from_env(
            env_file=None,
            environ={
                "LLM_API_KEY": "sk-test",
                "QQ_HARD_TRIGGER_RETRY_DELAYS": "-1.0",
            },
        )
