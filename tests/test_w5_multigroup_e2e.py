"""End-to-end W5 tests: compact low-latency prompts, QoS hard-over-soft preemption, per-group SubAgent quota, and Cross-Group Shared Cortex ("一魂多体") with strict privacy firewall."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

import pytest

from qq_agent.agent import AgentKernel
from qq_agent.channel_bridge import DirectMemoryBus
from qq_agent.guardrails import GuardrailManager
from qq_agent.memory import MemoryStore
from qq_agent.persona import PersonaManager
from qq_agent.provider import ProviderError
from qq_agent.session import SessionStore
from qq_agent.stickers import StickerRegistry
from qq_agent.tools.web_search import SearchResultItem, WebSearchClient
from qq_agent.trigger import RuleTrigger
from qq_agent.types import ChatMessage, ChatResult, InboundMessage, OutboundReply, ToolCall, ToolDef


def _write_rich_persona(tmp_path: Path) -> Path:
    persona_file = tmp_path / "persona.toml"
    slang_items = ", ".join(f'"口头禅碎片{i}"' for i in range(1, 15))
    examples_blocks = "\n".join(
        f'[[few_shot_examples]]\nuser = "测试群友提问样例编号{i}今天怎么说"\nassistant = "测试群友回答样例编号{i}简短接话"'
        for i in range(1, 15)
    )
    persona_file.write_text(
        f"""
name = "AME"
aliases = ["ame", "阿เม"]
role_prompt = "你是常驻熟人 QQ 群的真实群友 AME，说话简短自然。"
style_rules = [
    "单句简短干脆，不写长篇列表。",
    "禁止括号动作描写，用表情包代替。",
    "禁止客服腔与说教口吻。",
]
safety_boundaries = [
    "拒绝违法违规与危险指导。",
    "严格保护群友隐私与敏感凭证。",
]
slang_hints = [{slang_items}]
topic_keywords = ["战舰世界", "帝国时代4"]

{examples_blocks}
""".strip(),
        encoding="utf-8",
    )
    return persona_file


async def _drain_outbound_until(
    bus: DirectMemoryBus,
    expected_count: int,
    *,
    timeout_s: float = 3.0,
) -> list[OutboundReply]:
    collected: list[OutboundReply] = []
    deadline = time.monotonic() + timeout_s
    while len(collected) < expected_count and time.monotonic() < deadline:
        batch = await bus.poll_outbound(limit=16)
        for seq, reply in batch:
            await bus.ack_outbound(seq)
            collected.append(reply)
        if len(collected) < expected_count:
            await asyncio.sleep(0.01)
    return collected


def test_compact_prompt_slims_system_prompt_and_tool_schemas_without_losing_safety(
    tmp_path: Path,
) -> None:
    persona_path = _write_rich_persona(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    stickers = StickerRegistry(
        repo_root / "data" / "stickers",
        manifest_path=repo_root / "config" / "stickers.toml",
    )

    class _DummyProvider:
        async def chat(
            self,
            messages: list[ChatMessage],
            *,
            model: str | None = None,
            temperature: float = 0.7,
            max_tokens: int = 512,
            tools: tuple[ToolDef, ...] | list[ToolDef] = (),
            enable_thinking: bool | None = None,
        ) -> ChatResult:
            return ChatResult(content="ok", model=model or "glm-5.3-flash")

    with (
        SessionStore(tmp_path / "sessions.sqlite3") as sessions,
        MemoryStore(tmp_path / "memory.sqlite3") as memory,
        GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
    ):
        persona = PersonaManager(persona_path)
        bus = DirectMemoryBus()
        full_kernel = AgentKernel(
            bus=bus,
            provider=_DummyProvider(),
            persona=persona,
            sessions=sessions,
            memory=memory,
            trigger=RuleTrigger(bot_qq=999),
            guardrails=guardrails,
            stickers=stickers,
            compact_prompt=False,
        )
        compact_kernel = AgentKernel(
            bus=bus,
            provider=_DummyProvider(),
            persona=persona,
            sessions=sessions,
            memory=memory,
            trigger=RuleTrigger(bot_qq=999),
            guardrails=guardrails,
            stickers=stickers,
            compact_prompt=True,
        )

        full_prompt = full_kernel._render_turn_system_prompt(
            session_key="group:1001",
            chat_type="group",
            group_id=1001,
            user_id=2001,
            include_tool_guide=True,
        )
        slim_prompt = compact_kernel._render_turn_system_prompt(
            session_key="group:1001",
            chat_type="group",
            group_id=1001,
            user_id=2001,
            include_tool_guide=True,
        )

        assert len(slim_prompt) < len(full_prompt) * 0.58
        assert "AME" in slim_prompt
        assert "<untrusted_chat_message>" in slim_prompt
        assert "send_message" in slim_prompt
        # Top few-shots kept, tail few-shots trimmed
        assert "测试群友提问样例编号1" in slim_prompt
        assert "测试群友提问样例编号14" not in slim_prompt


def test_qos_hard_trigger_preempts_slow_soft_trigger_and_advances_ack_high_water(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        persona_path = _write_rich_persona(tmp_path)
        soft_started = asyncio.Event()
        release_soft = asyncio.Event()
        hard_captured_user_prompts: list[str] = []

        class _PreemptibleProvider:
            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 512,
                tools: tuple[ToolDef, ...] | list[ToolDef] = (),
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                user_text = messages[-1].content
                if "帮我看下报错" not in user_text:
                    soft_started.set()
                    await release_soft.wait()
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="soft_send",
                                name="send_message",
                                arguments={"bubbles": ["这条软触发慢回复绝不该发出"]},
                            ),
                        ),
                    )
                hard_captured_user_prompts.append(user_text)
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            id="hard_send",
                            name="send_message",
                            arguments={"bubbles": ["来了，报错发我看看"]},
                        ),
                    ),
                )

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            MemoryStore(tmp_path / "memory.sqlite3") as memory,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            bus = DirectMemoryBus()
            trigger = RuleTrigger(
                bot_qq=999,
                allowed_group_ids={590001},
                topic_prob=1.0,
                proactive_prob=0.0,
                proactive_cooldown_s=0.0,
            )
            kernel = AgentKernel(
                bus=bus,
                provider=_PreemptibleProvider(),
                persona=PersonaManager(persona_path),
                sessions=sessions,
                memory=memory,
                trigger=trigger,
                guardrails=guardrails,
                compact_prompt=True,
                group_debounce_s=0.0,
                bubble_typing_delay_s=0.0,
            )
            try:
                # seq=1: Passive soft trigger (group_topic_interest) enters slow LLM inference
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="101",
                        chat_type="group",
                        user_id=20001,
                        group_id=590001,
                        sender_name="路人甲",
                        text="最近战舰世界新线怎么样",
                        raw_message="最近战舰世界新线怎么样",
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()
                await asyncio.wait_for(soft_started.wait(), timeout=1.5)

                # seq=2: Another passive soft trigger while actor is busy -> dropped_soft immediately
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="102",
                        chat_type="group",
                        user_id=20003,
                        group_id=590001,
                        sender_name="路人丙",
                        text="战舰世界今晚组队吗",
                        raw_message="战舰世界今晚组队吗",
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()

                # seq=3: Explicit @AME hard trigger -> 0ms QoS preemption over seq=1!
                t0 = time.monotonic()
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="103",
                        chat_type="group",
                        user_id=20002,
                        group_id=590001,
                        sender_name="急事群友乙",
                        text="帮我看下报错",
                        raw_message="[CQ:at,qq=999] 帮我看下报错",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()

                replies = await _drain_outbound_until(bus, 1, timeout_s=1.0)
                elapsed_s = time.monotonic() - t0
                assert len(replies) == 1
                assert replies[0].text == "来了，报错发我看看"
                assert elapsed_s < 0.5, f"Hard trigger should preempt immediately, took {elapsed_s:.3f}s"

                await kernel.wait_idle(timeout=1.5)

                actor = kernel._actors["group:590001"]
                assert actor.preempted_count == 1
                assert kernel._dropped_soft_count >= 1

                # Ambient context of both dropped & preempted soft lines preserved for the @AME turn!
                assert hard_captured_user_prompts
                assert "战舰世界今晚组队吗" in hard_captured_user_prompts[0]
                assert "最近战舰世界新线怎么样" in hard_captured_user_prompts[0]

                # Zero leaked bus sequence numbers across preempted seq=1, dropped seq=2, completed seq=3
                assert kernel.ack_coordinator.contiguous_high_water == 3
                assert kernel.ack_coordinator.inflight_count == 0
            finally:
                release_soft.set()
                await kernel.aclose()

    asyncio.run(_run())


def test_multi_group_concurrent_turns_and_fair_per_group_subagent_quota(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        persona_path = _write_rich_persona(tmp_path)
        sub_gate = asyncio.Event()
        sub_started_goals: list[str] = []
        all_three_subs_started = asyncio.Event()

        class _MultiGroupProvider:
            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 512,
                tools: tuple[ToolDef, ...] | list[ToolDef] = (),
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                sys_text = messages[0].content
                user_text = messages[-1].content

                # Sub-agent slow-brain execution
                if "<research_goal>" in user_text:
                    sub_started_goals.append(user_text)
                    if len(sub_started_goals) >= 3:
                        all_three_subs_started.set()
                    await sub_gate.wait()
                    return ChatResult(
                        content=f"已核实简报：{user_text.splitlines()[1]}",
                        model=model or "glm-5.3-flash",
                    )

                # Fast-brain callback synthesis when sub-agent completes
                if "<subagent_research_brief" in user_text:
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="cb_send",
                                name="send_message",
                                arguments={"bubbles": ["查完啦，结果给你整理好了"]},
                            ),
                        ),
                    )

                # Tool result follow-up when 3rd spawn_subagent in group 590001 hits per-group cap
                if messages[-1].role == "tool":
                    err_text = " ".join(tr.content for tr in messages[-1].tool_results)
                    assert "per-session concurrency limit (2)" in err_text
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="cap_send",
                                name="send_message",
                                arguments={"bubbles": ["手头已经在查两个啦，等我先翻完"]},
                            ),
                        ),
                    )

                if "深查任务" in user_text:
                    tag = user_text.split("深查任务", 1)[1].split("<", 1)[0].strip()
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id=f"pad_{tag}",
                                name="send_message",
                                arguments={"bubbles": [f"行，我帮你翻下{tag}"]},
                            ),
                            ToolCall(
                                id=f"sp_{tag}",
                                name="spawn_subagent",
                                arguments={"goal": f"深查任务{tag}"},
                            ),
                        ),
                    )

                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            id="fast_chat",
                            name="send_message",
                            arguments={"bubbles": ["二群秒回不卡顿"]},
                        ),
                    ),
                )

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            MemoryStore(tmp_path / "memory.sqlite3") as memory,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            bus = DirectMemoryBus()
            kernel = AgentKernel(
                bus=bus,
                provider=_MultiGroupProvider(),
                persona=PersonaManager(persona_path),
                sessions=sessions,
                memory=memory,
                trigger=RuleTrigger(
                    bot_qq=999,
                    allowed_group_ids={590001, 916002},
                    topic_prob=0.0,
                    proactive_prob=0.0,
                ),
                guardrails=guardrails,
                admin_user_ids={88888},
                compact_prompt=True,
                subagent_enable_thinking=False,
                subagent_max_concurrent=6,
                subagent_max_per_session=2,
            )
            try:
                # Group 590001 spawns 2 subagents, then tries a 3rd (blocked by per_group=2),
                # while Group 916002 concurrently spawns its own subagent AND chats fast!
                for idx, tag in enumerate(("A1", "A2", "A3"), start=1):
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id=f"20{idx}",
                            chat_type="group",
                            user_id=30000 + idx,
                            group_id=590001,
                            sender_name=f"一群友{idx}",
                            text=f"深查任务{tag}",
                            raw_message=f"[CQ:at,qq=999] 深查任务{tag}",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="301",
                        chat_type="group",
                        user_id=40001,
                        group_id=916002,
                        sender_name="二群友甲",
                        text="深查任务B1",
                        raw_message="[CQ:at,qq=999] 深查任务B1",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="302",
                        chat_type="group",
                        user_id=40002,
                        group_id=916002,
                        sender_name="二群友乙",
                        text="在不在",
                        raw_message="[CQ:at,qq=999] 在不在",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )

                await kernel.dispatch_once()
                await asyncio.wait_for(all_three_subs_started.wait(), timeout=2.0)
                initial_replies = await _drain_outbound_until(bus, 6, timeout_s=2.0)
                texts = [r.text for r in initial_replies]
                assert "二群秒回不卡顿" in texts
                assert "手头已经在查两个啦，等我先翻完" in texts

                # Verify per-session fairness: group:590001 capped at 2, group:916002 got 1!
                assert kernel.subagent_runner.active_count() == 3
                assert kernel.subagent_runner.active_count(session_key="group:590001") == 2
                assert kernel.subagent_runner.active_count(session_key="group:916002") == 1

                # Admin checks /status in group:590001
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="299",
                        chat_type="group",
                        user_id=88888,
                        group_id=590001,
                        sender_name="管理员",
                        text="/status",
                        raw_message="/status",
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()
                status_out = (await _drain_outbound_until(bus, 1, timeout_s=1.0))[0]
                assert "全局 3/6 个" in status_out.text
                assert "当前会话 2 个" in status_out.text
                assert "单群限额=2" in status_out.text
                assert "慢脑思考=OFF" in status_out.text

                # Release all 3 subagents and verify clean delivery to respective groups
                sub_gate.set()
                followup_replies = await _drain_outbound_until(bus, 3, timeout_s=2.5)
                await kernel.wait_idle(timeout=2.0)

                g1_followups = [r for r in followup_replies if r.target_id == 590001]
                g2_followups = [r for r in followup_replies if r.target_id == 916002]
                assert len(g1_followups) == 2
                assert len(g2_followups) == 1
                assert kernel.ack_coordinator.inflight_count == 0
            finally:
                sub_gate.set()
                await kernel.aclose()

    asyncio.run(_run())


def test_cross_group_shared_cortex_sync_and_strict_privacy_firewall(
    tmp_path: Path,
) -> None:
    async def _run() -> None:
        persona_path = _write_rich_persona(tmp_path)
        group2_captured_messages: list[list[ChatMessage]] = []
        web_search_calls = 0

        class _FakeSearchClient(WebSearchClient):
            async def search_items(
                self,
                query: str,
                *,
                max_results: int = 5,
            ) -> list[SearchResultItem]:
                nonlocal web_search_calls
                web_search_calls += 1
                return [
                    SearchResultItem(
                        title="Rust 1.85.0 正式发布公告",
                        url="https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/",
                        snippet="Rust 1.85 正式稳定了 async closures 与 Rust 2024 Edition。",
                        source="bing_cn",
                    )
                ]

        class _SharedCortexProvider:
            def __init__(self) -> None:
                self._g1_sub_spawned = False

            async def chat(
                self,
                messages: list[ChatMessage],
                *,
                model: str | None = None,
                temperature: float = 0.7,
                max_tokens: int = 512,
                tools: tuple[ToolDef, ...] | list[ToolDef] = (),
                enable_thinking: bool | None = None,
            ) -> ChatResult:
                user_text = messages[-1].content

                if "<research_goal>" in user_text:
                    return ChatResult(
                        content="Rust 1.85 正式稳定了 async closures 与 Rust 2024 Edition。",
                        model=model or "glm-5.3-flash",
                    )

                if "<subagent_research_brief" in user_text:
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="g1_cb",
                                name="send_message",
                                arguments={"bubbles": ["翻完啦，Rust 1.85 稳定了 async closures 和 2024 Edition"]},
                            ),
                        ),
                    )

                if "私聊测试机密号码" in user_text:
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="dm_send",
                                name="send_message",
                                arguments={"bubbles": ["收到私聊，绝不外传"]},
                            ),
                        ),
                    )

                if "顺便深查下 Rust 1.85" in user_text and not self._g1_sub_spawned:
                    self._g1_sub_spawned = True
                    return ChatResult(
                        content="",
                        model=model or "glm-5.3-flash",
                        tool_calls=(
                            ToolCall(
                                id="g1_pad",
                                name="send_message",
                                arguments={"bubbles": ["等下我翻下 Rust 1.85 公告"]},
                            ),
                            ToolCall(
                                id="g1_sp",
                                name="spawn_subagent",
                                arguments={"goal": "Rust 1.85 核心更新内容"},
                            ),
                        ),
                    )

                # Group 2 query! Capture full prompt messages for Shared Cortex & Privacy Firewall audit
                group2_captured_messages.append(list(messages))
                return ChatResult(
                    content="",
                    model=model or "glm-5.3-flash",
                    tool_calls=(
                        ToolCall(
                            id="g2_send",
                            name="send_message",
                            arguments={"bubbles": ["刚查过，稳定了 async closures 和 2024 Edition"]},
                        ),
                    ),
                )

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            MemoryStore(tmp_path / "memory.sqlite3") as memory,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            # Seed isolated memories across scopes
            memory.add_memory("管理员服务器SSH密钥指纹XYZ-SECRET", scope="admin_private")
            memory.add_memory("590001群周末内部聚餐暗语ALPHA-ONLY", scope="group:590001")
            memory.add_memory("AME平时喜欢玩碧蓝航线", scope="public")

            bus = DirectMemoryBus()
            kernel = AgentKernel(
                bus=bus,
                provider=_SharedCortexProvider(),
                persona=PersonaManager(persona_path),
                sessions=sessions,
                memory=memory,
                trigger=RuleTrigger(
                    bot_qq=999,
                    allowed_group_ids={590001, 916002},
                    topic_prob=0.0,
                    proactive_prob=0.0,
                ),
                guardrails=guardrails,
                admin_user_ids={88888},
                compact_prompt=True,
                fact_cache_ttl_s=1800.0,
                web_search_client=_FakeSearchClient(),
            )
            try:
                # 1) In Group 590001: User 777001 mentions playing Age of Empires 4 & asks about Rust 1.85
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="501",
                        chat_type="group",
                        user_id=777001,
                        group_id=590001,
                        sender_name="舰长甲",
                        text="最近在玩帝国时代4，顺便深查下 Rust 1.85 有啥核心更新",
                        raw_message="[CQ:at,qq=999] 最近在玩帝国时代4，顺便深查下 Rust 1.85 有啥核心更新",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()
                await _drain_outbound_until(bus, 2, timeout_s=2.5)
                await kernel.wait_idle(timeout=2.0)

                # 2) Admin sends sensitive private DM -> must NEVER leak into bot_self_state or Group 916002
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="502",
                        chat_type="private",
                        user_id=88888,
                        group_id=None,
                        sender_name="管理员",
                        text="私聊测试机密号码 13812345678 @秘密联系人 TOP-DM-SECRET",
                        raw_message="私聊测试机密号码 13812345678 @秘密联系人 TOP-DM-SECRET",
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()
                await _drain_outbound_until(bus, 1, timeout_s=1.5)
                await kernel.wait_idle(timeout=1.5)

                # 3) Now in Group 916002: same acquaintance 777001 asks about Rust 1.85!
                await bus.publish_inbound(
                    InboundMessage(
                        event_id="601",
                        chat_type="group",
                        user_id=777001,
                        group_id=916002,
                        sender_name="舰长甲",
                        text="Rust 1.85 更新了啥来着",
                        raw_message="[CQ:at,qq=999] Rust 1.85 更新了啥来着",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )
                await kernel.dispatch_once()
                g2_replies = await _drain_outbound_until(bus, 1, timeout_s=1.5)
                assert len(g2_replies) == 1
                assert g2_replies[0].target_id == 916002

                assert len(group2_captured_messages) == 1
                g2_msgs = group2_captured_messages[0]
                g2_sys = g2_msgs[0].content
                g2_all_content = "\n".join(m.content for m in g2_msgs)

                # Shared Cortex Layer 1: Cached objective research fact injected in 0ms without re-searching!
                assert "【近期已查证客观资讯（可直接参考免重搜）】" in g2_sys
                assert "Rust 1.85 正式稳定了 async closures" in g2_sys
                assert web_search_calls == 0

                # Shared Cortex Layer 2: Cross-group acquaintance public hobby cue synced!
                assert "玩帝国时代" in g2_sys
                assert "关注编程/AI" in g2_sys

                # Shared Cortex Layer 3: Sanitized bot public self-state synced (with zero IDs)!
                assert "【你当下的自身近况" in g2_sys
                assert "590001" not in g2_sys
                self_state_now = memory.get_public_self_state()
                assert self_state_now
                assert "777001" not in self_state_now
                assert "590001" not in self_state_now
                assert "13812345678" not in self_state_now

                # Public memory shared as expected
                assert "AME平时喜欢玩碧蓝航线" in g2_sys

                # Strict Privacy Firewall: ZERO cross-group or private leaks!
                assert "XYZ-SECRET" not in g2_all_content
                assert "ALPHA-ONLY" not in g2_all_content
                assert "13812345678" not in g2_all_content
                assert "秘密联系人" not in g2_all_content
                assert "TOP-DM-SECRET" not in g2_all_content
            finally:
                await kernel.aclose()

    asyncio.run(_run())
