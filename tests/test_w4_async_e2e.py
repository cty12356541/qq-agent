"""Wave W4 end-to-end tests: Pi-Agent SessionActor concurrency, fast/slow depth=1 SubAgent, Use-Chat-as-Tool, steering, /stop, and decoupled llmos recovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import time
from typing import Any

import httpx
import pytest

from qq_agent import (
    AgentKernel,
    DirectMemoryBus,
    GuardrailManager,
    InboundMessage,
    ManagedLlmosDaemon,
    MemoryStore,
    OpenAICompatProvider,
    OutboundReply,
    PersonaManager,
    ResilientLlmosBus,
    RuleTrigger,
    SessionStore,
    WebPageFetcher,
    WebSearchClient,
    cli,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONA_PATH = REPO_ROOT / "config" / "persona.default.toml"
LLMOS_ROOT = Path(r"D:\llmos")


def _fake_public_resolver(host: str, port: int) -> list[Any]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


async def _drain_outbound_until(
    bus: DirectMemoryBus | ResilientLlmosBus,
    expected_count: int,
    *,
    timeout_s: float = 5.0,
) -> list[OutboundReply]:
    collected: list[OutboundReply] = []
    deadline = time.monotonic() + timeout_s
    while len(collected) < expected_count and time.monotonic() < deadline:
        batch = await bus.poll_outbound(limit=16)
        for seq, reply in batch:
            collected.append(reply)
            await bus.ack_outbound(seq)
        if len(collected) < expected_count:
            await asyncio.sleep(0.01)
    assert len(collected) >= expected_count, (
        f"Expected at least {expected_count} outbound replies, got {len(collected)}: {collected}"
    )
    return collected


def test_w4_use_chat_as_tool_multi_bubble_fast_web_search_and_soft_trigger_silence(
    tmp_path: Path,
) -> None:
    """Verify Use-Chat-as-Tool multi-bubble streaming, fast web_search loop, and soft-trigger stay_silent."""
    seen_requests: list[dict[str, Any]] = []

    def mock_search_transport(request: httpx.Request) -> httpx.Response:
        html_doc = """
        <html><body>
          <ol id="b_results">
            <li class="b_algo">
              <h2><a href="https://worldofwarships.asia/zh-tw/news/ranked-15-2/">战舰世界 15.2 排位赛公告</a></h2>
              <div class="b_caption"><p>新赛季排位赛将于下周四开启，黄金联盟采用 10 级 6v6 赛制。</p></div>
            </li>
          </ol>
        </body></html>
        """
        return httpx.Response(200, text=html_doc, headers={"Content-Type": "text/html; charset=utf-8"})

    def mock_llm_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_requests.append(body)
        idx = len(seen_requests)

        if idx == 1:
            # Hard @AME trigger: `stay_silent` MUST NOT be exposed!
            tool_names = {t["name"] for t in body.get("tools", [])}
            assert "send_message" in tool_names
            assert "web_search" in tool_names
            assert "stay_silent" not in tool_names
            return httpx.Response(
                200,
                json={
                    "id": "msg_w4_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_pad_1",
                            "name": "send_message",
                            "input": {
                                "bubbles": ["等下我搜下）"],
                                "reply_to_message_id": 201,
                            },
                        },
                        {
                            "type": "tool_use",
                            "id": "call_search_1",
                            "name": "web_search",
                            "input": {"query": "战舰世界 15.2 排位"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 110, "output_tokens": 35},
                },
            )

        if idx == 2:
            # Step 1 after web_search returns <untrusted_web_results>
            last_msg = body["messages"][-1]
            assert last_msg["role"] == "user"
            tool_res_text = json.dumps(last_msg["content"], ensure_ascii=False)
            assert "untrusted_web_results" in tool_res_text
            assert "10 级 6v6" in tool_res_text
            return httpx.Response(
                200,
                json={
                    "id": "msg_w4_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_ans_1",
                            "name": "send_message",
                            "input": {
                                "bubbles": ["查到了，下周四开新赛季", "黄金是10级6v6）"],
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 180, "output_tokens": 28},
                },
            )

        # idx == 3: Soft trigger (60s follow-up window without @AME) -> `stay_silent` IS exposed!
        tool_names_soft = {t["name"] for t in body.get("tools", [])}
        assert "stay_silent" in tool_names_soft
        return httpx.Response(
            200,
            json={
                "id": "msg_w4_3",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_silent_1",
                        "name": "stay_silent",
                        "input": {"reason": "群友在和室友确认外卖，不生硬插话"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 150, "output_tokens": 18},
            },
        )

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        persona = PersonaManager(DEFAULT_PERSONA_PATH)

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            MemoryStore(tmp_path / "memory.sqlite3") as memory,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            trigger = RuleTrigger(
                bot_qq=999,
                allowed_group_ids={70001},
                followup_window_s=60.0,
            )
            search_client = WebSearchClient(
                transport=httpx.MockTransport(mock_search_transport)
            )
            async with OpenAICompatProvider(
                api_key="sk-w4-mock",
                base_url="http://127.0.0.1:15722/api/anthropic",
                default_model="glm-5.3-flash",
                enable_thinking=False,
                transport=httpx.MockTransport(mock_llm_transport),
            ) as provider:
                kernel = AgentKernel(
                    bus=bus,
                    provider=provider,
                    persona=persona,
                    sessions=sessions,
                    memory=memory,
                    trigger=trigger,
                    guardrails=guardrails,
                    bot_qq=999,
                    admin_user_ids={88888},
                    bubble_typing_delay_s=0.0,
                    web_search_client=search_client,
                )
                stop_ev = asyncio.Event()
                kernel_task = asyncio.create_task(
                    kernel.run_forever(stop_event=stop_ev, poll_interval_s=0.01)
                )
                try:
                    # 1) Hard @AME trigger -> padding bubble + fast web_search + 2-bubble answer
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="201",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="舰长老王",
                            text="新版本15.2排位啥时候开",
                            raw_message="[CQ:at,qq=999] 新版本15.2排位啥时候开",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    bubbles = await _drain_outbound_until(bus, 3)
                    assert [b.text for b in bubbles] == [
                        "等下我搜下）",
                        "查到了，下周四开新赛季",
                        "黄金是10级6v6）",
                    ]
                    assert bubbles[0].reply_to_message_id == 201
                    assert bubbles[1].reply_to_message_id is None
                    assert bubbles[2].reply_to_message_id is None

                    await kernel.wait_idle(timeout=3.0)
                    win_after_search = sessions.get_window("group:70001")
                    assert len(win_after_search) == 2
                    assert "untrusted_web_results" not in win_after_search[1].content

                    # 2) Soft follow-up trigger (no @AME) -> model calls `stay_silent` -> 0 bubbles & 0 L1 pollution
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="202",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="舰长老王",
                            text="室友问我要不要加卤蛋",
                            raw_message="室友问我要不要加卤蛋",
                            mentioned_bot=False,
                            self_id=999,
                        )
                    )
                    deadline = time.monotonic() + 3.0
                    while len(seen_requests) < 3 and time.monotonic() < deadline:
                        await asyncio.sleep(0.01)
                    await kernel.wait_idle(timeout=3.0)

                    assert len(seen_requests) == 3
                    assert await bus.poll_outbound() == []
                    assert len(sessions.get_window("group:70001")) == 2
                finally:
                    stop_ev.set()
                    await asyncio.wait_for(kernel_task, timeout=5.0)
                    await bus.close()

    asyncio.run(_run())


def test_w4_intra_group_nonblocking_concurrency_with_depth1_subagent(
    tmp_path: Path,
) -> None:
    """Verify Fast Brain spawns depth=1 Slow Brain without blocking same-group chat, pins ACK, and quotes origin."""
    slow_fetch_entered = asyncio.Event()
    release_slow_fetch = asyncio.Event()
    llm_payloads: list[dict[str, Any]] = []

    def mock_search_transport(request: httpx.Request) -> httpx.Response:
        html_doc = """
        <html><body>
          <ol id="b_results">
            <li class="b_algo">
              <h2><a href="https://wows.example.org/devblog-noshiro-dm">开发者日志：能代与得梅因平衡调整</a></h2>
              <div class="b_caption"><p>详见完整开发者日志平衡参数对比。</p></div>
            </li>
          </ol>
        </body></html>
        """
        return httpx.Response(200, text=html_doc, headers={"Content-Type": "text/html; charset=utf-8"})

    async def slow_page_handler(request: httpx.Request) -> httpx.Response:
        slow_fetch_entered.set()
        await release_slow_fetch.wait()
        page_html = (
            "<html><head><title>DevBlog 15.2</title></head><body><main>"
            "<p>平衡调整正式服说明：巡洋舰能代（Noshiro）本轮无参数改动；"
            "巡洋舰得梅因（Des Moines）雷达作用时间下调 2 秒，其余装填保持不变。</p>"
            "<p>补充测试数据与服务器维护时间表，确保正文长度超过一百六十字符以跳过外部渲染回退。"
            "亚服与欧服同步于周四凌晨更新补丁，排位赛与随机战均适用上述平衡参数。</p>"
            "</main></body></html>"
        )
        return httpx.Response(
            200,
            text=page_html,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    def mock_llm_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        llm_payloads.append(body)
        sys_text = body.get("system", "")
        user_dump = json.dumps(body.get("messages", []), ensure_ascii=False)
        tool_names = {t["name"] for t in body.get("tools", [])}

        # Slow Brain (depth=1 Research Sub-Agent): thinking MUST NOT be disabled, strict research-only tools!
        if "Research Sub-Agent" in sys_text:
            assert body.get("thinking") != {"type": "disabled"}
            assert "spawn_subagent" not in tool_names
            assert "send_message" not in tool_names
            assert "fetch_webpage" in tool_names

            if "untrusted_webpage" not in user_dump:
                return httpx.Response(
                    200,
                    json={
                        "id": "sub_step_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "glm-5.3-flash",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "sub_fetch_1",
                                "name": "fetch_webpage",
                                "input": {"url": "https://wows.example.org/devblog-noshiro-dm"},
                            }
                        ],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 90, "output_tokens": 30},
                    },
                )

            return httpx.Response(
                200,
                json={
                    "id": "sub_step_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "text",
                            "text": "查证结论：DevBlog 15.2 确认能代无参数改动，得梅因雷达持续时间下调 2 秒。",
                        }
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 210, "output_tokens": 40},
                },
            )

        # Fast Brain (Main Persona Loop): thinking MUST be disabled!
        assert body.get("thinking") == {"type": "disabled"}

        if "subagent_research_brief" in user_dump:
            return httpx.Response(
                200,
                json={
                    "id": "fast_cb_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_cb_send",
                            "name": "send_message",
                            "input": {
                                "bubbles": ["刚翻完日志，能代这次没动", "得梅因雷达砍了两秒）"],
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 160, "output_tokens": 25},
                },
            )

        if "今晚吃啥" in user_dump:
            return httpx.Response(
                200,
                json={
                    "id": "fast_member_b",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_b_send",
                            "name": "send_message",
                            "input": {
                                "bubbles": ["刚点了个猪脚饭）"],
                                "reply_to_message_id": 303,
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 120, "output_tokens": 15},
                },
            )

        # Member A's initial complex question -> immediate bubble + spawn_subagent!
        return httpx.Response(
            200,
            json={
                "id": "fast_member_a",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_a_pad",
                        "name": "send_message",
                        "input": {
                            "bubbles": ["我翻下日志帖子）"],
                            "reply_to_message_id": 301,
                        },
                    },
                    {
                        "type": "tool_use",
                        "id": "call_a_spawn",
                        "name": "spawn_subagent",
                        "input": {
                            "goal": "核实 DevBlog 15.2 能代与得梅因平衡改动",
                            "context_hint": "战舰世界 15.2 开发者日志",
                        },
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 130, "output_tokens": 42},
            },
        )

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        persona = PersonaManager(DEFAULT_PERSONA_PATH)

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            MemoryStore(tmp_path / "memory.sqlite3") as memory,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            trigger = RuleTrigger(bot_qq=999, allowed_group_ids={70001})
            search_client = WebSearchClient(
                transport=httpx.MockTransport(mock_search_transport)
            )
            page_fetcher = WebPageFetcher(
                transport=httpx.MockTransport(slow_page_handler),
                resolver=_fake_public_resolver,
                enable_jina_fallback=False,
            )
            async with OpenAICompatProvider(
                api_key="sk-w4-dual",
                base_url="http://127.0.0.1:15722/api/anthropic",
                default_model="glm-5.3-flash",
                enable_thinking=False,
                transport=httpx.MockTransport(mock_llm_transport),
            ) as provider:
                kernel = AgentKernel(
                    bus=bus,
                    provider=provider,
                    persona=persona,
                    sessions=sessions,
                    memory=memory,
                    trigger=trigger,
                    guardrails=guardrails,
                    bot_qq=999,
                    admin_user_ids={88888},
                    bubble_typing_delay_s=0.0,
                    web_search_client=search_client,
                    webpage_fetcher=page_fetcher,
                    subagent_enabled=True,
                    subagent_enable_thinking=True,
                )
                stop_ev = asyncio.Event()
                kernel_task = asyncio.create_task(
                    kernel.run_forever(stop_event=stop_ev, poll_interval_s=0.01)
                )
                try:
                    # 1) Member A triggers deep research
                    seq_a = await bus.publish_inbound(
                        InboundMessage(
                            event_id="301",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="群友A",
                            text="帮我翻下15.2日志能代和得梅因改了啥",
                            raw_message="[CQ:at,qq=999] 帮我翻下15.2日志能代和得梅因改了啥",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    assert seq_a == 1

                    first_bubble = (await _drain_outbound_until(bus, 1))[0]
                    assert first_bubble.text == "我翻下日志帖子）"
                    assert first_bubble.reply_to_message_id == 301

                    # Wait until Slow Brain is actively blocked inside `fetch_webpage`
                    await asyncio.wait_for(slow_fetch_entered.wait(), timeout=5.0)

                    # Delayed ACK invariant: seq_a=1 is pinned by active sub-agent!
                    assert kernel.ack_coordinator.inflight_count == 1
                    assert kernel.ack_coordinator.contiguous_high_water == 0

                    # 2) Admin runs /status while Sub-Agent is in flight -> 0ms response!
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="302",
                            chat_type="group",
                            user_id=88888,
                            group_id=70001,
                            sender_name="观测枢",
                            text="/status",
                            raw_message="/status",
                            self_id=999,
                        )
                    )
                    status_reply = (await _drain_outbound_until(bus, 1))[0]
                    assert "当前会话 1 个" in status_reply.text
                    assert "慢脑思考=ON" in status_reply.text

                    # 3) Member B in the SAME group chats with AME while Sub-Agent is still running!
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="303",
                            chat_type="group",
                            user_id=10002,
                            group_id=70001,
                            sender_name="群友B",
                            text="今晚吃啥",
                            raw_message="[CQ:at,qq=999] 今晚吃啥",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    member_b_reply = (await _drain_outbound_until(bus, 1))[0]
                    assert member_b_reply.text == "刚点了个猪脚饭）"
                    assert member_b_reply.reply_to_message_id == 303
                    assert not release_slow_fetch.is_set()

                    # 4) Release Slow Brain -> SubAgentResult wakes SessionActor -> quotes Member A (301) & ACKs all!
                    release_slow_fetch.set()
                    final_bubbles = await _drain_outbound_until(bus, 2)
                    assert [b.text for b in final_bubbles] == [
                        "刚翻完日志，能代这次没动",
                        "得梅因雷达砍了两秒）",
                    ]
                    assert final_bubbles[0].reply_to_message_id == 301

                    await kernel.wait_idle(timeout=5.0)
                    assert kernel.ack_coordinator.inflight_count == 0
                    assert kernel.ack_coordinator.contiguous_high_water == 3
                finally:
                    release_slow_fetch.set()
                    stop_ev.set()
                    await asyncio.wait_for(kernel_task, timeout=5.0)
                    await bus.close()

    asyncio.run(_run())


def test_w4_mid_turn_steering_and_admin_stop_preemption(tmp_path: Path) -> None:
    """Verify mid-turn `steer()` injection during tool I/O and `/stop` cancelling background SubAgent cleanly."""
    search_entered = asyncio.Event()
    release_search = asyncio.Event()
    sub_blocked = asyncio.Event()
    seen_fast_calls: list[dict[str, Any]] = []

    async def controllable_search_handler(request: httpx.Request) -> httpx.Response:
        search_entered.set()
        await release_search.wait()
        html_doc = """
        <html><body>
          <ol id="b_results">
            <li class="b_algo">
              <h2><a href="https://wows.example.org/maint">战舰世界维护时间</a></h2>
              <div class="b_caption"><p>亚服 05:00 更新，欧服 13:00 更新。</p></div>
            </li>
          </ol>
        </body></html>
        """
        return httpx.Response(200, text=html_doc, headers={"Content-Type": "text/html; charset=utf-8"})

    async def hanging_page_handler(request: httpx.Request) -> httpx.Response:
        sub_blocked.set()
        await asyncio.sleep(60.0)
        return httpx.Response(200, text="<html><body>unreachable</body></html>")

    def mock_llm_transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        sys_text = body.get("system", "")
        msgs_dump = json.dumps(body.get("messages", []), ensure_ascii=False)

        if "Research Sub-Agent" in sys_text:
            return httpx.Response(
                200,
                json={
                    "id": "sub_hang_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "sub_hang_fetch",
                            "name": "fetch_webpage",
                            "input": {"url": "https://wows.example.org/slow-guide"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 60, "output_tokens": 20},
                },
            )

        seen_fast_calls.append(body)
        if "steering_update" in msgs_dump:
            assert "欧服" in msgs_dump
            return httpx.Response(
                200,
                json={
                    "id": "fast_steered_ans",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_steered_send",
                            "name": "send_message",
                            "input": {"bubbles": ["欧服是下午一点维护）"]},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 170, "output_tokens": 20},
                },
            )

        if "深查长贴" in msgs_dump:
            return httpx.Response(
                200,
                json={
                    "id": "fast_spawn_stop",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_stop_pad",
                            "name": "send_message",
                            "input": {"bubbles": ["我去翻下长贴）"]},
                        },
                        {
                            "type": "tool_use",
                            "id": "call_stop_spawn",
                            "name": "spawn_subagent",
                            "input": {"goal": "深查长贴"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 100, "output_tokens": 30},
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "fast_step0_search",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_s0",
                        "name": "web_search",
                        "input": {"query": "战舰世界今日维护时间"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 20},
            },
        )

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        persona = PersonaManager(DEFAULT_PERSONA_PATH)

        with (
            SessionStore(tmp_path / "sessions.sqlite3") as sessions,
            GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0) as guardrails,
        ):
            trigger = RuleTrigger(bot_qq=999, allowed_group_ids={70001})
            search_client = WebSearchClient(
                transport=httpx.MockTransport(controllable_search_handler)
            )
            page_fetcher = WebPageFetcher(
                transport=httpx.MockTransport(hanging_page_handler),
                resolver=_fake_public_resolver,
                enable_jina_fallback=False,
            )
            async with OpenAICompatProvider(
                api_key="sk-w4-steer",
                base_url="http://127.0.0.1:15722/api/anthropic",
                default_model="glm-5.3-flash",
                enable_thinking=False,
                transport=httpx.MockTransport(mock_llm_transport),
            ) as provider:
                kernel = AgentKernel(
                    bus=bus,
                    provider=provider,
                    persona=persona,
                    sessions=sessions,
                    trigger=trigger,
                    guardrails=guardrails,
                    bot_qq=999,
                    admin_user_ids={88888},
                    bubble_typing_delay_s=0.0,
                    web_search_client=search_client,
                    webpage_fetcher=page_fetcher,
                )
                stop_ev = asyncio.Event()
                kernel_task = asyncio.create_task(
                    kernel.run_forever(stop_event=stop_ev, poll_interval_s=0.01)
                )
                try:
                    # Part A: Mid-turn `steer()` from same speaker while `web_search` is in flight
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="401",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="舰长甲",
                            text="今天几点维护",
                            raw_message="[CQ:at,qq=999] 今天几点维护",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    await asyncio.wait_for(search_entered.wait(), timeout=5.0)

                    # Inject mid-turn correction before releasing `web_search`
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="402",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="舰长甲",
                            text="忘说了问的是欧服",
                            raw_message="[CQ:at,qq=999] 忘说了问的是欧服",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    # Let IngressDispatcher pick up msg 402 and steer the running SessionActor
                    deadline = time.monotonic() + 2.0
                    while kernel.ack_coordinator.contiguous_high_water < 0 and time.monotonic() < deadline:
                        await asyncio.sleep(0.01)
                    await asyncio.sleep(0.05)

                    release_search.set()
                    steered_reply = (await _drain_outbound_until(bus, 1))[0]
                    assert steered_reply.text == "欧服是下午一点维护）"
                    await kernel.wait_idle(timeout=3.0)
                    assert kernel.ack_coordinator.contiguous_high_water == 2

                    # Part B: Spawn hanging background sub-agent, then `/stop` cancels it with zero leaked seqs!
                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="403",
                            chat_type="group",
                            user_id=10001,
                            group_id=70001,
                            sender_name="舰长甲",
                            text="帮我深查长贴",
                            raw_message="[CQ:at,qq=999] 帮我深查长贴",
                            mentioned_bot=True,
                            self_id=999,
                        )
                    )
                    pad = (await _drain_outbound_until(bus, 1))[0]
                    assert pad.text == "我去翻下长贴）"
                    await asyncio.wait_for(sub_blocked.wait(), timeout=5.0)
                    assert len(kernel.subagent_runner.active_tasks()) == 1
                    assert kernel.ack_coordinator.inflight_count == 1

                    await bus.publish_inbound(
                        InboundMessage(
                            event_id="404",
                            chat_type="group",
                            user_id=88888,
                            group_id=70001,
                            sender_name="观测枢",
                            text="/stop",
                            raw_message="/stop",
                            self_id=999,
                        )
                    )
                    stop_reply = (await _drain_outbound_until(bus, 1))[0]
                    assert "取消后台子任务 1 个" in stop_reply.text
                    assert kernel.subagent_runner.active_tasks() == []

                    deadline = time.monotonic() + 3.0
                    while kernel.ack_coordinator.inflight_count > 0 and time.monotonic() < deadline:
                        await asyncio.sleep(0.01)
                    assert kernel.ack_coordinator.inflight_count == 0
                    assert kernel.ack_coordinator.contiguous_high_water == 4
                    assert await bus.poll_outbound() == []
                finally:
                    release_search.set()
                    stop_ev.set()
                    await asyncio.wait_for(kernel_task, timeout=5.0)
                    await bus.close()

    asyncio.run(_run())


@pytest.mark.skipif(
    not (LLMOS_ROOT / "sdk" / "python" / "nlos_sdk" / "channel.py").exists(),
    reason=f"llmos checkout with ChannelClient SDK not found at {LLMOS_ROOT}",
)
def test_w4_managed_llmos_daemon_clean_cwd_and_decoupled_adapter_brain_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ManagedLlmosDaemon absolute `--root` + clean cwd, CLI roles, and decoupled adapter/brain replay."""
    # 1) CLI parser & `--role adapter` no-API-key validation
    parser = cli.build_parser()
    parsed_args = parser.parse_args(
        ["--role", "adapter", "--llmos-endpoint", r"\\.\pipe\test-ep"]
    )
    assert parsed_args.role == "adapter"
    assert parsed_args.llmos_endpoint == r"\\.\pipe\test-ep"

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    cfg_adapter = cli._resolve_config_from_args(parsed_args)
    assert cfg_adapter.runtime_role == "adapter"

    def mock_llm_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_llmos_decoupled",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_llmos_1",
                        "name": "send_message",
                        "input": {
                            "bubbles": ["脑进程重启恢复成功", "持久化队列零丢包）"],
                            "reply_to_message_id": 801,
                        },
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 90, "output_tokens": 24},
            },
        )

    async def _run() -> None:
        daemon_root = tmp_path / "isolated_llmos_daemon"
        async with ManagedLlmosDaemon(
            LLMOS_ROOT,
            data_dir=tmp_path,
            root_dir=daemon_root,
        ) as daemon:
            assert daemon.root_dir == daemon_root.resolve()
            assert daemon.root_dir.is_absolute()
            if daemon.launch_mode.startswith("prebuilt"):
                assert daemon.last_cwd == REPO_ROOT.resolve()
                assert daemon.last_cwd != LLMOS_ROOT.resolve()

            # Simulate isolated `adapter` process buffering inbound while `brain` is offline!
            adapter_bus = ResilientLlmosBus(
                daemon.plain_endpoint,
                llmos_root=LLMOS_ROOT,
                role="adapter",
            )
            await adapter_bus.start()
            try:
                enq_seq = await adapter_bus.publish_inbound(
                    InboundMessage(
                        event_id="801",
                        chat_type="group",
                        user_id=20001,
                        group_id=70001,
                        sender_name="压测群友",
                        text="测试脑进程离线期间入队不丢消息",
                        raw_message="[CQ:at,qq=999] 测试脑进程离线期间入队不丢消息",
                        mentioned_bot=True,
                        self_id=999,
                    )
                )
                assert enq_seq == 1

                # Now boot isolated `brain` bus + AgentKernel attaching to the same llmos endpoint!
                brain_bus = ResilientLlmosBus(
                    daemon.plain_endpoint,
                    llmos_root=LLMOS_ROOT,
                    role="brain",
                )
                await brain_bus.start()
                try:
                    health = await brain_bus.inspect_health()
                    assert health.inbound_replayed is True
                    assert health.inbound_backlog_bytes > 0

                    persona = PersonaManager(DEFAULT_PERSONA_PATH)
                    with (
                        SessionStore(tmp_path / "sessions.sqlite3") as sessions,
                        GuardrailManager(
                            tmp_path / "guardrails.sqlite3",
                            user_cooldown_s=0.0,
                        ) as guardrails,
                    ):
                        trigger = RuleTrigger(bot_qq=999, allowed_group_ids={70001})
                        async with OpenAICompatProvider(
                            api_key="sk-llmos-brain",
                            base_url="http://127.0.0.1:15722/api/anthropic",
                            default_model="glm-5.3-flash",
                            enable_thinking=False,
                            transport=httpx.MockTransport(mock_llm_transport),
                        ) as provider:
                            kernel = AgentKernel(
                                bus=brain_bus,
                                provider=provider,
                                persona=persona,
                                sessions=sessions,
                                trigger=trigger,
                                guardrails=guardrails,
                                bot_qq=999,
                                bubble_typing_delay_s=0.0,
                                runtime_role="brain",
                            )
                            stop_ev = asyncio.Event()
                            ktask = asyncio.create_task(
                                kernel.run_forever(stop_event=stop_ev, poll_interval_s=0.01)
                            )
                            try:
                                # `adapter_bus` receives both bubbles emitted by `brain` over llmos ChannelControl!
                                replies = await _drain_outbound_until(adapter_bus, 2, timeout_s=8.0)
                                assert [r.text for r in replies] == [
                                    "脑进程重启恢复成功",
                                    "持久化队列零丢包）",
                                ]
                                assert replies[0].reply_to_message_id == 801
                                await kernel.wait_idle(timeout=3.0)
                                assert kernel.ack_coordinator.contiguous_high_water == 1
                            finally:
                                stop_ev.set()
                                await asyncio.wait_for(ktask, timeout=5.0)
                finally:
                    await brain_bus.close()
            finally:
                await adapter_bus.close()

    asyncio.run(_run())
