"""Tests for Wave W3-01: OB11 merged forward recursion, @other detection, reply cache, and WS RPC resolution."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import websockets

from qq_agent.channel_bridge import (
    DirectMemoryBus,
    MSG_CACHE_CAPACITY,
    NapCatWsAdapter,
)
from qq_agent.ob11 import (
    event_to_inbound,
    extract_unresolved_forward_ids,
    format_forward_nodes,
)
from qq_agent.types import InboundMessage, OutboundReply


async def _poll_one_inbound(
    bus: DirectMemoryBus,
    *,
    timeout_s: float = 5.0,
) -> InboundMessage:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        batch = await bus.poll_inbound(limit=1)
        if batch:
            seq, msg = batch[0]
            await bus.ack_inbound(seq)
            return msg
        await asyncio.sleep(0.01)
    raise AssertionError("Timed out waiting for inbound message on bus")


def test_format_forward_nodes_nested_two_layers_speaker_priority_and_image_dedup_cap() -> None:
    # Depth 4 (exceeds default max_depth=3 -> must not expand deeper)
    level4_nodes = [
        {
            "sender": {"card": "深层路人", "user_id": 10009},
            "message": [{"type": "text", "data": {"text": "第四层不应展开"}}],
        }
    ]
    # Depth 3 (2nd nested child forward -> MUST expand under default max_depth=3)
    level3_nodes = [
        {
            "sender": {"card": "", "nickname": "", "user_id": 10003},
            "message": [
                {"type": "text", "data": {"text": "最内层战报"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/pic3.png"},
                },
                {
                    "type": "forward",
                    "data": {"id": "fwd-l4", "content": level4_nodes},
                },
            ],
        },
        {
            "type": "node",
            "data": {
                "name": "节点包装群友",
                "uin": "10004",
                "content": [
                    {
                        "type": "image",
                        "data": {"url": "https://img.example.test/pic1.png"},  # duplicate of L1
                    },
                    {
                        "type": "image",
                        "data": {"file": "https://img.example.test/pic4.jpg"},
                    },
                    {
                        "type": "image",
                        "data": {"url": "https://img.example.test/pic5_dropped.png"},
                    },
                ],
            },
        },
    ]
    # Depth 2 (1st nested child forward)
    level2_nodes = [
        {
            "sender": {"card": "", "nickname": "昵称B", "user_id": 10002},
            "content": [
                {"type": "text", "data": {"text": "二层转述"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/pic2.png"},
                },
            ],
        },
        {
            "sender": {"card": "名片C", "nickname": "昵称C", "user_id": 10005},
            "message": [
                {
                    "type": "forward",
                    "data": {"id": "fwd-l3", "content": level3_nodes},
                }
            ],
        },
    ]
    # Depth 1 (root forward nodes)
    level1_nodes = [
        {
            "sender": {"card": "群名片A", "nickname": "昵称A", "user_id": 10001},
            "message": [
                {"type": "text", "data": {"text": "第一句话"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/pic1.png"},
                },
            ],
        },
        {
            "sender": {"nickname": "转发员", "user_id": 10006},
            "message": [
                {
                    "type": "forward",
                    "data": {"id": "fwd-l2", "content": level2_nodes},
                }
            ],
        },
    ]

    formatted, urls = format_forward_nodes(level1_nodes)
    assert formatted.startswith("[合并转发聊天记录]\n")
    assert "[群名片A]: 第一句话 [图片]" in formatted
    assert "  [昵称B]: 二层转述 [图片]" in formatted
    assert "    [10003]: 最内层战报 [图片] [合并转发消息]" in formatted
    assert "第四层不应展开" not in formatted
    # Deduplicated and capped at 4 image URLs
    assert urls == [
        "https://img.example.test/pic1.png",
        "https://img.example.test/pic2.png",
        "https://img.example.test/pic3.png",
        "https://img.example.test/pic4.jpg",
    ]

    # Verify event_to_inbound integrates inline forward content + deduplicates outer & inner image URLs
    inbound = event_to_inbound(
        {
            "post_type": "message",
            "message_type": "group",
            "message_id": 9001,
            "self_id": 99999,
            "user_id": 10001,
            "group_id": 88888,
            "message": [
                {"type": "at", "data": {"qq": "99999"}},
                {
                    "type": "image",
                    "data": {"url": "https://img.example.test/pic2.png"},
                },
                {
                    "type": "forward",
                    "data": {"id": "fwd-root", "content": level1_nodes},
                },
            ],
            "sender": {"card": "测试员", "user_id": 10001},
        }
    )
    assert inbound is not None
    assert inbound.mentioned_bot is True
    assert inbound.mentioned_other is False
    assert "[合并转发聊天记录]" in inbound.text
    assert "最内层战报" in inbound.text
    assert inbound.image_urls == (
        "https://img.example.test/pic2.png",
        "https://img.example.test/pic1.png",
        "https://img.example.test/pic3.png",
        "https://img.example.test/pic4.jpg",
    )


def test_format_forward_nodes_truncation_by_max_messages_and_max_chars() -> None:
    many_nodes = [
        {
            "sender": {"card": f"群友{i}", "user_id": 10000 + i},
            "message": [{"type": "text", "data": {"text": f"这是第 {i} 条聊天记录内容"}}],
        }
        for i in range(1, 10)
    ]

    by_count, _ = format_forward_nodes(many_nodes, max_messages=3)
    assert "[群友1]: 这是第 1 条聊天记录内容" in by_count
    assert "[群友3]: 这是第 3 条聊天记录内容" in by_count
    assert "[群友4]" not in by_count
    assert by_count.endswith("...（合并转发已截断）")

    by_chars, _ = format_forward_nodes(many_nodes, max_chars=90)
    assert len(by_chars) <= 90
    assert by_chars.startswith("[合并转发聊天记录]")
    assert by_chars.endswith("...（合并转发已截断）")

    # Single oversized message node also truncates safely within max_chars
    huge_nodes = [
        {
            "sender": {"card": "长文哥", "user_id": 10088},
            "message": [{"type": "text", "data": {"text": "哈" * 2000}}],
        }
    ]
    huge_text, _ = format_forward_nodes(huge_nodes, max_chars=120)
    assert len(huge_text) <= 120
    assert "[长文哥]: 哈哈哈" in huge_text
    assert huge_text.endswith("...（合并转发已截断）")


def test_extract_unresolved_forward_ids_and_placeholder_fallback() -> None:
    resolved_event = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 9100,
        "user_id": 10001,
        "group_id": 88888,
        "message": [
            {
                "type": "forward",
                "data": {
                    "id": "fwd-inline",
                    "content": [
                        {
                            "sender": {"nickname": "A", "user_id": 10002},
                            "message": [{"type": "text", "data": {"text": "已内联"}}],
                        }
                    ],
                },
            }
        ],
    }
    assert extract_unresolved_forward_ids(resolved_event) == []

    unresolved_array_event = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 9101,
        "user_id": 10001,
        "group_id": 88888,
        "message": [
            {"type": "forward", "data": {"id": "fwd-a"}},
            {"type": "forward", "data": {"id": "fwd-b", "content": []}},
            {"type": "forward", "data": {"id": "fwd-a"}},
        ],
    }
    assert extract_unresolved_forward_ids(unresolved_array_event) == ["fwd-a", "fwd-b"]
    msg_arr = event_to_inbound(unresolved_array_event, bot_qq=99999)
    assert msg_arr is not None
    assert "[合并转发消息]" in msg_arr.text

    cq_event = {
        "post_type": "message",
        "message_type": "group",
        "message_id": 9102,
        "user_id": 10001,
        "group_id": 88888,
        "raw_message": "看看这个 [CQ:forward,id=fwd-cq-77]",
    }
    assert extract_unresolved_forward_ids(cq_event) == ["fwd-cq-77"]
    msg_cq = event_to_inbound(cq_event, bot_qq=99999)
    assert msg_cq is not None
    assert msg_cq.text == "看看这个 [合并转发消息]"


def test_event_to_inbound_mentioned_other_matrix() -> None:
    def _parse(segments: list[dict[str, Any]]) -> InboundMessage:
        res = event_to_inbound(
            {
                "post_type": "message",
                "message_type": "group",
                "message_id": 9200,
                "self_id": 99999,
                "user_id": 10001,
                "group_id": 88888,
                "message": segments,
            }
        )
        assert res is not None
        return res

    only_bot = _parse(
        [
            {"type": "at", "data": {"qq": "99999"}},
            {"type": "text", "data": {"text": " 在吗"}},
        ]
    )
    assert only_bot.mentioned_bot is True
    assert only_bot.mentioned_other is False

    only_other = _parse(
        [
            {"type": "at", "data": {"qq": "10002"}},
            {"type": "text", "data": {"text": " 出来打游戏"}},
        ]
    )
    assert only_other.mentioned_bot is False
    assert only_other.mentioned_other is True

    at_all = _parse(
        [
            {"type": "at", "data": {"qq": "all"}},
            {"type": "text", "data": {"text": " 全体集合"}},
        ]
    )
    assert at_all.mentioned_bot is False
    assert at_all.mentioned_other is True

    both_bot_and_other = _parse(
        [
            {"type": "at", "data": {"qq": "99999"}},
            {"type": "at", "data": {"qq": "10002"}},
            {"type": "text", "data": {"text": " 你们俩评评理"}},
        ]
    )
    assert both_bot_and_other.mentioned_bot is True
    assert both_bot_and_other.mentioned_other is True

    no_mention = _parse([{"type": "text", "data": {"text": "普通闲聊"}}])
    assert no_mention.mentioned_bot is False
    assert no_mention.mentioned_other is False


def test_ws_outbound_receipt_cache_and_inbound_zero_rpc_reply_hit() -> None:
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        ws_ready = asyncio.Event()
        server_push_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        rpc_calls: list[dict[str, Any]] = []
        outbound_sent_ready = asyncio.Event()
        next_bot_msg_id = 6000

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            nonlocal next_bot_msg_id
            ws_ready.set()

            async def _pusher() -> None:
                while True:
                    ev = await server_push_q.get()
                    if ev is None:
                        return
                    await ws.send(json.dumps(ev, ensure_ascii=False))

            pusher = asyncio.create_task(_pusher())
            try:
                async for raw in ws:
                    frame = json.loads(str(raw))
                    action = frame.get("action")
                    if action in {"get_msg", "get_forward_msg"}:
                        rpc_calls.append(frame)
                        continue
                    if action in {"send_group_msg", "send_private_msg"}:
                        next_bot_msg_id += 1
                        await ws.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "retcode": 0,
                                    "data": {"message_id": next_bot_msg_id},
                                    "echo": frame.get("echo", ""),
                                }
                            )
                        )
                        outbound_sent_ready.set()
            finally:
                pusher.cancel()
                await asyncio.gather(pusher, return_exceptions=True)

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                bot_qq=99999,
                poll_interval_s=0.01,
            )
            stop_ev = asyncio.Event()
            task = asyncio.create_task(adapter.run_once_session(stop_event=stop_ev))
            try:
                await asyncio.wait_for(ws_ready.wait(), timeout=5.0)

                # 1) Bot sends proactive outbound message (empty correlation_id -> auto echo)
                await bus.publish_outbound(
                    OutboundReply(
                        chat_type="group",
                        target_id=88888,
                        text="今晚八点准时开黑！",
                        correlation_id="",
                    )
                )
                await asyncio.wait_for(outbound_sent_ready.wait(), timeout=5.0)
                await asyncio.sleep(0.03)  # allow receipt frame to land in _recv_loop

                # Group member quotes bot's message (message_id=6001) -> 0ms cache hit, 0 RPCs
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 6101,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "6001"}},
                            {"type": "text", "data": {"text": "算我一个"}},
                        ],
                        "sender": {"card": "队友甲", "user_id": 10001},
                    }
                )
                quoted_bot_msg = await _poll_one_inbound(bus)
                assert quoted_bot_msg.reply_to_id == 6001
                assert quoted_bot_msg.reply_to_user_id == 99999
                assert quoted_bot_msg.reply_to_sender_name == "阿橘"
                assert quoted_bot_msg.reply_to_text == "今晚八点准时开黑！"
                assert rpc_calls == []

                # 2) Member B sends an image message (cached on arrival), Member A quotes it with "看这张图"
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 6102,
                        "self_id": 99999,
                        "user_id": 10002,
                        "group_id": 88888,
                        "message": [
                            {
                                "type": "image",
                                "data": {"url": "https://img.example.test/meme_ship.png"},
                            }
                        ],
                        "sender": {"card": "发图侠", "user_id": 10002},
                    }
                )
                img_inbound = await _poll_one_inbound(bus)
                assert img_inbound.image_urls == ("https://img.example.test/meme_ship.png",)

                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 6103,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "6102"}},
                            {"type": "at", "data": {"qq": "99999"}},
                            {"type": "text", "data": {"text": " 看这张图是什么梗"}},
                        ],
                        "sender": {"card": "队友甲", "user_id": 10001},
                    }
                )
                quote_img_msg = await _poll_one_inbound(bus)
                assert quote_img_msg.reply_to_user_id == 10002
                assert quote_img_msg.reply_to_sender_name == "发图侠"
                assert quote_img_msg.reply_to_text == "[图片]"
                assert quote_img_msg.image_urls == ("https://img.example.test/meme_ship.png",)
                assert rpc_calls == []

                # 3) Verify _msg_cache capacity bound (LRU evicts oldest beyond 500)
                for mid in range(1, MSG_CACHE_CAPACITY + 25):
                    adapter._cache_put(mid, (10001, "U", f"m-{mid}", ()))
                assert len(adapter._msg_cache) == MSG_CACHE_CAPACITY
                assert 1 not in adapter._msg_cache
                assert (MSG_CACHE_CAPACITY + 24) in adapter._msg_cache
            finally:
                await server_push_q.put(None)
                stop_ev.set()
                await asyncio.wait_for(task, timeout=5.0)
                await bus.close()

    asyncio.run(_run())


def test_ws_uncached_reply_get_msg_rpc_with_image_and_forward_without_deadlock() -> None:
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        ws_ready = asyncio.Event()
        server_push_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        seen_rpc_actions: list[tuple[str, dict[str, Any]]] = []

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            ws_ready.set()

            async def _pusher() -> None:
                while True:
                    ev = await server_push_q.get()
                    if ev is None:
                        return
                    await ws.send(json.dumps(ev, ensure_ascii=False))

            pusher = asyncio.create_task(_pusher())
            try:
                async for raw in ws:
                    frame = json.loads(str(raw))
                    action = str(frame.get("action") or "")
                    params = frame.get("params") or {}
                    echo = frame.get("echo", "")
                    seen_rpc_actions.append((action, params))

                    if action == "get_msg" and params.get("message_id") == 4001:
                        await ws.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "retcode": 0,
                                    "data": {
                                        "message_id": 4001,
                                        "message_type": "group",
                                        "sender": {
                                            "user_id": 20001,
                                            "card": "老船长",
                                            "nickname": "Captain",
                                        },
                                        "message": [
                                            {
                                                "type": "text",
                                                "data": {"text": "历史海战战绩"},
                                            },
                                            {
                                                "type": "image",
                                                "data": {
                                                    "url": "https://img.example.test/history_score.png"
                                                },
                                            },
                                        ],
                                    },
                                    "echo": echo,
                                },
                                ensure_ascii=False,
                            )
                        )
                    elif action == "get_msg" and params.get("message_id") == 4002:
                        # Quoted message is itself an unresolved merged forward!
                        await ws.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "retcode": 0,
                                    "data": {
                                        "message_id": 4002,
                                        "message_type": "group",
                                        "sender": {
                                            "user_id": 20002,
                                            "nickname": "情报员",
                                        },
                                        "message": [
                                            {
                                                "type": "forward",
                                                "data": {"id": "fwd-quoted-99"},
                                            }
                                        ],
                                    },
                                    "echo": echo,
                                },
                                ensure_ascii=False,
                            )
                        )
                    elif (
                        action == "get_forward_msg"
                        and params.get("message_id") == "fwd-quoted-99"
                    ):
                        await ws.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "retcode": 0,
                                    "data": {
                                        "messages": [
                                            {
                                                "sender": {
                                                    "card": "前线A",
                                                    "user_id": 30001,
                                                },
                                                "message": [
                                                    {
                                                        "type": "text",
                                                        "data": {"text": "发现敌方航母"},
                                                    },
                                                    {
                                                        "type": "image",
                                                        "data": {
                                                            "url": "https://img.example.test/cv_spot.jpg"
                                                        },
                                                    },
                                                ],
                                            }
                                        ]
                                    },
                                    "echo": echo,
                                },
                                ensure_ascii=False,
                            )
                        )
            finally:
                pusher.cancel()
                await asyncio.gather(pusher, return_exceptions=True)

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                bot_qq=99999,
                poll_interval_s=0.01,
            )
            stop_ev = asyncio.Event()
            task = asyncio.create_task(adapter.run_once_session(stop_event=stop_ev))
            try:
                await asyncio.wait_for(ws_ready.wait(), timeout=5.0)

                # Case A: Cache miss -> get_msg resolves sender, text, and image_urls
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 7101,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "4001"}},
                            {"type": "at", "data": {"qq": "99999"}},
                            {"type": "text", "data": {"text": " 评价一下这张截图"}},
                        ],
                        "sender": {"card": "群友A", "user_id": 10001},
                    }
                )
                m1 = await _poll_one_inbound(bus)
                assert m1.reply_to_id == 4001
                assert m1.reply_to_user_id == 20001
                assert m1.reply_to_sender_name == "老船长"
                assert m1.reply_to_text == "历史海战战绩 [图片]"
                assert m1.image_urls == ("https://img.example.test/history_score.png",)

                # Second reply to 4001 MUST hit _msg_cache locally without calling get_msg again
                rpc_count_before = len(seen_rpc_actions)
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 7102,
                        "self_id": 99999,
                        "user_id": 10003,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "4001"}},
                            {"type": "text", "data": {"text": "确实厉害"}},
                        ],
                        "sender": {"card": "群友C", "user_id": 10003},
                    }
                )
                m1_cached = await _poll_one_inbound(bus)
                assert m1_cached.reply_to_user_id == 20001
                assert m1_cached.reply_to_text == "历史海战战绩 [图片]"
                assert len(seen_rpc_actions) == rpc_count_before

                # Case B: Cache miss on quoted merged forward -> get_msg + chained get_forward_msg
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 7103,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "4002"}},
                            {"type": "at", "data": {"qq": "99999"}},
                            {"type": "text", "data": {"text": " 总结下这条转发"}},
                        ],
                        "sender": {"card": "群友A", "user_id": 10001},
                    }
                )
                m2 = await _poll_one_inbound(bus)
                assert m2.reply_to_id == 4002
                assert m2.reply_to_user_id == 20002
                assert m2.reply_to_sender_name == "情报员"
                assert "[合并转发聊天记录]" in m2.reply_to_text
                assert "[前线A]: 发现敌方航母 [图片]" in m2.reply_to_text
                assert m2.image_urls == ("https://img.example.test/cv_spot.jpg",)
            finally:
                await server_push_q.put(None)
                stop_ev.set()
                await asyncio.wait_for(task, timeout=5.0)
                await bus.close()

    asyncio.run(_run())


def test_ws_unresolved_forward_get_forward_msg_rpc_and_timeout_fallback() -> None:
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        ws_ready = asyncio.Event()
        server_push_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            ws_ready.set()

            async def _pusher() -> None:
                while True:
                    ev = await server_push_q.get()
                    if ev is None:
                        return
                    await ws.send(json.dumps(ev, ensure_ascii=False))

            pusher = asyncio.create_task(_pusher())
            try:
                async for raw in ws:
                    frame = json.loads(str(raw))
                    action = frame.get("action")
                    params = frame.get("params") or {}
                    if (
                        action == "get_forward_msg"
                        and params.get("message_id") == "fwd-live-88"
                    ):
                        await ws.send(
                            json.dumps(
                                {
                                    "status": "ok",
                                    "retcode": 0,
                                    "data": {
                                        "messages": [
                                            {
                                                "sender": {
                                                    "card": "群友X",
                                                    "user_id": 10011,
                                                },
                                                "message": [
                                                    {
                                                        "type": "text",
                                                        "data": {"text": "合并转发第一条"},
                                                    }
                                                ],
                                            },
                                            {
                                                "sender": {
                                                    "nickname": "群友Y",
                                                    "user_id": 10012,
                                                },
                                                "message": [
                                                    {
                                                        "type": "text",
                                                        "data": {"text": "合并转发带图"},
                                                    },
                                                    {
                                                        "type": "image",
                                                        "data": {
                                                            "url": "https://img.example.test/fwd_y.png"
                                                        },
                                                    },
                                                ],
                                            },
                                        ]
                                    },
                                    "echo": frame.get("echo", ""),
                                },
                                ensure_ascii=False,
                            )
                        )
                    # Intentionally do NOT respond to fwd-timeout or message_id=4999 to verify timeout safety
            finally:
                pusher.cancel()
                await asyncio.gather(pusher, return_exceptions=True)

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                bot_qq=99999,
                poll_interval_s=0.01,
                rpc_timeout_s=0.12,
            )
            stop_ev = asyncio.Event()
            task = asyncio.create_task(adapter.run_once_session(stop_event=stop_ev))
            try:
                await asyncio.wait_for(ws_ready.wait(), timeout=5.0)

                # 1) Unresolved forward hydrated via get_forward_msg
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 8001,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "at", "data": {"qq": "99999"}},
                            {"type": "forward", "data": {"id": "fwd-live-88"}},
                        ],
                        "sender": {"card": "搬运工", "user_id": 10001},
                    }
                )
                hydrated = await _poll_one_inbound(bus)
                assert "[合并转发聊天记录]" in hydrated.text
                assert "[群友X]: 合并转发第一条" in hydrated.text
                assert "[群友Y]: 合并转发带图 [图片]" in hydrated.text
                assert hydrated.image_urls == ("https://img.example.test/fwd_y.png",)

                # 2) RPC timeout degrades gracefully without dropping the inbound message or crashing session
                await server_push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 8002,
                        "self_id": 99999,
                        "user_id": 10001,
                        "group_id": 88888,
                        "message": [
                            {"type": "reply", "data": {"id": "4999"}},
                            {"type": "forward", "data": {"id": "fwd-timeout"}},
                            {"type": "text", "data": {"text": " 超时兜底测试"}},
                        ],
                        "sender": {"card": "搬运工", "user_id": 10001},
                    }
                )
                degraded = await _poll_one_inbound(bus)
                assert degraded.event_id == "8002"
                assert degraded.reply_to_id == 4999
                assert degraded.reply_to_user_id is None
                assert degraded.text == "[合并转发消息] 超时兜底测试"
            finally:
                await server_push_q.put(None)
                stop_ev.set()
                await asyncio.wait_for(task, timeout=5.0)
                await bus.close()

    asyncio.run(_run())
