"""Unit & E2E tests for 奶龙/奶蛙 sticker registry, OB11 sticker frames, and multimodal vision."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
import websockets

from qq_agent import (
    AgentKernel,
    DirectMemoryBus,
    GuardrailManager,
    NapCatWsAdapter,
    OpenAICompatProvider,
    OutboundReply,
    PersonaManager,
    RuleTrigger,
    SessionStore,
    StickerRegistry,
)
from qq_agent.ob11 import build_send_actions, event_to_inbound
from qq_agent.provider import normalize_image_bytes_for_vision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSONA_PATH = REPO_ROOT / "config" / "persona.default.toml"
DEFAULT_STICKERS_MANIFEST = REPO_ROOT / "config" / "stickers.toml"


def _make_tiny_png(color: tuple[int, int, int] = (255, 220, 40)) -> bytes:
    im = Image.new("RGB", (32, 32), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def test_sticker_registry_manifest_fuzzy_resolve_and_path_traversal_guard(tmp_path: Path) -> None:
    sticker_dir = tmp_path / "stickers"
    sticker_dir.mkdir()
    (sticker_dir / "naiwa_point_laugh.png").write_bytes(_make_tiny_png())
    (sticker_dir / "nailong_stop.gif").write_bytes(_make_tiny_png((250, 200, 10)))
    (sticker_dir / "custom_奶蛙偷笑.png").write_bytes(_make_tiny_png((100, 220, 80)))
    secret_outside = tmp_path / "outside.png"
    secret_outside.write_bytes(_make_tiny_png((0, 0, 0)))

    manifest = tmp_path / "stickers.toml"
    manifest.write_text(
        """
[[stickers]]
id = "naiwa_point_laugh"
series = "奶蛙"
file = "naiwa_point_laugh.png"
label = "捂肚指人嘲笑"
tags = ["嘲笑", "小丑"]

[[stickers]]
id = "nailong_stop"
series = "奶龙"
file = "nailong_stop.gif"
label = "伸手打住拒绝"
tags = ["打住", "达咩"]

[[stickers]]
id = "evil_traversal"
series = "奶蛙"
file = "../outside.png"
label = "越权路径"
""",
        encoding="utf-8",
    )

    reg = StickerRegistry(sticker_dir, manifest_path=manifest)
    ids = {e.id for e in reg.available_entries()}
    assert "naiwa_point_laugh" in ids
    assert "nailong_stop" in ids
    assert "custom_奶蛙偷笑" in ids
    assert "evil_traversal" not in ids
    assert reg.read_base64_uri("evil_traversal") is None

    # Exact ID, tag match, series fallback, and pure-sticker extraction
    clean1, sid1 = reg.extract_from_reply("你这波操作真难绷 [[sticker:naiwa_point_laugh]]")
    assert clean1 == "你这波操作真难绷"
    assert sid1 == "naiwa_point_laugh"

    clean2, sid2 = reg.extract_from_reply("[[sticker:达咩]]")
    assert clean2 == ""
    assert sid2 == "nailong_stop"

    clean3, sid3 = reg.extract_from_reply("给你看个宝贝 [表情包:奶龙]")
    assert clean3 == "给你看个宝贝"
    assert sid3 == "nailong_stop"

    uri = reg.read_base64_uri("naiwa_point_laugh")
    assert uri is not None and uri.startswith("base64://")
    assert "【表情包（奶龙/奶蛙系列）使用指南】" in reg.render_prompt_guide()


def test_ob11_inbound_sticker_image_and_outbound_two_frame_split() -> None:
    inbound = event_to_inbound(
        {
            "post_type": "message",
            "message_type": "private",
            "message_id": 9101,
            "self_id": 999001,
            "user_id": 100001,
            "message": [
                {
                    "type": "image",
                    "data": {
                        "file": "abc.gif",
                        "sub_type": 1,
                        "summary": "[动画表情]",
                        "url": "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=xyz",
                    },
                }
            ],
            "sender": {"nickname": "测试好友"},
        }
    )
    assert inbound is not None
    assert inbound.text == "[动画表情]"
    assert inbound.image_urls == (
        "https://multimedia.nt.qq.com.cn/download?appid=1406&fileid=xyz",
    )

    # Pure sticker reply -> 1 borderless `[动画表情]` frame (sub_type=1), NO empty text bubble
    pure_sticker = OutboundReply(
        chat_type="private",
        target_id=100001,
        text="",
        sticker_id="naiwa_stare",
        correlation_id="c-1",
    )
    pure_frames = build_send_actions(
        pure_sticker,
        sticker_file="base64://AAAA",
        echo="c-1",
    )
    assert len(pure_frames) == 1
    assert pure_frames[0]["params"]["message"] == [
        {
            "type": "image",
            "data": {"file": "base64://AAAA", "sub_type": 1, "summary": "[动画表情]"},
        }
    ]

    # Text + sticker reply -> 2 frames (text bubble first, standalone sticker second)
    combo = OutboundReply(
        chat_type="group",
        target_id=700001,
        text="这谁顶得住",
        reply_to_message_id=9101,
        at_user_id=100001,
        sticker_id="nailong_desk_laugh",
        correlation_id="c-2",
    )
    combo_frames = build_send_actions(
        combo,
        sticker_file="base64://BBBB",
        echo="c-2",
    )
    assert len(combo_frames) == 2
    assert combo_frames[0]["params"]["message"][-1] == {
        "type": "text",
        "data": {"text": " 这谁顶得住"},
    }
    assert combo_frames[1]["params"]["message"] == [
        {
            "type": "image",
            "data": {"file": "base64://BBBB", "sub_type": 1, "summary": "[动画表情]"},
        }
    ]


def test_normalize_animated_gif_for_vision() -> None:
    f1 = Image.new("RGB", (900, 600), (255, 0, 0))
    f2 = Image.new("RGB", (900, 600), (0, 255, 0))
    buf = io.BytesIO()
    f1.save(buf, format="GIF", save_all=True, append_images=[f2])
    norm = normalize_image_bytes_for_vision(buf.getvalue(), max_side=512)
    assert norm is not None
    media_type, b64 = norm
    assert media_type == "image/jpeg"
    assert len(b64) > 20


def test_e2e_inbound_image_vision_and_outbound_sticker_ws_pipeline(tmp_path: Path) -> None:
    sticker_dir = tmp_path / "stickers"
    sticker_dir.mkdir()
    (sticker_dir / "naiwa_stare.png").write_bytes(_make_tiny_png())

    img_bytes = _make_tiny_png((20, 180, 90))
    seen_llm_payloads: list[dict[str, Any]] = []

    def mock_img_cdn(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=img_bytes, headers={"Content-Type": "image/png"})

    def mock_llm(request: httpx.Request) -> httpx.Response:
        seen_llm_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_e2e_sticker",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [
                    {"type": "text", "text": "你发这图啥意思 [[sticker:naiwa_stare]]"}
                ],
                "usage": {"input_tokens": 180, "output_tokens": 25},
            },
        )

    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        persona = PersonaManager(DEFAULT_PERSONA_PATH)
        assert persona.current().name == "UD"
        stickers = StickerRegistry(sticker_dir, manifest_path=DEFAULT_STICKERS_MANIFEST)
        sessions = SessionStore(tmp_path / "sessions.sqlite3")
        guardrails = GuardrailManager(tmp_path / "guardrails.sqlite3", user_cooldown_s=0.0)
        trigger = RuleTrigger(bot_qq=999001)
        provider = OpenAICompatProvider(
            api_key="pool-key",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3-flash",
            enable_thinking=False,
            transport=httpx.MockTransport(mock_llm),
        )
        kernel = AgentKernel(
            bus=bus,
            provider=provider,
            persona=persona,
            sessions=sessions,
            trigger=trigger,
            guardrails=guardrails,
            stickers=stickers,
            enable_vision=True,
            vision_transport=httpx.MockTransport(mock_img_cdn),
            bot_qq=999001,
        )

        ws_ready = asyncio.Event()
        push_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        recv_q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            ws_ready.set()

            async def _pusher() -> None:
                while True:
                    item = await push_q.get()
                    if item is None:
                        return
                    await ws.send(json.dumps(item, ensure_ascii=False))

            pt = asyncio.create_task(_pusher())
            try:
                async for raw in ws:
                    await recv_q.put(json.loads(str(raw)))
            finally:
                pt.cancel()

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as srv:
            port = srv.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                bot_qq=999001,
                stickers=stickers,
                sticker_delay_s=0.01,
                poll_interval_s=0.01,
            )
            stop_ev = asyncio.Event()
            t_ad = asyncio.create_task(adapter.run_forever(stop_event=stop_ev))
            t_kn = asyncio.create_task(kernel.run_forever(stop_event=stop_ev, poll_interval_s=0.01))
            try:
                await asyncio.wait_for(ws_ready.wait(), timeout=5.0)
                await push_q.put(
                    {
                        "post_type": "message",
                        "message_type": "private",
                        "message_id": 5001,
                        "self_id": 999001,
                        "user_id": 100001,
                        "message": [
                            {
                                "type": "image",
                                "data": {
                                    "sub_type": 1,
                                    "summary": "[动画表情]",
                                    "url": "https://img.example.test/meme.png",
                                },
                            }
                        ],
                        "sender": {"nickname": "测试好友"},
                    }
                )
                frame1 = await asyncio.wait_for(recv_q.get(), timeout=5.0)
                frame2 = await asyncio.wait_for(recv_q.get(), timeout=5.0)

                assert frame1["params"]["message"] == [
                    {"type": "text", "data": {"text": "你发这图啥意思"}}
                ]
                assert frame2["params"]["message"][0]["type"] == "image"
                assert frame2["params"]["message"][0]["data"]["sub_type"] == 1
                assert frame2["params"]["message"][0]["data"]["summary"] == "[动画表情]"
                assert frame2["params"]["message"][0]["data"]["file"].startswith("base64://")

                # Verify multimodal vision block + thinking=disabled reached LLM
                assert seen_llm_payloads[0]["thinking"] == {"type": "disabled"}
                user_content = seen_llm_payloads[0]["messages"][-1]["content"]
                assert isinstance(user_content, list)
                assert user_content[0]["type"] == "image"
            finally:
                await push_q.put(None)
                stop_ev.set()
                await asyncio.gather(t_ad, t_kn, return_exceptions=True)
                await provider.aclose()
                sessions.close()
                guardrails.close()

    asyncio.run(_run())
