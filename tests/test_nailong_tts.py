"""Unit tests for NaiLong-Voice-Clone (GPT-SoVITS) TTS engine, NapCat AI voice fallback, and OB11 voice frames."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from qq_agent.channel_bridge import DirectMemoryBus
from qq_agent.kernel_turn_helpers import _has_voice_intent
from qq_agent.ob11 import build_send_actions, build_voice_action
from qq_agent.tools.chat_tools import ChatTurnState
from qq_agent.tts import (
    DEFAULT_NAILONG_MODEL_DIR,
    DEFAULT_NAPCAT_VOICE_CHARACTER,
    NaiLongTtsEngine,
    build_send_voice_tool,
    resolve_nailong_assets,
)
from qq_agent.types import InboundMessage, OutboundReply


def test_resolve_nailong_assets_with_tmp_dir(tmp_path: Path) -> None:
    ref = tmp_path / "reference.wav"
    ckpt = tmp_path / "NaiLong-e30.ckpt"
    pth = tmp_path / "NaiLong_e20_s720.pth"
    ref.write_bytes(b"RIFF" + b"\x00" * 100)
    ckpt.write_bytes(b"ckpt")
    pth.write_bytes(b"pth")

    assets = resolve_nailong_assets(tmp_path)
    assert assets.is_ready is True
    assert assets.reference_wav == ref
    assert assets.gpt_weights == ckpt
    assert assets.sovits_weights == pth

    # If local data/tts/nailong was downloaded, verify it is also ready
    if DEFAULT_NAILONG_MODEL_DIR.is_dir():
        live_assets = resolve_nailong_assets(DEFAULT_NAILONG_MODEL_DIR)
        assert live_assets.is_ready is True


def test_nailong_tts_engine_local_gpt_sovits_and_fallback(tmp_path: Path) -> None:
    async def _run() -> None:
        ref = tmp_path / "reference.wav"
        ckpt = tmp_path / "NaiLong-e30.ckpt"
        pth = tmp_path / "NaiLong_e20_s720.pth"
        ref.write_bytes(b"RIFF" + b"\x00" * 100)
        ckpt.write_bytes(b"ckpt")
        pth.write_bytes(b"pth")

        called_paths: list[str] = []
        fake_wav = b"RIFF" + b"\x11" * 128

        def _mock_handler(request: httpx.Request) -> httpx.Response:
            called_paths.append(request.url.path)
            if request.url.path in ("/set_gpt_weights", "/set_sovits_weights"):
                return httpx.Response(200, json={"status": "ok"})
            if request.url.path == "/tts":
                return httpx.Response(200, content=fake_wav, headers={"content-type": "audio/wav"})
            return httpx.Response(404)

        engine = NaiLongTtsEngine(
            model_dir=tmp_path,
            transport=httpx.MockTransport(_mock_handler),
        )
        res = await engine.synthesize("鹅鹅鹅，曲项向天歌", chat_type="group")
        assert res.backend_used == "local_gpt_sovits_nailong"
        assert res.voice_uri is not None and res.voice_uri.startswith("base64://")
        assert res.ai_voice_character is None
        assert called_paths == ["/set_gpt_weights", "/set_sovits_weights", "/tts"]

        def _fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        fallback_engine = NaiLongTtsEngine(
            model_dir=tmp_path,
            transport=httpx.MockTransport(_fail_handler),
        )
        fb_res = await fallback_engine.synthesize("干嘛呀", chat_type="group")
        assert fb_res.backend_used == "napcat_group_ai_voice"
        assert fb_res.voice_uri is None
        assert fb_res.ai_voice_character == DEFAULT_NAPCAT_VOICE_CHARACTER

    asyncio.run(_run())


def test_send_voice_tool_and_ob11_voice_frames(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = DirectMemoryBus()
        turn_state = ChatTurnState()

        def _fail_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        engine = NaiLongTtsEngine(
            model_dir=tmp_path,
            transport=httpx.MockTransport(_fail_handler),
        )
        msg = InboundMessage(
            event_id="901",
            chat_type="group",
            user_id=123456,
            group_id=590176297,
            sender_name="群友",
            text="奶娃发条语音听听",
            raw_message="奶娃发条语音听听",
            mentioned_bot=True,
        )
        tool = build_send_voice_tool(
            bus,
            turn_state,
            tts_engine=engine,
            msg=msg,
            allowed_sticker_ids=["naiwa_stare"],
        )
        out_str = await tool.execute({"text": "本奶娃想听你背诗！", "sticker_id": "naiwa_stare"})
        assert "Sent voice" in out_str
        assert turn_state.sent_bubbles == ["[语音：我想听你背诗]"]
        assert turn_state.last_sticker_id == "naiwa_stare"

        batch = await bus.poll_outbound()
        assert len(batch) == 1
        _, reply = batch[0]
        assert reply.ai_voice_character == DEFAULT_NAPCAT_VOICE_CHARACTER
        assert reply.sticker_id == "naiwa_stare"

        restored = OutboundReply.from_json_bytes(reply.to_json_bytes())
        assert restored.ai_voice_character == DEFAULT_NAPCAT_VOICE_CHARACTER
        actions = build_send_actions(restored, sticker_file="base64://AAA", echo="e1")
        assert len(actions) == 2
        assert actions[0]["action"] == "send_group_ai_record"
        assert actions[0]["params"]["character"] == DEFAULT_NAPCAT_VOICE_CHARACTER
        assert actions[0]["params"]["text"] == "我想听你背诗"
        assert actions[1]["action"] == "send_group_msg"
        assert actions[1]["params"]["message"][0]["type"] == "image"

        wav_reply = OutboundReply(
            chat_type="group",
            target_id=590176297,
            text="床前明月光",
            voice_uri="base64://UklGRg==",
        )
        voice_act = build_voice_action(wav_reply, echo="v1")
        assert voice_act is not None
        assert voice_act["action"] == "send_group_msg"
        assert voice_act["params"]["message"] == [
            {"type": "record", "data": {"file": "base64://UklGRg=="}}
        ]

    asyncio.run(_run())


def test_voice_intent_detection() -> None:
    assert _has_voice_intent("奶娃发个语音听听") is True
    assert _has_voice_intent("说句话来听听") is True
    assert _has_voice_intent("今天吃什么") is False
