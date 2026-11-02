"""Local `NaiLong-Voice-Clone` (GPT-SoVITS v2proplus) TTS engine, NapCat AI voice fallback, and `send_voice` tool."""

from __future__ import annotations

import base64
from collections.abc import Collection
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

import httpx

from qq_agent.anti_injection import sanitize_outbound_bubble
from qq_agent.channel_bridge import MessageBus
from qq_agent.stickers import StickerRegistry
from qq_agent.tools import ToolArgumentError, ToolBinding
from qq_agent.tools.chat_tools import (
    ChatTurnState,
    _build_allowed_sticker_map,
    _resolve_target_route,
    _validate_sticker_id,
    extract_sticker_tag,
)
from qq_agent.types import ChatType, InboundMessage, OutboundReply, ToolDef

logger = logging.getLogger(__name__)

DEFAULT_NAILONG_MODEL_DIR = Path("data/tts/nailong")
DEFAULT_NAILONG_PROMPT_TEXT = "啊，我才不要这样，好害羞啊。"
DEFAULT_GPT_SOVITS_URL = "http://127.0.0.1:9880"
DEFAULT_NAPCAT_VOICE_CHARACTER = "lucy-voice-laibixiaoxin"
MAX_VOICE_TEXT_CHARS = 120
_LOCAL_TTS_PYTHON_CANDIDATES: tuple[Path, ...] = (
    Path(r"D:\llmos\media\.venv-tts\Scripts\python.exe"),
    Path(r"D:\AI\ComfyUI\python_embeded\python.exe"),
)
_LOCAL_TTS_SPAWN_ATTEMPTED = False


def ensure_local_tts_server_running(base_url: str = DEFAULT_GPT_SOVITS_URL) -> bool:
    """Auto-start `qq_agent.local_tts_server` on `127.0.0.1:9880` if local GPU TTS python env exists."""
    global _LOCAL_TTS_SPAWN_ATTEMPTED
    if _LOCAL_TTS_SPAWN_ATTEMPTED or os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if base_url.rstrip("/") != DEFAULT_GPT_SOVITS_URL:
        return False
    _LOCAL_TTS_SPAWN_ATTEMPTED = True
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve().parent / "local_tts_server.py"
    for py_bin in _LOCAL_TTS_PYTHON_CANDIDATES:
        if py_bin.is_file() and script_path.is_file():
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                subprocess.Popen(
                    [str(py_bin), "-P", str(script_path), "--port", "9880"],
                    cwd=str(repo_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                logger.info("Spawned local GPU NaiLong TTS server via %s", py_bin)
                return True
            except Exception as exc:
                logger.debug("Failed to spawn local TTS server via %s: %s", py_bin, exc)
    return False


@dataclass(slots=True, frozen=True)
class TtsSynthesisResult:
    """Result of synthesizing a short voice line via local GPT-SoVITS or NapCat AI voice fallback."""

    voice_uri: str | None = None
    ai_voice_character: str | None = None
    backend_used: str = "none"


@dataclass(slots=True, frozen=True)
class NaiLongModelAssets:
    """Resolved on-disk paths for `pengyichen/NaiLong-Voice-Clone` GPT-SoVITS weights and reference audio."""

    model_dir: Path
    reference_wav: Path
    gpt_weights: Path
    sovits_weights: Path

    @property
    def is_ready(self) -> bool:
        return (
            self.reference_wav.is_file()
            and self.gpt_weights.is_file()
            and self.sovits_weights.is_file()
        )


def resolve_nailong_assets(model_dir: Path | str = DEFAULT_NAILONG_MODEL_DIR) -> NaiLongModelAssets:
    """Locate `reference.wav`, `NaiLong-e30.ckpt`, and `NaiLong_e20_s720.pth` under `model_dir`."""
    root = Path(model_dir)
    ref_wav = root / "reference.wav"
    gpt_ckpt = root / "NaiLong-e30.ckpt"
    sovits_pth = root / "NaiLong_e20_s720.pth"

    if not gpt_ckpt.is_file() and root.is_dir():
        ckpts = sorted(root.glob("*.ckpt"))
        if ckpts:
            gpt_ckpt = ckpts[-1]
    if not sovits_pth.is_file() and root.is_dir():
        pths = sorted(root.glob("*.pth"))
        if pths:
            sovits_pth = pths[-1]

    return NaiLongModelAssets(
        model_dir=root,
        reference_wav=ref_wav,
        gpt_weights=gpt_ckpt,
        sovits_weights=sovits_pth,
    )


class NaiLongTtsEngine:
    """Local `NaiLong-Voice-Clone` (GPT-SoVITS) TTS client with zero-downtime NapCat AI voice fallback."""

    def __init__(
        self,
        *,
        model_dir: Path | str = DEFAULT_NAILONG_MODEL_DIR,
        gpt_sovits_url: str = DEFAULT_GPT_SOVITS_URL,
        prompt_text: str = DEFAULT_NAILONG_PROMPT_TEXT,
        prompt_lang: str = "zh",
        fallback_character: str = DEFAULT_NAPCAT_VOICE_CHARACTER,
        timeout_s: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._assets = resolve_nailong_assets(model_dir)
        self._base_url = gpt_sovits_url.rstrip("/")
        self._prompt_text = prompt_text
        self._prompt_lang = prompt_lang
        self._fallback_character = fallback_character.strip() or DEFAULT_NAPCAT_VOICE_CHARACTER
        self._timeout_s = max(0.5, float(timeout_s))
        self._transport = transport
        self._weights_initialized = False

    @property
    def assets(self) -> NaiLongModelAssets:
        return self._assets

    @property
    def fallback_character(self) -> str:
        return self._fallback_character

    async def _ensure_weights_bound(self, client: httpx.AsyncClient) -> None:
        if self._weights_initialized or not self._assets.is_ready:
            return
        await client.get(
            f"{self._base_url}/set_gpt_weights",
            params={"weights_path": str(self._assets.gpt_weights.resolve())},
        )
        await client.get(
            f"{self._base_url}/set_sovits_weights",
            params={"weights_path": str(self._assets.sovits_weights.resolve())},
        )
        self._weights_initialized = True

    async def synthesize(
        self,
        text: str,
        *,
        chat_type: ChatType = "group",
    ) -> TtsSynthesisResult:
        """Synthesize `text` via local GPT-SoVITS (`NaiLong-Voice-Clone`), falling back to NapCat AI voice."""
        cleaned = " ".join(text.split()).strip()[:MAX_VOICE_TEXT_CHARS]
        if not cleaned:
            return TtsSynthesisResult()

        if self._assets.reference_wav.is_file() or self._transport is not None:
            ref_path = (
                str(self._assets.reference_wav.resolve())
                if self._assets.reference_wav.is_file()
                else str(self._assets.reference_wav)
            )
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._timeout_s),
                    transport=self._transport,
                ) as client:
                    await self._ensure_weights_bound(client)
                    resp = await client.get(
                        f"{self._base_url}/tts",
                        params={
                            "text": cleaned,
                            "text_lang": "zh",
                            "ref_audio_path": ref_path,
                            "prompt_text": self._prompt_text,
                            "prompt_lang": self._prompt_lang,
                            "media_type": "wav",
                        },
                    )
                    if resp.status_code == 200 and len(resp.content) > 64:
                        b64_wav = base64.b64encode(resp.content).decode("ascii")
                        return TtsSynthesisResult(
                            voice_uri=f"base64://{b64_wav}",
                            backend_used="local_gpt_sovits_nailong",
                        )
            except Exception as exc:
                if self._transport is None:
                    ensure_local_tts_server_running(self._base_url)
                logger.debug(
                    "Local GPT-SoVITS NaiLong endpoint (%s) unavailable (%s); falling back to NapCat AI voice",
                    self._base_url,
                    exc,
                )

        if chat_type == "group" and self._fallback_character:
            return TtsSynthesisResult(
                ai_voice_character=self._fallback_character,
                backend_used="napcat_group_ai_voice",
            )

        return TtsSynthesisResult()


def build_send_voice_tool(
    bus: MessageBus,
    turn_state: ChatTurnState,
    tts_engine: NaiLongTtsEngine | None = None,
    *,
    msg: InboundMessage | None = None,
    chat_type: ChatType | None = None,
    target_id: int | None = None,
    correlation_id: str = "",
    allowed_sticker_ids: Collection[str] | None = None,
    stickers: StickerRegistry | None = None,
    compact_schema: bool = False,
) -> ToolBinding:
    """Build `send_voice` terminal tool for emitting a NaiLong/奶娃 voice bubble (`[CQ:record]`)."""
    route_chat_type, route_target_id, base_corr_id = _resolve_target_route(
        msg=msg,
        chat_type=chat_type,
        target_id=target_id,
        correlation_id=correlation_id,
    )
    engine = tts_engine if tts_engine is not None else NaiLongTtsEngine()

    tool_desc = (
        "向当前会话发送一条奶龙/唐诗奶娃搞怪语音条（仅当群友明确要求“发语音/说句话听听/语音念一下”时调用）。"
        if compact_schema
        else (
            "使用奶龙/唐诗奶娃特调音色向当前会话发送一条 QQ 语音消息（record）。"
            "仅在群友明确要求发语音、听声音或念台词时调用，日常普通文字聊天请继续使用 send_message。"
        )
    )

    definition = ToolDef(
        name="send_voice",
        description=tool_desc,
        input_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "maxLength": MAX_VOICE_TEXT_CHARS,
                    "description": "要转成语音说出的简短口语台词（2~60字，勿带标点符号堆砌）。",
                },
                "sticker_id": {
                    "type": "string",
                    "description": "可选，语音发送后顺带附上的一张合法表情包 ID。",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        is_chat_terminal=True,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_text = args.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ToolArgumentError("send_voice requires a non-empty string 'text'")

        stripped_text, inline_tag = extract_sticker_tag(raw_text)
        clean_text = sanitize_outbound_bubble(stripped_text.strip())[:MAX_VOICE_TEXT_CHARS]
        if not clean_text:
            raise ToolArgumentError("send_voice 'text' cannot be empty after sanitization")

        allowed_map = _build_allowed_sticker_map(allowed_sticker_ids, stickers)
        raw_sticker_arg = args.get("sticker_id")
        if isinstance(raw_sticker_arg, str) and raw_sticker_arg.strip():
            effective_sticker_id = _validate_sticker_id(raw_sticker_arg, allowed_map)
        elif inline_tag is not None:
            effective_sticker_id = _validate_sticker_id(inline_tag, allowed_map)
        else:
            effective_sticker_id = None

        synth = await engine.synthesize(clean_text, chat_type=route_chat_type)
        outbound = OutboundReply(
            chat_type=route_chat_type,
            target_id=route_target_id,
            text=clean_text,
            reply_to_message_id=None,
            at_user_id=None,
            correlation_id=base_corr_id,
            sticker_id=effective_sticker_id,
            voice_uri=synth.voice_uri,
            ai_voice_character=synth.ai_voice_character,
        )
        await bus.publish_outbound(outbound)
        turn_state.sent_bubbles.append(f"[语音：{clean_text}]")
        turn_state.stayed_silent = False
        turn_state.silence_reason = ""
        if effective_sticker_id is not None:
            turn_state.last_sticker_id = effective_sticker_id
        return f"Sent voice ({synth.backend_used}): {clean_text}"

    return ToolBinding(definition=definition, execute=_execute)


__all__ = [
    "DEFAULT_GPT_SOVITS_URL",
    "DEFAULT_NAILONG_MODEL_DIR",
    "DEFAULT_NAPCAT_VOICE_CHARACTER",
    "MAX_VOICE_TEXT_CHARS",
    "NaiLongModelAssets",
    "NaiLongTtsEngine",
    "TtsSynthesisResult",
    "build_send_voice_tool",
    "resolve_nailong_assets",
]
