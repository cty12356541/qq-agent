"""Local GPU TTS HTTP server (`127.0.0.1:9880`) for `NaiLong-Voice-Clone` reference audio synthesis."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REF_WAV = _REPO_ROOT / "data" / "tts" / "nailong" / "reference.wav"
DEFAULT_REF_TEXT = "啊，我才不要这样，好害羞啊。"


class LocalNaiLongSynthesizer:
    """Thread-safe GPU voice cloning synthesizer backed by F5-TTS + `data/tts/nailong/reference.wav`."""

    def __init__(
        self,
        *,
        ref_wav: Path = DEFAULT_REF_WAV,
        ref_text: str = DEFAULT_REF_TEXT,
        nfe_step: int = 16,
    ) -> None:
        self.ref_wav = ref_wav.resolve()
        self.ref_text = ref_text
        self.nfe_step = max(8, int(nfe_step))
        self.gpt_weights: str = str((_REPO_ROOT / "data" / "tts" / "nailong" / "NaiLong-e30.ckpt").resolve())
        self.sovits_weights: str = str((_REPO_ROOT / "data" / "tts" / "nailong" / "NaiLong_e20_s720.pth").resolve())
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._device: str = "cpu"
        self.ready: bool = False
        self.last_error: str = ""

    def warmup(self) -> None:
        """Load F5-TTS on CUDA (if available) and run a 1-word warmup pass."""
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch  # type: ignore[import-not-found]
                from f5_tts.api import F5TTS  # type: ignore[import-not-found]

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Loading NaiLong GPU zero-shot TTS engine on device=%s ...", self._device)
                self._model = F5TTS(device=self._device)
                if self.ref_wav.is_file():
                    self._model.infer(
                        ref_file=str(self.ref_wav),
                        ref_text=self.ref_text,
                        gen_text="好呀",
                        nfe_step=8,
                    )
                self.ready = True
                logger.info("NaiLong GPU TTS engine ready (ref=%s, device=%s)", self.ref_wav, self._device)
            except Exception as exc:
                self.last_error = str(exc)
                logger.exception("Failed to initialize F5-TTS GPU model: %s", exc)

    def synthesize_wav(self, text: str, *, ref_audio_path: str | None = None, prompt_text: str | None = None) -> bytes:
        """Synthesize `text` into 16-bit PCM WAV bytes using NaiLong reference voice."""
        cleaned = " ".join(text.split()).strip()[:120]
        if not cleaned:
            raise ValueError("Empty TTS text")

        if self._model is None:
            self.warmup()
        if self._model is None:
            raise RuntimeError(f"TTS model unavailable: {self.last_error}")

        target_ref = Path(ref_audio_path) if ref_audio_path else self.ref_wav
        if not target_ref.is_file():
            target_ref = self.ref_wav

        # Always pair reference.wav with its exact ground-truth transcript to avoid alignment drift
        effective_ref_text = (
            self.ref_text
            if target_ref.name.lower() == "reference.wav" or not prompt_text
            else prompt_text
        )

        import soundfile as sf  # type: ignore[import-not-found]

        with self._lock:
            t0 = time.perf_counter()
            wav, sr, _ = self._model.infer(
                ref_file=str(target_ref),
                ref_text=effective_ref_text,
                gen_text=cleaned,
                nfe_step=self.nfe_step,
            )
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()
            logger.info(
                "Synthesized NaiLong voice (%d chars -> %d bytes WAV in %.2fs): %s",
                len(cleaned),
                len(wav_bytes),
                time.perf_counter() - t0,
                cleaned,
            )
            return wav_bytes


def build_tts_handler(synth: LocalNaiLongSynthesizer) -> type[BaseHTTPRequestHandler]:
    """Create HTTP handler compatible with GPT-SoVITS `/tts`, `/set_gpt_weights`, `/set_sovits_weights`."""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, data: Any, status: int = 200) -> None:
            raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)

            if path in {"/", "/health"}:
                self._send_json(
                    {
                        "ok": True,
                        "ready": synth.ready,
                        "device": synth._device,
                        "ref_wav": str(synth.ref_wav),
                        "gpt_weights": synth.gpt_weights,
                        "sovits_weights": synth.sovits_weights,
                    }
                )
                return

            if path == "/set_gpt_weights":
                synth.gpt_weights = (qs.get("weights_path") or [synth.gpt_weights])[0]
                self._send_json({"ok": True, "gpt_weights": synth.gpt_weights})
                return

            if path == "/set_sovits_weights":
                synth.sovits_weights = (qs.get("weights_path") or [synth.sovits_weights])[0]
                self._send_json({"ok": True, "sovits_weights": synth.sovits_weights})
                return

            if path == "/tts":
                text = (qs.get("text") or [""])[0].strip()
                ref_audio = (qs.get("ref_audio_path") or [""])[0].strip() or None
                prompt_text = (qs.get("prompt_text") or [""])[0].strip() or None
                if not text:
                    self._send_json({"error": "missing text"}, status=400)
                    return
                try:
                    wav_bytes = synth.synthesize_wav(
                        text,
                        ref_audio_path=ref_audio,
                        prompt_text=prompt_text,
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(wav_bytes)))
                    self.end_headers()
                    self.wfile.write(wav_bytes)
                except Exception as exc:
                    logger.warning("TTS synthesis error: %s", exc)
                    self._send_json({"error": str(exc)}, status=500)
                return

            self._send_json({"error": f"unknown endpoint {path}"}, status=404)

    return _Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Local GPU NaiLong TTS HTTP Server (port 9880)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--nfe-step", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    synth = LocalNaiLongSynthesizer(nfe_step=args.nfe_step)
    synth.warmup()
    server = ThreadingHTTPServer((args.host, args.port), build_tts_handler(synth))
    logger.info("Local NaiLong TTS HTTP server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
