"""Agent kernel orchestration and managed llmos `system-control-daemon` lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from .channel_bridge import MessageBus
from .commands import AdminCommandRouter
from .guardrails import GuardrailManager
from .persona import PersonaConfig, PersonaManager
from .provider import LLMProvider, ProviderError, fetch_vision_images
from .session import SessionStore
from .stickers import StickerRegistry
from .trigger import RuleTrigger
from .types import ChatMessage, InboundMessage, OutboundReply

logger = logging.getLogger(__name__)

_ENDPOINT_SEQ = 0
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_ipc_endpoint(label: str) -> str:
    """Generate a collision-free local Named Pipe (Windows) or UDS path (POSIX)."""
    global _ENDPOINT_SEQ
    _ENDPOINT_SEQ += 1
    unique = f"{os.getpid():x}-{time.time_ns():x}-{_ENDPOINT_SEQ:x}-{label}"
    if sys.platform == "win32":
        return rf"\\.\pipe\qq-agent-cc-{unique}"
    return str(Path(tempfile.gettempdir()) / f"qq-agent-cc-{unique}.sock")


def resolve_daemon_command(
    *,
    llmos_root: Path,
    root_dir: Path,
    auth_endpoint: str,
    plain_endpoint: str,
) -> tuple[list[str], str]:
    """Prefer prebuilt `system-control-daemon[.exe]` binary; fallback to `cargo run`."""
    exe_name = (
        "system-control-daemon.exe" if sys.platform == "win32" else "system-control-daemon"
    )
    daemon_args = [
        "--root",
        str(root_dir),
        "--auth-socket",
        auth_endpoint,
        "--plain-socket",
        plain_endpoint,
    ]
    candidates = (
        llmos_root / "target" / "debug" / exe_name,
        llmos_root / "target" / "release" / exe_name,
        _REPO_ROOT / "data" / "bin" / exe_name,
        Path(tempfile.gettempdir()) / "qq-agent-llmos-target" / "debug" / exe_name,
    )
    for prebuilt in candidates:
        if prebuilt.exists():
            return [str(prebuilt), *daemon_args], f"prebuilt ({prebuilt})"

    cargo_cmd = ["cargo", "run", "--quiet"]
    if (llmos_root / "Cargo.lock").exists():
        cargo_cmd.append("--locked")
    cargo_cmd.extend(
        [
            "-p",
            "nlos-system-control",
            "--features",
            "daemon",
            "--bin",
            "system-control-daemon",
            "--",
            *daemon_args,
        ]
    )
    return cargo_cmd, "cargo run fallback"


async def _drain_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
    except asyncio.CancelledError:
        return
    except Exception:
        return


class ManagedLlmosDaemon:
    """Async context manager that bootstraps and supervises llmos `system-control-daemon`."""

    def __init__(
        self,
        llmos_root: Path | str = r"D:\llmos",
        *,
        data_dir: Path | str = "data",
        root_dir: Path | str | None = None,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.llmos_root = Path(llmos_root)
        self.root_dir = (
            Path(root_dir) if root_dir is not None else Path(data_dir) / "llmos_daemon"
        )
        self._startup_timeout_s = max(1.0, float(startup_timeout_s))

        self.auth_endpoint: str = _make_ipc_endpoint("auth")
        self.plain_endpoint: str = _make_ipc_endpoint("plain")
        self.launch_mode: str = ""
        self.ready_banner: str = ""

        self._process: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []

    async def start(self) -> str:
        """Start `system-control-daemon`, wait for `READY `, and return `plain_endpoint`."""
        if self._process is not None and self._process.returncode is None:
            return self.plain_endpoint

        self.root_dir.mkdir(parents=True, exist_ok=True)
        cmd, mode_desc = resolve_daemon_command(
            llmos_root=self.llmos_root,
            root_dir=self.root_dir,
            auth_endpoint=self.auth_endpoint,
            plain_endpoint=self.plain_endpoint,
        )
        self.launch_mode = mode_desc

        sub_env = dict(os.environ)
        if cmd[0] == "cargo" and "CARGO_TARGET_DIR" not in sub_env:
            # Keep read-only llmos checkout clean when compiling via cargo fallback
            sub_env["CARGO_TARGET_DIR"] = str(
                Path(tempfile.gettempdir()) / "qq-agent-llmos-target"
            )

        logger.info(
            "Starting llmos system-control-daemon via %s (root=%s, plain=%s)",
            mode_desc,
            self.root_dir,
            self.plain_endpoint,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=self.llmos_root,
            env=sub_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = proc
        assert proc.stdout is not None

        try:
            ready_raw = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=self._startup_timeout_s,
            )
        except BaseException:
            await self.stop()
            raise

        ready_text = ready_raw.decode("utf-8", errors="replace").strip()
        if not ready_text.startswith("READY ") or "plain_socket=" not in ready_text:
            stderr_text = ""
            if proc.stderr is not None:
                try:
                    stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=3.0)
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                except TimeoutError:
                    stderr_text = "<stderr read timed out>"
            await self.stop()
            raise RuntimeError(
                f"system-control-daemon failed before READY: stdout={ready_text!r} stderr={stderr_text!r}"
            )

        self.ready_banner = ready_text
        self._drain_tasks = [
            asyncio.create_task(_drain_stream(proc.stdout)),
            asyncio.create_task(_drain_stream(proc.stderr)),
        ]
        return self.plain_endpoint

    async def stop(self) -> None:
        """Terminate the managed daemon subprocess and wait for full OS reaping."""
        proc = self._process
        self._process = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

        if self._drain_tasks:
            for task in self._drain_tasks:
                task.cancel()
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
            self._drain_tasks.clear()

        if sys.platform != "win32":
            for sock_path in (self.auth_endpoint, self.plain_endpoint):
                try:
                    Path(sock_path).unlink()
                except FileNotFoundError:
                    pass

    async def __aenter__(self) -> ManagedLlmosDaemon:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()


class AgentKernel:
    """Core message-processing pipeline assembling trigger, commands, guardrails, memory, and LLM."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        provider: LLMProvider,
        persona: PersonaManager,
        sessions: SessionStore,
        trigger: RuleTrigger,
        guardrails: GuardrailManager,
        stickers: StickerRegistry | None = None,
        enable_vision: bool = True,
        vision_transport: Any | None = None,
        commands: AdminCommandRouter | None = None,
        bot_qq: int | None = None,
        admin_user_ids: set[int] | None = None,
        error_fallback_reply: str = "刚卡了一下，再发一遍",
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.persona = persona
        self.sessions = sessions
        self.trigger = trigger
        self.guardrails = guardrails
        self.stickers = stickers
        self.enable_vision = bool(enable_vision)
        self._vision_transport = vision_transport
        self.error_fallback_reply = error_fallback_reply

        effective_bot_qq = bot_qq if bot_qq is not None else self.trigger.bot_qq
        self.bot_qq = effective_bot_qq
        if self.trigger.bot_qq is None and effective_bot_qq is not None:
            self.trigger.bot_qq = effective_bot_qq

        if commands is not None:
            self.commands = commands
        else:
            self.commands = AdminCommandRouter(
                admin_user_ids=set(admin_user_ids) if admin_user_ids is not None else set(),
                on_reset=self.sessions.clear,
                on_quota=self.guardrails.snapshot,
                on_reload=self._reload_and_sync_persona,
            )

        self._sync_persona(self.persona.current())

    def _sync_persona(self, cfg: PersonaConfig) -> None:
        self.trigger.update_keywords((cfg.name, *cfg.aliases))
        self.guardrails.rate_refusal = cfg.cute_rate_refusal
        self.guardrails.budget_refusal = cfg.cute_budget_refusal

    def _reload_and_sync_persona(self) -> str:
        cfg = self.persona.reload()
        self._sync_persona(cfg)
        if self.stickers is not None:
            self.stickers.reload()
        return cfg.name

    @staticmethod
    def _build_outbound_reply(
        msg: InboundMessage,
        text: str,
        *,
        sticker_id: str | None = None,
    ) -> OutboundReply | None:
        if msg.chat_type == "group":
            if msg.group_id is None:
                return None
            reply_to_id = int(msg.event_id) if msg.event_id.isdigit() else None
            return OutboundReply(
                chat_type="group",
                target_id=msg.group_id,
                text=text,
                reply_to_message_id=reply_to_id,
                at_user_id=msg.user_id,
                correlation_id=msg.event_id,
                sticker_id=sticker_id,
            )
        return OutboundReply(
            chat_type="private",
            target_id=msg.user_id,
            text=text,
            reply_to_message_id=None,
            at_user_id=None,
            correlation_id=msg.event_id,
            sticker_id=sticker_id,
        )

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply | None:
        """Process a single inbound message through the strict 7-stage defense & reply pipeline."""
        # 1) Anti-self-loop guard
        effective_bot_qq = self.bot_qq if self.bot_qq is not None else self.trigger.bot_qq
        if (effective_bot_qq is not None and msg.user_id == effective_bot_qq) or (
            msg.self_id is not None and msg.user_id == msg.self_id
        ):
            return None

        # 2) Hot-sync persona keywords and cute refusal messages
        p_cfg = self.persona.current()
        self._sync_persona(p_cfg)

        # 3) Group whitelist pre-check (never respond to messages or commands in non-whitelisted groups)
        if (
            msg.chat_type == "group"
            and self.trigger.allowed_group_ids
            and msg.group_id not in self.trigger.allowed_group_ids
        ):
            return None

        # 4) Whitelisted admin command routing (bypasses LLM context & token accounting)
        cmd_res = self.commands.try_handle(msg)
        if cmd_res.handled:
            if cmd_res.reply_text:
                return self._build_outbound_reply(msg, cmd_res.reply_text)
            return None

        # 5) Tier-1 zero-cost rule trigger evaluation
        trig = self.trigger.evaluate(msg)
        if not trig.should_reply:
            return None

        # 6) Rate-limiting and daily token/micro-USD budget guardrails
        guard = self.guardrails.check(msg)
        if not guard.allowed:
            if guard.refusal_reply:
                return self._build_outbound_reply(msg, guard.refusal_reply)
            return None

        # 7) Assemble anti-injection prompt window (+ sticker guide + multimodal vision) and invoke LLM
        sys_prompt = self.persona.render_system_prompt()
        if self.stickers is not None:
            sticker_guide = self.stickers.render_prompt_guide()
            if sticker_guide:
                sys_prompt = f"{sys_prompt}\n\n{sticker_guide}"

        sys_msg = ChatMessage(role="system", content=sys_prompt)
        history = self.sessions.get_window(msg.session_key)
        wrapped_user = self.persona.wrap_user_message(msg, clean_text=trig.clean_text)

        vision_images: tuple[tuple[str, str], ...] = ()
        if self.enable_vision and msg.image_urls:
            vision_images = await fetch_vision_images(
                msg.image_urls,
                transport=self._vision_transport,
            )

        try:
            res = await self.provider.chat(
                [
                    sys_msg,
                    *history,
                    ChatMessage(role="user", content=wrapped_user, images=vision_images),
                ],
                model=guard.override_model,
            )
        except ProviderError as exc:
            logger.error(
                "LLM provider error on session=%s event_id=%s: %s",
                msg.session_key,
                msg.event_id,
                exc,
            )
            return self._build_outbound_reply(msg, self.error_fallback_reply)

        self.guardrails.record_usage(
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            total_tokens=res.total_tokens,
            cost_micros_usd=res.cost_micros_usd,
        )

        sticker_id: str | None = None
        if self.stickers is not None:
            clean_reply, sticker_id = self.stickers.extract_from_reply(res.content)
        else:
            clean_reply = res.content.strip()

        if not clean_reply and sticker_id is None:
            clean_reply = res.content.strip() or self.error_fallback_reply

        history_assistant = (
            f"{clean_reply} [[sticker:{sticker_id}]]".strip()
            if sticker_id is not None
            else clean_reply
        )
        self.sessions.append(msg.session_key, "user", wrapped_user)
        self.sessions.append(msg.session_key, "assistant", history_assistant)
        return self._build_outbound_reply(msg, clean_reply, sticker_id=sticker_id)

    async def process_once(self, limit: int = 16) -> int:
        """Poll up to `limit` inbound messages, handle & publish replies, and monotonically ACK."""
        batch = await self.bus.poll_inbound(limit=limit)
        if not batch:
            return 0

        processed = 0
        for seq, msg in batch:
            reply = await self.handle_inbound(msg)
            if reply is not None:
                await self.bus.publish_outbound(reply)
            await self.bus.ack_inbound(seq)
            processed += 1
        return processed

    async def run_forever(
        self,
        stop_event: asyncio.Event | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        """Continuously poll and process inbound messages until `stop_event` is set."""
        effective_stop = stop_event if stop_event is not None else asyncio.Event()
        interval = max(0.001, float(poll_interval_s))

        while not effective_stop.is_set():
            processed = await self.process_once()
            if processed == 0:
                try:
                    await asyncio.wait_for(effective_stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    pass


__all__ = [
    "AgentKernel",
    "ManagedLlmosDaemon",
    "resolve_daemon_command",
]
