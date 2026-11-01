"""Agent kernel orchestration, burst debouncing, ambient group context, memory wiring, and managed llmos daemon."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from .memory import MemoryStore
from .persona import PersonaConfig, PersonaManager
from .provider import LLMProvider, ProviderError, fetch_vision_images
from .session import SessionStore
from .stickers import StickerRegistry
from .trigger import RuleTrigger
from .types import ChatMessage, InboundMessage, OutboundReply

logger = logging.getLogger(__name__)

_ENDPOINT_SEQ = 0
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SYNTHETIC_PLACEHOLDERS: frozenset[str] = frozenset({"（戳了戳你）", "（回复了你）", "[图片]"})
_AMBIENT_RING_MAXLEN = 25
_AMBIENT_INJECT_LIMIT = 15
_L2_COMPRESS_THRESHOLD = 6


@dataclass(slots=True)
class _DebounceBucket:
    """In-flight burst message accumulator per `(session_key, user_id)`."""

    first_ts: float
    last_ts: float
    items: list[tuple[InboundMessage, str]] = field(default_factory=list)


def _coalesce_debounced_items(
    items: list[tuple[InboundMessage, str]],
) -> tuple[InboundMessage, str]:
    """Merge a rapid burst of messages from the same speaker into one coherent turn."""
    assert items
    if len(items) == 1:
        return items[0]

    last_msg = items[-1][0]
    real_lines: list[str] = []
    fallback_placeholders: list[str] = []
    merged_imgs: list[str] = []
    seen_imgs: set[str] = set()

    quote_source: InboundMessage | None = None
    any_mentioned_bot = False

    for msg, clean_text in items:
        if msg.mentioned_bot:
            any_mentioned_bot = True
        if quote_source is None and (
            msg.reply_to_id is not None or msg.reply_to_text.strip()
        ):
            quote_source = msg
        stripped = clean_text.strip()
        if stripped:
            if stripped in _SYNTHETIC_PLACEHOLDERS:
                fallback_placeholders.append(stripped)
            else:
                real_lines.append(stripped)
        for url in msg.image_urls:
            if url not in seen_imgs and len(merged_imgs) < 4:
                seen_imgs.add(url)
                merged_imgs.append(url)

    if real_lines:
        merged_clean = "\n".join(real_lines)
    elif fallback_placeholders:
        merged_clean = fallback_placeholders[0]
    else:
        merged_clean = ""

    merged_msg = replace(
        last_msg,
        text=merged_clean,
        mentioned_bot=any_mentioned_bot or last_msg.mentioned_bot,
        image_urls=tuple(merged_imgs),
        reply_to_id=(
            quote_source.reply_to_id if quote_source is not None else last_msg.reply_to_id
        ),
        reply_to_user_id=(
            quote_source.reply_to_user_id
            if quote_source is not None
            else last_msg.reply_to_user_id
        ),
        reply_to_sender_name=(
            quote_source.reply_to_sender_name
            if quote_source is not None
            else last_msg.reply_to_sender_name
        ),
        reply_to_text=(
            quote_source.reply_to_text
            if quote_source is not None
            else last_msg.reply_to_text
        ),
    )
    return merged_msg, merged_clean


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
    """Core message-processing pipeline assembling trigger, debounce, ambient buffer, memory, and LLM."""

    def __init__(
        self,
        *,
        bus: MessageBus,
        provider: LLMProvider,
        persona: PersonaManager,
        sessions: SessionStore,
        trigger: RuleTrigger,
        guardrails: GuardrailManager,
        memory: MemoryStore | None = None,
        stickers: StickerRegistry | None = None,
        enable_vision: bool = True,
        vision_transport: Any | None = None,
        commands: AdminCommandRouter | None = None,
        bot_qq: int | None = None,
        admin_user_ids: set[int] | None = None,
        private_debounce_s: float = 0.0,
        group_debounce_s: float = 0.0,
        max_debounce_wait_s: float = 6.0,
        max_debounce_batch: int = 8,
        clock: Callable[[], float] | None = None,
        error_fallback_reply: str = "刚卡了一下，再发一遍",
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.persona = persona
        self.sessions = sessions
        self.memory = memory
        self.trigger = trigger
        self.guardrails = guardrails
        self.stickers = stickers
        self.enable_vision = bool(enable_vision)
        self._vision_transport = vision_transport
        self.error_fallback_reply = error_fallback_reply

        self.private_debounce_s = max(0.0, float(private_debounce_s))
        self.group_debounce_s = max(0.0, float(group_debounce_s))
        self.max_debounce_wait_s = max(0.1, float(max_debounce_wait_s))
        self.max_debounce_batch = max(1, int(max_debounce_batch))
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

        self._admin_user_ids: set[int] = (
            set(admin_user_ids) if admin_user_ids is not None else set()
        )
        self._debounce_buckets: dict[tuple[str, int], _DebounceBucket] = {}
        self._ambient_buffers: dict[str, deque[str]] = {}
        self._ambient_unseen_counts: dict[str, int] = {}
        self._bg_tasks: set[asyncio.Task[Any]] = set()

        effective_bot_qq = bot_qq if bot_qq is not None else self.trigger.bot_qq
        self.bot_qq = effective_bot_qq
        if self.trigger.bot_qq is None and effective_bot_qq is not None:
            self.trigger.bot_qq = effective_bot_qq

        if commands is not None:
            self.commands = commands
        else:
            self.commands = AdminCommandRouter(
                admin_user_ids=self._admin_user_ids,
                on_reset=self._reset_session_and_summary,
                on_quota=self.guardrails.snapshot,
                on_reload=self._reload_and_sync_persona,
                on_persona_status=self._persona_status,
                on_persona_switch=self._switch_persona_profile,
                on_remember=(
                    (lambda content, scope: self.memory.add_memory(content, scope=scope).id)
                    if self.memory is not None
                    else None
                ),
                on_list_memories=(
                    (
                        lambda inc_priv: [
                            (f.id, f.scope, f.content)
                            for f in self.memory.list_memories(
                                include_admin_private=inc_priv
                            )
                        ]
                    )
                    if self.memory is not None
                    else None
                ),
                on_forget=(
                    self.memory.delete_memory if self.memory is not None else None
                ),
                on_profile=(
                    self._update_user_profile if self.memory is not None else None
                ),
            )

        self._sync_persona(self.persona.current())

    def _sync_persona(self, cfg: PersonaConfig) -> None:
        self.trigger.update_keywords((cfg.name, *cfg.aliases))
        if hasattr(self.trigger, "update_topic_keywords"):
            self.trigger.update_topic_keywords(cfg.topic_keywords)
        self.guardrails.rate_refusal = cfg.cute_rate_refusal
        self.guardrails.budget_refusal = cfg.cute_budget_refusal

    def _reload_and_sync_persona(self) -> str:
        cfg = self.persona.reload()
        self._sync_persona(cfg)
        if self.stickers is not None:
            self.stickers.reload()
        if cfg.active_profile:
            return f"{cfg.name}（激活档案: {cfg.active_profile}）"
        return cfg.name

    def _persona_status(self) -> tuple[str, str, list[str]]:
        cfg = self.persona.current()
        return cfg.name, cfg.active_profile, self.persona.list_profiles()

    def _switch_persona_profile(self, profile_name: str) -> str:
        cfg = self.persona.switch_profile(profile_name)
        self._sync_persona(cfg)
        return f"{cfg.name}（当前档案: {cfg.active_profile}）"

    def _update_user_profile(self, target_qq: int, notes: str) -> str:
        assert self.memory is not None
        prof = self.memory.set_user_profile(target_qq, notes)
        name_part = f"（{prof.display_name}）" if prof.display_name else ""
        return f"已更新用户 {prof.user_id}{name_part} 画像：{prof.traits_text}"

    def _reset_session_and_summary(self, session_key: str) -> int:
        cleared = self.sessions.clear(session_key)
        if self.memory is not None:
            self.memory.clear_summary(session_key)
        self._ambient_buffers.pop(session_key, None)
        self._ambient_unseen_counts.pop(session_key, None)
        stale_keys = [k for k in self._debounce_buckets if k[0] == session_key]
        for k in stale_keys:
            self._debounce_buckets.pop(k, None)
        return cleared

    def _record_ambient_group_line(self, msg: InboundMessage) -> None:
        if msg.chat_type != "group":
            return
        body = msg.text.strip()
        if not body and msg.image_urls:
            body = "[图片]"
        if not body:
            return
        compact = " ".join(body.split())
        if len(compact) > 220:
            compact = f"{compact[:220]}..."
        ring = self._ambient_buffers.setdefault(
            msg.session_key,
            deque(maxlen=_AMBIENT_RING_MAXLEN),
        )
        ring.append(f"[{msg.sender_name}]: {compact}")
        prev_unseen = self._ambient_unseen_counts.get(msg.session_key, 0)
        self._ambient_unseen_counts[msg.session_key] = min(
            _AMBIENT_RING_MAXLEN,
            prev_unseen + 1,
        )

    def _consume_ambient_context_xml(self, msg: InboundMessage) -> str:
        if msg.chat_type != "group":
            return ""
        unseen = self._ambient_unseen_counts.get(msg.session_key, 0)
        if unseen <= 0:
            return ""
        ring = self._ambient_buffers.get(msg.session_key)
        if not ring:
            self._ambient_unseen_counts[msg.session_key] = 0
            return ""
        take_n = min(unseen, _AMBIENT_INJECT_LIMIT, len(ring))
        recent_lines = list(ring)[-take_n:]
        self._ambient_unseen_counts[msg.session_key] = 0
        return self.persona.wrap_ambient_context(recent_lines)

    def _schedule_l2_summary(self, session_key: str, batch: list[ChatMessage]) -> None:
        if self.memory is None or not batch:
            return
        summary_model = self.guardrails.cheap_model
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self.memory.summarize_batch(
                session_key,
                batch,
                self.provider,
                model=summary_model,
            )
        )
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

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

    def _passes_pre_checks(self, msg: InboundMessage) -> bool:
        effective_bot_qq = self.bot_qq if self.bot_qq is not None else self.trigger.bot_qq
        if (effective_bot_qq is not None and msg.user_id == effective_bot_qq) or (
            msg.self_id is not None and msg.user_id == msg.self_id
        ):
            return False

        p_cfg = self.persona.current()
        self._sync_persona(p_cfg)

        if (
            msg.chat_type == "group"
            and self.trigger.allowed_group_ids
            and msg.group_id not in self.trigger.allowed_group_ids
        ):
            return False

        if self.memory is not None:
            self.memory.touch_user(msg.user_id, msg.sender_name)

        return True

    async def _execute_triggered_turn(
        self,
        msg: InboundMessage,
        *,
        clean_text: str,
    ) -> OutboundReply | None:
        # 6) Rate-limiting and daily token/micro-USD budget guardrails
        guard = self.guardrails.check(msg, now=self._clock())
        if not guard.allowed:
            if guard.refusal_reply:
                return self._build_outbound_reply(msg, guard.refusal_reply)
            return None

        # 7) Assemble anti-injection prompt window (+ scoped L2/L3 memory + sticker guide + ambient context + vision)
        sys_prompt = self.persona.render_system_prompt()
        if self.memory is not None:
            is_admin_priv = (
                msg.chat_type == "private" and msg.user_id in self._admin_user_ids
            )
            mem_block = self.memory.render_context_block(
                session_key=msg.session_key,
                chat_type=msg.chat_type,
                group_id=msg.group_id,
                user_id=msg.user_id,
                is_admin_private=is_admin_priv,
            )
            if mem_block:
                sys_prompt = f"{sys_prompt}\n\n{mem_block}"

        if self.stickers is not None:
            sticker_guide = self.stickers.render_prompt_guide()
            if sticker_guide:
                sys_prompt = f"{sys_prompt}\n\n{sticker_guide}"

        sys_msg = ChatMessage(role="system", content=sys_prompt)
        history = self.sessions.get_window(msg.session_key)
        wrapped_user = self.persona.wrap_user_message(msg, clean_text=clean_text)
        ambient_xml = self._consume_ambient_context_xml(msg)
        live_user_content = (
            f"{ambient_xml}\n{wrapped_user}" if ambient_xml else wrapped_user
        )

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
                    ChatMessage(
                        role="user",
                        content=live_user_content,
                        images=vision_images,
                    ),
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
        evicted_u = self.sessions.append(msg.session_key, "user", wrapped_user)
        evicted_a = self.sessions.append(msg.session_key, "assistant", history_assistant)

        if self.memory is not None and (evicted_u or evicted_a):
            batch_to_compress = self.memory.stage_evicted_messages(
                msg.session_key,
                [*evicted_u, *evicted_a],
                compress_threshold=_L2_COMPRESS_THRESHOLD,
            )
            if batch_to_compress:
                self._schedule_l2_summary(msg.session_key, batch_to_compress)

        if msg.chat_type == "group" and hasattr(self.trigger, "record_bot_reply"):
            self.trigger.record_bot_reply(msg.group_id, msg.user_id)

        return self._build_outbound_reply(msg, clean_reply, sticker_id=sticker_id)

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply | None:
        """Process a single inbound message immediately (without burst debounce delay)."""
        if not self._passes_pre_checks(msg):
            return None

        cmd_res = self.commands.try_handle(msg)
        if cmd_res.handled:
            if cmd_res.reply_text:
                return self._build_outbound_reply(msg, cmd_res.reply_text)
            return None

        trig = self.trigger.evaluate(msg)
        if not trig.should_reply:
            self._record_ambient_group_line(msg)
            return None

        return await self._execute_triggered_turn(msg, clean_text=trig.clean_text)

    def _debounce_window_for(self, chat_type: str) -> float:
        return (
            self.group_debounce_s
            if chat_type == "group"
            else self.private_debounce_s
        )

    async def _ingest_for_bus_loop(self, msg: InboundMessage) -> int:
        if not self._passes_pre_checks(msg):
            return 1

        # Slash admin commands always bypass debounce with 0ms latency
        cmd_res = self.commands.try_handle(msg)
        if cmd_res.handled:
            if cmd_res.reply_text:
                reply = self._build_outbound_reply(msg, cmd_res.reply_text)
                if reply is not None:
                    await self.bus.publish_outbound(reply)
            return 1

        bucket_key = (msg.session_key, msg.user_id)
        window_s = self._debounce_window_for(msg.chat_type)
        now = self._clock()

        existing_bucket = self._debounce_buckets.get(bucket_key)
        if existing_bucket is not None and not msg.mentioned_other:
            # Coalesce rapid follow-up lines from the same speaker into open burst bucket
            follow_text = msg.text.strip() or ("[图片]" if msg.image_urls else "")
            if follow_text or msg.image_urls:
                existing_bucket.items.append((msg, follow_text))
                existing_bucket.last_ts = now
                return 1

        trig = self.trigger.evaluate(msg)
        if not trig.should_reply:
            self._record_ambient_group_line(msg)
            return 1

        if window_s <= 0.0:
            reply = await self._execute_triggered_turn(msg, clean_text=trig.clean_text)
            if reply is not None:
                await self.bus.publish_outbound(reply)
            return 1

        self._debounce_buckets[bucket_key] = _DebounceBucket(
            first_ts=now,
            last_ts=now,
            items=[(msg, trig.clean_text)],
        )
        return 1

    async def _flush_ready_debounce_buckets(self, *, force_all: bool = False) -> int:
        if not self._debounce_buckets:
            return 0

        now = self._clock()
        ready_keys: list[tuple[str, int]] = []
        for key, bucket in self._debounce_buckets.items():
            if not bucket.items:
                ready_keys.append(key)
                continue
            chat_type = bucket.items[-1][0].chat_type
            window_s = self._debounce_window_for(chat_type)
            if (
                force_all
                or (now - bucket.last_ts) >= window_s
                or (now - bucket.first_ts) >= self.max_debounce_wait_s
                or len(bucket.items) >= self.max_debounce_batch
            ):
                ready_keys.append(key)

        dispatched = 0
        for key in ready_keys:
            bucket = self._debounce_buckets.pop(key, None)
            if bucket is None or not bucket.items:
                continue
            merged_msg, merged_clean = _coalesce_debounced_items(bucket.items)
            reply = await self._execute_triggered_turn(
                merged_msg,
                clean_text=merged_clean,
            )
            if reply is not None:
                await self.bus.publish_outbound(reply)
            dispatched += 1
        return dispatched

    async def wait_background_tasks(self) -> None:
        """Await any in-flight asynchronous L2 rolling summary tasks."""
        if self._bg_tasks:
            pending = list(self._bg_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

    async def process_once(self, limit: int = 16) -> int:
        """Poll up to `limit` inbound messages, ingest/debounce, flush ready bursts, and monotonically ACK."""
        batch = await self.bus.poll_inbound(limit=limit)
        processed = 0
        for seq, msg in batch:
            await self._ingest_for_bus_loop(msg)
            await self.bus.ack_inbound(seq)
            processed += 1

        flushed = await self._flush_ready_debounce_buckets()
        return processed + flushed

    async def run_forever(
        self,
        stop_event: asyncio.Event | None = None,
        poll_interval_s: float = 0.05,
    ) -> None:
        """Continuously poll and process inbound messages until `stop_event` is set."""
        effective_stop = stop_event if stop_event is not None else asyncio.Event()
        interval = max(0.001, float(poll_interval_s))

        try:
            while not effective_stop.is_set():
                processed = await self.process_once()
                if processed == 0:
                    try:
                        await asyncio.wait_for(effective_stop.wait(), timeout=interval)
                        return
                    except TimeoutError:
                        pass
        finally:
            await self.wait_background_tasks()


__all__ = [
    "AgentKernel",
    "ManagedLlmosDaemon",
    "resolve_daemon_command",
]
