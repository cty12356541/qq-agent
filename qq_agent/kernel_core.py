"""Core Pi-Agent kernel orchestrating per-session actors, SubAgentRunner, memory, and ingress loops."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import logging
from pathlib import Path
import time
from typing import Any

import httpx

from .channel_bridge import AckCoordinator, AdaptivePoller, BusQueueMetrics, MessageBus
from .commands import AdminCommandRouter
from .guardrails import GuardrailManager
from .kernel_execution import KernelExecutionMixin
from .kernel_maintenance import (
    consume_reload_sentinel,
    drain_kernel_for_graceful_reload,
    extract_soft_orphaned_seqs_from_actor,
)
from .kernel_turn_helpers import (
    _AMBIENT_INJECT_LIMIT,
    _AMBIENT_RING_MAXLEN,
    _L2_COMPRESS_THRESHOLD,
    _SOFT_TRIGGER_REASONS,
    _DebounceBucket,
    _TurnEnvelope,
    _build_default_admin_router,
    _coalesce_turn_envelopes,
    _extract_orphaned_seqs_from_actor,
    _has_cancel_steer_intent,
    _init_search_and_subagent_runner,
    _pop_ready_debounce_batches,
    _render_status_snapshot,
    _stop_actors_and_collect_orphans,
)
from .memory import MemoryStore, extract_public_interest_cues
from .persona import PersonaConfig, PersonaManager
from .pi_core import AmbientBuffer, CancelToken, SessionActor, SubAgentRunner, TurnPriority
from .provider import LLMProvider
from .session import SessionStore
from .stickers import StickerRegistry
from .tools import ChatTurnState, WebPageFetcher, WebSearchClient
from .trigger import RuleTrigger
from .types import ChatMessage, InboundMessage, OutboundReply, RuntimeRole


class AgentKernel(KernelExecutionMixin):
    """Async Pi-Agent kernel orchestrating per-session actors, fast/slow SubAgentRunner, tools, and memory."""

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
        vision_max_side: int = 512,
        vision_transport: Any | None = None,
        compact_prompt: bool = False,
        fact_cache_ttl_s: float = 1800.0,
        commands: AdminCommandRouter | None = None,
        bot_qq: int | None = None,
        admin_user_ids: set[int] | None = None,
        private_debounce_s: float = 0.0,
        group_debounce_s: float = 0.0,
        max_debounce_wait_s: float = 6.0,
        max_debounce_batch: int = 8,
        agent_max_steps: int = 3,
        agent_turn_timeout_s: float = 35.0,
        bubble_typing_delay_s: float = 0.0,
        enable_tools: bool = True,
        web_search_enabled: bool = True,
        web_search_timeout_s: float = 8.0,
        search_api_key: str | None = None,
        search_api_base_url: str | None = None,
        web_search_client: WebSearchClient | None = None,
        webpage_fetcher: WebPageFetcher | None = None,
        subagent_enabled: bool = True,
        subagent_model: str | None = None,
        subagent_enable_thinking: bool = True,
        subagent_max_steps: int = 4,
        subagent_timeout_s: float = 30.0,
        subagent_max_concurrent: int = 6,
        subagent_max_per_session: int | None = 2,
        runtime_role: RuntimeRole = "all-in-one",
        hard_trigger_retry_delays_s: Sequence[float] = (),
        reload_sentinel_path: Path | str | None = None,
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
        self.vision_max_side = max(64, int(vision_max_side))
        self._vision_transport = vision_transport
        self._vision_http_client: httpx.AsyncClient | None = None
        self.compact_prompt = bool(compact_prompt)
        self.fact_cache_ttl_s = max(0.0, float(fact_cache_ttl_s))
        self.error_fallback_reply = error_fallback_reply
        self.runtime_role: RuntimeRole = runtime_role

        self.private_debounce_s = max(0.0, float(private_debounce_s))
        self.group_debounce_s = max(0.0, float(group_debounce_s))
        self.max_debounce_wait_s = max(0.1, float(max_debounce_wait_s))
        self.max_debounce_batch = max(1, int(max_debounce_batch))
        self.agent_max_steps = max(1, int(agent_max_steps))
        self.agent_turn_timeout_s = max(1.0, float(agent_turn_timeout_s))
        self.bubble_typing_delay_s = max(0.0, float(bubble_typing_delay_s))
        self.hard_trigger_retry_delays_s: tuple[float, ...] = tuple(
            max(0.0, float(d)) for d in hard_trigger_retry_delays_s
        )
        self.reload_sentinel_path: Path | None = (
            Path(reload_sentinel_path) if reload_sentinel_path is not None else None
        )
        self._closing: bool = False

        self.enable_tools = bool(enable_tools)
        self.web_search_enabled = bool(web_search_enabled)
        self.subagent_enabled = bool(subagent_enabled)
        self.subagent_enable_thinking = bool(subagent_enable_thinking)
        self.subagent_model = subagent_model.strip() if subagent_model and subagent_model.strip() else None
        self.subagent_max_concurrent = max(1, int(subagent_max_concurrent))
        self.subagent_max_per_session: int | None = (
            int(subagent_max_per_session)
            if subagent_max_per_session is not None and int(subagent_max_per_session) > 0
            else None
        )

        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._admin_user_ids: set[int] = set(admin_user_ids) if admin_user_ids is not None else set()

        self._debounce_buckets: dict[tuple[str, int], _DebounceBucket] = {}
        self._ambient_buffers: dict[str, AmbientBuffer] = {}
        self._actors: dict[str, SessionActor] = {}
        self._active_turn_states: dict[str, ChatTurnState] = {}
        self._ack_coordinator = AckCoordinator(self.bus)
        self._last_bus_metrics: BusQueueMetrics | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._subagent_seq_counter = 0
        self._dropped_soft_count = 0
        self._last_turn_latency_ms: float = 0.0

        self._web_search_client, self._webpage_fetcher, self._subagent_runner = (
            _init_search_and_subagent_runner(
                provider=self.provider,
                web_search_enabled=self.web_search_enabled,
                web_search_timeout_s=web_search_timeout_s,
                search_api_key=search_api_key,
                search_api_base_url=search_api_base_url,
                web_search_client=web_search_client,
                webpage_fetcher=webpage_fetcher,
                subagent_model=self.subagent_model,
                subagent_enable_thinking=self.subagent_enable_thinking,
                subagent_max_steps=subagent_max_steps,
                subagent_timeout_s=subagent_timeout_s,
                subagent_max_concurrent=self.subagent_max_concurrent,
                subagent_max_per_session=self.subagent_max_per_session,
                before_step=self._check_followup_step_budget,
                after_step=self._record_step_usage,
            )
        )

        effective_bot_qq = bot_qq if bot_qq is not None else self.trigger.bot_qq
        self.bot_qq = effective_bot_qq
        if self.trigger.bot_qq is None and effective_bot_qq is not None:
            self.trigger.bot_qq = effective_bot_qq

        self.commands = commands if commands is not None else _build_default_admin_router(self)
        self._sync_persona(self.persona.current())

    @property
    def ack_coordinator(self) -> AckCoordinator:
        return self._ack_coordinator

    @property
    def subagent_runner(self) -> SubAgentRunner:
        return self._subagent_runner

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
        return f"{cfg.name}（激活档案: {cfg.active_profile}）" if cfg.active_profile else cfg.name

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

    def _get_ambient_buffer(self, session_key: str) -> AmbientBuffer:
        buf = self._ambient_buffers.get(session_key)
        if buf is None:
            buf = AmbientBuffer(maxlen=_AMBIENT_RING_MAXLEN)
            self._ambient_buffers[session_key] = buf
        return buf

    def _schedule_bg_task(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _mark_seqs_done(self, seqs: Sequence[int]) -> None:
        for seq in seqs:
            if seq > 0:
                await self._ack_coordinator.mark_done(seq)

    def _can_preempt_session_turn(self, session_key: str) -> bool:
        st = self._active_turn_states.get(session_key)
        if st is None:
            return True
        return not st.sent_bubbles and st.last_sticker_id is None

    def _on_actor_item_dropped(self, item: Any) -> None:
        if isinstance(item, _TurnEnvelope):
            if item.trigger_reason in _SOFT_TRIGGER_REASONS:
                self._dropped_soft_count += 1
                self._record_ambient_group_line(item.msg, item.trigger_reason)
            if item.seqs and not (self._closing and item.trigger_reason not in _SOFT_TRIGGER_REASONS):
                self._schedule_bg_task(self._mark_seqs_done(item.seqs))
            return
        if isinstance(item, list):
            for elem in item:
                self._on_actor_item_dropped(elem)

    def _extract_orphaned_actor_seqs(self, actor: SessionActor) -> list[int]:
        return _extract_orphaned_seqs_from_actor(actor)

    def stop_session_tasks(self, target_scope: str) -> tuple[bool, int]:
        """Interrupt active turn(s) and depth=1 background sub-agents for `session_key` or `'*'`."""
        any_main, total_sub, orphaned = _stop_actors_and_collect_orphans(
            self._actors,
            self._subagent_runner,
            target_scope,
            self._extract_orphaned_actor_seqs,
        )
        if orphaned:
            self._schedule_bg_task(self._mark_seqs_done(orphaned))
        return any_main, total_sub

    def render_runtime_status(self, session_key: str) -> str:
        """Render a live snapshot of SessionActor concurrency, SubAgent tasks, and AckCoordinator watermarks."""
        return _render_status_snapshot(self, session_key)

    def _reset_session_and_summary(self, session_key: str) -> int:
        self.stop_session_tasks(session_key)
        cleared = self.sessions.clear(session_key)
        if self.memory is not None:
            self.memory.clear_summary(session_key)
        buf = self._ambient_buffers.pop(session_key, None)
        if buf is not None:
            buf.clear()
        actor = self._actors.get(session_key)
        if actor is not None:
            actor.ambient.clear()
        for k in [k for k in self._debounce_buckets if k[0] == session_key]:
            self._debounce_buckets.pop(k, None)
        return cleared

    def _observe_group_public_cues(self, msg: InboundMessage, text_override: str | None = None) -> None:
        if self.memory is None or msg.chat_type != "group":
            return
        source = text_override if text_override is not None else msg.text
        cues = extract_public_interest_cues(source)
        if cues:
            self.memory.observe_public_traits(msg.user_id, msg.sender_name, cues)

    def _record_ambient_group_line(self, msg: InboundMessage, refund_reason: str = "") -> None:
        if msg.chat_type != "group":
            return
        if refund_reason and hasattr(self.trigger, "refund_soft_trigger"):
            self.trigger.refund_soft_trigger(msg.group_id, msg.user_id, reason=refund_reason)
        self._observe_group_public_cues(msg)
        self._get_ambient_buffer(msg.session_key).record_message(msg)

    def _consume_ambient_context_xml(self, msg: InboundMessage) -> str:
        if msg.chat_type != "group":
            return ""
        recent_lines = self._get_ambient_buffer(msg.session_key).consume_recent_lines(
            limit=_AMBIENT_INJECT_LIMIT
        )
        return self.persona.wrap_ambient_context(recent_lines) if recent_lines else ""

    def _schedule_l2_summary(self, session_key: str, batch: list[ChatMessage]) -> None:
        if self.memory is None or not batch:
            return
        self._schedule_bg_task(
            self.memory.summarize_batch(
                session_key,
                batch,
                self.provider,
                model=self.guardrails.cheap_model,
            )
        )

    def _append_l1_turn(
        self,
        session_key: str,
        *,
        user_content: str | None,
        assistant_content: str,
    ) -> None:
        evicted: list[ChatMessage] = []
        if user_content is not None:
            evicted.extend(self.sessions.append(session_key, "user", user_content))
        evicted.extend(self.sessions.append(session_key, "assistant", assistant_content))
        if self.memory is not None and evicted:
            batch_to_compress = self.memory.stage_evicted_messages(
                session_key,
                evicted,
                compress_threshold=_L2_COMPRESS_THRESHOLD,
            )
            if batch_to_compress:
                self._schedule_l2_summary(session_key, batch_to_compress)

    def _get_or_create_actor(self, session_key: str, chat_type: str) -> SessionActor:
        actor = self._actors.get(session_key)
        window_s = self._debounce_window_for(chat_type)
        if actor is None:
            actor = SessionActor(
                session_key,
                turn_handler=self._run_actor_turn,
                subagent_result_handler=self._handle_subagent_result,
                subagent_runner=self._subagent_runner,
                debounce_s=window_s,
                max_debounce_wait_s=self.max_debounce_wait_s,
                max_debounce_batch=self.max_debounce_batch,
                ambient_maxlen=_AMBIENT_RING_MAXLEN,
                coalesce_fn=_coalesce_turn_envelopes,
                can_preempt_active_fn=lambda sk=session_key: self._can_preempt_session_turn(sk),
                on_item_dropped=self._on_actor_item_dropped,
            )
            actor.ambient = self._get_ambient_buffer(session_key)
            self._actors[session_key] = actor
        else:
            actor.debounce_s = window_s
        return actor

    async def _run_actor_turn(
        self,
        envelope: _TurnEnvelope,
        cancel_token: CancelToken,
        drain_steering: Callable[[], Sequence[str]],
    ) -> None:
        pin_candidate = envelope.seqs[-1] if envelope.seqs else None
        reply: OutboundReply | None = None
        pinned_to_subagent = False
        try:
            reply, pinned_to_subagent = await self._execute_triggered_turn(
                envelope.msg,
                clean_text=envelope.clean_text,
                trigger_reason=envelope.trigger_reason,
                cancel_token=cancel_token,
                drain_steering=drain_steering,
                pin_inbound_seq=pin_candidate,
                publish_to_bus=True,
            )
        finally:
            preserve_unacked_hard = (
                self._closing
                and reply is None
                and not pinned_to_subagent
                and envelope.trigger_reason not in _SOFT_TRIGGER_REASONS
                and cancel_token.is_cancelled
                and cancel_token.reason == "actor_closed"
            )
            if not preserve_unacked_hard:
                seqs_to_ack = (
                    envelope.seqs[:-1] if (pinned_to_subagent and envelope.seqs) else envelope.seqs
                )
                for seq in seqs_to_ack:
                    await self._ack_coordinator.mark_done(seq)

    @staticmethod
    def _has_cancel_intent(text: str) -> bool:
        return _has_cancel_steer_intent(text)

    async def dispatch_once(self, limit: int = 16) -> int:
        """Non-blocking IngressDispatcher: poll `MessageBus` and fan out across `SessionActor` pool."""
        batch = await self.bus.poll_inbound(limit=limit)
        dispatched = 0

        for seq, msg in batch:
            if self._ack_coordinator.is_tracked(seq):
                continue

            self._ack_coordinator.track(seq, msg.session_key)
            dispatched += 1

            if not self._passes_pre_checks(msg):
                await self._ack_coordinator.mark_done(seq)
                continue

            # 0ms admin slash-command fast lane (never blocked behind busy SessionActor)
            cmd_res = self.commands.try_handle(msg)
            if cmd_res.handled:
                if cmd_res.reply_text:
                    cmd_reply = self._build_outbound_reply(msg, cmd_res.reply_text)
                    if cmd_reply is not None:
                        await self.bus.publish_outbound(cmd_reply)
                await self._ack_coordinator.mark_done(seq)
                continue

            actor = self._get_or_create_actor(msg.session_key, msg.chat_type)

            # Coalesce rapid un-tagged burst follow-up lines when same speaker already opened debounce
            if (
                actor.state == "IDLE"
                and actor._debounce_items
                and actor._debounce_user_id == msg.user_id
                and not msg.mentioned_other
                and not msg.mentioned_bot
            ):
                follow_text = msg.text.strip() or ("[图片]" if msg.image_urls else "")
                if follow_text or msg.image_urls:
                    env = _TurnEnvelope(
                        msg=msg,
                        clean_text=follow_text,
                        trigger_reason="burst_followup",
                        seqs=(seq,),
                    )
                    route = actor.submit(env, user_id=msg.user_id, priority=actor._debounce_priority)
                    if route in {"steered", "dropped_soft"}:
                        await self._ack_coordinator.mark_done(seq)
                    continue

            cancel_intent = self._has_cancel_intent(msg.text)
            trig = self.trigger.evaluate(msg)

            if not trig.should_reply:
                if (
                    cancel_intent
                    and actor.is_running
                    and actor.active_turn_user_id == msg.user_id
                    and not msg.mentioned_other
                ):
                    actor.steer(f"[{msg.sender_name}]: {msg.text.strip()}", cancel_active=True)
                    self._subagent_runner.cancel(session_key=msg.session_key)
                else:
                    self._record_ambient_group_line(msg)
                await self._ack_coordinator.mark_done(seq)
                continue

            if cancel_intent:
                self._subagent_runner.cancel(session_key=msg.session_key)

            prio: TurnPriority = "soft" if trig.reason in _SOFT_TRIGGER_REASONS else "hard"
            env = _TurnEnvelope(
                msg=msg,
                clean_text=trig.clean_text,
                trigger_reason=trig.reason,
                seqs=(seq,),
            )
            route = actor.submit(
                env,
                user_id=msg.user_id,
                priority=prio,
                steer_if_same_user=(prio == "hard"),
                steer_text=f"[{msg.sender_name} 补充]: {trig.clean_text}",
                cancel_active=cancel_intent,
            )
            if route in {"steered", "dropped_soft"}:
                await self._ack_coordinator.mark_done(seq)

        return dispatched

    async def handle_inbound(self, msg: InboundMessage) -> OutboundReply | None:
        """Process a single inbound message immediately (without burst debounce delay)."""
        if not self._passes_pre_checks(msg):
            return None

        cmd_res = self.commands.try_handle(msg)
        if cmd_res.handled:
            return self._build_outbound_reply(msg, cmd_res.reply_text) if cmd_res.reply_text else None

        trig = self.trigger.evaluate(msg)
        if not trig.should_reply:
            self._record_ambient_group_line(msg)
            return None

        reply, _ = await self._execute_triggered_turn(
            msg,
            clean_text=trig.clean_text,
            trigger_reason=trig.reason,
            publish_to_bus=False,
        )
        return reply

    def _debounce_window_for(self, chat_type: str) -> float:
        return self.group_debounce_s if chat_type == "group" else self.private_debounce_s

    async def _ingest_for_bus_loop(self, msg: InboundMessage) -> int:
        if not self._passes_pre_checks(msg):
            return 1

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
            await self._execute_triggered_turn(
                msg,
                clean_text=trig.clean_text,
                trigger_reason=trig.reason,
                publish_to_bus=True,
            )
            return 1

        self._debounce_buckets[bucket_key] = _DebounceBucket(
            first_ts=now,
            last_ts=now,
            items=[(msg, trig.clean_text)],
        )
        return 1

    async def _flush_ready_debounce_buckets(self, *, force_all: bool = False) -> int:
        ready_batches = _pop_ready_debounce_batches(
            self._debounce_buckets,
            now=self._clock(),
            window_for=self._debounce_window_for,
            max_wait_s=self.max_debounce_wait_s,
            max_batch=self.max_debounce_batch,
            force_all=force_all,
        )
        for merged_msg, merged_clean in ready_batches:
            await self._execute_triggered_turn(
                merged_msg,
                clean_text=merged_clean,
                trigger_reason="direct",
                publish_to_bus=True,
            )
        return len(ready_batches)

    async def wait_idle(self, timeout: float | None = None) -> None:
        """Wait until all active SessionActors, SubAgents, and background summary tasks settle."""
        actors = list(self._actors.values())
        if actors:
            await asyncio.gather(*(a.wait_idle(timeout=timeout) for a in actors))
        await self.wait_background_tasks()

    async def wait_background_tasks(self) -> None:
        """Await any in-flight asynchronous L2 rolling summary or ACK cleanup tasks."""
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

    async def aclose(self) -> None:
        """Gracefully close actors, SubAgentRunner, HTTP pools, and preserve un-replied hard triggers un-ACKed."""
        self._closing = True
        soft_orphaned: list[int] = []
        actors = list(self._actors.values())
        for actor in actors:
            soft_orphaned.extend(extract_soft_orphaned_seqs_from_actor(actor))
        if actors:
            await asyncio.gather(*(actor.aclose() for actor in actors), return_exceptions=True)
        await self._subagent_runner.aclose()
        if soft_orphaned:
            await self._mark_seqs_done(soft_orphaned)
        await self.wait_background_tasks()
        vision_client = self._vision_http_client
        self._vision_http_client = None
        close_jobs = [
            self._web_search_client.aclose(),
            self._webpage_fetcher.aclose(),
        ]
        if vision_client is not None and not vision_client.is_closed:
            close_jobs.append(vision_client.aclose())
        await asyncio.gather(*close_jobs, return_exceptions=True)

    async def process_once(self, limit: int = 16) -> int:
        """Deterministic single-tick poll/debounce/flush step (compatible with synthetic fake-clock tests)."""
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
        """Run non-blocking IngressDispatcher + per-session SessionActor pool until `stop_event` or reload sentinel."""
        effective_stop = stop_event if stop_event is not None else asyncio.Event()
        base_interval = max(0.001, float(poll_interval_s))
        min_poll = min(0.015, base_interval)
        max_poll = min(0.20, max(base_interval * 4.0, 0.04))
        poller = AdaptivePoller(min_interval_s=min_poll, max_interval_s=max_poll)
        last_health_ts = 0.0

        try:
            while not effective_stop.is_set():
                if consume_reload_sentinel(self.reload_sentinel_path):
                    await drain_kernel_for_graceful_reload(self)
                    return
                dispatched = await self.dispatch_once()
                if dispatched > 0:
                    poller.reset()
                else:
                    now = time.monotonic()
                    if (now - last_health_ts) >= 15.0 and hasattr(self.bus, "inspect_health"):
                        last_health_ts = now
                        try:
                            self._last_bus_metrics = await self.bus.inspect_health()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    delay = poller.next_delay(False)
                    try:
                        await asyncio.wait_for(effective_stop.wait(), timeout=delay)
                        return
                    except TimeoutError:
                        pass
        finally:
            await self.aclose()


__all__ = ["AgentKernel"]
