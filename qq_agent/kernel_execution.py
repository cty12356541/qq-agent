"""Turn execution, vision image pre-fetching, and background SubAgent synthesis mixin for AgentKernel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from typing import Any
import uuid

import httpx

from .anti_injection import build_dynamic_injection_anchor, detect_injection_signals
from .kernel_maintenance import run_agent_loop_with_maintenance_retry
from .kernel_turn_helpers import (
    _FAST_BRAIN_TOOL_GUIDE,
    _FAST_BRAIN_TOOL_GUIDE_COMPACT,
    _SOFT_TRIGGER_REASONS,
    _RecordingBusProxy,
    _bridge_tool_binding,
    _build_reply_for_inbound,
    _build_turn_registered_tools,
    _format_sent_history,
    _format_subagent_callback_prompt,
    _summarize_web_search_outcome,
)
from .pi_core import (
    BudgetBlockedError,
    CancelToken,
    RegisteredTool,
    ToolExecutionEvent,
    TurnEndEvent,
    agent_loop,
)
from .provider import ProviderError, fetch_vision_images
from .tools import (
    ChatTurnState,
    build_cancel_subagent_tool,
    build_remember_note_tool,
    build_send_message_tool,
    build_spawn_subagent_tool,
    build_stay_silent_tool,
    build_web_search_tool,
)
from .tools.chat_tools import _SILENT_FALLBACK_TOKENS
from .types import (
    ChatMessage,
    ChatResult,
    InboundMessage,
    OutboundReply,
    SubAgentRequest,
    SubAgentResult,
)

logger = logging.getLogger(__name__)


class KernelExecutionMixin:
    """Fast-brain Pi-Agent turn execution, slow-brain callback synthesis, vision fetching, and quota hooks."""

    bus: Any
    provider: Any
    persona: Any
    sessions: Any
    memory: Any
    trigger: Any
    guardrails: Any
    stickers: Any
    enable_vision: bool
    vision_max_side: int
    _vision_transport: Any
    _vision_http_client: httpx.AsyncClient | None
    compact_prompt: bool
    fact_cache_ttl_s: float
    error_fallback_reply: str
    agent_max_steps: int
    agent_turn_timeout_s: float
    bubble_typing_delay_s: float
    enable_tools: bool
    web_search_enabled: bool
    subagent_enabled: bool
    bot_qq: int | None
    _clock: Callable[[], float]
    _admin_user_ids: set[int]
    _actors: dict[str, Any]
    _active_turn_states: dict[str, ChatTurnState]
    _ack_coordinator: Any
    _subagent_seq_counter: int
    _last_turn_latency_ms: float
    _web_search_client: Any
    _subagent_runner: Any

    def _get_or_create_vision_client(self) -> httpx.AsyncClient | None:
        if self._vision_transport is not None:
            return None
        if self._vision_http_client is None or self._vision_http_client.is_closed:
            self._vision_http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(8.0),
                follow_redirects=True,
            )
        return self._vision_http_client

    async def _fetch_turn_vision_images(
        self,
        msg: InboundMessage,
    ) -> tuple[tuple[str, str], ...]:
        """Download and normalize inbound vision images when multimodal vision is enabled."""
        if not self.enable_vision or not msg.image_urls:
            return ()
        return await fetch_vision_images(
            msg.image_urls,
            max_side=self.vision_max_side,
            transport=self._vision_transport,
            client=self._get_or_create_vision_client(),
        )

    def _check_followup_step_budget(self, step: int) -> str | None:
        """Check daily hard/soft quota before extra LLM hops without re-triggering user 1s message cooldown."""
        snap = self.guardrails.snapshot()
        if (
            snap.calls_used >= snap.max_daily_calls
            or snap.tokens_used >= snap.max_daily_tokens
            or snap.cost_micros_used >= snap.hard_budget_micros
        ):
            raise BudgetBlockedError(
                self.guardrails.budget_refusal,
                reason="daily_hard_limit",
            )
        if (
            snap.cost_micros_used >= snap.soft_budget_micros
            and self.guardrails.cheap_model is not None
            and self.guardrails.cheap_model.strip()
        ):
            return self.guardrails.cheap_model.strip()
        return None

    def _record_step_usage(self, res: ChatResult) -> None:
        self.guardrails.record_usage(
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            total_tokens=res.total_tokens,
            cost_micros_usd=res.cost_micros_usd,
        )

    @staticmethod
    def _build_outbound_reply(
        msg: InboundMessage,
        text: str,
        *,
        sticker_id: str | None = None,
    ) -> OutboundReply | None:
        return _build_reply_for_inbound(msg, text, sticker_id=sticker_id)

    def _passes_pre_checks(self, msg: InboundMessage) -> bool:
        effective_bot_qq = self.bot_qq if self.bot_qq is not None else self.trigger.bot_qq
        if (effective_bot_qq is not None and msg.user_id == effective_bot_qq) or (
            msg.self_id is not None and msg.user_id == msg.self_id
        ):
            return False

        p_cfg = self.persona.current()
        self._sync_persona(p_cfg)  # type: ignore[attr-defined]

        if (
            msg.chat_type == "group"
            and self.trigger.allowed_group_ids
            and msg.group_id not in self.trigger.allowed_group_ids
        ):
            return False

        if self.memory is not None:
            self.memory.touch_user(msg.user_id, msg.sender_name)

        return True

    def _render_turn_system_prompt(
        self,
        *,
        session_key: str,
        chat_type: str,
        group_id: int | None,
        user_id: int,
        include_tool_guide: bool,
        query_text: str = "",
        sender_name: str = "",
    ) -> str:
        sys_prompt = self.persona.render_system_prompt(compact=self.compact_prompt)
        if self.memory is not None:
            is_admin_priv = chat_type == "private" and user_id in self._admin_user_ids
            mem_block = self.memory.render_context_block(
                session_key=session_key,
                chat_type=chat_type,
                group_id=group_id,
                user_id=user_id,
                is_admin_private=is_admin_priv,
                query_text=query_text,
            )
            if mem_block:
                sys_prompt = f"{sys_prompt}\n\n{mem_block}"

        if self.stickers is not None:
            sticker_guide = self.stickers.render_prompt_guide(compact=self.compact_prompt)
            if sticker_guide:
                sys_prompt = f"{sys_prompt}\n\n{sticker_guide}"

        if include_tool_guide:
            tool_guide = (
                _FAST_BRAIN_TOOL_GUIDE_COMPACT
                if self.compact_prompt
                else _FAST_BRAIN_TOOL_GUIDE
            )
            sys_prompt = f"{sys_prompt}\n\n{tool_guide}"

        inj_signals = detect_injection_signals(query_text, sender_name)
        if inj_signals:
            anchor = build_dynamic_injection_anchor(
                self.persona.current().name,
                signals=inj_signals,
                compact=self.compact_prompt,
            )
            sys_prompt = f"{sys_prompt}\n\n{anchor}"

        return sys_prompt

    def _next_subagent_task_id(self) -> str:
        self._subagent_seq_counter += 1
        return f"sub-{self._subagent_seq_counter}-{uuid.uuid4().hex[:6]}"

    async def _on_subagent_complete(self, res: SubAgentResult) -> None:
        if res.cancelled:
            if res.inbound_seq is not None:
                await self._ack_coordinator.mark_done(res.inbound_seq)
            return

        if (
            not res.is_error
            and res.brief.strip()
            and self.memory is not None
            and self.fact_cache_ttl_s > 0.0
        ):
            self.memory.record_shared_fact(
                res.goal,
                res.brief,
                ttl_s=self.fact_cache_ttl_s,
            )

        actor = self._actors.get(res.session_key)
        if actor is None or not actor.enqueue_subagent_result(res):
            if res.inbound_seq is not None:
                await self._ack_coordinator.mark_done(res.inbound_seq)

    async def _handle_subagent_result(
        self,
        res: SubAgentResult,
        cancel_token: CancelToken,
    ) -> None:
        """Wake Fast Brain (`AME`, `thinking=False`) to deliver background research brief naturally."""
        try:
            if res.cancelled or cancel_token.is_cancelled:
                return

            if res.session_key.startswith("group:"):
                chat_type = "group"
                target_id = int(res.session_key.split(":", 1)[1])
                group_id: int | None = target_id
            else:
                chat_type = "private"
                target_id = int(res.session_key.split(":", 1)[1])
                group_id = None

            sys_prompt = self._render_turn_system_prompt(
                session_key=res.session_key,
                chat_type=chat_type,
                group_id=group_id,
                user_id=res.origin_user_id,
                include_tool_guide=self.enable_tools,
                query_text=res.goal,
            )
            history = self.sessions.get_window(res.session_key)
            callback_user_text = _format_subagent_callback_prompt(res, chat_type)

            turn_state = ChatTurnState()
            bus_proxy = _RecordingBusProxy(self.bus)

            def _inject_default_quote(args: dict[str, Any]) -> dict[str, Any]:
                if (
                    chat_type == "group"
                    and res.reply_to_message_id is not None
                    and not turn_state.sent_bubbles
                    and args.get("reply_to_message_id") is None
                ):
                    args["reply_to_message_id"] = res.reply_to_message_id
                return args

            callback_tools: list[RegisteredTool] = []
            if self.enable_tools:
                callback_tools.append(
                    _bridge_tool_binding(
                        build_send_message_tool(
                            bus_proxy,  # type: ignore[arg-type]
                            turn_state,
                            chat_type=chat_type,  # type: ignore[arg-type]
                            target_id=target_id,
                            correlation_id=res.task_id,
                            stickers=self.stickers,
                            typing_delay_s=self.bubble_typing_delay_s,
                            compact_schema=self.compact_prompt,
                        ),
                        arg_preprocessor=_inject_default_quote,
                    )
                )

            def _set_cb_latency(ms: float) -> None:
                self._last_turn_latency_ms = ms

            try:
                end_event = await run_agent_loop_with_maintenance_retry(
                    provider=self.provider,
                    messages=[
                        ChatMessage(role="system", content=sys_prompt),
                        *history,
                        ChatMessage(role="user", content=callback_user_text),
                    ],
                    tools=callback_tools,
                    turn_state=turn_state,
                    retry_delays_s=self.hard_trigger_retry_delays_s,  # type: ignore[attr-defined]
                    allow_retry=True,
                    enable_thinking=False,
                    max_steps=1,
                    timeout_s=self.agent_turn_timeout_s,
                    cancel_token=cancel_token,
                    before_step=self._check_followup_step_budget,
                    after_step=self._record_step_usage,
                    on_latency_ms=_set_cb_latency,
                    session_label=f"subagent:{res.session_key}:{res.task_id}",
                )
            except ProviderError as exc:
                logger.error(
                    "Fast-brain subagent synthesis ProviderError on session=%s task=%s: %s",
                    res.session_key,
                    res.task_id,
                    exc,
                )
                return

            if end_event is None or end_event.cancelled or turn_state.stayed_silent:
                return

            if not turn_state.sent_bubbles and turn_state.last_sticker_id is None:
                raw_reply = end_event.final_text.strip()
                if not raw_reply:
                    return
                sticker_id: str | None = None
                if self.stickers is not None:
                    clean_reply, sticker_id = self.stickers.extract_from_reply(raw_reply)
                else:
                    clean_reply = raw_reply
                if not clean_reply and sticker_id is None:
                    return
                fallback_out = OutboundReply(
                    chat_type=chat_type,  # type: ignore[arg-type]
                    target_id=target_id,
                    text=clean_reply,
                    reply_to_message_id=res.reply_to_message_id if chat_type == "group" else None,
                    at_user_id=None,
                    correlation_id=res.task_id,
                    sticker_id=sticker_id,
                )
                await self.bus.publish_outbound(fallback_out)
                if clean_reply:
                    turn_state.sent_bubbles.append(clean_reply)
                turn_state.last_sticker_id = sticker_id

            if turn_state.sent_bubbles or turn_state.last_sticker_id is not None:
                hist_text = _format_sent_history(turn_state.sent_bubbles, turn_state.last_sticker_id)
                self._append_l1_turn(  # type: ignore[attr-defined]
                    res.session_key,
                    user_content=None,
                    assistant_content=hist_text,
                )
                if (
                    chat_type == "group"
                    and group_id is not None
                    and hasattr(self.trigger, "record_bot_reply")
                ):
                    self.trigger.record_bot_reply(group_id, res.origin_user_id or None)
                if chat_type == "group" and self.memory is not None:
                    self.memory.update_public_self_state(res.goal, chat_type="group")
        finally:
            if res.inbound_seq is not None:
                await self._ack_coordinator.mark_done(res.inbound_seq)

    async def _execute_triggered_turn(
        self,
        msg: InboundMessage,
        *,
        clean_text: str,
        trigger_reason: str = "direct",
        cancel_token: CancelToken | None = None,
        drain_steering: Callable[[], Sequence[str]] | None = None,
        pin_inbound_seq: int | None = None,
        publish_to_bus: bool = False,
    ) -> tuple[OutboundReply | None, bool]:
        """Run a Fast Brain Pi-Agent turn; returns `(last_outbound_reply, pinned_seq_transferred_to_subagent)`."""
        guard = self.guardrails.check(msg, now=self._clock())
        if not guard.allowed:
            if guard.refusal_reply:
                refusal_out = self._build_outbound_reply(msg, guard.refusal_reply)
                if publish_to_bus and refusal_out is not None:
                    await self.bus.publish_outbound(refusal_out)
                return refusal_out, False
            return None, False

        self._observe_group_public_cues(msg, clean_text)  # type: ignore[attr-defined]

        sys_prompt = self._render_turn_system_prompt(
            session_key=msg.session_key,
            chat_type=msg.chat_type,
            group_id=msg.group_id,
            user_id=msg.user_id,
            include_tool_guide=self.enable_tools,
            query_text=(f"{clean_text}\n{msg.reply_to_text}" if msg.reply_to_text else clean_text),
            sender_name=msg.sender_name,
        )
        sys_msg = ChatMessage(role="system", content=sys_prompt)
        history = self.sessions.get_window(msg.session_key)
        wrapped_user = self.persona.wrap_user_message(msg, clean_text=clean_text)
        ambient_xml = self._consume_ambient_context_xml(msg)  # type: ignore[attr-defined]
        live_user_content = f"{ambient_xml}\n{wrapped_user}" if ambient_xml else wrapped_user

        turn_state = ChatTurnState()
        self._active_turn_states[msg.session_key] = turn_state
        bus_proxy = _RecordingBusProxy(self.bus)
        pinned_seq_claimed = False
        spawned_count = 0

        try:
            vision_images = await self._fetch_turn_vision_images(msg)

            def _spawn_for_turn(
                goal: str,
                context_hint: str,
                reply_to_message_id: int | None,
            ) -> str:
                nonlocal pinned_seq_claimed, spawned_count
                self._get_or_create_actor(msg.session_key, msg.chat_type)  # type: ignore[attr-defined]
                seq_for_sub: int | None = (
                    pin_inbound_seq
                    if (pin_inbound_seq is not None and not pinned_seq_claimed)
                    else None
                )

                task_id = self._next_subagent_task_id()
                req = SubAgentRequest(
                    task_id=task_id,
                    session_key=msg.session_key,
                    goal=goal,
                    context_hint=context_hint,
                    origin_user_id=msg.user_id,
                    origin_sender_name=msg.sender_name,
                    reply_to_message_id=reply_to_message_id,
                    inbound_seq=seq_for_sub,
                )
                self._subagent_runner.spawn(req, on_complete=self._on_subagent_complete)
                if seq_for_sub is not None:
                    pinned_seq_claimed = True
                spawned_count += 1
                return task_id

            def _cancel_for_turn(task_id: str | None) -> int:
                return self._subagent_runner.cancel(task_id, session_key=msg.session_key)

            turn_tools = _build_turn_registered_tools(
                self,
                bus_proxy=bus_proxy,
                turn_state=turn_state,
                msg=msg,
                clean_text=clean_text,
                trigger_reason=trigger_reason,
                spawn_fn=_spawn_for_turn,
                cancel_fn=_cancel_for_turn,
            )

            def _before_step(step: int) -> str | None:
                if step == 0:
                    return guard.override_model
                return self._check_followup_step_budget(step)

            def _on_tool_event(event: ToolExecutionEvent) -> None:
                if (
                    event.call.name == "web_search"
                    and not event.outcome.is_error
                    and self.memory is not None
                    and self.fact_cache_ttl_s > 0.0
                ):
                    raw_q = str(event.call.arguments.get("query", "")).strip()
                    brief_txt = _summarize_web_search_outcome(event.outcome.content)
                    if raw_q and brief_txt:
                        self.memory.record_shared_fact(raw_q, brief_txt, ttl_s=self.fact_cache_ttl_s)

            def _set_turn_latency(ms: float) -> None:
                self._last_turn_latency_ms = ms

            try:
                end_event = await run_agent_loop_with_maintenance_retry(
                    provider=self.provider,
                    messages=[
                        sys_msg,
                        *history,
                        ChatMessage(role="user", content=live_user_content, images=vision_images),
                    ],
                    tools=turn_tools,
                    turn_state=turn_state,
                    retry_delays_s=self.hard_trigger_retry_delays_s,  # type: ignore[attr-defined]
                    allow_retry=(trigger_reason not in _SOFT_TRIGGER_REASONS),
                    model=guard.override_model,
                    max_steps=self.agent_max_steps if self.enable_tools else 1,
                    timeout_s=self.agent_turn_timeout_s,
                    cancel_token=cancel_token,
                    drain_steering=drain_steering,
                    before_step=_before_step,
                    after_step=self._record_step_usage,
                    on_tool_event=_on_tool_event,
                    on_latency_ms=_set_turn_latency,
                    session_label=f"{msg.session_key}:{msg.event_id}",
                )
            except ProviderError as exc:
                logger.error(
                    "LLM provider error on session=%s event_id=%s: %s",
                    msg.session_key,
                    msg.event_id,
                    exc,
                )
                if not turn_state.sent_bubbles and turn_state.last_sticker_id is None:
                    err_out = self._build_outbound_reply(msg, self.error_fallback_reply)
                    if publish_to_bus and err_out is not None:
                        await self.bus.publish_outbound(err_out)
                    return err_out, pinned_seq_claimed
                return (bus_proxy.published[-1] if bus_proxy.published else None), pinned_seq_claimed

            if end_event is None:
                return None, pinned_seq_claimed

            # 1) Soft-trigger intentional silence -> 0 outbound bubbles & 0 L1 session pollution
            if (
                turn_state.stayed_silent
                and not turn_state.sent_bubbles
                and turn_state.last_sticker_id is None
            ):
                self._record_ambient_group_line(msg, trigger_reason)  # type: ignore[attr-defined]
                return None, pinned_seq_claimed

            # 2) Cancelled mid-turn via `/stop`, `cancel_active` steering, or QoS hard preemption
            if end_event.cancelled:
                if turn_state.sent_bubbles or turn_state.last_sticker_id is not None:
                    hist_partial = _format_sent_history(
                        turn_state.sent_bubbles,
                        turn_state.last_sticker_id,
                    )
                    self._append_l1_turn(  # type: ignore[attr-defined]
                        msg.session_key,
                        user_content=wrapped_user,
                        assistant_content=hist_partial,
                    )
                elif trigger_reason in _SOFT_TRIGGER_REASONS:
                    self._record_ambient_group_line(msg, trigger_reason)  # type: ignore[attr-defined]
                return (bus_proxy.published[-1] if bus_proxy.published else None), pinned_seq_claimed

            # 3) Model streamed 1~3 bubbles via `send_message` tool
            if turn_state.sent_bubbles or turn_state.last_sticker_id is not None:
                history_assistant = _format_sent_history(
                    turn_state.sent_bubbles,
                    turn_state.last_sticker_id,
                )
                self._append_l1_turn(  # type: ignore[attr-defined]
                    msg.session_key,
                    user_content=wrapped_user,
                    assistant_content=history_assistant,
                )
                if msg.chat_type == "group" and hasattr(self.trigger, "record_bot_reply"):
                    self.trigger.record_bot_reply(msg.group_id, msg.user_id)
                if msg.chat_type == "group" and self.memory is not None:
                    self.memory.update_public_self_state(clean_text, chat_type=msg.chat_type)
                return (bus_proxy.published[-1] if bus_proxy.published else None), pinned_seq_claimed

            # 4) Hard quota hit on a follow-up hop before any bubble was sent
            if end_event.reason == "budget_blocked":
                refusal_text = end_event.final_text.strip() or self.guardrails.budget_refusal
                budget_out = self._build_outbound_reply(msg, refusal_text)
                if publish_to_bus and budget_out is not None:
                    await self.bus.publish_outbound(budget_out)
                return budget_out, pinned_seq_claimed

            raw_final = end_event.final_text.strip()

            # 5) Plain-text silent token on soft trigger
            if (
                trigger_reason in _SOFT_TRIGGER_REASONS
                and raw_final.lower() in _SILENT_FALLBACK_TOKENS
            ):
                self._record_ambient_group_line(msg, trigger_reason)  # type: ignore[attr-defined]
                return None, pinned_seq_claimed

            # 6) Model spawned a background sub-agent without sending an upfront padding bubble or text
            if spawned_count > 0 and not raw_final:
                self._append_l1_turn(  # type: ignore[attr-defined]
                    msg.session_key,
                    user_content=wrapped_user,
                    assistant_content="（正在后台翻资料…）",
                )
                return None, pinned_seq_claimed

            # 7) Plain-text fallback safety net (keeps M1/M2 mock tests & plain-text fast-model turns 100% intact)
            sticker_id: str | None = None
            if self.stickers is not None:
                clean_reply, sticker_id = self.stickers.extract_from_reply(end_event.final_text)
            else:
                clean_reply = raw_final

            if not clean_reply and sticker_id is None:
                if end_event.reason == "timeout":
                    err_out = self._build_outbound_reply(msg, self.error_fallback_reply)
                    if publish_to_bus and err_out is not None:
                        await self.bus.publish_outbound(err_out)
                    return err_out, pinned_seq_claimed
                clean_reply = raw_final or self.error_fallback_reply

            history_assistant = _format_sent_history(
                [clean_reply] if clean_reply else [],
                sticker_id,
            )
            self._append_l1_turn(  # type: ignore[attr-defined]
                msg.session_key,
                user_content=wrapped_user,
                assistant_content=history_assistant,
            )
            if msg.chat_type == "group" and hasattr(self.trigger, "record_bot_reply"):
                self.trigger.record_bot_reply(msg.group_id, msg.user_id)
            if msg.chat_type == "group" and self.memory is not None:
                self.memory.update_public_self_state(clean_text, chat_type=msg.chat_type)

            fallback_reply = self._build_outbound_reply(msg, clean_reply, sticker_id=sticker_id)
            if publish_to_bus and fallback_reply is not None:
                await self.bus.publish_outbound(fallback_reply)
            return fallback_reply, pinned_seq_claimed
        finally:
            if self._active_turn_states.get(msg.session_key) is turn_state:
                self._active_turn_states.pop(msg.session_key, None)


__all__ = ["KernelExecutionMixin"]
