"""Maintenance-window delayed-reply retry, graceful brain reload drain, and un-ACKed hard-trigger preservation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import logging
from pathlib import Path

from .kernel_turn_helpers import _SOFT_TRIGGER_REASONS, _TurnEnvelope
from .pi_core import (
    CancelToken,
    RegisteredTool,
    SessionActor,
    ToolExecutionEvent,
    TurnEndEvent,
    agent_loop,
)
from .provider import LLMProvider, ProviderError
from .tools import ChatTurnState
from .types import ChatMessage, ChatResult

logger = logging.getLogger(__name__)

_NON_RETRYABLE_PROVIDER_STATUS_CODES: frozenset[int] = frozenset(
    {400, 401, 403, 404, 413, 422}
)


def extract_soft_orphaned_seqs_from_actor(actor: SessionActor) -> list[int]:
    """Collect ONLY soft-trigger queued sequence IDs on shutdown so hard `@` triggers stay un-ACKed for delayed replay."""
    orphaned: list[int] = []
    for item in actor._debounce_items:
        if isinstance(item, _TurnEnvelope) and item.trigger_reason in _SOFT_TRIGGER_REASONS:
            orphaned.extend(item.seqs)
    for item in actor._followup_queue:
        if isinstance(item, _TurnEnvelope) and item.trigger_reason in _SOFT_TRIGGER_REASONS:
            orphaned.extend(item.seqs)
    return orphaned


def consume_reload_sentinel(path: Path | None) -> bool:
    """Atomically detect and unlink `brain.reload` sentinel file if present."""
    if path is None or not path.exists():
        return False
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return False
    return True


def request_brain_reload_sentinel(sentinel_path: Path) -> int:
    """Write the `brain.reload` sentinel file for `--reload-brain` CLI and print operator notice."""
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    sentinel_path.write_text("reload\n", encoding="utf-8")
    print(
        f"[qq-agent] 已发送 brain 无感排空热重载信号：{sentinel_path}\n"
        "在途对话将排空回复后优雅退出，维护窗口内收到的新 @ 消息保留在 llmos 队列并延迟回复。",
        flush=True,
    )
    return 0


async def drain_kernel_for_graceful_reload(kernel: object) -> None:
    """Stop polling new inbound messages and finish all in-flight turns before clean hot-reload exit."""
    logger.info("Graceful brain reload sentinel detected; draining active turns before exit...")
    flush_fn = getattr(kernel, "_flush_ready_debounce_buckets", None)
    if callable(flush_fn):
        await flush_fn(force_all=True)
    wait_fn = getattr(kernel, "wait_idle", None)
    timeout_s = float(getattr(kernel, "agent_turn_timeout_s", 35.0))
    if callable(wait_fn):
        try:
            await wait_fn(timeout=timeout_s)
        except TimeoutError:
            pass


def is_retryable_maintenance_error(exc: ProviderError) -> bool:
    """Return True when `ProviderError` reflects transient upstream maintenance/network/5xx/429 unavailability."""
    code = exc.status_code
    if code is None:
        return True
    if code in _NON_RETRYABLE_PROVIDER_STATUS_CODES:
        return False
    return code in {408, 409, 425, 429} or code >= 500


async def sleep_with_cancel(
    delay_s: float,
    cancel_token: CancelToken | None,
) -> bool:
    """Sleep `delay_s` waking immediately on `cancel_token`; return `True` if turn remains active."""
    if cancel_token is not None and cancel_token.is_cancelled:
        return False
    if delay_s <= 0.0:
        await asyncio.sleep(0)
        return not (cancel_token is not None and cancel_token.is_cancelled)
    if cancel_token is None:
        await asyncio.sleep(delay_s)
        return True
    try:
        await asyncio.wait_for(cancel_token.event.wait(), timeout=delay_s)
        return False
    except TimeoutError:
        return not cancel_token.is_cancelled


async def run_agent_loop_with_maintenance_retry(
    *,
    provider: LLMProvider,
    messages: list[ChatMessage],
    tools: Sequence[RegisteredTool],
    turn_state: ChatTurnState,
    retry_delays_s: Sequence[float] = (),
    allow_retry: bool = False,
    model: str | None = None,
    enable_thinking: bool | None = None,
    max_steps: int = 1,
    timeout_s: float = 35.0,
    cancel_token: CancelToken | None = None,
    drain_steering: Callable[[], Sequence[str]] | None = None,
    before_step: Callable[[int], str | None] | None = None,
    after_step: Callable[[ChatResult], None] | None = None,
    on_tool_event: Callable[[ToolExecutionEvent], None] | None = None,
    on_latency_ms: Callable[[float], None] | None = None,
    session_label: str = "",
) -> TurnEndEvent | None:
    """Execute `agent_loop` with delayed backoff retry for hard triggers hitting upstream maintenance windows."""
    delays = tuple(max(0.0, float(d)) for d in retry_delays_s) if allow_retry else ()
    attempt = 0

    while True:
        end_event: TurnEndEvent | None = None
        try:
            async for event in agent_loop(
                provider,
                messages,
                tools=tools,
                model=model,
                enable_thinking=enable_thinking,
                max_steps=max_steps,
                timeout_s=timeout_s,
                cancel_token=cancel_token,
                drain_steering=drain_steering,
                before_step=before_step,
                after_step=after_step,
            ):
                if isinstance(event, ToolExecutionEvent):
                    if on_tool_event is not None:
                        on_tool_event(event)
                elif isinstance(event, TurnEndEvent):
                    end_event = event
                    if event.latency_ms > 0.0 and on_latency_ms is not None:
                        on_latency_ms(event.latency_ms)
        except ProviderError as exc:
            can_retry = (
                attempt < len(delays)
                and is_retryable_maintenance_error(exc)
                and not turn_state.sent_bubbles
                and turn_state.last_sticker_id is None
                and not (cancel_token is not None and cancel_token.is_cancelled)
            )
            if not can_retry:
                raise
            wait_s = delays[attempt]
            attempt += 1
            logger.warning(
                "Hard-trigger hit maintenance/provider error on %s (retry %d/%d in %.2fs): %s",
                session_label or "turn",
                attempt,
                len(delays),
                wait_s,
                exc,
            )
            if not await sleep_with_cancel(wait_s, cancel_token):
                reason = (cancel_token.reason if cancel_token is not None else "") or "cancelled"
                return TurnEndEvent(
                    reason=reason,
                    final_text="",
                    steps_used=0,
                    total_tokens=0,
                    total_cost_micros=0,
                    cancelled=True,
                    latency_ms=0.0,
                )
            continue

        if (
            end_event is not None
            and end_event.reason == "timeout"
            and attempt < len(delays)
            and not turn_state.sent_bubbles
            and turn_state.last_sticker_id is None
            and not end_event.final_text.strip()
            and not (cancel_token is not None and cancel_token.is_cancelled)
        ):
            wait_s = delays[attempt]
            attempt += 1
            logger.warning(
                "Hard-trigger hit upstream timeout on %s (retry %d/%d in %.2fs)",
                session_label or "turn",
                attempt,
                len(delays),
                wait_s,
            )
            if not await sleep_with_cancel(wait_s, cancel_token):
                reason = (cancel_token.reason if cancel_token is not None else "") or "cancelled"
                return TurnEndEvent(
                    reason=reason,
                    final_text="",
                    steps_used=0,
                    total_tokens=0,
                    total_cost_micros=0,
                    cancelled=True,
                    latency_ms=0.0,
                )
            continue

        return end_event


__all__ = [
    "consume_reload_sentinel",
    "drain_kernel_for_graceful_reload",
    "extract_soft_orphaned_seqs_from_actor",
    "is_retryable_maintenance_error",
    "request_brain_reload_sentinel",
    "run_agent_loop_with_maintenance_retry",
    "sleep_with_cancel",
]
