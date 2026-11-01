"""Per-session resident async SessionActor orchestrating debounce, QoS preemption, steer, followUp, and SubAgent wakeups."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
import inspect
import logging
import time
from typing import Any, Literal

from qq_agent.pi_ambient import (
    AmbientBuffer,
    _extract_item_text,
    _extract_item_user_id,
    default_coalesce_items,
)
from qq_agent.pi_loop import CancelToken
from qq_agent.subagent_runner import SubAgentRunner
from qq_agent.types import SubAgentResult

logger = logging.getLogger(__name__)

ActorState = Literal["IDLE", "RUNNING", "CLOSED"]
TurnPriority = Literal["hard", "soft"]
SubmitRoute = Literal["debouncing", "steered", "followup", "dropped_soft"]
ActorTurnHandler = Callable[..., Awaitable[Any]]
CoalesceFn = Callable[[list[Any]], Any]
PreemptPredicate = Callable[[], bool]
ItemDroppedCallback = Callable[[Any], None]


def _resolve_handler_arity(fn: Callable[..., Any]) -> int:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return 3
    positional = 0
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return 3
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional += 1
    return positional


def _normalize_priority(priority: TurnPriority) -> TurnPriority:
    if priority == "soft":
        return "soft"
    if priority == "hard":
        return "hard"
    raise ValueError(f"Unsupported TurnPriority: {priority!r}")


class SessionActor:
    """Per-session resident async actor orchestrating debounce, QoS preemption, steer, followUp, and SubAgent wakeups."""

    def __init__(
        self,
        session_key: str,
        turn_handler: ActorTurnHandler | None = None,
        *,
        subagent_result_handler: ActorTurnHandler | None = None,
        subagent_runner: SubAgentRunner | None = None,
        debounce_s: float = 0.0,
        max_debounce_wait_s: float = 6.0,
        max_debounce_batch: int = 8,
        ambient_maxlen: int = 25,
        coalesce_fn: CoalesceFn | None = None,
        can_preempt_active_fn: PreemptPredicate | None = None,
        on_item_dropped: ItemDroppedCallback | None = None,
    ) -> None:
        self.session_key = session_key
        self.ambient = AmbientBuffer(maxlen=ambient_maxlen)

        self._turn_handler = turn_handler
        self._turn_arity = _resolve_handler_arity(turn_handler) if turn_handler is not None else 0
        self._subagent_result_handler = subagent_result_handler
        self._subagent_arity = (
            _resolve_handler_arity(subagent_result_handler)
            if subagent_result_handler is not None
            else 0
        )
        self._subagent_runner = subagent_runner
        self._debounce_s = max(0.0, float(debounce_s))
        self._max_debounce_wait_s = max(0.05, float(max_debounce_wait_s))
        self._max_debounce_batch = max(1, int(max_debounce_batch))
        self._coalesce_fn = coalesce_fn if coalesce_fn is not None else default_coalesce_items
        self._can_preempt_active_fn = can_preempt_active_fn
        self._on_item_dropped = on_item_dropped

        self._state: ActorState = "IDLE"
        self._closed = False
        self._active_cancel_token: CancelToken | None = None
        self._active_turn_user_id: int | None = None
        self._active_turn_priority: TurnPriority = "hard"
        self._preempted_count: int = 0

        self._steering_queue: deque[str] = deque()
        self._followup_queue: deque[Any] = deque()
        self._followup_priorities: deque[TurnPriority] = deque()
        self._subagent_queue: deque[SubAgentResult] = deque()

        self._debounce_items: list[Any] = []
        self._debounce_user_id: int | None = None
        self._debounce_priority: TurnPriority = "hard"
        self._debounce_first_ts: float = 0.0
        self._debounce_last_ts: float = 0.0

        self._wakeup_event = asyncio.Event()
        self._debounce_bump_event = asyncio.Event()
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        self._worker_task: asyncio.Task[None] | None = None

    @property
    def state(self) -> ActorState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == "RUNNING"

    @property
    def active_turn_user_id(self) -> int | None:
        return self._active_turn_user_id

    @property
    def active_cancel_token(self) -> CancelToken | None:
        return self._active_cancel_token

    @property
    def active_turn_priority(self) -> TurnPriority:
        return self._active_turn_priority

    @property
    def preempted_count(self) -> int:
        return self._preempted_count

    @property
    def debounce_s(self) -> float:
        return self._debounce_s

    @debounce_s.setter
    def debounce_s(self, value: float) -> None:
        self._debounce_s = max(0.0, float(value))

    def _notify_dropped(self, item: Any) -> None:
        if self._on_item_dropped is None:
            return
        try:
            self._on_item_dropped(item)
        except Exception:
            logger.exception(
                "Unhandled error in SessionActor(%s) on_item_dropped callback",
                self.session_key,
            )

    def _has_hard_followup(self) -> bool:
        if not self._followup_queue:
            return False
        if len(self._followup_priorities) < len(self._followup_queue):
            return True
        return any(p == "hard" for p in self._followup_priorities)

    def _purge_soft_followups(self) -> int:
        if not self._followup_queue:
            self._followup_priorities.clear()
            return 0
        items_list = list(self._followup_queue)
        prios_list = list(self._followup_priorities)
        kept_items: list[Any] = []
        kept_prios: list[TurnPriority] = []
        dropped_items: list[Any] = []
        for idx, q_item in enumerate(items_list):
            q_prio: TurnPriority = prios_list[idx] if idx < len(prios_list) else "hard"
            if q_prio == "soft":
                dropped_items.append(q_item)
            else:
                kept_items.append(q_item)
                kept_prios.append("hard")
        self._followup_queue.clear()
        self._followup_priorities.clear()
        self._followup_queue.extend(kept_items)
        self._followup_priorities.extend(kept_prios)
        for d_item in dropped_items:
            self._notify_dropped(d_item)
        return len(dropped_items)

    def start(self) -> None:
        """Ensure the resident actor loop task is running on the active event loop."""
        if self._closed:
            return
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._run_loop(),
                name=f"session-actor:{self.session_key}",
            )

    def record_ambient(
        self,
        sender_name: str,
        user_id: int,
        text: str,
        timestamp: float = 0.0,
    ) -> bool:
        """Record a non-triggered group message into the session's `AmbientBuffer`."""
        return self.ambient.record(sender_name, user_id, text, timestamp)

    def steer(self, line: str, *, cancel_active: bool = False) -> None:
        """Push a live mid-turn steering update; optionally cancel the active `CancelToken`."""
        stripped = line.strip()
        if stripped:
            self._steering_queue.append(stripped)
        if cancel_active and self._active_cancel_token is not None:
            self._active_cancel_token.cancel(reason="steered_cancel")

    def drain_steering(self) -> list[str]:
        """Atomically drain all pending mid-turn steering lines."""
        if not self._steering_queue:
            return []
        drained = list(self._steering_queue)
        self._steering_queue.clear()
        return drained

    def enqueue_followup(self, item: Any, *, priority: TurnPriority = "hard") -> None:
        """Queue a follow-up turn item to run seamlessly right after the active turn finishes."""
        if self._closed:
            self._notify_dropped(item)
            return
        prio = _normalize_priority(priority)
        if prio == "soft":
            if (
                self._state == "RUNNING"
                or (self._debounce_items and self._debounce_priority == "hard")
                or self._has_hard_followup()
            ):
                self._notify_dropped(item)
                return
        else:
            self._purge_soft_followups()
        self._followup_queue.append(item)
        self._followup_priorities.append(prio)
        self._idle_event.clear()
        self.start()
        self._wakeup_event.set()

    def enqueue_subagent_result(self, res: SubAgentResult) -> bool:
        """Push a non-cancelled `SubAgentResult` onto the high-priority callback queue and wake Actor."""
        if self._closed or res.cancelled:
            return False
        self._subagent_queue.append(res)
        self._idle_event.clear()
        self.start()
        self._wakeup_event.set()
        return True

    def submit(
        self,
        item: Any,
        *,
        user_id: int | None = None,
        steer_if_same_user: bool = False,
        steer_text: str | None = None,
        cancel_active: bool = False,
        priority: TurnPriority = "hard",
    ) -> SubmitRoute:
        """Route an inbound triggered item across IDLE debounce coalescing vs RUNNING QoS preemption/steer/followUp."""
        if self._closed:
            raise RuntimeError(f"SessionActor({self.session_key}) is closed")

        prio = _normalize_priority(priority)
        speaker_id = _extract_item_user_id(item, user_id)

        if self._state == "RUNNING":
            if prio == "soft":
                self._notify_dropped(item)
                return "dropped_soft"

            can_preempt = (
                self._can_preempt_active_fn is None
                or bool(self._can_preempt_active_fn())
            )
            already_preempting = (
                self._active_cancel_token is not None
                and self._active_cancel_token.is_cancelled
                and self._active_cancel_token.reason == "preempted_by_hard_trigger"
            )
            if self._active_turn_priority == "soft" and (can_preempt or already_preempting):
                if (
                    self._active_cancel_token is not None
                    and not self._active_cancel_token.is_cancelled
                ):
                    self._active_cancel_token.cancel(reason="preempted_by_hard_trigger")
                    self._preempted_count += 1
                self._steering_queue.clear()
                self._purge_soft_followups()
                self.enqueue_followup(item, priority="hard")
                return "followup"

            if cancel_active or (
                steer_if_same_user
                and speaker_id is not None
                and speaker_id == self._active_turn_user_id
            ):
                self._purge_soft_followups()
                line = steer_text if steer_text is not None else _extract_item_text(item)
                self.steer(line, cancel_active=cancel_active)
                return "steered"

            self._purge_soft_followups()
            self.enqueue_followup(item, priority="hard")
            return "followup"

        if prio == "soft" and self._has_hard_followup():
            self._notify_dropped(item)
            return "dropped_soft"

        now = time.monotonic()
        if not self._debounce_items:
            if prio == "hard":
                self._purge_soft_followups()
            self._debounce_items.append(item)
            self._debounce_user_id = speaker_id
            self._debounce_priority = prio
            self._debounce_first_ts = now
            self._debounce_last_ts = now
            self._idle_event.clear()
            self.start()
            self._wakeup_event.set()
            return "debouncing"

        same_known_speaker = (
            speaker_id is not None and speaker_id == self._debounce_user_id
        )

        if self._debounce_priority == "soft" and prio == "hard":
            if same_known_speaker:
                self._purge_soft_followups()
                self._debounce_items.append(item)
                self._debounce_priority = "hard"
                self._debounce_last_ts = now
                self._debounce_bump_event.set()
                self._wakeup_event.set()
                return "debouncing"

            evicted = list(self._debounce_items)
            self._debounce_items.clear()
            self._debounce_items.append(item)
            self._debounce_user_id = speaker_id
            self._debounce_priority = "hard"
            self._debounce_first_ts = now
            self._debounce_last_ts = now
            self._idle_event.clear()
            self.start()
            self._debounce_bump_event.set()
            self._wakeup_event.set()
            for ev_item in evicted:
                self._notify_dropped(ev_item)
            self._purge_soft_followups()
            return "debouncing"

        if self._debounce_priority == "hard" and prio == "soft":
            if not same_known_speaker:
                self._notify_dropped(item)
                return "dropped_soft"
            self._debounce_items.append(item)
            self._debounce_last_ts = now
            self._debounce_bump_event.set()
            return "debouncing"

        if speaker_id == self._debounce_user_id:
            self._debounce_items.append(item)
            self._debounce_last_ts = now
            self._debounce_bump_event.set()
            return "debouncing"

        self.enqueue_followup(item, priority=prio)
        return "followup"

    def stop(self, reason: str = "stopped") -> tuple[bool, int]:
        """Cancel active turn `CancelToken`, purge pending queues, and cancel session sub-agents."""
        cancelled_main = False
        if (
            self._active_cancel_token is not None
            and not self._active_cancel_token.is_cancelled
        ):
            self._active_cancel_token.cancel(reason=reason)
            cancelled_main = True

        dropped_debounce = list(self._debounce_items)
        self._debounce_items.clear()
        self._debounce_user_id = None
        self._debounce_priority = "hard"
        self._debounce_bump_event.set()

        dropped_followup = list(self._followup_queue)
        self._followup_queue.clear()
        self._followup_priorities.clear()
        self._steering_queue.clear()
        self._subagent_queue.clear()

        for d_item in dropped_debounce:
            self._notify_dropped(d_item)
        for d_item in dropped_followup:
            self._notify_dropped(d_item)

        cancelled_subagents = 0
        if self._subagent_runner is not None:
            cancelled_subagents = self._subagent_runner.cancel(
                session_key=self.session_key
            )

        if self._state == "IDLE":
            self._idle_event.set()
        return cancelled_main, cancelled_subagents

    async def wait_idle(self, timeout: float | None = None) -> None:
        """Wait until the actor finishes all active, debouncing, follow-up, and callback work."""
        if timeout is not None:
            await asyncio.wait_for(self._idle_event.wait(), timeout=timeout)
        else:
            await self._idle_event.wait()

    async def aclose(self) -> None:
        """Stop all work and shut down the resident actor task."""
        self._closed = True
        self.stop(reason="actor_closed")
        self._state = "CLOSED"
        self._wakeup_event.set()
        self._debounce_bump_event.set()
        self._idle_event.set()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            await asyncio.gather(self._worker_task, return_exceptions=True)
        self._worker_task = None

    async def _wait_and_pop_debounced_turn(self) -> tuple[Any, TurnPriority] | None:
        if self._debounce_s > 0.0:
            while self._debounce_items and not self._closed:
                if len(self._debounce_items) >= self._max_debounce_batch:
                    break
                now = time.monotonic()
                rem_burst = self._debounce_s - (now - self._debounce_last_ts)
                rem_max = self._max_debounce_wait_s - (now - self._debounce_first_ts)
                sleep_s = min(rem_burst, rem_max)
                if sleep_s <= 0:
                    break
                self._debounce_bump_event.clear()
                try:
                    await asyncio.wait_for(self._debounce_bump_event.wait(), timeout=sleep_s)
                except TimeoutError:
                    pass

        if not self._debounce_items or self._closed:
            if self._debounce_items:
                leftover = list(self._debounce_items)
                self._debounce_items.clear()
                self._debounce_user_id = None
                self._debounce_priority = "hard"
                for item in leftover:
                    self._notify_dropped(item)
            else:
                self._debounce_user_id = None
                self._debounce_priority = "hard"
            return None

        batch = list(self._debounce_items)
        turn_priority = self._debounce_priority
        self._debounce_items.clear()
        self._debounce_user_id = None
        self._debounce_priority = "hard"
        return self._coalesce_fn(batch), turn_priority

    async def _invoke_handler(
        self,
        fn: ActorTurnHandler,
        arity: int,
        payload: Any,
        token: CancelToken,
    ) -> None:
        if arity <= 1:
            await fn(payload)
        elif arity == 2:
            await fn(payload, token)
        else:
            await fn(payload, token, self.drain_steering)

    async def _execute_main_turn(
        self,
        item: Any,
        *,
        turn_priority: TurnPriority = "hard",
    ) -> None:
        self._state = "RUNNING"
        self._active_turn_priority = turn_priority
        self._active_turn_user_id = _extract_item_user_id(item)
        self._steering_queue.clear()
        token = CancelToken()
        self._active_cancel_token = token
        try:
            if self._turn_handler is not None:
                await self._invoke_handler(self._turn_handler, self._turn_arity, item, token)
        except asyncio.CancelledError:
            if self._closed:
                raise
        except Exception:
            logger.exception("Unhandled error in SessionActor(%s) main turn", self.session_key)
        finally:
            self._active_cancel_token = None
            self._active_turn_user_id = None
            self._active_turn_priority = "hard"
            if not self._closed:
                self._state = "IDLE"

    async def _execute_subagent_callback(self, res: SubAgentResult) -> None:
        if res.cancelled or self._closed:
            return
        self._state = "RUNNING"
        self._active_turn_priority = "hard"
        self._active_turn_user_id = res.origin_user_id or None
        self._steering_queue.clear()
        token = CancelToken()
        self._active_cancel_token = token
        try:
            if self._subagent_result_handler is not None:
                await self._invoke_handler(
                    self._subagent_result_handler,
                    self._subagent_arity,
                    res,
                    token,
                )
        except asyncio.CancelledError:
            if self._closed:
                raise
        except Exception:
            logger.exception(
                "Unhandled error in SessionActor(%s) subagent callback",
                self.session_key,
            )
        finally:
            self._active_cancel_token = None
            self._active_turn_user_id = None
            self._active_turn_priority = "hard"
            if not self._closed:
                self._state = "IDLE"

    async def _run_loop(self) -> None:
        try:
            while not self._closed:
                if self._subagent_queue:
                    sub_res = self._subagent_queue.popleft()
                    await self._execute_subagent_callback(sub_res)
                    continue

                if self._debounce_items:
                    popped = await self._wait_and_pop_debounced_turn()
                    if popped is not None:
                        turn_item, turn_prio = popped
                        await self._execute_main_turn(turn_item, turn_priority=turn_prio)
                    continue

                if self._followup_queue:
                    next_item = self._followup_queue.popleft()
                    next_prio: TurnPriority = (
                        self._followup_priorities.popleft()
                        if self._followup_priorities
                        else "hard"
                    )
                    await self._execute_main_turn(next_item, turn_priority=next_prio)
                    continue

                self._state = "IDLE"
                self._wakeup_event.clear()
                if (
                    self._subagent_queue
                    or self._debounce_items
                    or self._followup_queue
                    or self._closed
                ):
                    continue
                self._idle_event.set()
                await self._wakeup_event.wait()
        except asyncio.CancelledError:
            return


__all__ = [
    "ActorState",
    "ActorTurnHandler",
    "CoalesceFn",
    "ItemDroppedCallback",
    "PreemptPredicate",
    "SessionActor",
    "SubmitRoute",
    "TurnPriority",
]
