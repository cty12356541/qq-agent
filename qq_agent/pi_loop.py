"""Python-native Pi-Agent async event-stream loop, CancelToken, tool execution, and step hooks."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import time
from typing import Any, TypeVar

from qq_agent.provider import LLMProvider
from qq_agent.types import (
    ChatMessage,
    ChatResult,
    ToolCall,
    ToolDef,
    ToolOutcome,
)

_T = TypeVar("_T")

ToolHandler = Callable[[dict[str, Any], "CancelToken"], Awaitable[ToolOutcome | str]]
BeforeStepHook = Callable[[int], str | None | Awaitable[str | None]]
AfterStepHook = Callable[[ChatResult], Any]


class BudgetBlockedError(RuntimeError):
    """Raised by `before_step` guardrail hook when hard quota is exhausted before an LLM hop."""

    def __init__(
        self,
        refusal_reply: str | None = None,
        reason: str = "daily_hard_limit",
    ) -> None:
        super().__init__(refusal_reply or reason)
        self.refusal_reply = refusal_reply
        self.reason = reason


class CancelToken:
    """Cooperative + preemptive task cancellation token for in-flight LLM and tool I/O."""

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason: str = ""
        self._bound_tasks: set[asyncio.Task[Any]] = set()

    @property
    def event(self) -> asyncio.Event:
        return self._event

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        """Signal cancellation and synchronously `.cancel()` every bound in-flight `asyncio.Task`."""
        if not self._event.is_set():
            self._reason = reason or "cancelled"
            self._event.set()
        tasks = list(self._bound_tasks)
        self._bound_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()

    def bind_task(self, task: asyncio.Task[_T]) -> asyncio.Task[_T]:
        """Bind an in-flight `asyncio.Task` so `cancel()` interrupts it immediately."""
        if self._event.is_set():
            if not task.done():
                task.cancel()
            return task
        if not task.done():
            self._bound_tasks.add(task)
            task.add_done_callback(self._bound_tasks.discard)
        return task

    def unbind_task(self, task: asyncio.Task[Any]) -> None:
        self._bound_tasks.discard(task)

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise asyncio.CancelledError(self._reason or "cancelled")


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """Client-side tool binding pairing a schema `ToolDef` with an async execution handler."""

    definition: ToolDef
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class TurnStartEvent:
    """Emitted once when `agent_loop` begins a turn."""

    max_steps: int = 0


@dataclass(frozen=True, slots=True)
class LlmStepEvent:
    """Emitted after each `provider.chat` hop completes."""

    step: int
    result: ChatResult


@dataclass(frozen=True, slots=True)
class ToolExecutionEvent:
    """Emitted for each tool call outcome (in original model call order) within a step."""

    step: int
    call: ToolCall
    outcome: ToolOutcome


@dataclass(frozen=True, slots=True)
class SteeringInjectedEvent:
    """Emitted at a tool boundary when live `steer()` updates are injected before the next hop."""

    step: int
    texts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnEndEvent:
    """Terminal event emitted once when `agent_loop` concludes, short-circuits, times out, or cancels."""

    reason: str
    final_text: str
    steps_used: int = 0
    total_tokens: int = 0
    total_cost_micros: int = 0
    cancelled: bool = False
    latency_ms: float = 0.0


AgentEvent = (
    TurnStartEvent
    | LlmStepEvent
    | ToolExecutionEvent
    | SteeringInjectedEvent
    | TurnEndEvent
)


async def execute_tool_call(
    call: ToolCall,
    registry: Mapping[str, RegisteredTool],
    cancel_token: CancelToken,
    *,
    deadline: float | None = None,
) -> ToolOutcome:
    """Execute one `ToolCall` safely, wrapping str/exceptions into `ToolOutcome` while honoring `CancelToken`."""
    cancel_token.raise_if_cancelled()

    reg = registry.get(call.name)
    if reg is None:
        return ToolOutcome(
            tool_call_id=call.id,
            name=call.name,
            content=f"Unknown tool: {call.name}",
            is_error=True,
        )

    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("agent turn timeout exceeded before tool execution")
    else:
        remaining = None

    task: asyncio.Task[ToolOutcome | str] = asyncio.create_task(
        reg.handler(dict(call.arguments), cancel_token)
    )
    cancel_token.bind_task(task)

    try:
        if remaining is not None:
            raw_out = await asyncio.wait_for(task, timeout=remaining)
        else:
            raw_out = await task
    except TimeoutError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        raise
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        cur = asyncio.current_task()
        outer_cancelling = cur is not None and getattr(cur, "cancelling", lambda: 0)() > 0
        if cancel_token.is_cancelled or outer_cancelling:
            raise
        return ToolOutcome(
            tool_call_id=call.id,
            name=call.name,
            content="Tool execution cancelled.",
            is_error=True,
        )
    except Exception as exc:
        return ToolOutcome(
            tool_call_id=call.id,
            name=call.name,
            content=f"Tool error ({type(exc).__name__}): {exc}",
            is_error=True,
        )

    if isinstance(raw_out, ToolOutcome):
        return ToolOutcome(
            tool_call_id=call.id,
            name=raw_out.name or call.name,
            content=raw_out.content,
            is_error=raw_out.is_error,
        )
    return ToolOutcome(
        tool_call_id=call.id,
        name=call.name,
        content=str(raw_out),
        is_error=False,
    )


def _provider_accepts_kwarg(provider: LLMProvider, kw_name: str) -> bool:
    try:
        sig = inspect.signature(provider.chat)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kw_name:
            return True
    return False


async def _call_provider_chat(
    provider: LLMProvider,
    messages: list[ChatMessage],
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
    tools: Sequence[ToolDef] | None,
    enable_thinking: bool | None,
) -> ChatResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if _provider_accepts_kwarg(provider, "tools"):
        kwargs["tools"] = tools
    if _provider_accepts_kwarg(provider, "enable_thinking"):
        kwargs["enable_thinking"] = enable_thinking
    return await provider.chat(messages, **kwargs)


async def agent_loop(
    provider: LLMProvider,
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[RegisteredTool] = (),
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    enable_thinking: bool | None = None,
    max_steps: int = 3,
    timeout_s: float = 35.0,
    cancel_token: CancelToken | None = None,
    drain_steering: Callable[[], Sequence[str]] | None = None,
    before_step: BeforeStepHook | None = None,
    after_step: AfterStepHook | None = None,
) -> AsyncIterator[AgentEvent]:
    """Pi-Agent async generator loop with terminal short-circuit, parallel tools, and mid-turn steering."""
    loop_start = time.perf_counter()
    effective_max_steps = max(1, int(max_steps))
    token = cancel_token if cancel_token is not None else CancelToken()
    loop_messages: list[ChatMessage] = list(messages)
    tool_map: dict[str, RegisteredTool] = {t.definition.name: t for t in tools}
    all_tool_defs: list[ToolDef] = [t.definition for t in tools]
    deadline: float | None = (
        time.monotonic() + float(timeout_s) if timeout_s > 0 else None
    )

    steps_used = 0
    total_tokens = 0
    total_cost_micros = 0
    last_text = ""

    def _elapsed_ms() -> float:
        return max(0.0, (time.perf_counter() - loop_start) * 1000.0)

    yield TurnStartEvent(max_steps=effective_max_steps)

    if token.is_cancelled:
        yield TurnEndEvent(
            reason=token.reason or "cancelled",
            final_text="",
            steps_used=0,
            total_tokens=0,
            total_cost_micros=0,
            cancelled=True,
            latency_ms=_elapsed_ms(),
        )
        return

    for step in range(effective_max_steps):
        if token.is_cancelled:
            yield TurnEndEvent(
                reason=token.reason or "cancelled",
                final_text=last_text,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=True,
                latency_ms=_elapsed_ms(),
            )
            return

        if deadline is not None and (deadline - time.monotonic()) <= 0:
            yield TurnEndEvent(
                reason="timeout",
                final_text=last_text,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=False,
                latency_ms=_elapsed_ms(),
            )
            return

        step_model = model
        if before_step is not None:
            try:
                maybe_model = before_step(step)
                if inspect.isawaitable(maybe_model):
                    maybe_model = await maybe_model
                if isinstance(maybe_model, str) and maybe_model.strip():
                    step_model = maybe_model.strip()
            except BudgetBlockedError as exc:
                yield TurnEndEvent(
                    reason="budget_blocked",
                    final_text=exc.refusal_reply or "",
                    steps_used=steps_used,
                    total_tokens=total_tokens,
                    total_cost_micros=total_cost_micros,
                    cancelled=False,
                    latency_ms=_elapsed_ms(),
                )
                return

        # On the final step, strip `requires_followup=True` query tools to force summary/completion.
        if step == effective_max_steps - 1:
            step_tool_defs = [
                td
                for td in all_tool_defs
                if td.is_chat_terminal or not td.requires_followup
            ]
        else:
            step_tool_defs = all_tool_defs

        chat_task: asyncio.Task[ChatResult] = asyncio.create_task(
            _call_provider_chat(
                provider,
                list(loop_messages),
                model=step_model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=step_tool_defs if step_tool_defs else None,
                enable_thinking=enable_thinking,
            )
        )
        token.bind_task(chat_task)

        try:
            if deadline is not None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError("agent turn timeout")
                result = await asyncio.wait_for(chat_task, timeout=rem)
            else:
                result = await chat_task
        except TimeoutError:
            if not chat_task.done():
                chat_task.cancel()
                await asyncio.gather(chat_task, return_exceptions=True)
            yield TurnEndEvent(
                reason="timeout",
                final_text=last_text,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=False,
                latency_ms=_elapsed_ms(),
            )
            return
        except asyncio.CancelledError:
            if not chat_task.done():
                chat_task.cancel()
                await asyncio.gather(chat_task, return_exceptions=True)
            if token.is_cancelled:
                yield TurnEndEvent(
                    reason=token.reason or "cancelled",
                    final_text=last_text,
                    steps_used=steps_used,
                    total_tokens=total_tokens,
                    total_cost_micros=total_cost_micros,
                    cancelled=True,
                    latency_ms=_elapsed_ms(),
                )
                return
            raise

        steps_used += 1
        step_tokens = (
            result.total_tokens
            if result.total_tokens > 0
            else (result.prompt_tokens + result.completion_tokens)
        )
        total_tokens += step_tokens
        total_cost_micros += result.cost_micros_usd
        if result.content.strip():
            last_text = result.content

        if after_step is not None:
            after_out = after_step(result)
            if inspect.isawaitable(after_out):
                await after_out

        yield LlmStepEvent(step=step, result=result)

        if not result.tool_calls:
            yield TurnEndEvent(
                reason="end_turn",
                final_text=result.content,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=False,
                latency_ms=_elapsed_ms(),
            )
            return

        # Partition tool calls: terminal chat actions run sequentially first; non-terminal run in parallel.
        terminal_indexed: list[tuple[int, ToolCall]] = []
        parallel_indexed: list[tuple[int, ToolCall]] = []
        for idx, tc in enumerate(result.tool_calls):
            reg = tool_map.get(tc.name)
            if reg is not None and reg.definition.is_chat_terminal:
                terminal_indexed.append((idx, tc))
            else:
                parallel_indexed.append((idx, tc))

        outcomes_by_idx: dict[int, ToolOutcome] = {}
        parallel_tasks: list[asyncio.Task[ToolOutcome]] = []
        try:
            for idx, tc in terminal_indexed:
                outcomes_by_idx[idx] = await execute_tool_call(
                    tc,
                    tool_map,
                    token,
                    deadline=deadline,
                )

            if parallel_indexed:
                parallel_tasks = [
                    token.bind_task(
                        asyncio.create_task(
                            execute_tool_call(tc, tool_map, token, deadline=deadline)
                        )
                    )
                    for _, tc in parallel_indexed
                ]
                gathered_outcomes = await asyncio.gather(*parallel_tasks)
                for (idx, _), out in zip(parallel_indexed, gathered_outcomes, strict=True):
                    outcomes_by_idx[idx] = out
        except TimeoutError:
            for pt in parallel_tasks:
                if not pt.done():
                    pt.cancel()
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks, return_exceptions=True)
            yield TurnEndEvent(
                reason="timeout",
                final_text=last_text,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=False,
                latency_ms=_elapsed_ms(),
            )
            return
        except asyncio.CancelledError:
            for pt in parallel_tasks:
                if not pt.done():
                    pt.cancel()
            if parallel_tasks:
                await asyncio.gather(*parallel_tasks, return_exceptions=True)
            if token.is_cancelled:
                yield TurnEndEvent(
                    reason=token.reason or "cancelled",
                    final_text=last_text,
                    steps_used=steps_used,
                    total_tokens=total_tokens,
                    total_cost_micros=total_cost_micros,
                    cancelled=True,
                    latency_ms=_elapsed_ms(),
                )
                return
            raise

        outcomes: tuple[ToolOutcome, ...] = tuple(
            outcomes_by_idx[i] for i in range(len(result.tool_calls))
        )
        for tc, out in zip(result.tool_calls, outcomes, strict=True):
            yield ToolExecutionEvent(step=step, call=tc, outcome=out)

        if token.is_cancelled:
            yield TurnEndEvent(
                reason=token.reason or "cancelled",
                final_text=last_text,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=True,
                latency_ms=_elapsed_ms(),
            )
            return

        # Drain live steering queue once at tool boundary before evaluating terminal short-circuit.
        raw_steered = drain_steering() if drain_steering is not None else ()
        steered = tuple(s.strip() for s in raw_steered if isinstance(s, str) and s.strip())

        called_defs = [
            tool_map[tc.name].definition
            for tc in result.tool_calls
            if tc.name in tool_map
        ]
        all_known = len(called_defs) == len(result.tool_calls)
        has_terminal = any(td.is_chat_terminal for td in called_defs)
        has_followup = any(td.requires_followup for td in called_defs)
        all_ok = all(not out.is_error for out in outcomes)

        if all_known and has_terminal and not has_followup and all_ok and not steered:
            yield TurnEndEvent(
                reason="terminal_tool_short_circuit",
                final_text=result.content,
                steps_used=steps_used,
                total_tokens=total_tokens,
                total_cost_micros=total_cost_micros,
                cancelled=False,
                latency_ms=_elapsed_ms(),
            )
            return

        loop_messages.append(
            ChatMessage(
                role="assistant",
                content=result.content,
                tool_calls=result.tool_calls,
            )
        )
        loop_messages.append(
            ChatMessage(
                role="tool",
                content="",
                tool_results=outcomes,
            )
        )

        if steered:
            steering_payload = "<steering_update>\n" + "\n".join(steered) + "\n</steering_update>"
            loop_messages.append(ChatMessage(role="user", content=steering_payload))
            yield SteeringInjectedEvent(step=step, texts=steered)

    yield TurnEndEvent(
        reason="max_steps_reached",
        final_text=last_text,
        steps_used=steps_used,
        total_tokens=total_tokens,
        total_cost_micros=total_cost_micros,
        cancelled=token.is_cancelled,
        latency_ms=_elapsed_ms(),
    )


__all__ = [
    "AfterStepHook",
    "AgentEvent",
    "BeforeStepHook",
    "BudgetBlockedError",
    "CancelToken",
    "LlmStepEvent",
    "RegisteredTool",
    "SteeringInjectedEvent",
    "ToolExecutionEvent",
    "ToolHandler",
    "TurnEndEvent",
    "TurnStartEvent",
    "agent_loop",
    "execute_tool_call",
]
