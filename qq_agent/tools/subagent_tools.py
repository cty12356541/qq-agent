"""Fast/slow dual-brain background sub-agent dispatch (`spawn_subagent`) and cancellation (`cancel_subagent`) tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import inspect
from typing import Any

from qq_agent.tools import ToolArgumentError, ToolBinding
from qq_agent.types import InboundMessage, ToolDef

SpawnSubagentFn = Callable[[str, str, int | None], str | Awaitable[str]]
CancelSubagentFn = Callable[[str | None], int | Awaitable[int]]


def _try_parse_message_id(raw_val: Any) -> int | None:
    if raw_val is None or isinstance(raw_val, bool):
        return None
    if isinstance(raw_val, int):
        return raw_val if raw_val != 0 else None
    if isinstance(raw_val, str):
        stripped = raw_val.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
            return parsed if parsed != 0 else None
        except ValueError:
            return None
    return None


def build_spawn_subagent_tool(
    spawn_fn: SpawnSubagentFn,
    *,
    msg: InboundMessage | None = None,
    default_reply_to_message_id: int | None = None,
) -> ToolBinding:
    """Build `spawn_subagent` (`is_chat_terminal=False, requires_followup=False`).

    Setting `requires_followup=False` allows the fast brain to emit `send_message(["等下我翻翻帖子"])`
    and `spawn_subagent(goal=...)` in a single LLM turn and immediately short-circuit, releasing
    the per-session lock with zero blocking while the slow brain researches in the background.
    """
    fallback_reply_id: int | None = (
        default_reply_to_message_id
        if default_reply_to_message_id is not None
        else (_try_parse_message_id(msg.event_id) if msg is not None else None)
    )

    definition = ToolDef(
        name="spawn_subagent",
        description=(
            "当遇到需要多步深查网页、对比多篇资料或复杂推理核实的问题时，派发后台慢脑子任务异步深查。"
            "建议在同一轮先用 send_message 自然告知群友『等下我去查查』，再调用本工具派单。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "明确具体要后台深度检索、核实或对比的问题目标。",
                },
                "context_hint": {
                    "type": "string",
                    "description": "可选，补充当前群聊相关背景信息、游戏名/版本或上下文线索。",
                },
                "reply_to_message_id": {
                    "type": "integer",
                    "description": "可选，后台查完后回报时引用回复的消息 ID（默认自动绑定当前触发消息）。",
                },
            },
            "required": ["goal"],
            "additionalProperties": False,
        },
        is_chat_terminal=False,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_goal = args.get("goal")
        if not isinstance(raw_goal, str) or not raw_goal.strip():
            raise ToolArgumentError("spawn_subagent requires a non-empty string 'goal'")
        goal = raw_goal.strip()

        raw_hint = args.get("context_hint")
        context_hint = str(raw_hint).strip() if raw_hint is not None else ""

        explicit_reply_id = _try_parse_message_id(args.get("reply_to_message_id"))
        effective_reply_id = (
            explicit_reply_id if explicit_reply_id is not None else fallback_reply_id
        )

        maybe_task_id = spawn_fn(goal, context_hint, effective_reply_id)
        if inspect.isawaitable(maybe_task_id):
            task_id = str(await maybe_task_id)
        else:
            task_id = str(maybe_task_id)

        return f"Spawned background subagent task_id={task_id}"

    return ToolBinding(definition=definition, execute=_execute)


def build_cancel_subagent_tool(
    cancel_fn: CancelSubagentFn,
) -> ToolBinding:
    """Build `cancel_subagent` (`is_chat_terminal=False, requires_followup=False`)."""
    definition = ToolDef(
        name="cancel_subagent",
        description=(
            "取消当前会话正在后台运行的深查子任务（例如群友说『不用查了』『算了换个话题』时调用）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "可选，指定要取消的 task_id；留空则取消当前会话全部在途子任务。",
                },
            },
            "additionalProperties": False,
        },
        is_chat_terminal=False,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        raw_tid = args.get("task_id")
        task_id: str | None = None
        if raw_tid is not None:
            cleaned = str(raw_tid).strip()
            if cleaned:
                task_id = cleaned

        maybe_count = cancel_fn(task_id)
        if inspect.isawaitable(maybe_count):
            cancelled = int(await maybe_count)
        else:
            cancelled = int(maybe_count)

        if task_id is not None:
            return f"Cancelled {cancelled} subagent task(s) matching task_id={task_id}"
        return f"Cancelled {cancelled} active subagent task(s) in session"

    return ToolBinding(definition=definition, execute=_execute)


__all__ = [
    "CancelSubagentFn",
    "SpawnSubagentFn",
    "build_cancel_subagent_tool",
    "build_spawn_subagent_tool",
]
