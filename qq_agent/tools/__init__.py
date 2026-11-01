"""Unified tool binding specification and safe result/error formatting helpers for qq-agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
import json
import re
from typing import Any

from qq_agent.types import ToolCall, ToolDef, ToolOutcome

_BEARER_SECRET_RE = re.compile(r"(Bearer\s+)[^\s\"',;]+", flags=re.IGNORECASE)
_SK_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")
_API_KEY_PARAM_RE = re.compile(r"([?&](?:api_key|apikey|token|key)=)[^&\s\"']+", flags=re.IGNORECASE)


class ToolArgumentError(ValueError):
    """Raised when a model-supplied tool argument fails schema or type validation."""


@dataclass(frozen=True, slots=True)
class ToolBinding:
    """Lightweight client-side tool binding seamlessly bridgeable to `pi_core.RegisteredTool`."""

    definition: ToolDef
    execute: Callable[[dict[str, Any]], Awaitable[str]]

    @property
    def name(self) -> str:
        return self.definition.name


def redact_tool_secrets(text: str) -> str:
    """Redact accidental bearer tokens or API keys from tool outputs and error strings."""
    cleaned = _BEARER_SECRET_RE.sub(r"\1[REDACTED]", text)
    cleaned = _SK_SECRET_RE.sub("sk-[REDACTED]", cleaned)
    cleaned = _API_KEY_PARAM_RE.sub(r"\1[REDACTED]", cleaned)
    return cleaned


def coerce_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Coerce raw model tool arguments (`dict`, JSON `str`, or `None`) into a `dict[str, Any]`."""
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, Mapping):
        return {str(k): v for k, v in raw_arguments.items()}
    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(f"Tool arguments string is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ToolArgumentError(
                f"Tool arguments JSON must decode to an object, got {type(parsed).__name__}"
            )
        return parsed
    raise ToolArgumentError(
        f"Tool arguments must be a JSON object mapping, got {type(raw_arguments).__name__}"
    )


def format_tool_result(result: Any, *, max_chars: int = 12_000) -> str:
    """Safely serialize and cap a tool execution return value as a clean UTF-8 string."""
    limit = max(64, int(max_chars))
    if result is None:
        text = "OK"
    elif isinstance(result, str):
        text = result
    elif isinstance(result, (dict, list, tuple)):
        text = json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)

    text = redact_tool_secrets(text)
    if len(text) > limit:
        suffix = "\n...[truncated]"
        keep = max(0, limit - len(suffix))
        text = text[:keep] + suffix
    return text


def format_tool_error(exc: BaseException | str, *, tool_name: str = "") -> str:
    """Safely format parameter validation or execution errors into a sanitized string for LLM feedback."""
    is_validation = isinstance(exc, (ToolArgumentError, TypeError, ValueError))
    raw_msg = exc if isinstance(exc, str) else (str(exc).strip() or type(exc).__name__)
    safe_msg = redact_tool_secrets(" ".join(raw_msg.split()))
    kind = "tool_validation_error" if is_validation else "tool_error"
    prefix = f"[{kind}:{tool_name}]" if tool_name else f"[{kind}]"
    return f"{prefix} {safe_msg}"


async def execute_tool_call(
    binding: ToolBinding,
    call: ToolCall,
    *,
    max_chars: int = 12_000,
) -> ToolOutcome:
    """Execute a `ToolCall` against `ToolBinding`, catching argument/runtime errors into `ToolOutcome`."""
    try:
        args = coerce_tool_arguments(call.arguments)
        raw_out = await binding.execute(args)
        content = format_tool_result(raw_out, max_chars=max_chars)
        return ToolOutcome(
            tool_call_id=call.id,
            name=binding.definition.name,
            content=content,
            is_error=False,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return ToolOutcome(
            tool_call_id=call.id,
            name=binding.definition.name,
            content=format_tool_error(exc, tool_name=binding.definition.name),
            is_error=True,
        )


from .chat_tools import (  # noqa: E402
    ChatTurnState,
    build_send_message_tool,
    build_stay_silent_tool,
    extract_sticker_tag,
    flush_fallback_text_reply,
)
from .memory_tools import (  # noqa: E402
    build_remember_note_tool,
    resolve_locked_memory_scope,
)
from .subagent_tools import (  # noqa: E402
    build_cancel_subagent_tool,
    build_spawn_subagent_tool,
)
from .web_search import (  # noqa: E402
    ResponseTooLargeError,
    SearchResultItem,
    SsrfBlockedError,
    WebFetchError,
    WebPageFetcher,
    WebSearchClient,
    build_fetch_webpage_tool,
    build_web_search_tool,
    validate_public_http_url,
)

__all__ = [
    "ChatTurnState",
    "ResponseTooLargeError",
    "SearchResultItem",
    "SsrfBlockedError",
    "ToolArgumentError",
    "ToolBinding",
    "WebFetchError",
    "WebPageFetcher",
    "WebSearchClient",
    "build_cancel_subagent_tool",
    "build_fetch_webpage_tool",
    "build_remember_note_tool",
    "build_send_message_tool",
    "build_spawn_subagent_tool",
    "build_stay_silent_tool",
    "build_web_search_tool",
    "coerce_tool_arguments",
    "execute_tool_call",
    "extract_sticker_tag",
    "flush_fallback_text_reply",
    "format_tool_error",
    "format_tool_result",
    "redact_tool_secrets",
    "resolve_locked_memory_scope",
    "validate_public_http_url",
]
