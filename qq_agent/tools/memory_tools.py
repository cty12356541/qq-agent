"""Zero-trust scope-locked long-term memory & acquaintance profile tool (`remember_note`)."""

from __future__ import annotations

from collections.abc import Collection
import inspect
from typing import Any

from qq_agent.anti_injection import check_remember_note_safety
from qq_agent.memory import MemoryStore
from qq_agent.tools import ToolArgumentError, ToolBinding
from qq_agent.types import InboundMessage, ToolDef

MAX_REMEMBER_CONTENT_CHARS = 200
_VALID_MEMORY_KINDS: frozenset[str] = frozenset({"fact", "user_profile"})


def resolve_locked_memory_scope(
    msg: InboundMessage,
    *,
    admin_user_ids: Collection[int] = (),
) -> str:
    """Compute the immutable zero-trust storage scope from the inbound message context.

    Scope rules (never overridden by LLM arguments):
    - Group chat: `f"group:{group_id}"` (even if speaker is an admin!)
    - Admin private chat: `"admin_private"`
    - Non-admin private chat: `f"private:{user_id}"`
    """
    if msg.chat_type == "group":
        if msg.group_id is None:
            raise ValueError("Group InboundMessage is missing group_id for memory scope locking")
        return f"group:{int(msg.group_id)}"

    admin_set = {int(uid) for uid in admin_user_ids}
    if int(msg.user_id) in admin_set:
        return "admin_private"
    return f"private:{int(msg.user_id)}"


def _write_scoped_fact(memory: Any, *, content: str, locked_scope: str) -> int | None:
    """Write a fact using `MemoryStore.add_fact` if present, otherwise `MemoryStore.add_memory`."""
    add_fact_fn = getattr(memory, "add_fact", None)
    if callable(add_fact_fn):
        sig = inspect.signature(add_fact_fn)
        if "scope" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            rec = add_fact_fn(content, scope=locked_scope)
        else:
            rec = add_fact_fn(locked_scope, content)
        return getattr(rec, "id", None) if rec is not None else None

    rec = memory.add_memory(content, scope=locked_scope)
    return int(rec.id)


def _upsert_scoped_user_profile(
    memory: Any,
    *,
    user_id: int,
    traits_note: str,
    display_name: str,
    locked_scope: str,
) -> Any:
    """Upsert user profile via `upsert_user_profile` if present, else merge into `set_user_profile`."""
    upsert_fn = getattr(memory, "upsert_user_profile", None)
    if callable(upsert_fn):
        sig = inspect.signature(upsert_fn)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        kwargs: dict[str, Any] = {}
        if "display_name" in params or has_varkw:
            kwargs["display_name"] = display_name
        if "scope" in params or has_varkw:
            kwargs["scope"] = locked_scope
        return upsert_fn(user_id, traits_note, **kwargs)

    existing = memory.get_user_profile(user_id) if hasattr(memory, "get_user_profile") else None
    merged_traits = traits_note
    if existing is not None and getattr(existing, "traits_text", "").strip():
        old_traits = str(existing.traits_text).strip()
        if traits_note not in old_traits:
            combined = f"{old_traits}；{traits_note}"
            merged_traits = combined[-240:] if len(combined) > 240 else combined
        else:
            merged_traits = old_traits

    return memory.set_user_profile(
        user_id,
        merged_traits,
        display_name=display_name,
    )


def build_remember_note_tool(
    memory: MemoryStore,
    msg: InboundMessage,
    *,
    admin_user_ids: Collection[int] = (),
    max_content_chars: int = MAX_REMEMBER_CONTENT_CHARS,
) -> ToolBinding:
    """Build `remember_note` (`is_chat_terminal=False, requires_followup=False`) with closure-locked `scope`."""
    locked_scope = resolve_locked_memory_scope(msg, admin_user_ids=admin_user_ids)
    admin_set = {int(uid) for uid in admin_user_ids}
    speaker_is_admin = int(msg.user_id) in admin_set
    char_cap = max(20, int(max_content_chars))

    definition = ToolDef(
        name="remember_note",
        description=(
            "将值得长期记住的重要圈子事实、群内约定梗或群友个人特征写入长期记忆。"
            "仅在出现明确新事实或稳定偏好时调用，日常寒暄无需记录（单条不超过 200 字）。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "maxLength": char_cap,
                    "description": f"简明客观的事实备忘或群友特征（<= {char_cap} 字）。",
                },
                "kind": {
                    "type": "string",
                    "enum": ["fact", "user_profile"],
                    "default": "fact",
                    "description": "记忆类型：'fact' 为会话/群内事件与梗，'user_profile' 为具体群友画像特征。",
                },
                "target_user_id": {
                    "type": "integer",
                    "description": "可选，当 kind='user_profile' 时指定目标 QQ 号（默认记录当前发言人）。",
                },
                "display_name": {
                    "type": "string",
                    "description": "可选，目标群友的常用称呼或群名片。",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        is_chat_terminal=False,
        requires_followup=False,
    )

    async def _execute(args: dict[str, Any]) -> str:
        # Zero-trust: any rogue `scope` key injected by the model is strictly ignored.
        raw_content = args.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            raise ToolArgumentError("remember_note requires a non-empty string 'content'")

        cleaned_content = " ".join(raw_content.split()).strip()
        if len(cleaned_content) > char_cap:
            cleaned_content = cleaned_content[:char_cap].rstrip()

        raw_kind = args.get("kind", "fact")
        kind = str(raw_kind).strip().lower() if raw_kind is not None else "fact"
        if not kind:
            kind = "fact"
        if kind not in _VALID_MEMORY_KINDS:
            raise ToolArgumentError(
                f"Invalid 'kind' {raw_kind!r}: expected 'fact' or 'user_profile'"
            )

        if kind == "fact":
            safety_err = check_remember_note_safety(
                cleaned_content,
                kind="fact",
                speaker_user_id=int(msg.user_id),
                target_user_id=int(msg.user_id),
                is_admin=speaker_is_admin,
            )
            if safety_err is not None:
                return safety_err
            fact_id = _write_scoped_fact(
                memory,
                content=cleaned_content,
                locked_scope=locked_scope,
            )
            id_suffix = f" (id={fact_id})" if fact_id is not None else ""
            return f"Stored fact in scope={locked_scope}{id_suffix}"

        raw_target_uid = args.get("target_user_id")
        if raw_target_uid is None or raw_target_uid == "":
            target_uid = int(msg.user_id)
        elif isinstance(raw_target_uid, bool):
            raise ToolArgumentError("'target_user_id' must be an integer QQ ID, not boolean")
        else:
            try:
                target_uid = int(raw_target_uid)
            except (TypeError, ValueError) as exc:
                raise ToolArgumentError(
                    f"'target_user_id' must be an integer QQ ID, got {raw_target_uid!r}"
                ) from exc
            if target_uid <= 0:
                raise ToolArgumentError("'target_user_id' must be a positive integer")

        safety_err = check_remember_note_safety(
            cleaned_content,
            kind="user_profile",
            speaker_user_id=int(msg.user_id),
            target_user_id=target_uid,
            is_admin=speaker_is_admin,
        )
        if safety_err is not None:
            return safety_err

        raw_display = args.get("display_name")
        if isinstance(raw_display, str) and raw_display.strip():
            display_name = raw_display.strip()[:64]
        elif target_uid == int(msg.user_id):
            display_name = msg.sender_name.strip()[:64]
        else:
            display_name = ""

        _upsert_scoped_user_profile(
            memory,
            user_id=target_uid,
            traits_note=cleaned_content,
            display_name=display_name,
            locked_scope=locked_scope,
        )
        return f"Updated user_profile for user_id={target_uid} (scope={locked_scope})"

    return ToolBinding(definition=definition, execute=_execute)


__all__ = [
    "MAX_REMEMBER_CONTENT_CHARS",
    "build_remember_note_tool",
    "resolve_locked_memory_scope",
]
