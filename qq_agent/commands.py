"""Independent admin whitelist command router bypassing LLM context (§4.5, §4.6)."""

from __future__ import annotations

import re
from collections.abc import Callable

from qq_agent.guardrails import QuotaSnapshot
from qq_agent.types import CommandResult, InboundMessage

_BUILTIN_COMMANDS: frozenset[str] = frozenset({"ping", "reset", "quota", "reload", "help"})
_PERMISSION_DENIED_TEXT = "权限不足：该管理指令仅限机器人管理员使用。"
_LEADING_CQ_AT_RE = re.compile(r"^(?:\[CQ:at,[^\]]+\]\s*)+")

_HELP_TEXT = (
    "管理命令列表（仅管理员可用）：\n"
    "/ping - 检查机器人在线状态\n"
    "/reset - 清理当前会话上下文记忆窗口\n"
    "/quota - 查看当日调用次数、Token 用量与美元花费额度\n"
    "/reload - 热重载人设配置与唤醒词\n"
    "/help - 查看管理命令帮助说明"
)


def _format_micros_usd(micros: int) -> str:
    return f"${micros / 1_000_000:.4f}"


class AdminCommandRouter:
    """Pre-LLM whitelist command router ensuring admin commands never enter LLM context."""

    def __init__(
        self,
        *,
        admin_user_ids: set[int],
        on_reset: Callable[[str], int] | None = None,
        on_quota: Callable[[], QuotaSnapshot] | None = None,
        on_reload: Callable[[], str] | None = None,
    ) -> None:
        self.admin_user_ids: set[int] = set(admin_user_ids)
        self._on_reset = on_reset
        self._on_quota = on_quota
        self._on_reload = on_reload

    @staticmethod
    def _extract_builtin_command(msg: InboundMessage) -> str | None:
        text = msg.text.strip()
        if not text:
            return None

        # Tolerate leading CQ at-segment or "@bot /cmd" when mentioned_bot is True
        text = _LEADING_CQ_AT_RE.sub("", text).strip()
        if msg.mentioned_bot and text.startswith("@"):
            parts = text.split(maxsplit=1)
            text = parts[1].strip() if len(parts) > 1 else ""

        if not text.startswith("/"):
            return None

        first_token = text.split(maxsplit=1)[0].lower()
        cmd = first_token[1:]
        if cmd in _BUILTIN_COMMANDS:
            return cmd
        return None

    def try_handle(self, msg: InboundMessage) -> CommandResult:
        cmd = self._extract_builtin_command(msg)
        if cmd is None:
            return CommandResult(handled=False)

        if msg.user_id not in self.admin_user_ids:
            return CommandResult(
                handled=True,
                reply_text=_PERMISSION_DENIED_TEXT,
                command_name=cmd,
            )

        if cmd == "ping":
            return CommandResult(
                handled=True,
                reply_text="pong! qq-agent 在线运行中。",
                command_name="ping",
            )

        if cmd == "reset":
            if self._on_reset is None:
                return CommandResult(
                    handled=True,
                    reply_text="会话重置功能尚未绑定。",
                    command_name="reset",
                )
            cleared = self._on_reset(msg.session_key)
            return CommandResult(
                handled=True,
                reply_text=f"已清理当前会话（{msg.session_key}）上下文，共移除 {cleared} 条消息。",
                command_name="reset",
            )

        if cmd == "quota":
            if self._on_quota is None:
                return CommandResult(
                    handled=True,
                    reply_text="配额查询功能尚未绑定。",
                    command_name="quota",
                )
            snap = self._on_quota()
            used_str = _format_micros_usd(snap.cost_micros_used)
            soft_str = _format_micros_usd(snap.soft_budget_micros)
            hard_str = _format_micros_usd(snap.hard_budget_micros)
            reply = (
                f"【当日配额 {snap.day}】\n"
                f"调用次数: {snap.calls_used} / {snap.max_daily_calls}\n"
                f"Token 用量: {snap.tokens_used} / {snap.max_daily_tokens}\n"
                f"花费额度: {used_str} / {hard_str}（软限 {soft_str}）"
            )
            return CommandResult(
                handled=True,
                reply_text=reply,
                command_name="quota",
            )

        if cmd == "reload":
            if self._on_reload is None:
                return CommandResult(
                    handled=True,
                    reply_text="人设热重载功能尚未绑定。",
                    command_name="reload",
                )
            persona_name = self._on_reload()
            return CommandResult(
                handled=True,
                reply_text=f"人设热重载成功！当前角色：{persona_name}",
                command_name="reload",
            )

        # cmd == "help"
        return CommandResult(
            handled=True,
            reply_text=_HELP_TEXT,
            command_name="help",
        )
