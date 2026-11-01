"""Independent admin whitelist command router bypassing LLM context (§4.5, §4.6, M2/W4 admin commands)."""

from __future__ import annotations

from collections.abc import Callable
import re

from qq_agent.guardrails import QuotaSnapshot
from qq_agent.types import CommandResult, InboundMessage

_BUILTIN_COMMANDS: frozenset[str] = frozenset(
    {
        "ping",
        "status",
        "stop",
        "reset",
        "quota",
        "reload",
        "persona",
        "remember",
        "memories",
        "forget",
        "profile",
        "help",
    }
)
_PERMISSION_DENIED_TEXT = "权限不足：该管理指令仅限机器人管理员使用。"
_LEADING_CQ_AT_RE = re.compile(r"^(?:\[CQ:at,[^\]]+\]\s*)+")
_SINGLE_CQ_AT_QQ_RE = re.compile(r"^\[CQ:at,qq=(\d+)[^\]]*\]$")

_HELP_TEXT = (
    "管理命令列表（仅管理员可用）：\n"
    "/ping - 检查机器人在线状态\n"
    "/status - 查看异步会话池、后台子任务与队列健康快照\n"
    "/stop [all] - 立即中断当前会话（或全部会话）在途推理与后台搜索子任务\n"
    "/reset - 清理当前会话上下文记忆窗口\n"
    "/quota - 查看当日调用次数、Token 用量与美元花费额度\n"
    "/reload - 热重载人设配置与唤醒词\n"
    "/persona [use <档案名>] - 查看当前人设状态或热切换人设档案\n"
    "/remember [--public|--private] <内容> - 写入长期记忆\n"
    "/memories - 查看当前可见的长期记忆列表\n"
    "/forget <ID> - 按数字 ID 删除长期记忆\n"
    "/profile <QQ号> <画像备注> - 设置群友跨会话画像备注\n"
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
        on_persona_status: Callable[[], tuple[str, str, list[str]]] | None = None,
        on_persona_switch: Callable[[str], str] | None = None,
        on_remember: Callable[[str, str], int] | None = None,
        on_list_memories: Callable[[bool], list[tuple[int, str, str]]] | None = None,
        on_forget: Callable[[int], bool] | None = None,
        on_profile: Callable[[int, str], str] | None = None,
        on_stop: Callable[[str], tuple[bool, int]] | None = None,
        on_status: Callable[[str], str] | None = None,
    ) -> None:
        self.admin_user_ids: set[int] = set(admin_user_ids)
        self._on_reset = on_reset
        self._on_quota = on_quota
        self._on_reload = on_reload
        self._on_persona_status = on_persona_status
        self._on_persona_switch = on_persona_switch
        self._on_remember = on_remember
        self._on_list_memories = on_list_memories
        self._on_forget = on_forget
        self._on_profile = on_profile
        self._on_stop = on_stop
        self._on_status = on_status

    @staticmethod
    def _extract_command_and_args(msg: InboundMessage) -> tuple[str, str] | None:
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

        parts = text.split(maxsplit=1)
        cmd = parts[0][1:].lower()
        if cmd not in _BUILTIN_COMMANDS:
            return None
        arg_text = parts[1].strip() if len(parts) > 1 else ""
        return cmd, arg_text

    @classmethod
    def _extract_builtin_command(cls, msg: InboundMessage) -> str | None:
        extracted = cls._extract_command_and_args(msg)
        return extracted[0] if extracted is not None else None

    def _handle_stop(self, msg: InboundMessage, arg_text: str) -> CommandResult:
        if self._on_stop is None:
            return CommandResult(
                handled=True,
                reply_text="任务停止功能尚未绑定。",
                command_name="stop",
            )
        target_scope = "*" if arg_text.strip().lower() == "all" else msg.session_key
        cancelled_main, cancelled_subs = self._on_stop(target_scope)
        scope_label = "全部会话" if target_scope == "*" else f"当前会话（{msg.session_key}）"
        if not cancelled_main and cancelled_subs == 0:
            reply = f"{scope_label}当前没有在途推理或后台子任务。"
        else:
            main_desc = "已中断主循环" if cancelled_main else "主循环空闲"
            reply = f"已停止{scope_label}在途任务：{main_desc}，取消后台子任务 {cancelled_subs} 个。"
        return CommandResult(
            handled=True,
            reply_text=reply,
            command_name="stop",
        )

    def _handle_status(self, msg: InboundMessage) -> CommandResult:
        if self._on_status is None:
            return CommandResult(
                handled=True,
                reply_text="运行状态查询功能尚未绑定。",
                command_name="status",
            )
        report = self._on_status(msg.session_key)
        return CommandResult(
            handled=True,
            reply_text=report,
            command_name="status",
        )

    def _handle_persona(self, arg_text: str) -> CommandResult:
        if not arg_text:
            if self._on_persona_status is None:
                return CommandResult(
                    handled=True,
                    reply_text="人设档案状态查询功能尚未绑定。",
                    command_name="persona",
                )
            name, active_profile, profiles = self._on_persona_status()
            active_display = active_profile if active_profile else "default"
            profiles_display = "、".join(profiles) if profiles else "无"
            reply = (
                f"当前角色：{name}\n"
                f"激活档案：{active_display}\n"
                f"可选档案：{profiles_display}"
            )
            return CommandResult(
                handled=True,
                reply_text=reply,
                command_name="persona",
            )

        tokens = arg_text.split(maxsplit=1)
        if tokens[0].lower() == "use":
            if len(tokens) < 2 or not tokens[1].strip():
                return CommandResult(
                    handled=True,
                    reply_text="用法：/persona 或 /persona use <档案名>",
                    command_name="persona",
                )
            target_profile = tokens[1].strip()
        else:
            target_profile = arg_text.strip()

        if self._on_persona_switch is None:
            return CommandResult(
                handled=True,
                reply_text="人设档案切换功能尚未绑定。",
                command_name="persona",
            )

        try:
            new_status = self._on_persona_switch(target_profile)
        except (ValueError, FileNotFoundError) as exc:
            return CommandResult(
                handled=True,
                reply_text=f"人设档案切换失败：{exc}",
                command_name="persona",
            )

        if target_profile in new_status:
            reply = f"人设档案切换成功：{new_status}"
        else:
            reply = f"人设档案已切换为 {target_profile}：{new_status}"
        return CommandResult(
            handled=True,
            reply_text=reply,
            command_name="persona",
        )

    def _handle_remember(self, msg: InboundMessage, arg_text: str) -> CommandResult:
        if msg.chat_type == "group" and msg.group_id is not None:
            scope = f"group:{msg.group_id}"
        else:
            scope = "admin_private"

        content = arg_text.strip()
        if content:
            parts = content.split(maxsplit=1)
            flag = parts[0].lower()
            if flag == "--public":
                scope = "public"
                content = parts[1].strip() if len(parts) > 1 else ""
            elif flag == "--private":
                scope = "admin_private"
                content = parts[1].strip() if len(parts) > 1 else ""

        if not content:
            return CommandResult(
                handled=True,
                reply_text="用法：/remember [--public|--private] <记忆内容>",
                command_name="remember",
            )

        if self._on_remember is None:
            return CommandResult(
                handled=True,
                reply_text="长期记忆写入功能尚未绑定。",
                command_name="remember",
            )

        fact_id = self._on_remember(content, scope)
        if msg.chat_type == "group" and scope == "admin_private":
            reply = f"已记录长期记忆 #{fact_id}（作用域：{scope}）。"
        else:
            reply = f"已记录长期记忆 #{fact_id}（作用域：{scope}）：{content}"
        return CommandResult(
            handled=True,
            reply_text=reply,
            command_name="remember",
        )

    def _handle_memories(self, msg: InboundMessage) -> CommandResult:
        if self._on_list_memories is None:
            return CommandResult(
                handled=True,
                reply_text="长期记忆查询功能尚未绑定。",
                command_name="memories",
            )

        include_admin_private = msg.chat_type == "private"
        items = self._on_list_memories(include_admin_private)
        if not include_admin_private:
            items = [
                (fid, scope, content)
                for fid, scope, content in items
                if scope != "admin_private"
            ]

        if not items:
            return CommandResult(
                handled=True,
                reply_text="当前暂无可展示的长期记忆。",
                command_name="memories",
            )

        lines = [f"长期记忆列表（共 {len(items)} 条）："]
        for fid, scope, content in items:
            lines.append(f"#{fid} [{scope}] {content}")
        return CommandResult(
            handled=True,
            reply_text="\n".join(lines),
            command_name="memories",
        )

    def _handle_forget(self, arg_text: str) -> CommandResult:
        token = arg_text.strip()
        if token.startswith("#"):
            token = token[1:].strip()
        if not token:
            return CommandResult(
                handled=True,
                reply_text="用法：/forget <记忆数字ID>",
                command_name="forget",
            )
        try:
            fact_id = int(token)
        except ValueError:
            return CommandResult(
                handled=True,
                reply_text=f"无效的记忆 ID（须为整数）：{arg_text.strip()}",
                command_name="forget",
            )

        if self._on_forget is None:
            return CommandResult(
                handled=True,
                reply_text="长期记忆删除功能尚未绑定。",
                command_name="forget",
            )

        deleted = self._on_forget(fact_id)
        if deleted:
            reply = f"已删除长期记忆 #{fact_id}。"
        else:
            reply = f"未找到长期记忆 #{fact_id}，无需删除。"
        return CommandResult(
            handled=True,
            reply_text=reply,
            command_name="forget",
        )

    def _handle_profile(self, arg_text: str) -> CommandResult:
        parts = arg_text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return CommandResult(
                handled=True,
                reply_text="用法：/profile <QQ号> <画像备注>",
                command_name="profile",
            )

        raw_qq = parts[0].strip()
        cq_match = _SINGLE_CQ_AT_QQ_RE.match(raw_qq)
        if cq_match is not None:
            raw_qq = cq_match.group(1)
        elif raw_qq.startswith("@"):
            raw_qq = raw_qq[1:].strip()

        try:
            target_qq = int(raw_qq)
        except ValueError:
            return CommandResult(
                handled=True,
                reply_text="用法：/profile <QQ号> <画像备注>（QQ号须为纯数字）",
                command_name="profile",
            )
        if target_qq <= 0:
            return CommandResult(
                handled=True,
                reply_text="用法：/profile <QQ号> <画像备注>（QQ号须为正整数）",
                command_name="profile",
            )

        notes = parts[1].strip()
        if self._on_profile is None:
            return CommandResult(
                handled=True,
                reply_text="群友画像设置功能尚未绑定。",
                command_name="profile",
            )

        confirm = self._on_profile(target_qq, notes).strip()
        if not confirm:
            reply = f"已更新用户 {target_qq} 的画像备注：{notes}"
        elif str(target_qq) in confirm:
            reply = confirm
        else:
            reply = f"已更新用户 {target_qq} 画像：{confirm}"
        return CommandResult(
            handled=True,
            reply_text=reply,
            command_name="profile",
        )

    def try_handle(self, msg: InboundMessage) -> CommandResult:
        extracted = self._extract_command_and_args(msg)
        if extracted is None:
            return CommandResult(handled=False)

        cmd, arg_text = extracted

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

        if cmd == "status":
            return self._handle_status(msg)

        if cmd == "stop":
            return self._handle_stop(msg, arg_text)

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

        if cmd == "persona":
            return self._handle_persona(arg_text)

        if cmd == "remember":
            return self._handle_remember(msg, arg_text)

        if cmd == "memories":
            return self._handle_memories(msg)

        if cmd == "forget":
            return self._handle_forget(arg_text)

        if cmd == "profile":
            return self._handle_profile(arg_text)

        # cmd == "help"
        return CommandResult(
            handled=True,
            reply_text=_HELP_TEXT,
            command_name="help",
        )
