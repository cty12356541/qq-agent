"""Command-line entry point for `python -m qq_agent` (DOGFOOD-FULL & `--direct-ws` modes)."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
import logging
from pathlib import Path
import sys
import tempfile

from .agent import AgentKernel, ManagedLlmosDaemon
from .channel_bridge import (
    DirectMemoryBus,
    LlmosChannelBus,
    MessageBus,
    NapCatWsAdapter,
    _load_channel_client_cls,
)
from .config import AgentConfig
from .guardrails import GuardrailManager
from .persona import PersonaManager
from .provider import OpenAICompatProvider
from .session import SessionStore
from .trigger import RuleTrigger

_REPO_DEFAULT_PERSONA = Path(__file__).resolve().parents[1] / "config" / "persona.default.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qq-agent",
        description="跑在 llmos 上的 QQ 群友智能体（支持 DOGFOOD-FULL 与 --direct-ws 双模运行）",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="环境变量文件路径（默认: .env；不存在时仅读取当前进程环境变量）",
    )
    parser.add_argument(
        "--direct-ws",
        action="store_true",
        help="切换为单进程内存消息总线模式（不依赖 llmos system-control-daemon）",
    )
    parser.add_argument(
        "--ws-url",
        default=None,
        help="覆盖 NapCat OneBot v11 正向 WebSocket 地址（例如 ws://127.0.0.1:3001）",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="覆盖人设 TOML 配置文件路径（默认: config/persona.default.toml）",
    )
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="执行配置、人设 TOML、SQLite 存储与 llmos SDK 健康自检后退出 0（无需 LLM_API_KEY）",
    )
    return parser


def _resolve_config_from_args(args: argparse.Namespace) -> AgentConfig:
    cfg = AgentConfig.from_env(args.env_file, direct_ws=bool(args.direct_ws))
    if args.ws_url:
        cfg.napcat_ws_url = str(args.ws_url).strip()
    if args.persona:
        cfg.persona_path = Path(args.persona)
    elif (
        cfg.persona_path == Path("config/persona.default.toml")
        and not cfg.persona_path.exists()
        and _REPO_DEFAULT_PERSONA.exists()
    ):
        cfg.persona_path = _REPO_DEFAULT_PERSONA
    return cfg


def _run_dry_check(cfg: AgentConfig) -> int:
    persona = PersonaManager(cfg.persona_path)
    p_cfg = persona.current()

    if cfg.data_dir.exists():
        with (
            SessionStore(cfg.sessions_db_path) as sessions,
            GuardrailManager(
                cfg.guardrails_db_path,
                max_daily_calls=cfg.daily_max_calls,
                max_daily_tokens=cfg.daily_max_tokens,
                soft_daily_budget_micros=cfg.daily_soft_cost_micros_usd,
                hard_daily_budget_micros=cfg.daily_hard_cost_micros_usd,
                cheap_model=cfg.llm_cheap_model,
            ) as guardrails,
        ):
            sessions.get_window("__dry_check__")
            snap = guardrails.snapshot()
        sqlite_desc = f"OK (data_dir={cfg.data_dir})"
    else:
        parent_dir = cfg.data_dir.resolve().parent
        if not parent_dir.exists():
            raise FileNotFoundError(
                f"QQ_DATA_DIR 父目录不存在，无法初始化 SQLite: {parent_dir}"
            )
        with tempfile.TemporaryDirectory(prefix="qq-agent-dry-check-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            with (
                SessionStore(tmp_root / "sessions.sqlite3") as sessions,
                GuardrailManager(
                    tmp_root / "guardrails.sqlite3",
                    max_daily_calls=cfg.daily_max_calls,
                    max_daily_tokens=cfg.daily_max_tokens,
                    soft_daily_budget_micros=cfg.daily_soft_cost_micros_usd,
                    hard_daily_budget_micros=cfg.daily_hard_cost_micros_usd,
                    cheap_model=cfg.llm_cheap_model,
                ) as guardrails,
            ):
                sessions.get_window("__dry_check__")
                snap = guardrails.snapshot()
        sqlite_desc = f"OK (临时自检库已通过 WAL 初始化校验，目标 data_dir={cfg.data_dir})"

    if cfg.mode == "llmos":
        channel_cls = _load_channel_client_cls(cfg.llmos_root)
        req_fn = getattr(channel_cls, "service_requirement", None)
        if callable(req_fn):
            req = req_fn()
            bus_desc = (
                f"llmos DOGFOOD-FULL (nlos_sdk.ChannelClient OK, "
                f"schema={req.schema_name} v{req.major}.{req.minimum_minor}, root={cfg.llmos_root})"
            )
        else:
            bus_desc = f"llmos DOGFOOD-FULL (nlos_sdk.ChannelClient OK, root={cfg.llmos_root})"
    else:
        bus_desc = "direct-ws (DirectMemoryBus 单进程内存双队列)"

    key_status = "已配置" if cfg.llm_api_key.strip() else "未配置（--dry-check 豁免，正式启动前需填写）"
    groups_desc = (
        ",".join(str(g) for g in sorted(cfg.allowed_group_ids))
        if cfg.allowed_group_ids
        else "未限制（允许所有所在群）"
    )
    aliases_desc = "、".join(p_cfg.aliases) if p_cfg.aliases else "无"

    summary = (
        "=== qq-agent 健康自检 (--dry-check) 通过 ===\n"
        f"运行模式   : {bus_desc}\n"
        f"NapCat WS  : {cfg.napcat_ws_url}\n"
        f"机器人 QQ  : {cfg.bot_qq if cfg.bot_qq is not None else '自动跟随 OB11 self_id'}\n"
        f"管理员数量 : {len(cfg.admin_user_ids)}\n"
        f"群聊白名单 : {groups_desc}\n"
        f"当前人设   : {p_cfg.name}（别名: {aliases_desc} | 文件: {cfg.persona_path}）\n"
        f"LLM 接口   : {cfg.llm_base_url} (model={cfg.llm_model}, API Key={key_status})\n"
        f"SQLite 护栏: {sqlite_desc} | 当日调用 {snap.calls_used}/{snap.max_daily_calls}"
    )
    print(summary, flush=True)
    return 0


async def run_runtime(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Assemble and run `NapCatWsAdapter` + `AgentKernel` concurrently until stopped."""
    effective_stop = stop_event if stop_event is not None else asyncio.Event()

    async with AsyncExitStack() as stack:
        persona = PersonaManager(cfg.persona_path)
        sessions = stack.enter_context(SessionStore(cfg.sessions_db_path))
        guardrails = stack.enter_context(
            GuardrailManager(
                cfg.guardrails_db_path,
                max_daily_calls=cfg.daily_max_calls,
                max_daily_tokens=cfg.daily_max_tokens,
                soft_daily_budget_micros=cfg.daily_soft_cost_micros_usd,
                hard_daily_budget_micros=cfg.daily_hard_cost_micros_usd,
                cheap_model=cfg.llm_cheap_model,
            )
        )
        trigger = RuleTrigger(
            bot_qq=cfg.bot_qq,
            allowed_group_ids=cfg.allowed_group_ids,
        )
        provider = await stack.enter_async_context(
            OpenAICompatProvider(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                default_model=cfg.llm_model,
            )
        )

        bus: MessageBus
        if cfg.mode == "direct-ws":
            bus = DirectMemoryBus()
        else:
            if cfg.llmos_plain_endpoint:
                endpoint = cfg.llmos_plain_endpoint
            else:
                daemon = await stack.enter_async_context(
                    ManagedLlmosDaemon(cfg.llmos_root, data_dir=cfg.data_dir)
                )
                endpoint = daemon.plain_endpoint
            bus = LlmosChannelBus(endpoint, llmos_root=cfg.llmos_root)

        await bus.start()
        stack.push_async_callback(bus.close)

        kernel = AgentKernel(
            bus=bus,
            provider=provider,
            persona=persona,
            sessions=sessions,
            trigger=trigger,
            guardrails=guardrails,
            bot_qq=cfg.bot_qq,
            admin_user_ids=cfg.admin_user_ids,
        )
        adapter = NapCatWsAdapter(
            cfg.napcat_ws_url,
            bus,
            bot_qq=cfg.bot_qq,
            access_token=cfg.napcat_access_token,
        )

        adapter_task = asyncio.create_task(adapter.run_forever(stop_event=effective_stop))
        kernel_task = asyncio.create_task(kernel.run_forever(stop_event=effective_stop))
        tasks = {adapter_task, kernel_task}

        try:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            effective_stop.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            effective_stop.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code (0 on success, non-zero on failure)."""
    if argv is None and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        cfg = _resolve_config_from_args(args)
    except Exception as exc:
        print(f"[qq-agent] 配置加载失败：{exc}", file=sys.stderr, flush=True)
        return 2

    if args.dry_check:
        try:
            return _run_dry_check(cfg)
        except Exception as exc:
            print(f"[qq-agent] 健康自检失败：{exc}", file=sys.stderr, flush=True)
            return 1

    if not cfg.llm_api_key.strip():
        print(
            "[qq-agent] 启动失败：未检测到有效的 LLM_API_KEY。\n"
            "请先将 .env.example 复制为 .env 并填入 LLM_API_KEY（或设置进程环境变量 LLM_API_KEY），\n"
            "也可先运行 `python -m qq_agent --dry-check` 进行无密钥健康自检。",
            file=sys.stderr,
            flush=True,
        )
        return 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(run_runtime(cfg))
    except KeyboardInterrupt:
        print("\n[qq-agent] 收到中断信号，已安全停机退出。", flush=True)
        return 0
    except Exception as exc:
        print(f"[qq-agent] 运行时异常退出：{exc}", file=sys.stderr, flush=True)
        return 1

    return 0


__all__ = [
    "build_parser",
    "main",
    "run_runtime",
]
