"""Command-line entry point for `python -m qq_agent` (3-process DOGFOOD-FULL supervisor/adapter/brain & `--direct-ws`)."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from .agent import AgentKernel, ManagedLlmosDaemon
from .channel_bridge import (
    DirectMemoryBus,
    LlmosChannelBus,
    MessageBus,
    NapCatWsAdapter,
    ResilientLlmosBus,
    _load_channel_client_cls,
)
from .config import AgentConfig
from .guardrails import GuardrailManager
from .memory import MemoryStore
from .persona import PersonaManager
from .provider import OpenAICompatProvider
from .session import SessionStore
from .stickers import StickerRegistry
from .tools import WebPageFetcher, WebSearchClient
from .trigger import RuleTrigger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_DEFAULT_PERSONA = _REPO_ROOT / "config" / "persona.default.toml"
_REPO_DEFAULT_STICKERS = _REPO_ROOT / "config" / "stickers.toml"

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qq-agent",
        description="跑在 llmos 上的全异步 QQ 群友智能体（支持三进程故障隔离 DOGFOOD-FULL 与 --direct-ws 模式）",
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
        "--role",
        choices=("supervisor", "adapter", "brain", "all-in-one"),
        default=None,
        help="进程角色：supervisor（默认三进程编排守护）、adapter（独立 WS 桥接）、brain（独立异步内核）、all-in-one（单进程合体）",
    )
    parser.add_argument(
        "--llmos-endpoint",
        default=None,
        help="指定或附着已运行的 llmos system-control-daemon plain_socket 端点路径",
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
    if args.role:
        cfg.runtime_role = args.role
    elif cfg.mode == "direct-ws":
        cfg.runtime_role = "all-in-one"

    if args.llmos_endpoint:
        cfg.llmos_plain_endpoint = str(args.llmos_endpoint).strip()
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
    if (
        cfg.stickers_manifest_path == Path("config/stickers.toml")
        and not cfg.stickers_manifest_path.exists()
        and _REPO_DEFAULT_STICKERS.exists()
    ):
        cfg.stickers_manifest_path = _REPO_DEFAULT_STICKERS
    return cfg


def _run_dry_check(cfg: AgentConfig) -> int:
    persona = PersonaManager(
        cfg.persona_path,
        local_overlay_path=cfg.effective_persona_local_path,
        personas_dir=cfg.effective_personas_dir,
    )
    p_cfg = persona.current()
    profiles = persona.list_profiles()
    stickers = StickerRegistry(cfg.stickers_dir, manifest_path=cfg.stickers_manifest_path)
    sticker_count = len(stickers.available_entries())

    if cfg.data_dir.exists():
        with (
            SessionStore(cfg.sessions_db_path, max_window=24) as sessions,
            MemoryStore(cfg.memory_db_path) as memory,
            GuardrailManager(
                cfg.guardrails_db_path,
                user_cooldown_s=1.0,
                max_daily_calls=cfg.daily_max_calls,
                max_daily_tokens=cfg.daily_max_tokens,
                soft_daily_budget_micros=cfg.daily_soft_cost_micros_usd,
                hard_daily_budget_micros=cfg.daily_hard_cost_micros_usd,
                cheap_model=cfg.llm_cheap_model,
            ) as guardrails,
        ):
            sessions.get_window("__dry_check__")
            memory.get_summary("__dry_check__")
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
                SessionStore(tmp_root / "sessions.sqlite3", max_window=24) as sessions,
                MemoryStore(tmp_root / "memory.sqlite3") as memory,
                GuardrailManager(
                    tmp_root / "guardrails.sqlite3",
                    user_cooldown_s=1.0,
                    max_daily_calls=cfg.daily_max_calls,
                    max_daily_tokens=cfg.daily_max_tokens,
                    soft_daily_budget_micros=cfg.daily_soft_cost_micros_usd,
                    hard_daily_budget_micros=cfg.daily_hard_cost_micros_usd,
                    cheap_model=cfg.llm_cheap_model,
                ) as guardrails,
            ):
                sessions.get_window("__dry_check__")
                memory.get_summary("__dry_check__")
                snap = guardrails.snapshot()
        sqlite_desc = f"OK (临时自检库已通过 WAL 初始化校验，目标 data_dir={cfg.data_dir})"

    if cfg.mode == "llmos":
        channel_cls = _load_channel_client_cls(cfg.llmos_root)
        req_fn = getattr(channel_cls, "service_requirement", None)
        if callable(req_fn):
            req = req_fn()
            bus_desc = (
                f"llmos DOGFOOD-FULL (role={cfg.runtime_role}, nlos_sdk.ChannelClient OK, "
                f"schema={req.schema_name} v{req.major}.{req.minimum_minor}, daemon_root={cfg.llmos_daemon_root})"
            )
        else:
            bus_desc = (
                f"llmos DOGFOOD-FULL (role={cfg.runtime_role}, nlos_sdk.ChannelClient OK, "
                f"daemon_root={cfg.llmos_daemon_root})"
            )
    else:
        bus_desc = f"direct-ws (role={cfg.runtime_role}, DirectMemoryBus 单进程内存双队列)"

    key_status = "已配置" if cfg.llm_api_key.strip() else "未配置（--dry-check / --role adapter 豁免，brain 启动前需填写）"
    groups_desc = (
        ",".join(str(g) for g in sorted(cfg.allowed_group_ids))
        if cfg.allowed_group_ids
        else "未限制（允许所有所在群）"
    )
    aliases_desc = "、".join(p_cfg.aliases) if p_cfg.aliases else "无"
    profile_desc = p_cfg.active_profile if p_cfg.active_profile else "default"
    archived_desc = "、".join(profiles) if profiles else "无"
    search_engine_desc = "API+BingCN+DDG" if (cfg.search_api_key and cfg.search_api_key.strip()) else "BingCN+DDG (免Key)"

    summary = (
        "=== qq-agent 健康自检 (--dry-check) 通过 ===\n"
        f"运行模式   : {bus_desc}\n"
        f"NapCat WS  : {cfg.napcat_ws_url}\n"
        f"机器人 QQ  : {cfg.bot_qq if cfg.bot_qq is not None else '自动跟随 OB11 self_id'}\n"
        f"管理员数量 : {len(cfg.admin_user_ids)}\n"
        f"群聊白名单 : {groups_desc}\n"
        f"当前人设   : {p_cfg.name}（别名: {aliases_desc} | 激活档案: {profile_desc} | 归档库: {archived_desc}）\n"
        f"快慢双脑   : 快脑 max_steps={cfg.agent_max_steps} (thinking={'ON' if cfg.llm_enable_thinking else 'OFF'}, compact={'ON' if cfg.compact_prompt else 'OFF'}) | "
        f"慢脑={'ON' if cfg.subagent_enabled else 'OFF'} (model={cfg.effective_subagent_model}, "
        f"thinking={'ON' if cfg.subagent_enable_thinking else 'OFF'}, max_steps={cfg.subagent_max_steps}, "
        f"concurrency={cfg.subagent_max_concurrent}/per_group={cfg.subagent_max_per_session}, timeout={cfg.subagent_timeout_s}s)\n"
        f"联网搜索   : {'ON' if cfg.web_search_enabled else 'OFF'} ({search_engine_desc}, timeout={cfg.web_search_timeout_s}s, fact_ttl={cfg.fact_cache_ttl_s:.0f}s)\n"
        f"群聊触发率 : 追问窗口={cfg.group_followup_window_s}s, 兴趣词={cfg.group_topic_prob:.0%}, 随机插话={cfg.group_proactive_prob:.0%}, 保底={cfg.group_proactive_every_n}条\n"
        f"输入防抖   : 私聊={cfg.private_debounce_s}s, 群聊={cfg.group_debounce_s}s, 气泡打字延迟={cfg.bubble_typing_delay_s}s\n"
        f"表情包图库 : 已就绪 {sticker_count} 张（目录: {cfg.stickers_dir}）\n"
        f"LLM 接口   : {cfg.llm_base_url} (model={cfg.llm_model}, vision={'ON' if cfg.enable_vision else 'OFF'}@{cfg.vision_max_side}px, API Key={key_status})\n"
        f"SQLite 护栏: {sqlite_desc} | 当日调用 {snap.calls_used}/{snap.max_daily_calls}"
    )
    print(summary, flush=True)
    return 0


def _build_kernel_from_config(
    cfg: AgentConfig,
    *,
    bus: MessageBus,
    provider: OpenAICompatProvider,
    persona: PersonaManager,
    stickers: StickerRegistry,
    sessions: SessionStore,
    memory: MemoryStore,
    guardrails: GuardrailManager,
    trigger: RuleTrigger,
    runtime_role: str,
) -> AgentKernel:
    search_client = WebSearchClient(
        timeout_s=cfg.web_search_timeout_s,
        search_api_key=cfg.search_api_key,
        search_api_base_url=cfg.search_api_base_url,
    )
    page_fetcher = WebPageFetcher(timeout_s=max(8.0, float(cfg.web_search_timeout_s)))

    return AgentKernel(
        bus=bus,
        provider=provider,
        persona=persona,
        sessions=sessions,
        memory=memory,
        trigger=trigger,
        guardrails=guardrails,
        stickers=stickers,
        enable_vision=cfg.enable_vision,
        vision_max_side=cfg.vision_max_side,
        compact_prompt=cfg.compact_prompt,
        fact_cache_ttl_s=cfg.fact_cache_ttl_s,
        bot_qq=cfg.bot_qq,
        admin_user_ids=cfg.admin_user_ids,
        private_debounce_s=cfg.private_debounce_s,
        group_debounce_s=cfg.group_debounce_s,
        agent_max_steps=cfg.agent_max_steps,
        agent_turn_timeout_s=cfg.agent_turn_timeout_s,
        bubble_typing_delay_s=cfg.bubble_typing_delay_s,
        web_search_enabled=cfg.web_search_enabled,
        web_search_timeout_s=cfg.web_search_timeout_s,
        search_api_key=cfg.search_api_key,
        search_api_base_url=cfg.search_api_base_url,
        web_search_client=search_client,
        webpage_fetcher=page_fetcher,
        subagent_enabled=cfg.subagent_enabled,
        subagent_model=cfg.subagent_model,
        subagent_enable_thinking=cfg.subagent_enable_thinking,
        subagent_max_steps=cfg.subagent_max_steps,
        subagent_timeout_s=cfg.subagent_timeout_s,
        subagent_max_concurrent=cfg.subagent_max_concurrent,
        subagent_max_per_session=cfg.subagent_max_per_session,
        runtime_role=runtime_role,  # type: ignore[arg-type]
    )


async def run_adapter_only(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run isolated NapCat OB11 WebSocket <-> `ResilientLlmosBus` adapter worker process."""
    if not cfg.llmos_plain_endpoint:
        raise ValueError("--role adapter requires --llmos-endpoint or LLMOS_PLAIN_ENDPOINT")

    effective_stop = stop_event if stop_event is not None else asyncio.Event()
    persona = PersonaManager(
        cfg.persona_path,
        local_overlay_path=cfg.effective_persona_local_path,
        personas_dir=cfg.effective_personas_dir,
    )
    stickers = StickerRegistry(cfg.stickers_dir, manifest_path=cfg.stickers_manifest_path)

    bus = ResilientLlmosBus(
        cfg.llmos_plain_endpoint,
        llmos_root=cfg.llmos_root,
        role="adapter",
    )
    await bus.start()
    try:
        p_now = persona.current()
        adapter = NapCatWsAdapter(
            cfg.napcat_ws_url,
            bus,
            bot_qq=cfg.bot_qq,
            bot_name=p_now.name,
            access_token=cfg.napcat_access_token,
            auto_approve_friend_ids=cfg.admin_user_ids,
            auto_approve_group_ids=cfg.allowed_group_ids,
            stickers=stickers,
        )
        logger.info(
            "qq-agent [adapter] worker ready (pid=%d, ws=%s, endpoint=%s, bot_name=%s, stickers=%d)",
            os.getpid(),
            cfg.napcat_ws_url,
            cfg.llmos_plain_endpoint,
            p_now.name,
            len(stickers.available_entries()),
        )
        await adapter.run_forever(stop_event=effective_stop)
    finally:
        await bus.close()


async def run_brain_only(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run isolated Pi-Agent `AgentKernel` + `SubAgentRunner` <-> `ResilientLlmosBus` brain worker process."""
    if not cfg.llmos_plain_endpoint:
        raise ValueError("--role brain requires --llmos-endpoint or LLMOS_PLAIN_ENDPOINT")

    effective_stop = stop_event if stop_event is not None else asyncio.Event()

    async with AsyncExitStack() as stack:
        persona = PersonaManager(
            cfg.persona_path,
            local_overlay_path=cfg.effective_persona_local_path,
            personas_dir=cfg.effective_personas_dir,
        )
        stickers = StickerRegistry(cfg.stickers_dir, manifest_path=cfg.stickers_manifest_path)
        sessions = stack.enter_context(SessionStore(cfg.sessions_db_path, max_window=24))
        memory = stack.enter_context(MemoryStore(cfg.memory_db_path))
        guardrails = stack.enter_context(
            GuardrailManager(
                cfg.guardrails_db_path,
                user_cooldown_s=1.0,
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
            followup_window_s=cfg.group_followup_window_s,
            topic_prob=cfg.group_topic_prob,
            proactive_prob=cfg.group_proactive_prob,
            proactive_every_n=cfg.group_proactive_every_n,
            proactive_cooldown_s=cfg.group_proactive_cooldown_s,
        )
        provider = await stack.enter_async_context(
            OpenAICompatProvider(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                default_model=cfg.llm_model,
                enable_thinking=cfg.llm_enable_thinking,
            )
        )

        bus = ResilientLlmosBus(
            cfg.llmos_plain_endpoint,
            llmos_root=cfg.llmos_root,
            role="brain",
        )
        await bus.start()
        stack.push_async_callback(bus.close)

        kernel = _build_kernel_from_config(
            cfg,
            bus=bus,
            provider=provider,
            persona=persona,
            stickers=stickers,
            sessions=sessions,
            memory=memory,
            guardrails=guardrails,
            trigger=trigger,
            runtime_role="brain",
        )
        p_now = persona.current()
        logger.info(
            "qq-agent [brain] worker ready (pid=%d, endpoint=%s, model=%s, fast_thinking=%s, compact=%s, debounce=%.2fs/%.2fs, subagent=%s/%s[max=%d/group=%d], web_search=%s[ttl=%.0fs], vision_max=%dpx, profile=%s)",
            os.getpid(),
            cfg.llmos_plain_endpoint,
            cfg.llm_model,
            "on" if cfg.llm_enable_thinking else "off",
            "on" if cfg.compact_prompt else "off",
            cfg.private_debounce_s,
            cfg.group_debounce_s,
            "on" if cfg.subagent_enabled else "off",
            "thinking:on" if cfg.subagent_enable_thinking else "thinking:off",
            cfg.subagent_max_concurrent,
            cfg.subagent_max_per_session,
            "on" if cfg.web_search_enabled else "off",
            cfg.fact_cache_ttl_s,
            cfg.vision_max_side,
            p_now.active_profile or "default",
        )
        await kernel.run_forever(stop_event=effective_stop)


async def run_all_in_one(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Single-process runtime running `NapCatWsAdapter` + `AgentKernel` concurrently."""
    effective_stop = stop_event if stop_event is not None else asyncio.Event()

    async with AsyncExitStack() as stack:
        persona = PersonaManager(
            cfg.persona_path,
            local_overlay_path=cfg.effective_persona_local_path,
            personas_dir=cfg.effective_personas_dir,
        )
        stickers = StickerRegistry(cfg.stickers_dir, manifest_path=cfg.stickers_manifest_path)
        sessions = stack.enter_context(SessionStore(cfg.sessions_db_path, max_window=24))
        memory = stack.enter_context(MemoryStore(cfg.memory_db_path))
        guardrails = stack.enter_context(
            GuardrailManager(
                cfg.guardrails_db_path,
                user_cooldown_s=1.0,
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
            followup_window_s=cfg.group_followup_window_s,
            topic_prob=cfg.group_topic_prob,
            proactive_prob=cfg.group_proactive_prob,
            proactive_every_n=cfg.group_proactive_every_n,
            proactive_cooldown_s=cfg.group_proactive_cooldown_s,
        )
        provider = await stack.enter_async_context(
            OpenAICompatProvider(
                api_key=cfg.llm_api_key,
                base_url=cfg.llm_base_url,
                default_model=cfg.llm_model,
                enable_thinking=cfg.llm_enable_thinking,
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
                    ManagedLlmosDaemon(
                        cfg.llmos_root,
                        data_dir=cfg.data_dir,
                        root_dir=cfg.llmos_daemon_root,
                    )
                )
                endpoint = daemon.plain_endpoint
            bus = LlmosChannelBus(endpoint, llmos_root=cfg.llmos_root, role="all-in-one")

        await bus.start()
        stack.push_async_callback(bus.close)

        kernel = _build_kernel_from_config(
            cfg,
            bus=bus,
            provider=provider,
            persona=persona,
            stickers=stickers,
            sessions=sessions,
            memory=memory,
            guardrails=guardrails,
            trigger=trigger,
            runtime_role="all-in-one",
        )
        p_now = persona.current()
        adapter = NapCatWsAdapter(
            cfg.napcat_ws_url,
            bus,
            bot_qq=cfg.bot_qq,
            bot_name=p_now.name,
            access_token=cfg.napcat_access_token,
            auto_approve_friend_ids=cfg.admin_user_ids,
            auto_approve_group_ids=cfg.allowed_group_ids,
            stickers=stickers,
        )

        logger.info(
            "qq-agent [all-in-one] runtime ready (mode=%s, ws=%s, model=%s, thinking=%s, subagent=%s, search=%s, vision=%s, profile=%s, archives=%s, debounce=%.1fs/%.1fs, stickers=%d, admins=%d, groups=%s)",
            cfg.mode,
            cfg.napcat_ws_url,
            cfg.llm_model,
            "on" if cfg.llm_enable_thinking else "off",
            "on" if cfg.subagent_enabled else "off",
            "on" if cfg.web_search_enabled else "off",
            "on" if cfg.enable_vision else "off",
            p_now.active_profile or "default",
            persona.list_profiles(),
            cfg.private_debounce_s,
            cfg.group_debounce_s,
            len(stickers.available_entries()),
            len(cfg.admin_user_ids),
            sorted(cfg.allowed_group_ids) if cfg.allowed_group_ids else "ALL",
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


async def _forward_child_stream(
    label: str,
    stream: asyncio.StreamReader | None,
) -> None:
    if stream is None:
        return
    try:
        while True:
            raw_line = await stream.readline()
            if not raw_line:
                break
            text = raw_line.decode("utf-8", errors="replace").rstrip()
            if text:
                print(f"[{label}] {text}", file=sys.stderr, flush=True)
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def _terminate_child_process(
    proc: asyncio.subprocess.Process | None,
    *,
    grace_s: float = 3.0,
) -> None:
    if proc is None or proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


def _write_runtime_state(
    path: Path,
    *,
    status: str,
    plain_endpoint: str,
    daemon_root: Path,
    daemon_pid: int | None,
    adapter_pid: int | None,
    brain_pid: int | None,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status,
            "supervisor_pid": os.getpid(),
            "daemon_pid": daemon_pid,
            "adapter_pid": adapter_pid,
            "brain_pid": brain_pid,
            "plain_endpoint": plain_endpoint,
            "llmos_daemon_root": str(daemon_root),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except Exception as exc:
        logger.debug("Failed writing llmos_runtime.json (%s): %s", path, exc)


async def run_supervisor(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
    env_file: Path | str = ".env",
) -> None:
    """Orchestrate and independently auto-heal 3 isolated OS processes: `llmos daemon` + `adapter` + `brain`."""
    effective_stop = stop_event if stop_event is not None else asyncio.Event()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    managed_daemon: ManagedLlmosDaemon | None = None
    if cfg.llmos_plain_endpoint:
        endpoint = cfg.llmos_plain_endpoint
        auth_endpoint: str | None = None
    else:
        managed_daemon = ManagedLlmosDaemon(
            cfg.llmos_root,
            data_dir=cfg.data_dir,
            root_dir=cfg.llmos_daemon_root,
        )
        endpoint = await managed_daemon.start()
        auth_endpoint = managed_daemon.auth_endpoint

    # Pre-initialize both deterministic durable queues once so adapter & brain attach race-free.
    bootstrap_bus = ResilientLlmosBus(
        endpoint,
        llmos_root=cfg.llmos_root,
        role="supervisor",
    )
    await bootstrap_bus.start()
    await bootstrap_bus.close()

    child_env = dict(os.environ)
    child_env["PYTHONUTF8"] = "1"
    child_env["LLMOS_ROOT"] = str(cfg.llmos_root.resolve())
    child_env["LLMOS_PLAIN_ENDPOINT"] = endpoint
    child_env["QQ_DATA_DIR"] = str(cfg.data_dir.resolve())
    child_env["QQ_PERSONA_PATH"] = str(cfg.persona_path.resolve())
    child_env["QQ_STICKERS_MANIFEST"] = str(cfg.stickers_manifest_path.resolve())
    child_env["NAPCAT_WS_URL"] = cfg.napcat_ws_url
    if cfg.llm_api_key:
        child_env["LLM_API_KEY"] = cfg.llm_api_key
    child_env["LLM_BASE_URL"] = cfg.llm_base_url
    child_env["LLM_MODEL"] = cfg.llm_model
    if cfg.llm_cheap_model:
        child_env["LLM_CHEAP_MODEL"] = cfg.llm_cheap_model

    resolved_env_arg = str(Path(env_file))
    stream_tasks: set[asyncio.Task[None]] = set()

    async def _spawn_worker(role: str) -> asyncio.subprocess.Process:
        cmd = [
            sys.executable,
            "-m",
            "qq_agent",
            "--env-file",
            resolved_env_arg,
            "--role",
            role,
            "--llmos-endpoint",
            endpoint,
            "--ws-url",
            cfg.napcat_ws_url,
            "--persona",
            str(cfg.persona_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=_REPO_ROOT,
            env=child_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        for stream in (proc.stdout, proc.stderr):
            t = asyncio.create_task(_forward_child_stream(role, stream))
            stream_tasks.add(t)
            t.add_done_callback(stream_tasks.discard)
        logger.info("qq-agent [supervisor] spawned %s worker (pid=%s)", role, proc.pid)
        return proc

    adapter_proc: asyncio.subprocess.Process | None = None
    brain_proc: asyncio.subprocess.Process | None = None
    adapter_failures = 0
    brain_failures = 0
    adapter_spawn_ts = time.monotonic()
    brain_spawn_ts = time.monotonic()

    try:
        adapter_proc = await _spawn_worker("adapter")
        adapter_spawn_ts = time.monotonic()
        brain_proc = await _spawn_worker("brain")
        brain_spawn_ts = time.monotonic()

        _write_runtime_state(
            cfg.llmos_runtime_state_path,
            status="running",
            plain_endpoint=endpoint,
            daemon_root=cfg.llmos_daemon_root,
            daemon_pid=managed_daemon.pid if managed_daemon is not None else None,
            adapter_pid=adapter_proc.pid,
            brain_pid=brain_proc.pid,
        )
        logger.info(
            "qq-agent [supervisor] 3-process DOGFOOD-FULL online (daemon_pid=%s, adapter_pid=%s, brain_pid=%s, root=%s)",
            managed_daemon.pid if managed_daemon is not None else "external",
            adapter_proc.pid,
            brain_proc.pid,
            cfg.llmos_daemon_root,
        )

        stop_wait_task = asyncio.create_task(effective_stop.wait())
        try:
            while not effective_stop.is_set():
                adapter_wait = asyncio.create_task(adapter_proc.wait())
                brain_wait = asyncio.create_task(brain_proc.wait())
                wait_set: set[asyncio.Task[Any]] = {stop_wait_task, adapter_wait, brain_wait}
                daemon_wait: asyncio.Task[Any] | None = None
                if managed_daemon is not None and managed_daemon.is_running:
                    daemon_wait = asyncio.create_task(managed_daemon.wait())
                    wait_set.add(daemon_wait)

                done, pending = await asyncio.wait(
                    wait_set,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    if p is not stop_wait_task:
                        p.cancel()
                await asyncio.gather(
                    *(p for p in pending if p is not stop_wait_task),
                    return_exceptions=True,
                )

                if effective_stop.is_set() or stop_wait_task in done:
                    break

                # 1) Auto-heal system-control-daemon on identical Named Pipe endpoint + durable root
                if (
                    daemon_wait is not None
                    and daemon_wait in done
                    and managed_daemon is not None
                    and not effective_stop.is_set()
                ):
                    logger.error(
                        "llmos system-control-daemon exited unexpectedly; restarting on same endpoint=%s",
                        endpoint,
                    )
                    await managed_daemon.stop()
                    managed_daemon = ManagedLlmosDaemon(
                        cfg.llmos_root,
                        data_dir=cfg.data_dir,
                        root_dir=cfg.llmos_daemon_root,
                        plain_endpoint=endpoint,
                        auth_endpoint=auth_endpoint,
                    )
                    await managed_daemon.start()

                # 2) Auto-heal adapter worker without interrupting brain!
                if adapter_proc.returncode is not None and not effective_stop.is_set():
                    uptime = time.monotonic() - adapter_spawn_ts
                    adapter_failures = 1 if uptime > 30.0 else (adapter_failures + 1)
                    delay = min(15.0, 0.8 * (2 ** min(adapter_failures - 1, 4)))
                    logger.warning(
                        "adapter worker exited (rc=%s, uptime=%.1fs); respawning in %.1fs (brain stays online)",
                        adapter_proc.returncode,
                        uptime,
                        delay,
                    )
                    try:
                        await asyncio.wait_for(effective_stop.wait(), timeout=delay)
                        break
                    except TimeoutError:
                        pass
                    adapter_proc = await _spawn_worker("adapter")
                    adapter_spawn_ts = time.monotonic()

                # 3) Auto-heal brain worker without dropping NapCat WS connection!
                if brain_proc.returncode is not None and not effective_stop.is_set():
                    uptime = time.monotonic() - brain_spawn_ts
                    brain_failures = 1 if uptime > 30.0 else (brain_failures + 1)
                    delay = min(15.0, 0.8 * (2 ** min(brain_failures - 1, 4)))
                    logger.warning(
                        "brain worker exited (rc=%s, uptime=%.1fs); respawning in %.1fs (adapter WS stays online)",
                        brain_proc.returncode,
                        uptime,
                        delay,
                    )
                    try:
                        await asyncio.wait_for(effective_stop.wait(), timeout=delay)
                        break
                    except TimeoutError:
                        pass
                    brain_proc = await _spawn_worker("brain")
                    brain_spawn_ts = time.monotonic()

                _write_runtime_state(
                    cfg.llmos_runtime_state_path,
                    status="running",
                    plain_endpoint=endpoint,
                    daemon_root=cfg.llmos_daemon_root,
                    daemon_pid=managed_daemon.pid if managed_daemon is not None else None,
                    adapter_pid=adapter_proc.pid if adapter_proc is not None else None,
                    brain_pid=brain_proc.pid if brain_proc is not None else None,
                )
        finally:
            if not stop_wait_task.done():
                stop_wait_task.cancel()
                await asyncio.gather(stop_wait_task, return_exceptions=True)
    finally:
        await asyncio.gather(
            _terminate_child_process(brain_proc),
            _terminate_child_process(adapter_proc),
            return_exceptions=True,
        )
        if stream_tasks:
            for st in list(stream_tasks):
                st.cancel()
            await asyncio.gather(*stream_tasks, return_exceptions=True)
        if managed_daemon is not None:
            await managed_daemon.stop()
        _write_runtime_state(
            cfg.llmos_runtime_state_path,
            status="stopped",
            plain_endpoint=endpoint,
            daemon_root=cfg.llmos_daemon_root,
            daemon_pid=None,
            adapter_pid=None,
            brain_pid=None,
        )


async def run_runtime(
    cfg: AgentConfig,
    *,
    stop_event: asyncio.Event | None = None,
    env_file: Path | str = ".env",
) -> None:
    """Dispatch runtime execution according to `cfg.mode` and `cfg.runtime_role`."""
    if cfg.mode == "direct-ws" or cfg.runtime_role == "all-in-one":
        await run_all_in_one(cfg, stop_event=stop_event)
        return
    if cfg.runtime_role == "adapter":
        await run_adapter_only(cfg, stop_event=stop_event)
        return
    if cfg.runtime_role == "brain":
        await run_brain_only(cfg, stop_event=stop_event)
        return
    await run_supervisor(cfg, stop_event=stop_event, env_file=env_file)


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

    if cfg.runtime_role != "adapter" and not cfg.llm_api_key.strip():
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
        asyncio.run(run_runtime(cfg, env_file=args.env_file))
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
    "run_adapter_only",
    "run_all_in_one",
    "run_brain_only",
    "run_runtime",
    "run_supervisor",
]
