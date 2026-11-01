"""3-process DOGFOOD-FULL supervisor orchestrating and auto-healing `llmos daemon`, `adapter`, and `brain`."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from .channel_bridge import ResilientLlmosBus
from .config import AgentConfig
from .daemon_manager import ManagedLlmosDaemon

_REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


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


__all__ = [
    "_forward_child_stream",
    "_terminate_child_process",
    "_write_runtime_state",
    "run_supervisor",
]
