"""Lifecycle management for llmos `system-control-daemon` subprocess and local IPC endpoints."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

logger = logging.getLogger(__name__)

_ENDPOINT_SEQ = 0
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_ipc_endpoint(label: str) -> str:
    """Generate a collision-free local Named Pipe (Windows) or UDS path (POSIX)."""
    global _ENDPOINT_SEQ
    _ENDPOINT_SEQ += 1
    unique = f"{os.getpid():x}-{time.time_ns():x}-{_ENDPOINT_SEQ:x}-{label}"
    if sys.platform == "win32":
        return rf"\\.\pipe\qq-agent-cc-{unique}"
    return str(Path(tempfile.gettempdir()) / f"qq-agent-cc-{unique}.sock")


def resolve_daemon_command(
    *,
    llmos_root: Path,
    root_dir: Path,
    auth_endpoint: str,
    plain_endpoint: str,
) -> tuple[list[str], str]:
    """Prefer prebuilt `system-control-daemon[.exe]` binary; fallback to `cargo run`."""
    resolved_llmos = Path(llmos_root).resolve()
    resolved_root = Path(root_dir).resolve()
    exe_name = (
        "system-control-daemon.exe" if sys.platform == "win32" else "system-control-daemon"
    )
    daemon_args = [
        "--root",
        str(resolved_root),
        "--auth-socket",
        auth_endpoint,
        "--plain-socket",
        plain_endpoint,
    ]
    candidates = (
        resolved_llmos / "target" / "debug" / exe_name,
        resolved_llmos / "target" / "release" / exe_name,
        _REPO_ROOT / "data" / "bin" / exe_name,
        Path(tempfile.gettempdir()) / "qq-agent-llmos-target" / "debug" / exe_name,
    )
    for prebuilt in candidates:
        if prebuilt.exists():
            return [str(prebuilt), *daemon_args], f"prebuilt ({prebuilt})"

    cargo_cmd = ["cargo", "run", "--quiet"]
    if (resolved_llmos / "Cargo.lock").exists():
        cargo_cmd.append("--locked")
    cargo_cmd.extend(
        [
            "-p",
            "nlos-system-control",
            "--features",
            "daemon",
            "--bin",
            "system-control-daemon",
            "--",
            *daemon_args,
        ]
    )
    return cargo_cmd, "cargo run fallback"


async def _drain_stream(stream: asyncio.StreamReader | None) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.readline()
            if not chunk:
                break
    except asyncio.CancelledError:
        return
    except Exception:
        return


class ManagedLlmosDaemon:
    """Async context manager that bootstraps and supervises llmos `system-control-daemon`."""

    def __init__(
        self,
        llmos_root: Path | str = r"D:\llmos",
        *,
        data_dir: Path | str = "data",
        root_dir: Path | str | None = None,
        plain_endpoint: str | None = None,
        auth_endpoint: str | None = None,
        startup_timeout_s: float = 180.0,
    ) -> None:
        self.llmos_root: Path = Path(llmos_root).resolve()
        self.root_dir: Path = (
            Path(root_dir) if root_dir is not None else Path(data_dir) / "llmos_daemon"
        ).resolve()
        self._startup_timeout_s = max(1.0, float(startup_timeout_s))

        self.auth_endpoint: str = (
            auth_endpoint.strip() if auth_endpoint and auth_endpoint.strip() else _make_ipc_endpoint("auth")
        )
        self.plain_endpoint: str = (
            plain_endpoint.strip() if plain_endpoint and plain_endpoint.strip() else _make_ipc_endpoint("plain")
        )
        self.launch_mode: str = ""
        self.ready_banner: str = ""
        self.last_cwd: Path | None = None

        self._process: asyncio.subprocess.Process | None = None
        self._drain_tasks: list[asyncio.Task[None]] = []

    @property
    def pid(self) -> int | None:
        if self._process is not None and self._process.returncode is None:
            return self._process.pid
        return None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def wait(self) -> int | None:
        """Wait for the underlying daemon OS process to exit and return its exit code."""
        if self._process is None:
            return None
        return await self._process.wait()

    async def start(self) -> str:
        """Start `system-control-daemon`, wait for `READY `, and return `plain_endpoint`."""
        if self._process is not None and self._process.returncode is None:
            return self.plain_endpoint

        self.root_dir.mkdir(parents=True, exist_ok=True)
        cmd, mode_desc = resolve_daemon_command(
            llmos_root=self.llmos_root,
            root_dir=self.root_dir,
            auth_endpoint=self.auth_endpoint,
            plain_endpoint=self.plain_endpoint,
        )
        self.launch_mode = mode_desc

        sub_env = dict(os.environ)
        if cmd[0] == "cargo":
            run_cwd = self.llmos_root
            if "CARGO_TARGET_DIR" not in sub_env:
                sub_env["CARGO_TARGET_DIR"] = str(
                    Path(tempfile.gettempdir()) / "qq-agent-llmos-target"
                )
        else:
            # Never run prebuilt binaries with D:\llmos as cwd so D:\llmos stays 100% clean.
            run_cwd = _REPO_ROOT

        self.last_cwd = run_cwd

        logger.info(
            "Starting llmos system-control-daemon via %s (root=%s, cwd=%s, plain=%s)",
            mode_desc,
            self.root_dir,
            run_cwd,
            self.plain_endpoint,
        )
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=run_cwd,
            env=sub_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = proc
        assert proc.stdout is not None

        try:
            ready_raw = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=self._startup_timeout_s,
            )
        except BaseException:
            await self.stop()
            raise

        ready_text = ready_raw.decode("utf-8", errors="replace").strip()
        if not ready_text.startswith("READY ") or "plain_socket=" not in ready_text:
            stderr_text = ""
            if proc.stderr is not None:
                try:
                    stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=3.0)
                    stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
                except TimeoutError:
                    stderr_text = "<stderr read timed out>"
            await self.stop()
            raise RuntimeError(
                f"system-control-daemon failed before READY: stdout={ready_text!r} stderr={stderr_text!r}"
            )

        self.ready_banner = ready_text
        self._drain_tasks = [
            asyncio.create_task(_drain_stream(proc.stdout)),
            asyncio.create_task(_drain_stream(proc.stderr)),
        ]
        return self.plain_endpoint

    async def stop(self) -> None:
        """Terminate the managed daemon subprocess and wait for full OS reaping."""
        proc = self._process
        self._process = None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()

        if self._drain_tasks:
            for task in self._drain_tasks:
                task.cancel()
            await asyncio.gather(*self._drain_tasks, return_exceptions=True)
            self._drain_tasks.clear()

        if sys.platform != "win32":
            for sock_path in (self.auth_endpoint, self.plain_endpoint):
                try:
                    Path(sock_path).unlink()
                except FileNotFoundError:
                    pass

    async def __aenter__(self) -> ManagedLlmosDaemon:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.stop()


__all__ = [
    "ManagedLlmosDaemon",
    "_drain_stream",
    "_make_ipc_endpoint",
    "resolve_daemon_command",
]
