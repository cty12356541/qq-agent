"""Read-only, zero-pollution `nlos_sdk` loader and deterministic 16-byte SABI identity helpers.

Strictly preserves `D:\\llmos` as read-only (never writes `.egg-info` or build artifacts into `llmos_root`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import site
import sys
from typing import Any


@dataclass(frozen=True, slots=True)
class LlmosSdkBundle:
    """Resolved NLOS Python SDK symbols and metadata."""

    version: str
    ChannelClient: Any
    ChannelControlError: Any
    LocalRpcClient: Any
    IpcError: Any
    TransportConfig: Any
    ServiceDirectoryClient: Any
    ServiceRequirement: Any


def derive_deterministic_id16(domain: str, *parts: str | bytes | int) -> bytes:
    """Derive a non-zero 16-byte deterministic identifier via domain-separated SHA-256."""
    hasher = hashlib.sha256()
    hasher.update(domain.encode("utf-8"))
    for part in parts:
        hasher.update(b"\x00")
        if isinstance(part, bytes):
            hasher.update(part)
        elif isinstance(part, int):
            hasher.update(str(part).encode("utf-8"))
        else:
            hasher.update(str(part).encode("utf-8"))
    raw16 = hasher.digest()[:16]
    if raw16 == b"\x00" * 16:
        return b"\x00" * 15 + b"\x01"
    return raw16


INBOUND_CHANNEL_CREATE_KEY: bytes = derive_deterministic_id16("qq-agent/channel/inbound/v1")
OUTBOUND_CHANNEL_CREATE_KEY: bytes = derive_deterministic_id16("qq-agent/channel/outbound/v1")


def process_id_for_role(role: str) -> bytes:
    """Return a deterministic 16-byte SABI `process_id` for `adapter`, `brain`, or `all-in-one`."""
    return derive_deterministic_id16("qq-agent/process/v1", role.strip().lower() or "default")


def fiber_id_for_session(session_key: str) -> bytes:
    """Return a deterministic 16-byte SABI consumer `fiber_id` scoped to a chat session."""
    return derive_deterministic_id16("qq-agent/fiber/session/v1", session_key)


def fiber_id_for_subagent(session_key: str, task_id: str) -> bytes:
    """Return a deterministic 16-byte SABI `fiber_id` scoped to a depth=1 background sub-agent."""
    return derive_deterministic_id16("qq-agent/fiber/subagent/v1", session_key, task_id)


def _inject_readonly_sdk_paths(llmos_root: Path) -> None:
    root = llmos_root.resolve()
    sdk_dir = root / "sdk" / "python"
    gen_dir = root / "gen" / "python"
    for candidate in (gen_dir, sdk_dir):
        if candidate.is_dir():
            cand_str = str(candidate)
            if cand_str not in sys.path:
                sys.path.insert(0, cand_str)


def ensure_local_venv_pth(llmos_root: Path | str) -> Path | None:
    """Best-effort write of a `.pth` pointer inside the active virtualenv's site-packages.

    Never writes anything inside `llmos_root`. Returns the `.pth` path if written/present.
    """
    if sys.prefix == sys.base_prefix:
        return None
    root = Path(llmos_root).resolve()
    sdk_dir = root / "sdk" / "python"
    gen_dir = root / "gen" / "python"
    if not sdk_dir.is_dir():
        return None

    for sp in site.getsitepackages():
        sp_path = Path(sp)
        if sp_path.is_dir() and str(sp_path).startswith(str(Path(sys.prefix).resolve())):
            pth_file = sp_path / "qq_agent_nlos_sdk.pth"
            lines = [str(sdk_dir)]
            if gen_dir.is_dir():
                lines.append(str(gen_dir))
            desired = "\n".join(lines) + "\n"
            try:
                if not pth_file.is_file() or pth_file.read_text(encoding="utf-8") != desired:
                    pth_file.write_text(desired, encoding="utf-8")
                return pth_file
            except OSError:
                return None
    return None


def load_nlos_sdk(llmos_root: Path | str = r"D:\llmos") -> LlmosSdkBundle:
    """Import `nlos_sdk` cleanly without mutating `llmos_root` on disk."""
    root = Path(llmos_root)
    try:
        import nlos_sdk  # type: ignore[import-not-found]
    except ImportError:
        _inject_readonly_sdk_paths(root)
        import nlos_sdk  # type: ignore[import-not-found]

    return LlmosSdkBundle(
        version=str(getattr(nlos_sdk, "__version__", "0.0.0")),
        ChannelClient=nlos_sdk.ChannelClient,
        ChannelControlError=nlos_sdk.ChannelControlError,
        LocalRpcClient=nlos_sdk.LocalRpcClient,
        IpcError=nlos_sdk.IpcError,
        TransportConfig=nlos_sdk.TransportConfig,
        ServiceDirectoryClient=nlos_sdk.ServiceDirectoryClient,
        ServiceRequirement=nlos_sdk.ServiceRequirement,
    )


__all__ = [
    "INBOUND_CHANNEL_CREATE_KEY",
    "LlmosSdkBundle",
    "OUTBOUND_CHANNEL_CREATE_KEY",
    "derive_deterministic_id16",
    "ensure_local_venv_pth",
    "fiber_id_for_session",
    "fiber_id_for_subagent",
    "load_nlos_sdk",
    "process_id_for_role",
]
