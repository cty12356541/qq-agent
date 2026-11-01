"""Codebase-wide monolith governance, acyclic import, and OSS hygiene guardrails (W6-06)."""

from __future__ import annotations

import importlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "qq_agent"

# Permanent hard ceiling on any single production Python file under `qq_agent/`.
MAX_PROD_FILE_LINES = 700

_FACADE_MODULES: tuple[str, ...] = (
    "qq_agent",
    "qq_agent.agent",
    "qq_agent.cli",
    "qq_agent.channel_bridge",
    "qq_agent.pi_core",
    "qq_agent.memory",
    "qq_agent.tools",
    "qq_agent.tools.web_search",
)


def _iter_prod_py_files() -> list[Path]:
    return sorted(
        p
        for p in PKG_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def test_all_production_modules_stay_under_loc_ceiling() -> None:
    """Prevent multi-thousand-line monoliths from ever returning to `qq_agent/`."""
    files = _iter_prod_py_files()
    assert len(files) >= 35

    violations: list[str] = []
    for path in files:
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_PROD_FILE_LINES:
            rel = path.relative_to(REPO_ROOT).as_posix()
            violations.append(f"{rel}: {line_count} lines (max {MAX_PROD_FILE_LINES})")

    assert not violations, (
        "Production file(s) exceeded the monolith governance ceiling:\n  - "
        + "\n  - ".join(violations)
    )


def test_all_production_modules_import_cleanly_and_expose_facades() -> None:
    """Every module under `qq_agent/` must import cleanly and facades must resolve `__all__`."""
    for path in _iter_prod_py_files():
        rel_parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        if rel_parts[-1] == "__main__":
            continue
        if rel_parts[-1] == "__init__":
            rel_parts.pop()
        mod_name = ".".join(rel_parts)
        mod = importlib.import_module(mod_name)
        assert mod is not None

    for facade_name in _FACADE_MODULES:
        mod = importlib.import_module(facade_name)
        exported = getattr(mod, "__all__", None)
        if exported is not None:
            assert len(exported) > 0, f"{facade_name}.__all__ should not be empty"
            for sym in exported:
                assert hasattr(mod, sym), f"{facade_name} missing __all__ symbol {sym!r}"


def test_production_source_contains_no_hardcoded_live_ids() -> None:
    """Ensure default `AgentConfig` ships with empty/neutral OSS defaults (zero personal IDs)."""
    from qq_agent.config import AgentConfig

    cfg = AgentConfig(llm_api_key="test-key")
    assert cfg.bot_qq is None
    assert cfg.admin_user_ids == set()
    assert cfg.allowed_group_ids is None
    assert cfg.napcat_access_token is None

