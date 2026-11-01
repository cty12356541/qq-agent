"""Wave W6-01 governance tests for AgentKernel & CLI supervisor modular split (`<= 680` LOC)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

LANE_A_PRODUCTION_MODULES: tuple[str, ...] = (
    "qq_agent/agent.py",
    "qq_agent/daemon_manager.py",
    "qq_agent/kernel_turn_helpers.py",
    "qq_agent/kernel_execution.py",
    "qq_agent/kernel_core.py",
    "qq_agent/cli.py",
    "qq_agent/supervisor.py",
)


@pytest.mark.parametrize("rel_path", LANE_A_PRODUCTION_MODULES)
def test_w6_01_production_module_loc_under_680(rel_path: str) -> None:
    """Every W6-01 modified or newly created `qq_agent/*.py` module must be `<= 680` UTF-8 lines."""
    target = REPO_ROOT / rel_path
    assert target.is_file(), f"Expected module not found: {target}"
    line_count = len(target.read_text(encoding="utf-8").splitlines())
    assert line_count <= 680, f"{rel_path} has {line_count} lines (exceeds 680 LOC limit)"
    if rel_path == "qq_agent/agent.py":
        assert line_count <= 90, f"Facade {rel_path} has {line_count} lines (exceeds 90 LOC facade cap)"
    if rel_path == "qq_agent/cli.py":
        assert line_count <= 620, f"{rel_path} has {line_count} lines (exceeds 620 LOC CLI cap)"


def test_w6_01_facade_reexports_and_symbol_identity() -> None:
    """Public and internal helper exports must preserve exact object identity across facade and submodules."""
    from qq_agent import AgentKernel as PkgAgentKernel, ManagedLlmosDaemon as PkgManagedLlmosDaemon, cli
    from qq_agent import agent as agent_facade
    from qq_agent.agent import AgentKernel, ManagedLlmosDaemon, resolve_daemon_command
    from qq_agent.daemon_manager import (
        ManagedLlmosDaemon as DmManagedLlmosDaemon,
        _drain_stream,
        _make_ipc_endpoint,
        resolve_daemon_command as dm_resolve_daemon_command,
    )
    from qq_agent.kernel_core import AgentKernel as CoreAgentKernel
    from qq_agent.kernel_execution import KernelExecutionMixin
    from qq_agent import kernel_turn_helpers as turn_helpers
    from qq_agent.provider import OpenAICompatProvider
    from qq_agent.supervisor import (
        _forward_child_stream,
        _terminate_child_process,
        _write_runtime_state,
        run_supervisor,
    )

    assert AgentKernel is PkgAgentKernel
    assert AgentKernel is CoreAgentKernel
    assert issubclass(AgentKernel, KernelExecutionMixin)

    assert ManagedLlmosDaemon is PkgManagedLlmosDaemon
    assert ManagedLlmosDaemon is DmManagedLlmosDaemon
    assert resolve_daemon_command is dm_resolve_daemon_command
    assert agent_facade._make_ipc_endpoint is _make_ipc_endpoint
    assert agent_facade._drain_stream is _drain_stream

    for helper_name in (
        "_DebounceBucket",
        "_TurnEnvelope",
        "_RecordingBusProxy",
        "_bridge_tool_binding",
        "_coalesce_debounced_items",
        "_coalesce_turn_envelopes",
        "_summarize_web_search_outcome",
    ):
        assert getattr(agent_facade, helper_name) is getattr(turn_helpers, helper_name)

    assert cli.run_supervisor is run_supervisor
    assert cli._forward_child_stream is _forward_child_stream
    assert cli._terminate_child_process is _terminate_child_process
    assert cli._write_runtime_state is _write_runtime_state
    assert cli.OpenAICompatProvider is OpenAICompatProvider


def test_w6_01_cli_provider_monkeypatch_compatibility(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Monkeypatching `cli.OpenAICompatProvider` must intercept CLI provider instantiation."""
    from qq_agent import cli
    from qq_agent.channel_bridge import DirectMemoryBus
    from qq_agent.config import AgentConfig
    from qq_agent.guardrails import GuardrailManager
    from qq_agent.memory import MemoryStore
    from qq_agent.persona import PersonaManager
    from qq_agent.session import SessionStore
    from qq_agent.stickers import StickerRegistry
    from qq_agent.trigger import RuleTrigger

    captured_kwargs: dict[str, Any] = {}

    class _SentinelProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(cli, "OpenAICompatProvider", _SentinelProvider)

    cfg = AgentConfig(
        bot_qq=999,
        llm_api_key="sk-w6-test",
        llm_base_url="http://127.0.0.1:9999/v1",
        llm_model="test-model",
        data_dir=tmp_path / "data",
        persona_path=REPO_ROOT / "config" / "persona.default.toml",
        stickers_manifest_path=REPO_ROOT / "config" / "stickers.toml",
        mode="direct-ws",
    )
    persona = PersonaManager(cfg.persona_path)
    stickers = StickerRegistry(cfg.stickers_dir, manifest_path=cfg.stickers_manifest_path)
    bus = DirectMemoryBus()
    with (
        SessionStore(tmp_path / "sessions.sqlite3") as sessions,
        MemoryStore(tmp_path / "memory.sqlite3") as memory,
        GuardrailManager(tmp_path / "guardrails.sqlite3") as guardrails,
    ):
        trigger = RuleTrigger(bot_qq=999)
        kernel = cli._build_kernel_from_config(
            cfg,
            bus=bus,
            persona=persona,
            stickers=stickers,
            sessions=sessions,
            memory=memory,
            guardrails=guardrails,
            trigger=trigger,
            runtime_role="all-in-one",
        )
        assert isinstance(kernel.provider, _SentinelProvider)
        assert captured_kwargs["api_key"] == "sk-w6-test"
        assert captured_kwargs["default_model"] == "test-model"


@pytest.mark.parametrize(
    "entry_import",
    [
        "import qq_agent.daemon_manager",
        "import qq_agent.kernel_turn_helpers",
        "import qq_agent.kernel_execution",
        "import qq_agent.kernel_core",
        "import qq_agent.agent",
        "import qq_agent.supervisor",
        "import qq_agent.cli",
        "from qq_agent import AgentKernel, ManagedLlmosDaemon, cli",
    ],
)
def test_w6_01_isolated_import_orders_have_no_circular_imports(entry_import: str) -> None:
    """Every new and refactored module must import cleanly as the very first entry point in a fresh interpreter."""
    proc = subprocess.run(
        [sys.executable, "-c", entry_import],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, f"Failed `{entry_import}`:\nstdout={proc.stdout}\nstderr={proc.stderr}"
