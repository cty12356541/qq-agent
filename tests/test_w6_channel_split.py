# tests/test_w6_channel_split.py
"""Wave W6-02 governance tests for modularized channel_bridge split."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

import qq_agent
from qq_agent import bus_base, channel_bridge, llmos_bus, napcat_ws

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "qq_agent"

SPLIT_MODULES = (
    "bus_base.py",
    "llmos_bus.py",
    "napcat_ws.py",
    "channel_bridge.py",
)
MAX_MODULE_LINES = 680
MAX_FACADE_LINES = 90


@pytest.mark.parametrize("filename", SPLIT_MODULES)
def test_w6_channel_split_file_line_budgets(filename: str) -> None:
    path = PKG_DIR / filename
    assert path.is_file(), f"Missing expected module: {path}"
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert 1 <= line_count <= MAX_MODULE_LINES, (
        f"{filename} has {line_count} lines, exceeding <= {MAX_MODULE_LINES} budget"
    )
    if filename == "channel_bridge.py":
        assert line_count <= MAX_FACADE_LINES, (
            f"channel_bridge.py facade has {line_count} lines, exceeding <= {MAX_FACADE_LINES}"
        )


def test_w6_channel_bridge_facade_reexports_identical_references() -> None:
    # bus_base core primitives
    assert channel_bridge.BusQueueMetrics is bus_base.BusQueueMetrics
    assert channel_bridge.MessageBus is bus_base.MessageBus
    assert channel_bridge._MonotonicAckQueue is bus_base._MonotonicAckQueue
    assert channel_bridge.DirectMemoryBus is bus_base.DirectMemoryBus
    assert channel_bridge.AckCoordinator is bus_base.AckCoordinator
    assert channel_bridge.AdaptivePoller is bus_base.AdaptivePoller
    assert channel_bridge.DEFAULT_CONSUMER_FIBER_ID is bus_base.DEFAULT_CONSUMER_FIBER_ID

    # llmos_bus SABI / IPC spine & helpers
    assert channel_bridge.LlmosChannelBus is llmos_bus.LlmosChannelBus
    assert channel_bridge.ResilientLlmosBus is llmos_bus.ResilientLlmosBus
    assert llmos_bus.ResilientLlmosBus is llmos_bus.LlmosChannelBus
    assert channel_bridge._load_channel_client_cls is llmos_bus._load_channel_client_cls
    assert channel_bridge._try_parse_int is llmos_bus._try_parse_int
    assert channel_bridge._random_nonzero_id16 is llmos_bus._random_nonzero_id16
    assert channel_bridge._validate_nonzero_id16 is llmos_bus._validate_nonzero_id16
    assert llmos_bus._validate_nonzero_id16 is bus_base._validate_nonzero_id16
    assert channel_bridge._extract_acked_seq is llmos_bus._extract_acked_seq
    assert channel_bridge._extract_trim_seq is llmos_bus._extract_trim_seq
    assert channel_bridge._extract_backlog_bytes is llmos_bus._extract_backlog_bytes
    assert channel_bridge._accepts_kwarg is llmos_bus._accepts_kwarg
    assert channel_bridge.CONSUMER_FIBER_GENERATION == llmos_bus.CONSUMER_FIBER_GENERATION
    assert channel_bridge.MAX_REGISTERED_SEQ_CACHE == llmos_bus.MAX_REGISTERED_SEQ_CACHE

    # napcat_ws OneBot v11 WebSocket adapter & constants
    assert channel_bridge.NapCatWsAdapter is napcat_ws.NapCatWsAdapter
    assert channel_bridge.MSG_CACHE_CAPACITY == napcat_ws.MSG_CACHE_CAPACITY
    assert channel_bridge.OUTBOUND_ECHO_CACHE_CAPACITY == napcat_ws.OUTBOUND_ECHO_CACHE_CAPACITY
    assert channel_bridge.MsgCacheEntry == napcat_ws.MsgCacheEntry

    # Top-level package re-exports remain identical
    assert qq_agent.AckCoordinator is bus_base.AckCoordinator
    assert qq_agent.AdaptivePoller is bus_base.AdaptivePoller
    assert qq_agent.BusQueueMetrics is bus_base.BusQueueMetrics
    assert qq_agent.DirectMemoryBus is bus_base.DirectMemoryBus
    assert qq_agent.LlmosChannelBus is llmos_bus.LlmosChannelBus
    assert qq_agent.MessageBus is bus_base.MessageBus
    assert qq_agent.NapCatWsAdapter is napcat_ws.NapCatWsAdapter
    assert qq_agent.ResilientLlmosBus is llmos_bus.ResilientLlmosBus


def test_w6_channel_split_import_dag_has_no_cycles() -> None:
    sibling_names = {"bus_base", "llmos_bus", "napcat_ws", "channel_bridge"}
    edges: dict[str, set[str]] = {name: set() for name in sibling_names}

    for mod_name in sibling_names:
        tree = ast.parse((PKG_DIR / f"{mod_name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level == 1 and node.module in sibling_names:
                    edges[mod_name].add(node.module)
                elif node.level == 0 and node.module and node.module.startswith("qq_agent."):
                    sub = node.module.split(".", 1)[1]
                    if sub in sibling_names:
                        edges[mod_name].add(sub)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("qq_agent."):
                        sub = alias.name.split(".", 1)[1]
                        if sub in sibling_names:
                            edges[mod_name].add(sub)

    assert edges["bus_base"] == set()
    assert edges["llmos_bus"] <= {"bus_base"}
    assert edges["napcat_ws"] <= {"bus_base", "llmos_bus"}
    assert edges["channel_bridge"] == {"bus_base", "llmos_bus", "napcat_ws"}


@pytest.mark.parametrize(
    "entry_module",
    (
        "qq_agent.bus_base",
        "qq_agent.llmos_bus",
        "qq_agent.napcat_ws",
        "qq_agent.channel_bridge",
    ),
)
def test_w6_channel_split_clean_interpreter_import_without_circular_error(
    entry_module: str,
) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {entry_module}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Importing {entry_module} in fresh interpreter failed:\n{proc.stderr}"
    )
