"""Backward-compatible facade for AgentKernel, ManagedLlmosDaemon, and turn orchestration helpers."""

from __future__ import annotations

from .daemon_manager import (
    ManagedLlmosDaemon,
    _drain_stream,
    _make_ipc_endpoint,
    resolve_daemon_command,
)
from .kernel_core import AgentKernel
from .kernel_execution import KernelExecutionMixin
from .kernel_turn_helpers import (
    _AMBIENT_INJECT_LIMIT,
    _AMBIENT_RING_MAXLEN,
    _CANCEL_STEER_KEYWORDS,
    _FAST_BRAIN_TOOL_GUIDE,
    _FAST_BRAIN_TOOL_GUIDE_COMPACT,
    _L2_COMPRESS_THRESHOLD,
    _SOFT_TRIGGER_REASONS,
    _SYNTHETIC_PLACEHOLDERS,
    _DebounceBucket,
    _RecordingBusProxy,
    _TurnEnvelope,
    _bridge_tool_binding,
    _coalesce_debounced_items,
    _coalesce_turn_envelopes,
    _summarize_web_search_outcome,
)

__all__ = [
    "AgentKernel",
    "KernelExecutionMixin",
    "ManagedLlmosDaemon",
    "_AMBIENT_INJECT_LIMIT",
    "_AMBIENT_RING_MAXLEN",
    "_CANCEL_STEER_KEYWORDS",
    "_DebounceBucket",
    "_FAST_BRAIN_TOOL_GUIDE",
    "_FAST_BRAIN_TOOL_GUIDE_COMPACT",
    "_L2_COMPRESS_THRESHOLD",
    "_RecordingBusProxy",
    "_SOFT_TRIGGER_REASONS",
    "_SYNTHETIC_PLACEHOLDERS",
    "_TurnEnvelope",
    "_bridge_tool_binding",
    "_coalesce_debounced_items",
    "_coalesce_turn_envelopes",
    "_drain_stream",
    "_make_ipc_endpoint",
    "_summarize_web_search_outcome",
    "resolve_daemon_command",
]
