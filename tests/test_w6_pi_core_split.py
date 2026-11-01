"""Wave W6-03 governance tests: pi_core modular split line limits, symbol identity, and acyclic imports."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from qq_agent import pi_ambient, pi_core, pi_loop, session_actor, subagent_runner
from qq_agent.types import ChatMessage, ChatResult, SubAgentRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "qq_agent"

TARGET_MODULES: tuple[str, ...] = (
    "pi_loop.py",
    "subagent_runner.py",
    "pi_ambient.py",
    "session_actor.py",
    "pi_core.py",
)

MAX_MODULE_LINES = 680
MAX_FACADE_LINES = 95


@pytest.mark.parametrize("filename", TARGET_MODULES)
def test_w6_pi_core_split_line_count_redline(filename: str) -> None:
    """Every split module in Lane C must stay <= 680 UTF-8 lines, and pi_core.py facade <= 95 lines."""
    path = PKG_DIR / filename
    assert path.is_file(), f"Missing expected module: {path}"
    line_count = len(path.read_text(encoding="utf-8").splitlines())
    assert line_count > 0
    assert line_count <= MAX_MODULE_LINES, (
        f"{filename} has {line_count} lines, exceeding {MAX_MODULE_LINES}-line redline"
    )
    if filename == "pi_core.py":
        assert line_count <= MAX_FACADE_LINES, (
            f"pi_core.py facade has {line_count} lines, exceeding {MAX_FACADE_LINES}-line limit"
        )


def test_w6_pi_core_facade_symbol_identity() -> None:
    """Symbols re-exported by qq_agent.pi_core must be identical (`is`) to their submodule definitions."""
    expected_origins: dict[str, Any] = {
        # pi_loop
        "AfterStepHook": pi_loop,
        "AgentEvent": pi_loop,
        "BeforeStepHook": pi_loop,
        "BudgetBlockedError": pi_loop,
        "CancelToken": pi_loop,
        "LlmStepEvent": pi_loop,
        "RegisteredTool": pi_loop,
        "SteeringInjectedEvent": pi_loop,
        "ToolExecutionEvent": pi_loop,
        "ToolHandler": pi_loop,
        "TurnEndEvent": pi_loop,
        "TurnStartEvent": pi_loop,
        "agent_loop": pi_loop,
        "execute_tool_call": pi_loop,
        "_provider_accepts_kwarg": pi_loop,
        "_call_provider_chat": pi_loop,
        # subagent_runner
        "DEFAULT_SUBAGENT_SYSTEM_PROMPT": subagent_runner,
        "SubAgentCompleteCallback": subagent_runner,
        "SubAgentHandle": subagent_runner,
        "SubAgentRunner": subagent_runner,
        # pi_ambient
        "AmbientBuffer": pi_ambient,
        "AmbientRecord": pi_ambient,
        "default_coalesce_items": pi_ambient,
        "_SYNTHETIC_PLACEHOLDERS": pi_ambient,
        "_extract_item_user_id": pi_ambient,
        "_extract_item_text": pi_ambient,
        "_coalesce_inbound_pairs": pi_ambient,
        # session_actor
        "ActorState": session_actor,
        "ActorTurnHandler": session_actor,
        "CoalesceFn": session_actor,
        "ItemDroppedCallback": session_actor,
        "PreemptPredicate": session_actor,
        "SessionActor": session_actor,
        "SubmitRoute": session_actor,
        "TurnPriority": session_actor,
        "_resolve_handler_arity": session_actor,
        "_normalize_priority": session_actor,
    }

    required_public = {
        "AmbientBuffer",
        "AmbientRecord",
        "BudgetBlockedError",
        "CancelToken",
        "LlmStepEvent",
        "RegisteredTool",
        "SessionActor",
        "SteeringInjectedEvent",
        "SubAgentHandle",
        "SubAgentRunner",
        "SubmitRoute",
        "ToolExecutionEvent",
        "TurnEndEvent",
        "TurnPriority",
        "TurnStartEvent",
        "agent_loop",
        "default_coalesce_items",
        "execute_tool_call",
    }
    assert required_public.issubset(set(pi_core.__all__))

    for sym_name, origin_mod in expected_origins.items():
        assert hasattr(origin_mod, sym_name), f"{origin_mod.__name__} missing {sym_name}"
        assert hasattr(pi_core, sym_name), f"qq_agent.pi_core missing {sym_name}"
        assert getattr(pi_core, sym_name) is getattr(origin_mod, sym_name), (
            f"Identity mismatch for {sym_name} between pi_core and {origin_mod.__name__}"
        )


def test_w6_pi_core_internal_import_dag_is_acyclic() -> None:
    """AST import graph among pi_loop, subagent_runner, pi_ambient, session_actor, and pi_core must be a strict DAG."""
    module_names = {Path(name).stem for name in TARGET_MODULES}
    edges: dict[str, set[str]] = {mod: set() for mod in module_names}

    for filename in TARGET_MODULES:
        stem = Path(filename).stem
        tree = ast.parse((PKG_DIR / filename).read_text(encoding="utf-8"), filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target: str | None = None
                if node.level == 1 and node.module in module_names:
                    target = node.module
                elif node.level == 0 and node.module:
                    parts = node.module.split(".")
                    if len(parts) == 2 and parts[0] == "qq_agent" and parts[1] in module_names:
                        target = parts[1]
                elif node.level == 1 and node.module is None:
                    for alias in node.names:
                        if alias.name in module_names:
                            edges[stem].add(alias.name)
                if target is not None and target != stem:
                    edges[stem].add(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) == 2 and parts[0] == "qq_agent" and parts[1] in module_names:
                        if parts[1] != stem:
                            edges[stem].add(parts[1])

    # Submodules must never import back from the pi_core facade.
    for submod in module_names - {"pi_core"}:
        assert "pi_core" not in edges[submod], f"{submod} must not import from pi_core facade"

    # DFS cycle detection (0=unvisited, 1=visiting, 2=visited).
    state: dict[str, int] = {mod: 0 for mod in module_names}

    def _visit(node: str, trail: list[str]) -> None:
        state[node] = 1
        for dep in sorted(edges[node]):
            if state[dep] == 1:
                cycle = " -> ".join([*trail, node, dep])
                raise AssertionError(f"Circular import detected: {cycle}")
            if state[dep] == 0:
                _visit(dep, [*trail, node])
        state[node] = 2

    for mod in sorted(module_names):
        if state[mod] == 0:
            _visit(mod, [])


@pytest.mark.parametrize(
    "import_target",
    [
        "qq_agent.pi_loop",
        "qq_agent.pi_ambient",
        "qq_agent.subagent_runner",
        "qq_agent.session_actor",
        "qq_agent.pi_core",
    ],
)
def test_w6_pi_core_fresh_interpreter_isolated_imports(import_target: str) -> None:
    """Importing any of the 5 modules first in a fresh Python interpreter succeeds without circular import errors."""
    proc = subprocess.run(
        [sys.executable, "-c", f"import {import_target}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Isolated import of {import_target} failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_w6_pi_core_direct_submodules_collaboration() -> None:
    """Direct imports from split submodules interoperate seamlessly across loop, SubAgentRunner, AmbientBuffer, and SessionActor."""

    class _StubProvider:
        async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
            return ChatResult(
                content="查证结论：模块化拆分正常",
                model=str(kwargs.get("model") or "stub"),
                total_tokens=12,
            )

    async def _scenario() -> None:
        coalesced = pi_ambient.default_coalesce_items(["第一句", "第二句"])
        assert coalesced == "第一句\n第二句"

        observed_turns: list[str] = []
        observed_sub_briefs: list[str] = []

        runner = subagent_runner.SubAgentRunner(
            _StubProvider(),
            max_concurrent=2,
            max_per_session=1,
        )

        async def _on_turn(item: str, token: pi_loop.CancelToken) -> None:
            token.raise_if_cancelled()
            events = [
                ev
                async for ev in pi_loop.agent_loop(
                    _StubProvider(),
                    [ChatMessage(role="user", content=item)],
                    cancel_token=token,
                )
            ]
            end_ev = events[-1]
            assert isinstance(end_ev, pi_loop.TurnEndEvent)
            observed_turns.append(end_ev.final_text)

        async def _on_sub(res: Any) -> None:
            observed_sub_briefs.append(res.brief)

        actor = session_actor.SessionActor(
            "group:60300",
            _on_turn,
            subagent_result_handler=_on_sub,
            subagent_runner=runner,
            debounce_s=0.0,
        )
        try:
            assert actor.record_ambient("Alice", 1001, "路过打卡", 1.0)
            assert actor.ambient.consume_recent_lines() == ["[Alice]: 路过打卡"]

            route = actor.submit("测试主循环", user_id=1001)
            assert route == "debouncing"
            handle = runner.spawn(
                SubAgentRequest(
                    task_id="w6-sub-1",
                    session_key="group:60300",
                    goal="核验拆分",
                ),
                on_complete=actor.enqueue_subagent_result,
            )
            await handle.task
            await actor.wait_idle(timeout=3.0)
        finally:
            await actor.aclose()
            await runner.aclose()

        assert observed_turns == ["查证结论：模块化拆分正常"]
        assert observed_sub_briefs == ["查证结论：模块化拆分正常"]

    import asyncio

    asyncio.run(_scenario())
