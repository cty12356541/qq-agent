"""Governance and compatibility tests for Wave W6-05 memory & Shared Cortex modularization."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import qq_agent.cortex_cues as cortex_cues_mod
import qq_agent.memory as memory_facade_mod
import qq_agent.memory_store as memory_store_mod
import qq_agent.shared_cortex as shared_cortex_mod

_PKG_DIR = Path(memory_facade_mod.__file__).resolve().parent

_LANE_MODULES: tuple[str, ...] = (
    "cortex_cues.py",
    "shared_cortex.py",
    "memory_store.py",
    "memory.py",
)


def test_w6_05_memory_modules_line_budgets() -> None:
    """Every W6-05 `qq_agent/*.py` module must stay `<= 680` lines and `memory.py` facade `<= 85` lines."""
    for filename in _LANE_MODULES:
        path = _PKG_DIR / filename
        assert path.is_file(), f"Missing expected module: {path}"
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= 680, f"{filename} exceeded 680-line ceiling ({line_count} lines)"

    facade_lines = len((_PKG_DIR / "memory.py").read_text(encoding="utf-8").splitlines())
    assert facade_lines <= 85, f"memory.py facade exceeded 85-line ceiling ({facade_lines} lines)"


def test_w6_05_facade_and_submodule_symbol_identity() -> None:
    """Core public and internal symbols imported from `qq_agent.memory` must be identical objects to submodules."""
    assert memory_facade_mod.MemoryStore is memory_store_mod.MemoryStore
    assert memory_facade_mod.SharedCortexMixin is shared_cortex_mod.SharedCortexMixin
    assert issubclass(memory_store_mod.MemoryStore, shared_cortex_mod.SharedCortexMixin)

    for sym_name in (
        "MemoryFact",
        "UserProfile",
        "SharedFact",
        "extract_public_interest_cues",
        "_compile_public_cue_rules",
        "_sanitize_memory_text",
        "_normalize_topic_key",
        "_is_high_distinctiveness_token",
        "_QueryFeatures",
        "_is_ignored_alnum_token",
        "_extract_query_features",
        "_is_safe_core_containment",
        "_sanitize_public_self_topic",
    ):
        assert hasattr(memory_facade_mod, sym_name), f"Facade missing symbol {sym_name}"
        assert hasattr(cortex_cues_mod, sym_name), f"cortex_cues missing symbol {sym_name}"
        assert getattr(memory_facade_mod, sym_name) is getattr(cortex_cues_mod, sym_name)


def test_w6_05_acyclic_dependency_graph_and_isolated_imports() -> None:
    """Submodules must form a strict DAG with zero circular imports regardless of import order."""
    lane_mod_names = {
        "qq_agent.cortex_cues",
        "qq_agent.shared_cortex",
        "qq_agent.memory_store",
        "qq_agent.memory",
    }

    edges: dict[str, set[str]] = {}
    for filename in _LANE_MODULES:
        mod_full = f"qq_agent.{filename[:-3]}"
        tree = ast.parse((_PKG_DIR / filename).read_text(encoding="utf-8"), filename=filename)
        deps: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                target = (
                    f"qq_agent.{node.module}"
                    if node.level == 1
                    else node.module
                )
                if target in lane_mod_names and target != mod_full:
                    deps.add(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in lane_mod_names and alias.name != mod_full:
                        deps.add(alias.name)
        edges[mod_full] = deps

    # Verify expected clean layering
    assert edges["qq_agent.cortex_cues"] == set()
    assert edges["qq_agent.shared_cortex"] == {"qq_agent.cortex_cues"}
    assert edges["qq_agent.memory_store"] == {
        "qq_agent.cortex_cues",
        "qq_agent.shared_cortex",
    }
    assert edges["qq_agent.memory"] == {
        "qq_agent.cortex_cues",
        "qq_agent.shared_cortex",
        "qq_agent.memory_store",
    }

    # Verify cold-start isolated import of each module as entry point
    repo_root = str(_PKG_DIR.parent)
    for entry_mod in (
        "qq_agent.cortex_cues",
        "qq_agent.shared_cortex",
        "qq_agent.memory_store",
        "qq_agent.memory",
    ):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {repo_root!r}); import {entry_mod}",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"Cold import of {entry_mod} failed: {proc.stderr or proc.stdout}"
        )


def test_w6_05_modularized_store_end_to_end_smoke(tmp_path: Path) -> None:
    """Smoke-test L2/L3 + SharedCortexMixin composition through `MemoryStore`."""
    t0 = 1_750_000_000.0
    with memory_store_mod.MemoryStore(tmp_path / "w6_smoke.sqlite3") as store:
        store.add_memory("管理员私有备注", scope="admin_private")
        store.add_memory("群公共设定", scope="public")
        cues = cortex_cues_mod.extract_public_interest_cues("今晚继续打战舰世界和写Python")
        assert "聊过战舰世界" in cues
        assert "关注编程/AI" in cues

        prof = store.observe_public_traits(10001, "测试群友", cues)
        assert prof is not None
        assert "聊过战舰世界" in prof.traits_text

        fact = store.record_shared_fact("Python 3.13 更新", "Python 3.13 引入自由线程实验模式", now=t0)
        assert fact is not None
        matched = store.find_relevant_shared_facts("Python 3.13 有什么更新", now=t0 + 10)
        assert len(matched) == 1
        assert matched[0].topic_key == fact.topic_key

        state = store.update_public_self_state("刚刚聊完 Python 3.13 @某人 12345678", chat_type="group", now=t0)
        assert "12345678" not in state
        assert "@某人" not in state

        ctx_group = store.render_context_block(
            session_key="group:9001",
            chat_type="group",
            group_id=9001,
            user_id=10001,
            query_text="Python 3.13 怎么样",
            now=t0 + 20,
        )
        assert "管理员私有备注" not in ctx_group
        assert "群公共设定" in ctx_group
        assert "自由线程" in ctx_group
        assert state in ctx_group
