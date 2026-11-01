"""Tests for Wave W5 low-latency & multi-group shared-cortex configuration contracts."""

from __future__ import annotations

from qq_agent.config import AgentConfig


def test_w5_low_latency_and_multigroup_defaults():
    cfg = AgentConfig.from_env(env_file=None, environ={})
    assert cfg.private_debounce_s == 0.30
    assert cfg.group_debounce_s == 0.40
    assert cfg.bubble_typing_delay_s == 0.15
    assert cfg.subagent_enable_thinking is False
    assert cfg.subagent_max_concurrent == 6
    assert cfg.subagent_max_per_session == 2
    assert cfg.vision_max_side == 512
    assert cfg.compact_prompt is True
    assert cfg.fact_cache_ttl_s == 1800.0


def test_w5_env_overrides():
    cfg = AgentConfig.from_env(
        env_file=None,
        environ={
            "QQ_VISION_MAX_SIDE": "768",
            "QQ_COMPACT_PROMPT": "false",
            "QQ_FACT_CACHE_TTL_S": "900",
            "SUBAGENT_MAX_CONCURRENT": "8",
            "SUBAGENT_MAX_PER_SESSION": "3",
        },
    )
    assert cfg.vision_max_side == 768
    assert cfg.compact_prompt is False
    assert cfg.fact_cache_ttl_s == 900.0
    assert cfg.subagent_max_concurrent == 8
    assert cfg.subagent_max_per_session == 3
