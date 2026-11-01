from __future__ import annotations

from pathlib import Path

from qq_agent.commands import AdminCommandRouter
from qq_agent.guardrails import GuardrailManager, QuotaSnapshot
from qq_agent.trigger import RuleTrigger
from qq_agent.types import CommandResult, GuardDecision, InboundMessage, TriggerDecision


def _make_msg(
    *,
    chat_type: str = "group",
    user_id: int = 1001,
    group_id: int | None = 88888,
    text: str = "你好",
    mentioned_bot: bool = False,
    self_id: int | None = 99999,
) -> InboundMessage:
    actual_group = group_id if chat_type == "group" else None
    return InboundMessage(
        event_id="evt-1",
        chat_type="group" if chat_type == "group" else "private",
        user_id=user_id,
        group_id=actual_group,
        sender_name="测试群友",
        text=text,
        raw_message=text,
        mentioned_bot=mentioned_bot,
        self_id=self_id,
    )


# ---------------------------------------------------------------------------
# RuleTrigger tests
# ---------------------------------------------------------------------------


def test_trigger_ignores_self_by_bot_qq_and_self_id() -> None:
    trigger_by_cfg = RuleTrigger(bot_qq=99999)
    self_msg = _make_msg(user_id=99999, text="我自己说的话", mentioned_bot=True, self_id=None)
    assert trigger_by_cfg.evaluate(self_msg) == TriggerDecision(False, "ignored_self")

    trigger_by_event = RuleTrigger(bot_qq=None)
    self_event_msg = _make_msg(chat_type="private", user_id=77777, text="私聊回声", self_id=77777)
    assert trigger_by_event.evaluate(self_event_msg) == TriggerDecision(False, "ignored_self")


def test_trigger_group_whitelist() -> None:
    trigger = RuleTrigger(bot_qq=99999, allowed_group_ids={10001, 10002})

    blocked = _make_msg(group_id=55555, text="在吗", mentioned_bot=True)
    assert trigger.evaluate(blocked) == TriggerDecision(False, "group_not_whitelisted")

    allowed = _make_msg(group_id=10001, text="  在吗  ", mentioned_bot=True)
    assert trigger.evaluate(allowed) == TriggerDecision(True, "group_at", "在吗")

    # Empty set means whitelist not active (allow all groups)
    open_trigger = RuleTrigger(bot_qq=99999, allowed_group_ids=set())
    assert open_trigger.evaluate(blocked) == TriggerDecision(True, "group_at", "在吗")


def test_trigger_private_chat_rules() -> None:
    trigger = RuleTrigger(bot_qq=99999, allow_private=True)
    assert trigger.evaluate(_make_msg(chat_type="private", text="  晚上好  ")) == TriggerDecision(
        True, "private_direct", "晚上好"
    )
    assert trigger.evaluate(_make_msg(chat_type="private", text="   ")) == TriggerDecision(
        False, "ignored_empty"
    )

    disabled_trigger = RuleTrigger(bot_qq=99999, allow_private=False)
    assert disabled_trigger.evaluate(
        _make_msg(chat_type="private", text="晚上好")
    ) == TriggerDecision(False, "private_disabled")


def test_trigger_group_at_and_empty_poke() -> None:
    trigger = RuleTrigger(bot_qq=99999)
    poke = _make_msg(text="   ", mentioned_bot=True)
    assert trigger.evaluate(poke) == TriggerDecision(True, "group_at", "（戳了戳你）")

    at_question = _make_msg(text="  今天打战舰吗？ ", mentioned_bot=True)
    assert trigger.evaluate(at_question) == TriggerDecision(True, "group_at", "今天打战舰吗？")


def test_trigger_group_keywords_hot_reload_and_silence() -> None:
    trigger = RuleTrigger(bot_qq=99999, keywords=["", "   ", "小Q", "QBot"])

    # Empty non-@ group message
    assert trigger.evaluate(_make_msg(text="   ", mentioned_bot=False)) == TriggerDecision(
        False, "ignored_empty"
    )
    # Normal group chatter remains silent
    assert trigger.evaluate(_make_msg(text="今晚吃火锅吧", mentioned_bot=False)) == TriggerDecision(
        False, "not_triggered"
    )
    # Case-insensitive keyword match
    assert trigger.evaluate(
        _make_msg(text="  呼叫 qbot 出来冒泡 ", mentioned_bot=False)
    ) == TriggerDecision(True, "group_keyword", "呼叫 qbot 出来冒泡")

    # Hot-update keywords after persona reload
    trigger.update_keywords(["舰长助手"])
    assert trigger.evaluate(
        _make_msg(text="呼叫 qbot 出来冒泡", mentioned_bot=False)
    ) == TriggerDecision(False, "not_triggered")
    assert trigger.evaluate(
        _make_msg(text="舰长助手帮我配个加点", mentioned_bot=False)
    ) == TriggerDecision(True, "group_keyword", "舰长助手帮我配个加点")


# ---------------------------------------------------------------------------
# GuardrailManager tests
# ---------------------------------------------------------------------------


def test_guardrail_user_cooldown_per_session_and_user() -> None:
    gm = GuardrailManager(":memory:", user_cooldown_s=3.0)
    try:
        m_u1 = _make_msg(group_id=100, user_id=10)
        m_u2 = _make_msg(group_id=100, user_id=20)
        m_u1_priv = _make_msg(chat_type="private", user_id=10)

        assert gm.check(m_u1, now=0.0, today="2026-11-01") == GuardDecision(True, "ok")
        # Rapid double-click from same user in same group -> silent cooldown drop
        assert gm.check(m_u1, now=1.2, today="2026-11-01") == GuardDecision(
            False, "user_cooldown", refusal_reply=None
        )
        # Different user in same group is unaffected
        assert gm.check(m_u2, now=1.2, today="2026-11-01") == GuardDecision(True, "ok")
        # Same user in private session is unaffected
        assert gm.check(m_u1_priv, now=1.2, today="2026-11-01") == GuardDecision(True, "ok")
        # Cooldown expires relative to the last accepted trigger (t=0.0 -> t=3.0)
        assert gm.check(m_u1, now=3.0, today="2026-11-01") == GuardDecision(True, "ok")
    finally:
        gm.close()


def test_guardrail_group_rate_limit_and_antispam_notice() -> None:
    gm = GuardrailManager(
        ":memory:",
        user_cooldown_s=0.0,
        group_max_per_minute=3,
        rate_refusal="说话太快啦，让我喘口气~",
    )
    try:
        g1 = _make_msg(group_id=100, user_id=10)
        priv = _make_msg(chat_type="private", user_id=10)

        assert gm.check(g1, now=10.0, today="2026-11-01") == GuardDecision(True, "ok")
        assert gm.check(g1, now=20.0, today="2026-11-01") == GuardDecision(True, "ok")
        assert gm.check(g1, now=30.0, today="2026-11-01") == GuardDecision(True, "ok")

        # 4th message within 60s hits limit and emits refusal once
        assert gm.check(g1, now=35.0, today="2026-11-01") == GuardDecision(
            False, "group_rate_limit", refusal_reply="说话太快啦，让我喘口气~"
        )
        # Within 30s of refusal notice -> silent drop
        assert gm.check(g1, now=50.0, today="2026-11-01") == GuardDecision(
            False, "group_rate_limit", refusal_reply=None
        )
        # Private chat does not use group_max_per_minute
        assert gm.check(priv, now=50.0, today="2026-11-01") == GuardDecision(True, "ok")

        # >30s after refusal notice, still within 60s of t=10.0 -> prompts once more
        assert gm.check(g1, now=66.0, today="2026-11-01") == GuardDecision(
            False, "group_rate_limit", refusal_reply="说话太快啦，让我喘口气~"
        )
        # At t=70.1, oldest entry (t=10.0) falls out of 60s window -> allowed again
        assert gm.check(g1, now=70.1, today="2026-11-01") == GuardDecision(True, "ok")
    finally:
        gm.close()


def test_guardrail_soft_budget_switches_to_cheap_model() -> None:
    gm = GuardrailManager(
        ":memory:",
        user_cooldown_s=0.0,
        soft_daily_budget_micros=1_000,
        hard_daily_budget_micros=3_000,
        cheap_model="claude-haiku-5-5",
    )
    try:
        msg = _make_msg(group_id=100, user_id=10)
        assert gm.check(msg, now=1.0, today="2026-11-01") == GuardDecision(True, "ok")

        snap = gm.record_usage(
            prompt_tokens=120,
            completion_tokens=80,
            total_tokens=0,
            cost_micros_usd=1_200,
            today="2026-11-01",
        )
        assert snap.calls_used == 1
        assert snap.tokens_used == 200
        assert snap.cost_micros_used == 1_200

        assert gm.check(msg, now=5.0, today="2026-11-01") == GuardDecision(
            True,
            "daily_budget_soft_downgrade",
            override_model="claude-haiku-5-5",
        )

        # When total_tokens is explicitly provided, do not double-count prompt+completion
        snap2 = gm.record_usage(
            prompt_tokens=50,
            completion_tokens=50,
            total_tokens=100,
            cost_micros_usd=100,
            today="2026-11-01",
        )
        assert snap2.calls_used == 2
        assert snap2.tokens_used == 300
    finally:
        gm.close()

    # Without cheap_model configured, crossing soft budget keeps default model ("ok")
    with GuardrailManager(
        ":memory:",
        user_cooldown_s=0.0,
        soft_daily_budget_micros=500,
        hard_daily_budget_micros=2_000,
        cheap_model=None,
    ) as gm_no_cheap:
        gm_no_cheap.record_usage(total_tokens=100, cost_micros_usd=800, today="2026-11-01")
        assert gm_no_cheap.check(
            _make_msg(group_id=100, user_id=10), now=1.0, today="2026-11-01"
        ) == GuardDecision(True, "ok")


def test_guardrail_hard_limit_persistence_and_antispam(tmp_path: Path) -> None:
    db_file = tmp_path / "guardrails.sqlite3"

    gm1 = GuardrailManager(
        db_file,
        max_daily_calls=10,
        max_daily_tokens=10_000,
        soft_daily_budget_micros=1_000,
        hard_daily_budget_micros=2_000,
    )
    gm1.record_usage(total_tokens=600, cost_micros_usd=1_500, today="2026-11-01")
    gm1.close()

    # Re-open a fresh instance pointing at the same SQLite database (simulating process restart)
    gm2 = GuardrailManager(
        db_file,
        user_cooldown_s=0.0,
        max_daily_calls=10,
        max_daily_tokens=10_000,
        soft_daily_budget_micros=1_000,
        hard_daily_budget_micros=2_000,
        cheap_model="glm-4-flash",
        budget_refusal="今日脑细胞额度已耗尽，明天再聊叭~",
    )
    try:
        persisted = gm2.snapshot(today="2026-11-01")
        assert persisted == QuotaSnapshot(
            day="2026-11-01",
            calls_used=1,
            max_daily_calls=10,
            tokens_used=600,
            max_daily_tokens=10_000,
            cost_micros_used=1_500,
            soft_budget_micros=1_000,
            hard_budget_micros=2_000,
        )

        g1 = _make_msg(group_id=100, user_id=10)
        g2 = _make_msg(group_id=200, user_id=20)
        assert gm2.check(g1, now=10.0, today="2026-11-01") == GuardDecision(
            True, "daily_budget_soft_downgrade", override_model="glm-4-flash"
        )

        # Cross hard budget threshold
        gm2.record_usage(total_tokens=400, cost_micros_usd=600, today="2026-11-01")

        # First hard-limit rejection in group 100 replies with budget_refusal
        assert gm2.check(g1, now=20.0, today="2026-11-01") == GuardDecision(
            False, "daily_hard_limit", refusal_reply="今日脑细胞额度已耗尽，明天再聊叭~"
        )
        # Repeat in group 100 within 60s -> silent refusal_reply=None
        assert gm2.check(g1, now=50.0, today="2026-11-01") == GuardDecision(
            False, "daily_hard_limit", refusal_reply=None
        )
        # Different session (group 200) gets its own initial refusal notice
        assert gm2.check(g2, now=50.0, today="2026-11-01") == GuardDecision(
            False, "daily_hard_limit", refusal_reply="今日脑细胞额度已耗尽，明天再聊叭~"
        )
        # After 60s in group 100 -> refusal notice allowed again
        assert gm2.check(g1, now=80.0, today="2026-11-01") == GuardDecision(
            False, "daily_hard_limit", refusal_reply="今日脑细胞额度已耗尽，明天再聊叭~"
        )
        # Next calendar day starts clean
        assert gm2.check(g1, now=90.0, today="2026-11-02") == GuardDecision(True, "ok")
    finally:
        gm2.close()


def test_guardrail_hard_limit_by_calls_and_tokens() -> None:
    gm_calls = GuardrailManager(":memory:", user_cooldown_s=0.0, max_daily_calls=1)
    gm_tokens = GuardrailManager(":memory:", user_cooldown_s=0.0, max_daily_tokens=500)
    try:
        msg = _make_msg(group_id=100, user_id=10)
        gm_calls.record_usage(total_tokens=10, cost_micros_usd=1, today="2026-11-01")
        assert gm_calls.check(msg, now=1.0, today="2026-11-01").reason == "daily_hard_limit"

        gm_tokens.record_usage(total_tokens=500, cost_micros_usd=1, today="2026-11-01")
        assert gm_tokens.check(msg, now=1.0, today="2026-11-01").reason == "daily_hard_limit"
    finally:
        gm_calls.close()
        gm_tokens.close()


# ---------------------------------------------------------------------------
# AdminCommandRouter tests
# ---------------------------------------------------------------------------


def test_commands_non_command_and_non_admin_interception() -> None:
    called = {"reset": False, "quota": False, "reload": False}

    router = AdminCommandRouter(
        admin_user_ids={42},
        on_reset=lambda _sk: (called.__setitem__("reset", True) or 5),  # type: ignore[func-returns-value]
        on_quota=lambda: (
            called.__setitem__("quota", True)  # type: ignore[func-returns-value]
            or QuotaSnapshot("2026-11-01", 0, 300, 0, 500_000, 0, 1_500_000, 3_000_000)
        ),
        on_reload=lambda: (called.__setitem__("reload", True) or "小Q"),  # type: ignore[func-returns-value]
    )

    # Normal text & unknown slash commands pass through (handled=False)
    assert router.try_handle(_make_msg(user_id=999, text="你好呀")) == CommandResult(handled=False)
    assert router.try_handle(_make_msg(user_id=999, text="/notacommand")) == CommandResult(
        handled=False
    )

    # Non-admin attempting built-in admin commands is strictly intercepted and never reaches callbacks or LLM
    for raw_text, expected_cmd in [
        ("/ping", "ping"),
        (" /reset ", "reset"),
        ("/quota", "quota"),
        ("/reload", "reload"),
        ("/help", "help"),
    ]:
        res = router.try_handle(_make_msg(user_id=999, text=raw_text, mentioned_bot=True))
        assert res == CommandResult(
            handled=True,
            reply_text="权限不足：该管理指令仅限机器人管理员使用。",
            command_name=expected_cmd,
        )

    assert called == {"reset": False, "quota": False, "reload": False}


def test_commands_admin_executes_all_five_commands() -> None:
    reset_sessions: list[str] = []
    snapshot = QuotaSnapshot(
        day="2026-11-01",
        calls_used=12,
        max_daily_calls=300,
        tokens_used=45_000,
        max_daily_tokens=500_000,
        cost_micros_used=250_500,
        soft_budget_micros=1_500_000,
        hard_budget_micros=3_000_000,
    )

    router = AdminCommandRouter(
        admin_user_ids={888},
        on_reset=lambda sk: (reset_sessions.append(sk) or 18),  # type: ignore[func-returns-value]
        on_quota=lambda: snapshot,
        on_reload=lambda: "毒舌舰长·小Q",
    )

    ping_res = router.try_handle(_make_msg(user_id=888, text="/ping"))
    assert ping_res.handled is True
    assert ping_res.command_name == "ping"
    assert ping_res.reply_text is not None and len(ping_res.reply_text) > 0

    reset_res = router.try_handle(_make_msg(user_id=888, group_id=777, text="/reset"))
    assert reset_res.handled is True
    assert reset_res.command_name == "reset"
    assert reset_sessions == ["group:777"]
    assert reset_res.reply_text is not None and "18" in reset_res.reply_text

    # Simulates extracted text after "@bot /quota" in group
    quota_res = router.try_handle(
        _make_msg(user_id=888, group_id=777, text="  /quota  ", mentioned_bot=True)
    )
    assert quota_res.handled is True
    assert quota_res.command_name == "quota"
    assert quota_res.reply_text is not None
    assert "12" in quota_res.reply_text and "300" in quota_res.reply_text
    assert "45000" in quota_res.reply_text.replace(",", "")
    assert "$0.2505" in quota_res.reply_text
    assert "$3.0000" in quota_res.reply_text

    reload_res = router.try_handle(_make_msg(user_id=888, text="/reload"))
    assert reload_res.handled is True
    assert reload_res.command_name == "reload"
    assert reload_res.reply_text is not None and "毒舌舰长·小Q" in reload_res.reply_text

    help_res = router.try_handle(_make_msg(user_id=888, text="/help"))
    assert help_res.handled is True
    assert help_res.command_name == "help"
    assert help_res.reply_text is not None
    for cmd_token in ("/ping", "/reset", "/quota", "/reload", "/help"):
        assert cmd_token in help_res.reply_text
