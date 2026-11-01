"""Unit tests for M2 AdminCommandRouter persona & long-term memory commands (W3-03)."""

from __future__ import annotations

from qq_agent.commands import AdminCommandRouter
from qq_agent.types import CommandResult, InboundMessage

ADMIN_QQ = 88888
NORMAL_QQ = 10001
TARGET_QQ = 20002
GROUP_ID = 70001
BOT_QQ = 99999


def _msg(
    text: str,
    *,
    user_id: int = ADMIN_QQ,
    chat_type: str = "group",
    group_id: int | None = GROUP_ID,
    mentioned_bot: bool = False,
) -> InboundMessage:
    effective_group = group_id if chat_type == "group" else None
    return InboundMessage(
        event_id="2001",
        chat_type="group" if chat_type == "group" else "private",
        user_id=user_id,
        group_id=effective_group,
        sender_name="测试用户",
        text=text,
        raw_message=text,
        mentioned_bot=mentioned_bot,
        self_id=BOT_QQ,
    )


def test_m2_commands_non_admin_interception_blocks_all_callbacks() -> None:
    invoked: list[str] = []

    router = AdminCommandRouter(
        admin_user_ids={ADMIN_QQ},
        on_persona_status=lambda: (invoked.append("persona_status") or ("UD", "default", ["default"])),  # type: ignore[func-returns-value]
        on_persona_switch=lambda p: (invoked.append(f"switch:{p}") or p),  # type: ignore[func-returns-value]
        on_remember=lambda c, s: (invoked.append(f"remember:{s}:{c}") or 1),  # type: ignore[func-returns-value]
        on_list_memories=lambda inc: (invoked.append(f"list:{inc}") or []),  # type: ignore[func-returns-value]
        on_forget=lambda fid: (invoked.append(f"forget:{fid}") or True),  # type: ignore[func-returns-value]
        on_profile=lambda qq, n: (invoked.append(f"profile:{qq}:{n}") or "ok"),  # type: ignore[func-returns-value]
    )

    cases = [
        ("/persona", "persona"),
        ("/persona use wows", "persona"),
        (f"[CQ:at,qq={BOT_QQ}] /remember --public 秘密", "remember"),
        ("@UD /memories", "memories"),
        ("/forget 1", "forget"),
        (f"/profile {TARGET_QQ} 备注", "profile"),
    ]
    for raw_text, expected_cmd in cases:
        res = router.try_handle(
            _msg(raw_text, user_id=NORMAL_QQ, chat_type="group", mentioned_bot=True)
        )
        assert res == CommandResult(
            handled=True,
            reply_text="权限不足：该管理指令仅限机器人管理员使用。",
            command_name=expected_cmd,
        )

    assert invoked == []


def test_persona_status_switch_and_friendly_errors() -> None:
    state = {"active": "default", "profiles": ["default", "wows", "late_night"]}

    def _status() -> tuple[str, str, list[str]]:
        return ("UD", str(state["active"]), list(state["profiles"]))  # type: ignore[arg-type]

    def _switch(profile: str) -> str:
        if profile == "bad_schema":
            raise ValueError("TOML 字段缺失: role_prompt")
        if profile not in state["profiles"]:
            raise FileNotFoundError(f"未找到档案 {profile}.toml")
        state["active"] = profile
        return f"UD（档案：{profile}）"

    router = AdminCommandRouter(
        admin_user_ids={ADMIN_QQ},
        on_persona_status=_status,
        on_persona_switch=_switch,
    )

    # 1) /persona without arguments shows current name, active profile, and available profiles
    st_res = router.try_handle(_msg("/persona"))
    assert st_res.handled is True
    assert st_res.command_name == "persona"
    assert st_res.reply_text is not None
    assert "UD" in st_res.reply_text
    assert "default" in st_res.reply_text
    assert "wows" in st_res.reply_text
    assert "late_night" in st_res.reply_text

    # 2) /persona use <profile> switches active profile
    sw_res = router.try_handle(_msg("@UD /persona use wows", mentioned_bot=True))
    assert sw_res.handled is True
    assert sw_res.command_name == "persona"
    assert state["active"] == "wows"
    assert sw_res.reply_text is not None and "wows" in sw_res.reply_text

    # 3) Shorthand /persona <profile> also switches active profile
    sw_short = router.try_handle(_msg("/persona late_night"))
    assert sw_short.handled is True
    assert state["active"] == "late_night"
    assert sw_short.reply_text is not None and "late_night" in sw_short.reply_text

    # 4) Missing profile argument after `use` returns usage prompt
    usage_res = router.try_handle(_msg("/persona use   "))
    assert usage_res.handled is True
    assert usage_res.reply_text is not None and "用法" in usage_res.reply_text

    # 5) FileNotFoundError and ValueError are caught and returned cleanly
    fnf_res = router.try_handle(_msg("/persona use nonexistent"))
    assert fnf_res.handled is True
    assert fnf_res.command_name == "persona"
    assert fnf_res.reply_text is not None and "nonexistent" in fnf_res.reply_text

    val_res = router.try_handle(_msg("/persona use bad_schema"))
    assert val_res.handled is True
    assert val_res.command_name == "persona"
    assert val_res.reply_text is not None and "role_prompt" in val_res.reply_text


def test_remember_default_scopes_and_flag_overrides() -> None:
    recorded: list[tuple[int, str, str]] = []

    def _remember(content: str, scope: str) -> int:
        new_id = len(recorded) + 1
        recorded.append((new_id, scope, content))
        return new_id

    router = AdminCommandRouter(
        admin_user_ids={ADMIN_QQ},
        on_remember=_remember,
    )

    # 1) Private chat defaults to scope="admin_private"
    r_priv = router.try_handle(_msg("/remember 下周三考专升本模拟", chat_type="private"))
    assert r_priv.handled is True
    assert r_priv.command_name == "remember"
    assert recorded[-1] == (1, "admin_private", "下周三考专升本模拟")
    assert r_priv.reply_text is not None and "#1" in r_priv.reply_text

    # 2) Group chat defaults to scope=f"group:{group_id}"
    r_grp = router.try_handle(_msg("/remember 每周五晚八点打军团战", chat_type="group", group_id=GROUP_ID))
    assert r_grp.handled is True
    assert recorded[-1] == (2, f"group:{GROUP_ID}", "每周五晚八点打军团战")
    assert r_grp.reply_text is not None and f"group:{GROUP_ID}" in r_grp.reply_text

    # 3) --public overrides default in both private and group chat
    r_pub = router.try_handle(_msg("/remember --public UD最常开的巡洋舰是能代", chat_type="private"))
    assert r_pub.handled is True
    assert recorded[-1] == (3, "public", "UD最常开的巡洋舰是能代")
    assert r_pub.reply_text is not None and "public" in r_pub.reply_text

    # 4) --private overrides default in group chat AND avoids echoing secret content in group reply
    r_grp_priv = router.try_handle(
        _msg("/remember --private 管理员私密备忘口令", chat_type="group", group_id=GROUP_ID)
    )
    assert r_grp_priv.handled is True
    assert recorded[-1] == (4, "admin_private", "管理员私密备忘口令")
    assert r_grp_priv.reply_text is not None
    assert "admin_private" in r_grp_priv.reply_text
    assert "管理员私密备忘口令" not in r_grp_priv.reply_text

    # 5) Empty content (with or without flag) prompts usage without writing
    for empty_cmd in ("/remember", "/remember   ", "/remember --public", "/remember --private  "):
        empty_res = router.try_handle(_msg(empty_cmd, chat_type="private"))
        assert empty_res.handled is True
        assert empty_res.command_name == "remember"
        assert empty_res.reply_text is not None and "用法" in empty_res.reply_text
    assert len(recorded) == 4


def test_memories_group_never_echoes_admin_private_while_private_includes_it() -> None:
    seen_flags: list[bool] = []
    all_entries: list[tuple[int, str, str]] = [
        (1, "public", "公开记忆：UD喜欢能代"),
        (2, f"group:{GROUP_ID}", "本群记忆：周五军团战"),
        (3, "admin_private", "绝密记忆：仅限私聊可见的私人备忘"),
    ]

    def _list_memories(include_admin_private: bool) -> list[tuple[int, str, str]]:
        seen_flags.append(include_admin_private)
        # Intentionally return unfiltered list to also verify router's defense-in-depth filter
        return list(all_entries)

    router = AdminCommandRouter(
        admin_user_ids={ADMIN_QQ},
        on_list_memories=_list_memories,
    )

    # In group chat: include_admin_private MUST be False, and admin_private must NEVER appear
    grp_res = router.try_handle(_msg("/memories", chat_type="group", group_id=GROUP_ID))
    assert grp_res.handled is True
    assert grp_res.command_name == "memories"
    assert seen_flags == [False]
    assert grp_res.reply_text is not None
    assert "公开记忆：UD喜欢能代" in grp_res.reply_text
    assert "本群记忆：周五军团战" in grp_res.reply_text
    assert "admin_private" not in grp_res.reply_text
    assert "绝密记忆" not in grp_res.reply_text

    # In private chat: include_admin_private is True, all memories including admin_private appear
    priv_res = router.try_handle(_msg("/memories", chat_type="private"))
    assert priv_res.handled is True
    assert priv_res.command_name == "memories"
    assert seen_flags == [False, True]
    assert priv_res.reply_text is not None
    assert "绝密记忆：仅限私聊可见的私人备忘" in priv_res.reply_text
    assert "admin_private" in priv_res.reply_text


def test_forget_profile_unbound_fallbacks_and_updated_help() -> None:
    existing_ids = {1, 2}
    profiles: dict[int, str] = {}

    router = AdminCommandRouter(
        admin_user_ids={ADMIN_QQ},
        on_forget=lambda fid: (fid in existing_ids and (existing_ids.remove(fid) or True)),  # type: ignore[func-returns-value]
        on_profile=lambda qq, notes: (profiles.__setitem__(qq, notes) or f"已更新 {qq} 画像：{notes}"),  # type: ignore[func-returns-value]
    )

    # /forget valid IDs (with and without leading '#') & missing ID & invalid arguments
    f_ok = router.try_handle(_msg("/forget 1"))
    assert f_ok.handled is True and f_ok.command_name == "forget"
    assert 1 not in existing_ids
    assert f_ok.reply_text is not None and "#1" in f_ok.reply_text

    f_hash = router.try_handle(_msg("/forget #2"))
    assert f_hash.handled is True and 2 not in existing_ids

    f_miss = router.try_handle(_msg("/forget 999"))
    assert f_miss.handled is True
    assert f_miss.reply_text is not None and "未找到" in f_miss.reply_text

    for bad_forget in ("/forget", "/forget abc"):
        f_bad = router.try_handle(_msg(bad_forget))
        assert f_bad.handled is True and f_bad.command_name == "forget"

    # /profile valid QQ, @QQ, CQ:at, and invalid arguments
    p_ok = router.try_handle(_msg(f"/profile {TARGET_QQ} 战舰大和绝活哥，正在备考专升本"))
    assert p_ok.handled is True
    assert p_ok.command_name == "profile"
    assert profiles[TARGET_QQ] == "战舰大和绝活哥，正在备考专升本"
    assert p_ok.reply_text is not None and str(TARGET_QQ) in p_ok.reply_text

    p_cq = router.try_handle(_msg(f"/profile [CQ:at,qq={NORMAL_QQ}] 帝国4快攻老手"))
    assert p_cq.handled is True
    assert profiles[NORMAL_QQ] == "帝国4快攻老手"

    for bad_profile in ("/profile", f"/profile {TARGET_QQ}", "/profile abc 备注", "/profile -5 备注"):
        p_bad = router.try_handle(_msg(bad_profile))
        assert p_bad.handled is True and p_bad.command_name == "profile"
        assert p_bad.reply_text is not None and "用法" in p_bad.reply_text

    # /help lists all 10 commands
    help_res = router.try_handle(_msg("/help"))
    assert help_res.handled is True
    assert help_res.reply_text is not None
    for cmd_token in (
        "/ping",
        "/reset",
        "/quota",
        "/reload",
        "/persona",
        "/remember",
        "/memories",
        "/forget",
        "/profile",
        "/help",
    ):
        assert cmd_token in help_res.reply_text

    # Minimal router without M2 callbacks bound returns graceful unbound notices
    bare_router = AdminCommandRouter(admin_user_ids={ADMIN_QQ})
    for bare_cmd in (
        "/persona",
        "/persona use wows",
        "/remember 测试",
        "/memories",
        "/forget 1",
        f"/profile {TARGET_QQ} 备注",
    ):
        bare_res = bare_router.try_handle(_msg(bare_cmd))
        assert bare_res.handled is True
        assert bare_res.reply_text is not None and "尚未绑定" in bare_res.reply_text
