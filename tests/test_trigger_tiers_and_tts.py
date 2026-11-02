"""Unit tests for per-group trigger tiers (`1028505110:low`), `/tier` admin command, and local NaiLong TTS."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from qq_agent.channel_bridge import DirectMemoryBus
from qq_agent.commands import AdminCommandRouter, handle_trigger_tier_command
from qq_agent.config import AgentConfig
from qq_agent.tools.chat_tools import ChatTurnState
from qq_agent.trigger import BUILTIN_TRIGGER_TIERS, RuleTrigger
from qq_agent.tts import (
    DEFAULT_NAILONG_PROMPT_TEXT,
    NaiLongTtsEngine,
    build_send_voice_tool,
    resolve_nailong_assets,
)
from qq_agent.types import InboundMessage


def _group_msg(
    text: str,
    *,
    group_id: int = 1028505110,
    user_id: int = 10001,
    mentioned_bot: bool = False,
) -> InboundMessage:
    return InboundMessage(
        event_id="9001",
        chat_type="group",
        user_id=user_id,
        group_id=group_id,
        sender_name="群友A",
        text=text,
        raw_message=text,
        mentioned_bot=mentioned_bot,
    )


def test_config_parses_group_trigger_tiers() -> None:
    cfg = AgentConfig.from_env(
        env_file=None,
        environ={
            "QQ_GROUP_TRIGGER_TIERS": "1028505110:low, 590176297:high, 916748723:passive",
        },
    )
    assert cfg.group_trigger_tiers == {
        1028505110: "low",
        590176297: "high",
        916748723: "passive",
    }


def test_group_trigger_tier_low_reduces_frequency_for_1028505110() -> None:
    clock_now = [100.0]
    trig = RuleTrigger(
        bot_qq=3777852533,
        topic_keywords=("游戏",),
        topic_prob=0.88,
        proactive_prob=0.40,
        proactive_every_n=4,
        group_tiers={1028505110: "low"},
        rng=lambda: 0.20,  # 0.20 > low proactive_prob (0.02), < default (0.40)
        clock=lambda: clock_now[0],
    )

    low_prof = trig.profile_for_group(1028505110)
    assert low_prof == BUILTIN_TRIGGER_TIERS["low"]
    assert low_prof.proactive_every_n == 0
    assert low_prof.proactive_cooldown_s == 45.0
    assert low_prof.max_followup_turns == 3

    default_prof = trig.profile_for_group(590176297)
    assert default_prof.tier == "default"
    assert default_prof.proactive_every_n == 4

    # In group 590176297 (default), rng=0.20 < 0.40 triggers random proactive reply
    d_default = trig.evaluate(_group_msg("今天天气不错呀", group_id=590176297))
    assert d_default.should_reply is True
    assert d_default.reason == "group_proactive_random"

    # In group 1028505110 (low), rng=0.20 > 0.02 does NOT trigger random reply on message #1
    d_low = trig.evaluate(_group_msg("今天天气不错呀", group_id=1028505110))
    assert d_low.should_reply is False

    # Followup in group 1028505110 (low) expires after 3 turns
    trig.record_bot_reply(1028505110, 10001)
    clock_now[0] += 5.0
    f1 = trig.evaluate(_group_msg("第一句追问", group_id=1028505110, user_id=10001))
    assert f1.should_reply is True and f1.reason == "group_followup_window"
    f2 = trig.evaluate(_group_msg("第二句追问", group_id=1028505110, user_id=10001))
    assert f2.should_reply is True and f2.reason == "group_followup_window"
    f3 = trig.evaluate(_group_msg("第三句追问", group_id=1028505110, user_id=10001))
    assert f3.should_reply is True and f3.reason == "group_followup_window"
    f4 = trig.evaluate(_group_msg("第四句超过low档3轮上限", group_id=1028505110, user_id=10001))
    assert f4.should_reply is False


def test_admin_tier_command_inspect_and_switch() -> None:
    trig = RuleTrigger(bot_qq=3777852533, group_tiers={1028505110: "low"})
    router = AdminCommandRouter(
        admin_user_ids={2823611930},
        on_tier=lambda msg, arg: handle_trigger_tier_command(trig, msg, arg),
    )

    res_view = router.try_handle(
        _group_msg("/tier", group_id=1028505110, user_id=2823611930)
    )
    assert res_view.handled is True
    assert "1028505110 激发档位：low" in res_view.reply_text

    res_switch = router.try_handle(
        _group_msg("/tier passive", group_id=1028505110, user_id=2823611930)
    )
    assert res_switch.handled is True
    assert "passive" in res_switch.reply_text
    assert trig.profile_for_group(1028505110).tier == "passive"


def test_nailong_tts_engine_and_send_voice_tool() -> None:
    assets = resolve_nailong_assets(Path("data/tts/nailong"))
    assert assets.is_ready is True

    seen_params: dict[str, str] = {}

    async def _mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/tts":
            seen_params.update(dict(request.url.params))
            return httpx.Response(200, content=b"RIFF" + (b"\x00" * 128))
        return httpx.Response(200, json={"ok": True})

    engine = NaiLongTtsEngine(transport=httpx.MockTransport(_mock_handler))

    async def _run() -> None:
        synth = await engine.synthesize("鹅鹅鹅，曲项向天歌")
        assert synth.backend_used == "local_gpt_sovits_nailong"
        assert synth.voice_uri is not None and synth.voice_uri.startswith("base64://")
        assert seen_params.get("prompt_text") == DEFAULT_NAILONG_PROMPT_TEXT

        bus = DirectMemoryBus()
        await bus.start()
        try:
            state = ChatTurnState()
            binding = build_send_voice_tool(
                bus,
                state,
                tts_engine=engine,
                msg=_group_msg("念首诗听听", group_id=1028505110),
            )
            out = await binding.execute({"text": "鹅鹅鹅，曲项向天歌"})
            assert "local_gpt_sovits_nailong" in out
            batch = await bus.poll_outbound(limit=4)
            assert len(batch) == 1
            _, reply = batch[0]
            assert reply.voice_uri is not None and reply.voice_uri.startswith("base64://")
        finally:
            await bus.close()

    asyncio.run(_run())
