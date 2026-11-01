from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from qq_agent.provider import OpenAICompatProvider, ProviderError
from qq_agent.types import ChatMessage, ToolCall, ToolDef, ToolOutcome


def _sample_tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="send_message",
            description="Send natural QQ chat bubbles",
            input_schema={
                "type": "object",
                "properties": {
                    "bubbles": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["bubbles"],
            },
            is_chat_terminal=True,
            requires_followup=False,
        ),
        ToolDef(
            name="web_search",
            description="Search recent web information",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            is_chat_terminal=False,
            requires_followup=True,
        ),
    ]


def test_anthropic_tools_payload_and_empty_text_tool_use_parsing() -> None:
    captured_bodies: list[dict[str, Any]] = []
    call_idx = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_idx
        call_idx += 1
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        if call_idx == 1:
            # Pure thinking + tool_use response with zero text blocks
            return httpx.Response(
                200,
                json={
                    "id": "msg_tool_01",
                    "type": "message",
                    "role": "assistant",
                    "model": "glm-5.3-flash",
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "thinking", "thinking": "Need to search WoWS patch notes first."},
                        {
                            "type": "tool_use",
                            "id": "toolu_01",
                            "name": "web_search",
                            "input": {"query": "战舰世界 莫斯科 传奇插"},
                        },
                    ],
                    "usage": {"input_tokens": 180, "output_tokens": 45},
                },
            )
        # Empty content & no tool_use -> must raise ProviderError
        return httpx.Response(
            200,
            json={
                "id": "msg_empty_02",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "stop_reason": "end_turn",
                "content": [{"type": "thinking", "thinking": "Only thinking, no text or tools."}],
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
        )

    async def run() -> None:
        tools = _sample_tools()
        async with OpenAICompatProvider(
            api_key="glm-pool-key",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3-flash",
            transport=httpx.MockTransport(handler),
        ) as provider:
            res = await provider.chat(
                [ChatMessage(role="user", content="莫斯科传奇插怎么样？")],
                tools=tools,
            )
            assert res.content == ""
            assert res.stop_reason == "tool_use"
            assert res.tool_calls == (
                ToolCall(
                    id="toolu_01",
                    name="web_search",
                    arguments={"query": "战舰世界 莫斯科 传奇插"},
                ),
            )
            assert res.total_tokens == 225

            with pytest.raises(ProviderError, match="no tool_use"):
                await provider.chat([ChatMessage(role="user", content="test empty")])

        assert captured_bodies[0]["tools"] == [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    asyncio.run(run())


def test_anthropic_consecutive_tool_result_and_user_turn_merging() -> None:
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_merge_01",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_send",
                        "name": "send_message",
                        "input": {"bubbles": ["莫斯科传奇插加射程和精度，很稳）"]},
                    }
                ],
                "usage": {"input_tokens": 320, "output_tokens": 40},
            },
        )

    async def run() -> None:
        history = [
            ChatMessage(role="system", content="你是群友UD"),
            ChatMessage(role="user", content="帮我查下莫斯科传奇插"),
            # Assistant turn 1: empty text + tool_calls (must NOT emit empty {"type":"text","text":""})
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(
                    ToolCall("t_search", "web_search", {"query": "莫斯科传奇插"}),
                    ToolCall("t_fetch", "web_search", {"query": "斯大林对比"}),
                ),
            ),
            # Tool results followed immediately by a live <steering_update> user message
            ChatMessage(
                role="tool",
                content="",
                tool_results=(
                    ToolOutcome("t_search", "web_search", "主炮精度-7%，射程+8%", is_error=False),
                    ToolOutcome("t_fetch", "web_search", "upstream timeout", is_error=True),
                ),
            ),
            ChatMessage(
                role="user",
                content="<steering_update>\n顺便只说结论就行\n</steering_update>",
            ),
            # Assistant turn 2: non-empty text + tool_calls
            ChatMessage(
                role="assistant",
                content="我再确认一条细节",
                tool_calls=(ToolCall("t_verify", "web_search", {"query": "莫斯科射速"}),),
            ),
            # User turn followed by another user-role tool_results message (reverse merge direction)
            ChatMessage(role="user", content="补充：国服最新版本"),
            ChatMessage(
                role="user",
                content="",
                tool_results=(ToolOutcome("t_verify", "web_search", "装填不变", is_error=False),),
            ),
        ]

        async with OpenAICompatProvider(
            api_key="glm-pool-key",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3-flash",
            transport=httpx.MockTransport(handler),
        ) as provider:
            res = await provider.chat(history, tools=_sample_tools())
            assert res.tool_calls[0].name == "send_message"

        msgs = captured_bodies[0]["messages"]
        # Anthropic strictly forbids consecutive turns with the same "user" role
        for prev_m, next_m in zip(msgs, msgs[1:]):
            assert not (prev_m["role"] == "user" and next_m["role"] == "user")

        # Turn 0: initial user
        assert msgs[0] == {"role": "user", "content": "帮我查下莫斯科传奇插"}
        # Turn 1: assistant with pure tool_use blocks (no empty text block)
        assert msgs[1] == {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t_search", "name": "web_search", "input": {"query": "莫斯科传奇插"}},
                {"type": "tool_use", "id": "t_fetch", "name": "web_search", "input": {"query": "斯大林对比"}},
            ],
        }
        # Turn 2: merged tool_results + steering user text block
        assert msgs[2] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t_search",
                    "content": "主炮精度-7%，射程+8%",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t_fetch",
                    "content": "upstream timeout",
                    "is_error": True,
                },
                {
                    "type": "text",
                    "text": "<steering_update>\n顺便只说结论就行\n</steering_update>",
                },
            ],
        }
        # Turn 3: assistant with text + tool_use
        assert msgs[3] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我再确认一条细节"},
                {"type": "tool_use", "id": "t_verify", "name": "web_search", "input": {"query": "莫斯科射速"}},
            ],
        }
        # Turn 4: user text merged with subsequent tool_result block
        assert msgs[4] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "补充：国服最新版本"},
                {"type": "tool_result", "tool_use_id": "t_verify", "content": "装填不变"},
            ],
        }

    asyncio.run(run())


def test_openai_tools_serialization_and_fault_tolerant_tool_calls_parsing() -> None:
    captured_bodies: list[dict[str, Any]] = []
    step = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal step
        step += 1
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        if step == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tools-1",
                    "model": "deepseek-chat",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_ok",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": '{"query": "Python 3.13 GIL"}',
                                        },
                                    },
                                    {
                                        "id": "call_bad_json",
                                        "type": "function",
                                        "function": {
                                            "name": "web_search",
                                            "arguments": "{broken-json",
                                        },
                                    },
                                    {
                                        "id": "call_non_dict",
                                        "type": "function",
                                        "function": {
                                            "name": "send_message",
                                            "arguments": '["bubble"]',
                                        },
                                    },
                                    {
                                        "id": "call_empty",
                                        "type": "function",
                                        "function": {
                                            "name": "send_message",
                                            "arguments": "   ",
                                        },
                                    },
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 150, "completion_tokens": 60, "total_tokens": 210},
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-empty",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            },
        )

    async def run() -> None:
        tools = _sample_tools()
        messages = [
            ChatMessage(role="system", content="你是小助手"),
            ChatMessage(role="user", content="查一下 Python 3.13"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=(ToolCall("c_prev", "web_search", {"query": "PEP 703"}),),
            ),
            ChatMessage(
                role="tool",
                content="",
                tool_results=(
                    ToolOutcome("c_prev", "web_search", "Free-threaded CPython"),
                    ToolOutcome("c_prev_2", "web_search", "Experimental flag"),
                ),
            ),
        ]

        async with OpenAICompatProvider(
            api_key="sk-openai-test",
            base_url="https://api.deepseek.com/v1",
            default_model="deepseek-chat",
            transport=httpx.MockTransport(handler),
        ) as provider:
            res = await provider.chat(messages, tools=tools)
            assert res.content == ""
            assert res.stop_reason == "tool_calls"
            assert len(res.tool_calls) == 4
            assert res.tool_calls[0] == ToolCall(
                id="call_ok",
                name="web_search",
                arguments={"query": "Python 3.13 GIL"},
            )
            assert res.tool_calls[1] == ToolCall(
                id="call_bad_json",
                name="web_search",
                arguments={"_raw": "{broken-json"},
            )
            assert res.tool_calls[2] == ToolCall(
                id="call_non_dict",
                name="send_message",
                arguments={"_raw": ["bubble"]},
            )
            assert res.tool_calls[3] == ToolCall(
                id="call_empty",
                name="send_message",
                arguments={},
            )

            with pytest.raises(ProviderError, match="no tool_calls"):
                await provider.chat([ChatMessage(role="user", content="empty check")])

        req_body = captured_bodies[0]
        assert req_body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]
        assert req_body["messages"][2] == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c_prev",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "PEP 703"}',
                    },
                }
            ],
        }
        assert req_body["messages"][3] == {
            "role": "tool",
            "tool_call_id": "c_prev",
            "content": "Free-threaded CPython",
        }
        assert req_body["messages"][4] == {
            "role": "tool",
            "tool_call_id": "c_prev_2",
            "content": "Experimental flag",
        }

    asyncio.run(run())


def test_per_call_enable_thinking_override_across_protocols() -> None:
    anthropic_bodies: list[dict[str, Any]] = []
    openai_bodies: list[dict[str, Any]] = []

    def anthropic_handler(request: httpx.Request) -> httpx.Response:
        anthropic_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "glm-5.3-flash",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    def openai_handler(request: httpx.Request) -> httpx.Response:
        openai_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "model": "glm-5.3-flash",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    async def run() -> None:
        # 1) Shared Anthropic pool initialized with enable_thinking=True, overridden per-call
        async with OpenAICompatProvider(
            api_key="pool-key",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3-flash",
            enable_thinking=True,
            transport=httpx.MockTransport(anthropic_handler),
        ) as anth_provider:
            # Default inherits True -> thinking enabled (no disabled block)
            await anth_provider.chat([ChatMessage(role="user", content="slow default")])
            # Fast-brain override False -> thinking disabled
            await anth_provider.chat(
                [ChatMessage(role="user", content="fast override")],
                enable_thinking=False,
            )
            # Slow-brain explicit True -> thinking enabled
            await anth_provider.chat(
                [ChatMessage(role="user", content="slow explicit")],
                enable_thinking=True,
            )

        assert "thinking" not in anthropic_bodies[0]
        assert anthropic_bodies[1]["thinking"] == {"type": "disabled"}
        assert "thinking" not in anthropic_bodies[2]

        # 2) Shared OpenAI-compat GLM pool initialized with enable_thinking=False, overridden per-call
        async with OpenAICompatProvider(
            api_key="pool-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            default_model="glm-5.3-flash",
            enable_thinking=False,
            transport=httpx.MockTransport(openai_handler),
        ) as oai_provider:
            await oai_provider.chat([ChatMessage(role="user", content="fast default")])
            await oai_provider.chat(
                [ChatMessage(role="user", content="slow subagent")],
                enable_thinking=True,
            )
            await oai_provider.chat(
                [ChatMessage(role="user", content="fast explicit")],
                enable_thinking=False,
            )

        assert openai_bodies[0]["thinking"] == {"type": "disabled"}
        assert "thinking" not in openai_bodies[1]
        assert openai_bodies[2]["thinking"] == {"type": "disabled"}

    asyncio.run(run())
