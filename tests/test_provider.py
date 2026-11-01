from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from qq_agent.provider import (
    LLMProvider,
    ModelPricing,
    OpenAICompatProvider,
    ProviderError,
    estimate_cost_micros,
)
from qq_agent.types import ChatMessage


def test_estimate_cost_micros_builtin_and_fallback():
    # Built-in models produce positive integer micro-USD costs proportional to tokens
    cost_ds = estimate_cost_micros("deepseek-chat", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost_flash = estimate_cost_micros("glm-4-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost_plus = estimate_cost_micros("glm-4-plus", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost_unknown = estimate_cost_micros("custom-unknown-llm", prompt_tokens=1_000_000, completion_tokens=500_000)

    assert cost_ds > 0
    assert cost_flash > 0
    assert cost_plus > cost_flash
    assert cost_unknown > 0
    assert estimate_cost_micros("deepseek-chat", prompt_tokens=0, completion_tokens=0) == 0

    pricing = ModelPricing(input_micros_per_1m=200_000, output_micros_per_1m=800_000)
    assert pricing.input_micros_per_1m == 200_000
    assert pricing.output_micros_per_1m == 800_000


def test_normal_chat_usage_and_cost_accounting():
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-001",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "今晚开黑打战舰！"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            },
        )

    async def run() -> None:
        provider = OpenAICompatProvider(
            api_key="sk-test-key-001",
            base_url="https://api.deepseek.com/v1/",
            default_model="deepseek-chat",
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(provider, LLMProvider)
        try:
            result = await provider.chat(
                [
                    ChatMessage(role="system", content="你是群友阿橘"),
                    ChatMessage(role="user", content="今晚玩啥？"),
                ],
                temperature=0.6,
                max_tokens=512,
            )
        finally:
            await provider.aclose()

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert str(req.url) == "https://api.deepseek.com/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer sk-test-key-001"
        body: dict[str, Any] = json.loads(req.content.decode("utf-8"))
        assert body["model"] == "deepseek-chat"
        assert body["temperature"] == 0.6
        assert body["max_tokens"] == 512
        assert body["messages"] == [
            {"role": "system", "content": "你是群友阿橘"},
            {"role": "user", "content": "今晚玩啥？"},
        ]

        assert result.content == "今晚开黑打战舰！"
        assert result.model == "deepseek-chat"
        assert result.prompt_tokens == 1000
        assert result.completion_tokens == 500
        assert result.total_tokens == 1500
        assert result.cost_micros_usd == estimate_cost_micros("deepseek-chat", 1000, 500)
        assert result.cost_micros_usd > 0
        assert result.latency_ms >= 0.0

    asyncio.run(run())


def test_model_override_parameter():
    seen_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_models.append(payload["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "已切换模型"}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 1000},
            },
        )

    async def run() -> None:
        provider = OpenAICompatProvider(
            api_key="sk-override-test",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            default_model="glm-4-flash",
            transport=httpx.MockTransport(handler),
        )
        try:
            result = await provider.chat(
                [ChatMessage(role="user", content="帮我分析这段代码")],
                model="glm-4-plus",
            )
        finally:
            await provider.aclose()

        assert seen_models == ["glm-4-plus"]
        assert result.model == "glm-4-plus"
        assert result.total_tokens == 3000
        assert result.cost_micros_usd == estimate_cost_micros("glm-4-plus", 2000, 1000)

    asyncio.run(run())


def test_429_and_transport_error_retry_then_succeed():
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        if attempts == 2:
            raise httpx.ConnectError("Temporary upstream reset", request=request)
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [{"message": {"role": "assistant", "content": "重试成功啦"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )

    async def run() -> None:
        provider = OpenAICompatProvider(
            api_key="sk-retry-key",
            max_retries=2,
            transport=httpx.MockTransport(handler),
        )
        try:
            result = await provider.chat([ChatMessage(role="user", content="在吗")])
        finally:
            await provider.aclose()

        assert attempts == 3
        assert result.content == "重试成功啦"

    asyncio.run(run())


def test_401_no_retry_and_never_leaks_api_key():
    secret_key = "sk-top-secret-never-leak-998877"
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        # Simulate an upstream server that echoes the offending API key in its 401 error body
        return httpx.Response(
            401,
            json={
                "error": {
                    "message": f"Incorrect API key provided: {secret_key}. Authorization Bearer {secret_key} rejected."
                }
            },
        )

    async def run() -> None:
        provider = OpenAICompatProvider(
            api_key=secret_key,
            max_retries=3,
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(ProviderError) as exc_info:
                await provider.chat([ChatMessage(role="user", content="hello")])
        finally:
            await provider.aclose()

        err = exc_info.value
        assert attempts == 1  # 401 must fail immediately with zero retries
        assert err.status_code == 401
        assert secret_key not in str(err)
        assert secret_key not in repr(err)
        assert secret_key not in repr(provider)
        assert err.__cause__ is None or secret_key not in repr(err.__cause__)

    asyncio.run(run())


def test_503_exhausts_retries_and_400_403_fail_fast():
    secret_key = "sk-exhaust-secret-554433"
    calls_503 = 0

    def handler_503(request: httpx.Request) -> httpx.Response:
        nonlocal calls_503
        calls_503 += 1
        return httpx.Response(503, text=f"Upstream overloaded for key {secret_key}")

    async def run() -> None:
        async with OpenAICompatProvider(
            api_key=secret_key,
            max_retries=2,
            retry_base_delay_s=0.01,
            transport=httpx.MockTransport(handler_503),
        ) as provider:
            with pytest.raises(ProviderError) as exc_info:
                await provider.chat([ChatMessage(role="user", content="ping")])
            assert calls_503 == 3  # 1 initial + 2 retries
            assert exc_info.value.status_code == 503
            assert secret_key not in str(exc_info.value)

        for code in (400, 403):
            fast_calls = 0

            def handler_fast(request: httpx.Request, status: int = code) -> httpx.Response:
                nonlocal fast_calls
                fast_calls += 1
                return httpx.Response(status, json={"error": {"message": "Bad request or forbidden"}})

            async with OpenAICompatProvider(
                api_key=secret_key,
                max_retries=2,
                retry_base_delay_s=0.01,
                transport=httpx.MockTransport(handler_fast),
            ) as provider:
                with pytest.raises(ProviderError) as exc_fast:
                    await provider.chat([ChatMessage(role="user", content="ping")])
                assert fast_calls == 1
                assert exc_fast.value.status_code == code

    asyncio.run(run())


def test_anthropic_compat_messages_endpoint_and_thinking_block_filtering():
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_glm_pool_001",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3",
                "content": [
                    {"type": "thinking", "thinking": "Internal reasoning should not leak to QQ chat."},
                    {"type": "text", "text": "在呢！今晚一起开黑呀~"},
                ],
                "usage": {
                    "input_tokens": 120,
                    "cache_read_input_tokens": 80,
                    "output_tokens": 250,
                },
            },
        )

    async def run() -> None:
        async with OpenAICompatProvider(
            api_key="glm-pool-token",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3[1m]",
            transport=httpx.MockTransport(handler),
        ) as provider:
            res = await provider.chat(
                [
                    ChatMessage(role="system", content="你是小助"),
                    ChatMessage(role="user", content="在吗？"),
                ]
            )

        assert len(captured_requests) == 1
        req = captured_requests[0]
        assert str(req.url) == "http://127.0.0.1:15722/api/anthropic/v1/messages"
        assert req.headers["x-api-key"] == "glm-pool-token"
        assert req.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(req.content.decode("utf-8"))
        assert body["model"] == "glm-5.3"
        assert body["system"] == "你是小助"
        assert body["thinking"] == {"type": "disabled"}
        assert body["messages"] == [{"role": "user", "content": "在吗？"}]

        assert res.content == "在呢！今晚一起开黑呀~"
        assert "Internal reasoning" not in res.content
        assert res.prompt_tokens == 200
        assert res.completion_tokens == 250
        assert res.total_tokens == 450
        assert res.cost_micros_usd == estimate_cost_micros("glm-5.3", 200, 250)

    asyncio.run(run())


def test_multimodal_vision_messages_and_thinking_toggle():
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "msg_vision_01",
                "type": "message",
                "role": "assistant",
                "model": "glm-5.3-flash",
                "content": [{"type": "text", "text": "看到奶蛙问号图了"}],
                "usage": {"input_tokens": 300, "output_tokens": 20},
            },
        )

    async def run() -> None:
        async with OpenAICompatProvider(
            api_key="glm-pool-token",
            base_url="http://127.0.0.1:15722/api/anthropic",
            default_model="glm-5.3-flash",
            enable_thinking=False,
            transport=httpx.MockTransport(handler),
        ) as provider:
            res = await provider.chat(
                [
                    ChatMessage(
                        role="user",
                        content="看看这张图",
                        images=(("image/jpeg", "ZmFrZS1qcGVn"),),
                    )
                ]
            )
            assert res.content == "看到奶蛙问号图了"
            # Override enable_thinking=True omits disabled block
            await provider.chat(
                [ChatMessage(role="user", content="深思一下")],
                enable_thinking=True,
            )

        assert captured_bodies[0]["thinking"] == {"type": "disabled"}
        assert captured_bodies[0]["messages"] == [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "ZmFrZS1qcGVn",
                        },
                    },
                    {"type": "text", "text": "看看这张图"},
                ],
            }
        ]
        assert "thinking" not in captured_bodies[1]

    asyncio.run(run())
