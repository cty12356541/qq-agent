"""Async OpenAI-compatible & Anthropic-compatible LLM provider with retry, micro-USD accounting, and secret redaction."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import re
import time
from typing import Any, Protocol, runtime_checkable

import httpx

from qq_agent.types import ChatMessage, ChatResult

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_BEARER_RE = re.compile(r"(Bearer\s+)[^\s\"',;]+", flags=re.IGNORECASE)
_SK_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


class ProviderError(RuntimeError):
    """Sanitized LLM provider failure with optional HTTP status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Token pricing denominated in micro-USD ($1e-6) per 1,000,000 tokens."""

    input_micros_per_1m: int
    output_micros_per_1m: int

    def compute_micros(self, prompt_tokens: int, completion_tokens: int) -> int:
        in_toks = max(0, int(prompt_tokens))
        out_toks = max(0, int(completion_tokens))
        if in_toks == 0 and out_toks == 0:
            return 0
        numerator = in_toks * self.input_micros_per_1m + out_toks * self.output_micros_per_1m
        return (numerator + 500_000) // 1_000_000

    @staticmethod
    def estimate_cost_micros(model: str, prompt_tokens: int, completion_tokens: int) -> int:
        return estimate_cost_micros(model, prompt_tokens, completion_tokens)


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(input_micros_per_1m=270_000, output_micros_per_1m=1_100_000),
    "deepseek-reasoner": ModelPricing(input_micros_per_1m=550_000, output_micros_per_1m=2_190_000),
    "glm-5.3-flash": ModelPricing(input_micros_per_1m=100_000, output_micros_per_1m=400_000),
    "glm-5.3": ModelPricing(input_micros_per_1m=1_400_000, output_micros_per_1m=4_400_000),
    "glm-4-flash": ModelPricing(input_micros_per_1m=15_000, output_micros_per_1m=15_000),
    "glm-4-air": ModelPricing(input_micros_per_1m=70_000, output_micros_per_1m=70_000),
    "glm-4-plus": ModelPricing(input_micros_per_1m=700_000, output_micros_per_1m=700_000),
    "claude-haiku-5-5": ModelPricing(input_micros_per_1m=100_000, output_micros_per_1m=500_000),
    "claude-sonnet-5-5": ModelPricing(input_micros_per_1m=2_000_000, output_micros_per_1m=10_000_000),
    "gpt-4o-mini": ModelPricing(input_micros_per_1m=150_000, output_micros_per_1m=600_000),
    "gpt-4o": ModelPricing(input_micros_per_1m=2_500_000, output_micros_per_1m=10_000_000),
}

FALLBACK_MODEL_PRICING = ModelPricing(
    input_micros_per_1m=500_000,
    output_micros_per_1m=1_500_000,
)


def get_model_pricing(model: str) -> ModelPricing:
    """Resolve pricing for a model ID using exact match, longest prefix match, or fallback."""
    normalized = model.strip().lower()
    if normalized in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[normalized]
    best_prefix = ""
    best_pricing: ModelPricing | None = None
    for known_name, pricing in DEFAULT_MODEL_PRICING.items():
        if normalized.startswith(known_name) and len(known_name) > len(best_prefix):
            best_prefix = known_name
            best_pricing = pricing
    return best_pricing if best_pricing is not None else FALLBACK_MODEL_PRICING


def estimate_cost_micros(model: str, prompt_tokens: int, completion_tokens: int) -> int:
    """Estimate completion cost in micro-USD ($1e-6) for the given model and token counts."""
    return get_model_pricing(model).compute_micros(prompt_tokens, completion_tokens)


def _sanitize_error_message(raw: str, api_key: str) -> str:
    """Redact API keys and bearer tokens before truncating error text."""
    sanitized = raw
    if api_key:
        sanitized = sanitized.replace(api_key, "***REDACTED***")
        stripped = api_key.strip()
        if stripped and stripped != api_key:
            sanitized = sanitized.replace(stripped, "***REDACTED***")
    sanitized = _BEARER_RE.sub(r"\1***REDACTED***", sanitized)
    sanitized = _SK_TOKEN_RE.sub("sk-***REDACTED***", sanitized)
    sanitized = " ".join(sanitized.split())
    if len(sanitized) > 320:
        sanitized = sanitized[:317] + "..."
    return sanitized


def _extract_error_detail(response: httpx.Response, api_key: str) -> str:
    """Extract a concise, secret-redacted error summary from an HTTP error response."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err_obj = payload.get("error")
            if isinstance(err_obj, dict) and err_obj.get("message"):
                detail = str(err_obj["message"])
            elif isinstance(err_obj, str):
                detail = err_obj
            elif payload.get("message"):
                detail = str(payload["message"])
    except ValueError:
        detail = ""
    if not detail:
        detail = response.text or f"HTTP status {response.status_code}"
    return _sanitize_error_message(detail, api_key)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol implemented by all chat completion providers."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResult: ...


class OpenAICompatProvider:
    """Async chat completion client for OpenAI-compatible and Anthropic-compatible (`/v1/messages`) endpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        default_model: str = "deepseek-chat",
        timeout_s: float = 90.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_base_delay_s: float = 0.15,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._default_model = default_model
        self._max_retries = max(0, int(max_retries))
        self._retry_base_delay_s = max(0.0, float(retry_base_delay_s))
        lower_url = self._base_url.lower()
        self._is_anthropic = ("/anthropic" in lower_url) or ("api.anthropic.com" in lower_url)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    def __repr__(self) -> str:
        return (
            f"OpenAICompatProvider(base_url={self._base_url!r}, "
            f"default_model={self._default_model!r}, api_key='***')"
        )

    async def __aenter__(self) -> OpenAICompatProvider:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient."""
        await self._client.aclose()

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        stripped = name.strip()
        if stripped.endswith("[1m]") and len(stripped) > 4:
            return stripped[:-4]
        return stripped

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResult:
        raw_model = model or self._default_model
        target_model = (
            self._normalize_model_name(raw_model) if self._is_anthropic else raw_model
        )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._is_anthropic:
            url = (
                f"{self._base_url}/messages"
                if self._base_url.endswith("/v1")
                else f"{self._base_url}/v1/messages"
            )
            headers["anthropic-version"] = "2023-06-01"
            if self._api_key:
                headers["x-api-key"] = self._api_key
                headers["Authorization"] = f"Bearer {self._api_key}"
            sys_parts = [m.content for m in messages if m.role == "system" and m.content]
            conv_msgs = [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ]
            if not conv_msgs:
                conv_msgs = [{"role": "user", "content": ""}]
            payload: dict[str, Any] = {
                "model": target_model,
                "messages": conv_msgs,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if sys_parts:
                payload["system"] = "\n\n".join(sys_parts)
        else:
            url = f"{self._base_url}/chat/completions"
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            payload = {
                "model": target_model,
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        max_attempts = 1 + self._max_retries
        start_perf = time.perf_counter()
        response: httpx.Response | None = None

        for attempt in range(max_attempts):
            try:
                response = await self._client.post(url, headers=headers, json=payload)
            except httpx.TransportError as exc:
                detail = _sanitize_error_message(str(exc), self._api_key)
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self._retry_base_delay_s * (2**attempt))
                    continue
                raise ProviderError(
                    f"LLM transport error ({type(exc).__name__}) after {max_attempts} attempts: {detail}",
                    status_code=None,
                ) from None

            if 200 <= response.status_code < 300:
                break

            detail = _extract_error_detail(response, self._api_key)
            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(self._retry_base_delay_s * (2**attempt))
                    continue
                raise ProviderError(
                    f"LLM request failed with HTTP {response.status_code} after {max_attempts} attempts: {detail}",
                    status_code=response.status_code,
                )

            raise ProviderError(
                f"LLM request rejected with HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
            )

        assert response is not None
        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        try:
            data = response.json()
        except ValueError:
            detail = _sanitize_error_message(response.text, self._api_key)
            raise ProviderError(
                f"Invalid JSON from LLM provider: {detail}",
                status_code=response.status_code,
            ) from None

        if not isinstance(data, dict):
            raise ProviderError(
                "Malformed completion payload: expected JSON object",
                status_code=response.status_code,
            )

        resolved_model = str(data.get("model") or target_model)
        usage_obj = data.get("usage")
        usage = usage_obj if isinstance(usage_obj, dict) else {}

        if self._is_anthropic:
            blocks = data.get("content")
            if not isinstance(blocks, list) or not blocks:
                raise ProviderError(
                    "Malformed Anthropic payload: missing non-empty 'content' array",
                    status_code=response.status_code,
                )
            text_parts = [
                str(b.get("text") or "")
                for b in blocks
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "".join(text_parts).strip()
            prompt_tokens = int(usage.get("input_tokens") or 0) + int(
                usage.get("cache_read_input_tokens") or 0
            )
            completion_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        else:
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ProviderError(
                    "Malformed completion payload: missing non-empty 'choices'",
                    status_code=response.status_code,
                )

            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            message_obj = first_choice.get("message") if isinstance(first_choice, dict) else {}
            if not isinstance(message_obj, dict):
                raise ProviderError(
                    "Malformed completion payload: missing 'choices[0].message'",
                    status_code=response.status_code,
                )

            raw_content = message_obj.get("content")
            content = "" if raw_content is None else str(raw_content)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        cost_micros = estimate_cost_micros(resolved_model, prompt_tokens, completion_tokens)

        return ChatResult(
            content=content,
            model=resolved_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_micros_usd=cost_micros,
            latency_ms=latency_ms,
        )
