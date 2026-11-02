# qq_agent/llmos_bus.py
"""基于 `llmos` `nlos.sabi.ChannelControl` 持久化队列的崩溃安全双通道总线与 SABI/IPC 辅助工具。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
import dataclasses
import hashlib
import inspect
import logging
from pathlib import Path
import secrets
from typing import Any, Callable, TypeVar

from .bus_base import BusQueueMetrics, DEFAULT_CONSUMER_FIBER_ID, _validate_nonzero_id16
from .llmos_sdk import (
    INBOUND_CHANNEL_CREATE_KEY,
    OUTBOUND_CHANNEL_CREATE_KEY,
    derive_deterministic_id16,
    fiber_id_for_session,
    load_nlos_sdk,
    process_id_for_role,
)
from .types import InboundMessage, OutboundReply

logger = logging.getLogger(__name__)

R = TypeVar("R")

CONSUMER_FIBER_GENERATION: int = 1
MAX_REGISTERED_SEQ_CACHE: int = 4096
_VOICE_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "tts" / "cache"


def _spill_large_voice_uri(reply: OutboundReply) -> OutboundReply:
    """Spill large `base64://` voice payloads (>16KB) to disk so IPC stays well under 64KB."""
    uri = reply.voice_uri
    if not uri or len(uri) <= 16384 or not uri.startswith("base64://"):
        return reply
    try:
        _VOICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(uri.encode("ascii", errors="ignore")).hexdigest()[:20]
        blob_path = (_VOICE_CACHE_DIR / f"voice_{digest}.b64").resolve()
        blob_path.write_text(uri, encoding="ascii")
        return dataclasses.replace(reply, voice_uri=f"localb64://{blob_path}")
    except Exception:
        return reply


def _rehydrate_voice_uri(reply: OutboundReply) -> OutboundReply:
    """Restore `localb64://` spilled voice payload back into `base64://` in the adapter."""
    uri = reply.voice_uri
    if not uri or not uri.startswith("localb64://"):
        return reply
    blob_path = Path(uri[len("localb64://") :])
    try:
        content = blob_path.read_text(encoding="ascii").strip()
        blob_path.unlink(missing_ok=True)
        if content.startswith("base64://"):
            return dataclasses.replace(reply, voice_uri=content)
    except Exception:
        pass
    return reply


def _try_parse_int(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    text = str(val).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _random_nonzero_id16() -> bytes:
    token = secrets.token_bytes(16)
    while token == b"\x00" * 16:
        token = secrets.token_bytes(16)
    return token


def _extract_acked_seq(state: Any) -> int:
    if hasattr(state, "acknowledged_through_sequence"):
        return int(state.acknowledged_through_sequence)
    if hasattr(state, "consume_high_water"):
        return int(state.consume_high_water)
    return 0


def _extract_trim_seq(state: Any, fallback_acked: int) -> int:
    if hasattr(state, "trim_high_water"):
        return int(state.trim_high_water)
    if hasattr(state, "compacted_through_sequence"):
        return int(state.compacted_through_sequence)
    return fallback_acked


def _extract_backlog_bytes(state: Any) -> int:
    if hasattr(state, "backlog_bytes"):
        return int(state.backlog_bytes)
    if hasattr(state, "outstanding_bytes"):
        return int(state.outstanding_bytes)
    return 0


def _accepts_kwarg(fn: Any, kwarg_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kwarg_name and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _load_channel_client_cls(llmos_root: Path) -> Any:
    """无污染加载 `<llmos_root>/sdk/python` 中的 `nlos_sdk.ChannelClient`。"""
    return load_nlos_sdk(llmos_root).ChannelClient


class LlmosChannelBus:
    """基于 `llmos` `nlos.sabi.ChannelControl` 持久化队列的崩溃安全双通道总线（DOGFOOD-FULL 模式）。"""

    def __init__(
        self,
        endpoint: str,
        *,
        llmos_root: Path | str = r"D:\llmos",
        capacity_bytes: int = 65536,
        client_factory: Callable[[str], Any] | None = None,
        role: str = "all-in-one",
        inbound_create_key: bytes | None = INBOUND_CHANNEL_CREATE_KEY,
        outbound_create_key: bytes | None = OUTBOUND_CHANNEL_CREATE_KEY,
        compact_every_n: int = 32,
        max_ipc_retries: int = 3,
        retry_base_delay_s: float = 0.02,
    ) -> None:
        if int(compact_every_n) < 1:
            raise ValueError("compact_every_n must be >= 1")
        self._endpoint = endpoint
        self._llmos_root = Path(llmos_root)
        self._capacity_bytes = capacity_bytes
        self._client_factory = client_factory
        self._role = str(role)
        self._process_id: bytes = process_id_for_role(self._role)
        self._compact_every_n = int(compact_every_n)
        self._max_ipc_retries = max(0, int(max_ipc_retries))
        self._retry_base_delay_s = max(0.0, float(retry_base_delay_s))

        if inbound_create_key is None:
            self._inbound_create_key = _random_nonzero_id16()
        else:
            self._inbound_create_key = _validate_nonzero_id16(
                inbound_create_key, "inbound_create_key"
            )

        if outbound_create_key is None:
            cand = _random_nonzero_id16()
            while cand == self._inbound_create_key:
                cand = _random_nonzero_id16()
            self._outbound_create_key = cand
        else:
            self._outbound_create_key = _validate_nonzero_id16(
                outbound_create_key, "outbound_create_key"
            )

        if self._outbound_create_key == self._inbound_create_key:
            raise ValueError("inbound_create_key and outbound_create_key must be distinct")

        self._consumer_fiber_id: bytes = DEFAULT_CONSUMER_FIBER_ID
        assert len(self._consumer_fiber_id) == 16

        self._client: Any | None = None
        self._started: bool = False
        self._inbound_record: Any | None = None
        self._outbound_record: Any | None = None
        self._inbound_high_water: int = 0
        self._outbound_high_water: int = 0
        self._last_inbound_compact_seq: int = 0
        self._last_outbound_compact_seq: int = 0
        self._registered_inbound_seqs: set[int] = set()
        self._registered_outbound_seqs: set[int] = set()
        self._ipc_error_cls: type[BaseException] | None = None
        self._rpc_lock = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def role(self) -> str:
        return self._role

    @property
    def process_id(self) -> bytes:
        return self._process_id

    @property
    def inbound_channel_id(self) -> bytes | None:
        return bytes(self._inbound_record.channel_id) if self._inbound_record is not None else None

    @property
    def outbound_channel_id(self) -> bytes | None:
        return (
            bytes(self._outbound_record.channel_id) if self._outbound_record is not None else None
        )

    @property
    def inbound_high_water(self) -> int:
        return self._inbound_high_water

    @property
    def outbound_high_water(self) -> int:
        return self._outbound_high_water

    def _build_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self._endpoint)
        sdk = load_nlos_sdk(self._llmos_root)
        self._ipc_error_cls = sdk.IpcError
        return sdk.ChannelClient(self._endpoint, process_id=self._process_id)

    async def _close_client_unlocked(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                maybe_coro = close_fn()
                if inspect.isawaitable(maybe_coro):
                    await maybe_coro
            except Exception:
                logger.debug("Ignored error closing stale ChannelClient", exc_info=True)

    def _is_retryable_ipc_error(self, exc: BaseException) -> bool:
        if isinstance(exc, (TimeoutError, OSError)):
            return True
        if self._ipc_error_cls is None:
            try:
                self._ipc_error_cls = load_nlos_sdk(self._llmos_root).IpcError
            except Exception:
                pass
        if self._ipc_error_cls is not None and isinstance(exc, self._ipc_error_cls):
            return True
        return type(exc).__name__ == "IpcError"

    async def _call_with_retry_locked(
        self,
        op_name: str,
        fn: Callable[[Any], Awaitable[R]],
    ) -> R:
        attempt = 0
        while True:
            try:
                if self._client is None:
                    self._client = self._build_client()
                return await fn(self._client)
            except Exception as exc:
                if not self._is_retryable_ipc_error(exc) or attempt >= self._max_ipc_retries:
                    raise
                delay = self._retry_base_delay_s * (2**attempt)
                logger.warning(
                    "LlmosChannelBus.%s transient IPC failure (retry %d/%d in %.3fs): %s",
                    op_name,
                    attempt + 1,
                    self._max_ipc_retries,
                    delay,
                    exc,
                )
                await self._close_client_unlocked()
                attempt += 1
                if delay > 0:
                    await asyncio.sleep(delay)
                self._client = self._build_client()

    async def _call_with_retry(
        self,
        fn: Callable[[Any], Awaitable[R]],
        *,
        op_name: str = "rpc",
    ) -> R:
        async with self._rpc_lock:
            return await self._call_with_retry_locked(op_name, fn)

    def _prune_registered_inbound_seqs(self) -> None:
        if not self._registered_inbound_seqs:
            return
        hw = self._inbound_high_water
        self._registered_inbound_seqs = {s for s in self._registered_inbound_seqs if s > hw}
        if len(self._registered_inbound_seqs) > MAX_REGISTERED_SEQ_CACHE:
            ordered = sorted(self._registered_inbound_seqs)
            self._registered_inbound_seqs = set(ordered[-MAX_REGISTERED_SEQ_CACHE:])

    def _prune_registered_outbound_seqs(self) -> None:
        if not self._registered_outbound_seqs:
            return
        hw = self._outbound_high_water
        self._registered_outbound_seqs = {s for s in self._registered_outbound_seqs if s > hw}
        if len(self._registered_outbound_seqs) > MAX_REGISTERED_SEQ_CACHE:
            ordered = sorted(self._registered_outbound_seqs)
            self._registered_outbound_seqs = set(ordered[-MAX_REGISTERED_SEQ_CACHE:])

    def _effective_compact_every_n(self) -> int:
        if (
            self._client is not None
            and self._compact_every_n == 32
            and not hasattr(self._client, "inspect_queue")
            and not hasattr(self._client, "enqueue_registered")
        ):
            return 1
        return self._compact_every_n

    async def start(self) -> None:
        async with self._rpc_lock:
            in_rec = await self._call_with_retry_locked(
                "create_channel[inbound]",
                lambda c: c.create_channel(
                    self._capacity_bytes, idempotency_key=self._inbound_create_key
                ),
            )
            out_rec = await self._call_with_retry_locked(
                "create_channel[outbound]",
                lambda c: c.create_channel(
                    self._capacity_bytes, idempotency_key=self._outbound_create_key
                ),
            )
            self._inbound_record = in_rec
            self._outbound_record = out_rec

            if self._client is not None and hasattr(self._client, "inspect_queue"):
                in_state = await self._call_with_retry_locked(
                    "inspect_queue[inbound]",
                    lambda c: c.inspect_queue(in_rec.channel_id),
                )
                self._inbound_high_water = _extract_acked_seq(in_state)
                self._last_inbound_compact_seq = _extract_trim_seq(
                    in_state, self._inbound_high_water
                )
                self._prune_registered_inbound_seqs()

                out_state = await self._call_with_retry_locked(
                    "inspect_queue[outbound]",
                    lambda c: c.inspect_queue(out_rec.channel_id),
                )
                self._outbound_high_water = _extract_acked_seq(out_state)
                self._last_outbound_compact_seq = _extract_trim_seq(
                    out_state, self._outbound_high_water
                )
                self._prune_registered_outbound_seqs()

            self._started = True

    async def close(self) -> None:
        async with self._rpc_lock:
            self._started = False
            await self._close_client_unlocked()

    def _require_started_records(self) -> tuple[Any, Any]:
        if not self._started or self._inbound_record is None or self._outbound_record is None:
            raise RuntimeError("LlmosChannelBus is not started; call `await bus.start()` first")
        return self._inbound_record, self._outbound_record

    def _resolve_fiber_id(self, fiber_id: bytes | None) -> bytes:
        if fiber_id is None:
            return self._consumer_fiber_id
        return _validate_nonzero_id16(fiber_id, "fiber_id")

    @staticmethod
    def _consumption_key(channel_id: bytes, sequence: int, fiber_id: bytes) -> bytes:
        return derive_deterministic_id16("qq-agent/consume/v1", channel_id, sequence, fiber_id)

    @staticmethod
    async def _enqueue_entry(
        client: Any,
        rec: Any,
        *,
        payload: bytes,
        producer_fiber: bytes,
        idempotency_key: bytes,
    ) -> Any:
        if hasattr(client, "enqueue_registered"):
            return await client.enqueue_registered(
                rec.channel_id,
                rec.generation,
                rec.fencing_token,
                payload,
                producer_fiber,
                CONSUMER_FIBER_GENERATION,
                idempotency_key=idempotency_key,
            )
        if _accepts_kwarg(client.enqueue, "idempotency_key"):
            return await client.enqueue(
                rec.channel_id,
                rec.generation,
                rec.fencing_token,
                payload,
                idempotency_key=idempotency_key,
            )
        return await client.enqueue(
            rec.channel_id,
            rec.generation,
            rec.fencing_token,
            payload,
        )

    async def publish_inbound(self, msg: InboundMessage) -> int:
        payload = msg.to_json_bytes()
        producer_fiber = fiber_id_for_session(msg.session_key)
        if msg.event_id.strip():
            idem_key = derive_deterministic_id16(
                "qq-agent/enqueue/in/v1",
                msg.session_key,
                msg.event_id,
                msg.timestamp,
            )
        else:
            idem_key = _random_nonzero_id16()

        async with self._rpc_lock:
            inbound_rec, _ = self._require_started_records()
            entry = await self._call_with_retry_locked(
                "enqueue[inbound]",
                lambda c: self._enqueue_entry(
                    c,
                    inbound_rec,
                    payload=payload,
                    producer_fiber=producer_fiber,
                    idempotency_key=idem_key,
                ),
            )
            return int(entry.sequence)

    async def poll_inbound(self, limit: int = 16) -> list[tuple[int, InboundMessage]]:
        if limit <= 0:
            return []
        async with self._rpc_lock:
            inbound_rec, _ = self._require_started_records()
            entries = await self._call_with_retry_locked(
                "receive[inbound]",
                lambda c: c.receive(inbound_rec.channel_id, limit=limit),
            )
        return [
            (int(entry.sequence), InboundMessage.from_json_bytes(bytes(entry.payload)))
            for entry in entries
        ]

    async def register_inbound_consumption(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        if sequence <= 0:
            raise ValueError("sequence must be >= 1")
        resolved_fiber = self._resolve_fiber_id(fiber_id)
        async with self._rpc_lock:
            inbound_rec, _ = self._require_started_records()
            if sequence <= self._inbound_high_water or sequence in self._registered_inbound_seqs:
                self._prune_registered_inbound_seqs()
                return
            idem_key = self._consumption_key(inbound_rec.channel_id, sequence, resolved_fiber)
            await self._call_with_retry_locked(
                "register_queue_consumption[inbound]",
                lambda c: c.register_queue_consumption(
                    inbound_rec.channel_id,
                    inbound_rec.generation,
                    sequence,
                    resolved_fiber,
                    CONSUMER_FIBER_GENERATION,
                    idempotency_key=idem_key,
                ),
            )
            self._registered_inbound_seqs.add(sequence)
            self._prune_registered_inbound_seqs()

    async def ack_inbound(self, sequence: int, *, fiber_id: bytes | None = None) -> None:
        resolved_fiber = self._resolve_fiber_id(fiber_id)
        async with self._rpc_lock:
            inbound_rec, _ = self._require_started_records()
            if sequence <= self._inbound_high_water:
                return
            for seq in range(self._inbound_high_water + 1, sequence + 1):
                if seq in self._registered_inbound_seqs:
                    continue
                idem_key = self._consumption_key(inbound_rec.channel_id, seq, resolved_fiber)
                await self._call_with_retry_locked(
                    "register_queue_consumption[inbound]",
                    lambda c, s=seq, k=idem_key: c.register_queue_consumption(
                        inbound_rec.channel_id,
                        inbound_rec.generation,
                        s,
                        resolved_fiber,
                        CONSUMER_FIBER_GENERATION,
                        idempotency_key=k,
                    ),
                )
                self._registered_inbound_seqs.add(seq)

            await self._call_with_retry_locked(
                "ack[inbound]",
                lambda c: c.ack(inbound_rec.channel_id, inbound_rec.generation, sequence),
            )
            self._inbound_high_water = sequence
            self._prune_registered_inbound_seqs()

            if (sequence - self._last_inbound_compact_seq) >= self._effective_compact_every_n():
                await self._call_with_retry_locked(
                    "compact[inbound]",
                    lambda c: c.compact(inbound_rec.channel_id, inbound_rec.generation),
                )
                self._last_inbound_compact_seq = sequence

    async def publish_outbound(self, reply: OutboundReply) -> int:
        reply = _spill_large_voice_uri(reply)
        payload = reply.to_json_bytes()
        session_key = (
            f"group:{reply.target_id}"
            if reply.chat_type == "group"
            else f"private:{reply.target_id}"
        )
        producer_fiber = fiber_id_for_session(session_key)
        if reply.correlation_id.strip():
            idem_key = derive_deterministic_id16(
                "qq-agent/enqueue/out/v1",
                session_key,
                reply.correlation_id,
                payload,
            )
        else:
            idem_key = _random_nonzero_id16()

        async with self._rpc_lock:
            _, outbound_rec = self._require_started_records()
            entry = await self._call_with_retry_locked(
                "enqueue[outbound]",
                lambda c: self._enqueue_entry(
                    c,
                    outbound_rec,
                    payload=payload,
                    producer_fiber=producer_fiber,
                    idempotency_key=idem_key,
                ),
            )
            return int(entry.sequence)

    async def poll_outbound(self, limit: int = 16) -> list[tuple[int, OutboundReply]]:
        if limit <= 0:
            return []
        async with self._rpc_lock:
            _, outbound_rec = self._require_started_records()
            entries = await self._call_with_retry_locked(
                "receive[outbound]",
                lambda c: c.receive(outbound_rec.channel_id, limit=limit),
            )
        return [
            (int(entry.sequence), _rehydrate_voice_uri(OutboundReply.from_json_bytes(bytes(entry.payload))))
            for entry in entries
        ]

    async def ack_outbound(self, sequence: int, *, fiber_id: bytes | None = None) -> None:
        resolved_fiber = self._resolve_fiber_id(fiber_id)
        async with self._rpc_lock:
            _, outbound_rec = self._require_started_records()
            if sequence <= self._outbound_high_water:
                return
            for seq in range(self._outbound_high_water + 1, sequence + 1):
                if seq in self._registered_outbound_seqs:
                    continue
                idem_key = self._consumption_key(outbound_rec.channel_id, seq, resolved_fiber)
                await self._call_with_retry_locked(
                    "register_queue_consumption[outbound]",
                    lambda c, s=seq, k=idem_key: c.register_queue_consumption(
                        outbound_rec.channel_id,
                        outbound_rec.generation,
                        s,
                        resolved_fiber,
                        CONSUMER_FIBER_GENERATION,
                        idempotency_key=k,
                    ),
                )
                self._registered_outbound_seqs.add(seq)

            await self._call_with_retry_locked(
                "ack[outbound]",
                lambda c: c.ack(outbound_rec.channel_id, outbound_rec.generation, sequence),
            )
            self._outbound_high_water = sequence
            self._prune_registered_outbound_seqs()

            if (sequence - self._last_outbound_compact_seq) >= self._effective_compact_every_n():
                await self._call_with_retry_locked(
                    "compact[outbound]",
                    lambda c: c.compact(outbound_rec.channel_id, outbound_rec.generation),
                )
                self._last_outbound_compact_seq = sequence

    async def inspect_health(self) -> BusQueueMetrics:
        async with self._rpc_lock:
            inbound_rec, outbound_rec = self._require_started_records()
            in_replayed = bool(getattr(inbound_rec, "replayed", False))
            out_replayed = bool(getattr(outbound_rec, "replayed", False))
            if self._client is not None and hasattr(self._client, "inspect_queue"):
                in_state = await self._call_with_retry_locked(
                    "inspect_queue[inbound]",
                    lambda c: c.inspect_queue(inbound_rec.channel_id),
                )
                out_state = await self._call_with_retry_locked(
                    "inspect_queue[outbound]",
                    lambda c: c.inspect_queue(outbound_rec.channel_id),
                )
                in_acked = _extract_acked_seq(in_state)
                out_acked = _extract_acked_seq(out_state)
                self._inbound_high_water = max(self._inbound_high_water, in_acked)
                self._outbound_high_water = max(self._outbound_high_water, out_acked)
                return BusQueueMetrics(
                    inbound_backlog_bytes=_extract_backlog_bytes(in_state),
                    inbound_acked_seq=in_acked,
                    outbound_backlog_bytes=_extract_backlog_bytes(out_state),
                    outbound_acked_seq=out_acked,
                    inbound_replayed=in_replayed,
                    outbound_replayed=out_replayed,
                )
            return BusQueueMetrics(
                inbound_backlog_bytes=0,
                inbound_acked_seq=self._inbound_high_water,
                outbound_backlog_bytes=0,
                outbound_acked_seq=self._outbound_high_water,
                inbound_replayed=in_replayed,
                outbound_replayed=out_replayed,
            )


ResilientLlmosBus = LlmosChannelBus

__all__ = [
    "CONSUMER_FIBER_GENERATION",
    "DEFAULT_CONSUMER_FIBER_ID",
    "LlmosChannelBus",
    "MAX_REGISTERED_SEQ_CACHE",
    "ResilientLlmosBus",
    "_load_channel_client_cls",
]
