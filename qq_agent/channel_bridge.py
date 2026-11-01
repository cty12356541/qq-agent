# qq_agent/channel_bridge.py
"""双队列消息总线（DirectMemoryBus / LlmosChannelBus）与 NapCat OB11 正向 WS 适配器。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Awaitable
import dataclasses
import inspect
import json
import logging
import secrets
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

import websockets

from .llmos_sdk import (
    INBOUND_CHANNEL_CREATE_KEY,
    OUTBOUND_CHANNEL_CREATE_KEY,
    derive_deterministic_id16,
    fiber_id_for_session,
    load_nlos_sdk,
    process_id_for_role,
)
from .ob11 import (
    build_send_action,
    build_send_actions,
    event_to_inbound,
    extract_unresolved_forward_ids,
)
from .stickers import StickerRegistry
from .types import InboundMessage, OutboundReply

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_CONSUMER_FIBER_ID: bytes = b"qq-agent-bus-fib"  # 16 non-zero bytes
CONSUMER_FIBER_GENERATION: int = 1
MSG_CACHE_CAPACITY: int = 500
OUTBOUND_ECHO_CACHE_CAPACITY: int = 256
MAX_REGISTERED_SEQ_CACHE: int = 4096

MsgCacheEntry = tuple[int, str, str, tuple[str, ...]]


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


def _validate_nonzero_id16(raw: bytes, label: str) -> bytes:
    val = bytes(raw)
    if len(val) != 16 or val == b"\x00" * 16:
        raise ValueError(f"{label} must be 16 non-zero bytes")
    return val


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


@dataclasses.dataclass(frozen=True, slots=True)
class BusQueueMetrics:
    """入站与出站双队列积压字节、持久化确认高水位及幂等重放标记。"""

    inbound_backlog_bytes: int = 0
    inbound_acked_seq: int = 0
    outbound_backlog_bytes: int = 0
    outbound_acked_seq: int = 0
    inbound_replayed: bool = False
    outbound_replayed: bool = False


@runtime_checkable
class MessageBus(Protocol):
    """入站（Adapter -> Agent）与出站（Agent -> Adapter）双队列消息总线协议。"""

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def publish_inbound(self, msg: InboundMessage) -> int: ...

    async def poll_inbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, InboundMessage]]: ...

    async def ack_inbound(self, sequence: int) -> None: ...

    async def publish_outbound(self, reply: OutboundReply) -> int: ...

    async def poll_outbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, OutboundReply]]: ...

    async def ack_outbound(self, sequence: int) -> None: ...


class _MonotonicAckQueue(Generic[T]):
    """内存单调序号队列：poll 非破坏重读，ack 单调幂等高水位裁剪（对齐 llmos Channel 语义）。"""

    __slots__ = ("_entries", "_high_water", "_next_sequence")

    def __init__(self) -> None:
        self._entries: deque[tuple[int, T]] = deque()
        self._next_sequence: int = 1
        self._high_water: int = 0

    def enqueue(self, item: T) -> int:
        seq = self._next_sequence
        self._next_sequence += 1
        self._entries.append((seq, item))
        return seq

    def receive(self, limit: int = 16) -> list[tuple[int, T]]:
        if limit <= 0:
            return []
        return list(islice(self._entries, limit))

    def ack(self, sequence: int) -> None:
        if sequence <= self._high_water:
            return
        max_published = self._next_sequence - 1
        if sequence > max_published:
            raise ValueError(
                f"ack sequence {sequence} exceeds max published sequence {max_published}"
            )
        self._high_water = sequence
        while self._entries and self._entries[0][0] <= self._high_water:
            self._entries.popleft()


class DirectMemoryBus:
    """进程内内存双队列实现（用于 `--direct-ws` 单进程降级模式与快速单元测试）。"""

    def __init__(self) -> None:
        self._inbound: _MonotonicAckQueue[InboundMessage] = _MonotonicAckQueue()
        self._outbound: _MonotonicAckQueue[OutboundReply] = _MonotonicAckQueue()
        self._started: bool = False

    @property
    def inbound_high_water(self) -> int:
        return self._inbound._high_water

    @property
    def outbound_high_water(self) -> int:
        return self._outbound._high_water

    async def start(self) -> None:
        self._started = True

    async def close(self) -> None:
        self._started = False

    async def publish_inbound(self, msg: InboundMessage) -> int:
        return self._inbound.enqueue(msg)

    async def poll_inbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, InboundMessage]]:
        return self._inbound.receive(limit=limit)

    async def register_inbound_consumption(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        _ = sequence, fiber_id

    async def ack_inbound(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        _ = fiber_id
        self._inbound.ack(sequence)

    async def publish_outbound(self, reply: OutboundReply) -> int:
        return self._outbound.enqueue(reply)

    async def poll_outbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, OutboundReply]]:
        return self._outbound.receive(limit=limit)

    async def ack_outbound(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        _ = fiber_id
        self._outbound.ack(sequence)

    async def inspect_health(self) -> BusQueueMetrics:
        in_bytes = sum(len(msg.to_json_bytes()) for _, msg in self._inbound._entries)
        out_bytes = sum(len(rep.to_json_bytes()) for _, rep in self._outbound._entries)
        return BusQueueMetrics(
            inbound_backlog_bytes=in_bytes,
            inbound_acked_seq=self._inbound._high_water,
            outbound_backlog_bytes=out_bytes,
            outbound_acked_seq=self._outbound._high_water,
            inbound_replayed=False,
            outbound_replayed=False,
        )


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
                inbound_create_key,
                "inbound_create_key",
            )

        if outbound_create_key is None:
            cand = _random_nonzero_id16()
            while cand == self._inbound_create_key:
                cand = _random_nonzero_id16()
            self._outbound_create_key = cand
        else:
            self._outbound_create_key = _validate_nonzero_id16(
                outbound_create_key,
                "outbound_create_key",
            )

        if self._outbound_create_key == self._inbound_create_key:
            raise ValueError(
                "inbound_create_key and outbound_create_key must be distinct"
            )

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
        return (
            bytes(self._inbound_record.channel_id)
            if self._inbound_record is not None
            else None
        )

    @property
    def outbound_channel_id(self) -> bytes | None:
        return (
            bytes(self._outbound_record.channel_id)
            if self._outbound_record is not None
            else None
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
                if (
                    not self._is_retryable_ipc_error(exc)
                    or attempt >= self._max_ipc_retries
                ):
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
        self._registered_inbound_seqs = {
            s for s in self._registered_inbound_seqs if s > hw
        }
        if len(self._registered_inbound_seqs) > MAX_REGISTERED_SEQ_CACHE:
            ordered = sorted(self._registered_inbound_seqs)
            self._registered_inbound_seqs = set(ordered[-MAX_REGISTERED_SEQ_CACHE:])

    def _prune_registered_outbound_seqs(self) -> None:
        if not self._registered_outbound_seqs:
            return
        hw = self._outbound_high_water
        self._registered_outbound_seqs = {
            s for s in self._registered_outbound_seqs if s > hw
        }
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
                    self._capacity_bytes,
                    idempotency_key=self._inbound_create_key,
                ),
            )
            out_rec = await self._call_with_retry_locked(
                "create_channel[outbound]",
                lambda c: c.create_channel(
                    self._capacity_bytes,
                    idempotency_key=self._outbound_create_key,
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
                    in_state,
                    self._inbound_high_water,
                )
                self._prune_registered_inbound_seqs()

                out_state = await self._call_with_retry_locked(
                    "inspect_queue[outbound]",
                    lambda c: c.inspect_queue(out_rec.channel_id),
                )
                self._outbound_high_water = _extract_acked_seq(out_state)
                self._last_outbound_compact_seq = _extract_trim_seq(
                    out_state,
                    self._outbound_high_water,
                )
                self._prune_registered_outbound_seqs()

            self._started = True

    async def close(self) -> None:
        async with self._rpc_lock:
            self._started = False
            await self._close_client_unlocked()

    def _require_started_records(self) -> tuple[Any, Any]:
        if (
            not self._started
            or self._inbound_record is None
            or self._outbound_record is None
        ):
            raise RuntimeError(
                "LlmosChannelBus is not started; call `await bus.start()` first"
            )
        return self._inbound_record, self._outbound_record

    def _resolve_fiber_id(self, fiber_id: bytes | None) -> bytes:
        if fiber_id is None:
            return self._consumer_fiber_id
        return _validate_nonzero_id16(fiber_id, "fiber_id")

    @staticmethod
    def _consumption_key(
        channel_id: bytes,
        sequence: int,
        fiber_id: bytes,
    ) -> bytes:
        return derive_deterministic_id16(
            "qq-agent/consume/v1",
            channel_id,
            sequence,
            fiber_id,
        )

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

    async def poll_inbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, InboundMessage]]:
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
            if (
                sequence <= self._inbound_high_water
                or sequence in self._registered_inbound_seqs
            ):
                self._prune_registered_inbound_seqs()
                return
            idem_key = self._consumption_key(
                inbound_rec.channel_id,
                sequence,
                resolved_fiber,
            )
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

    async def ack_inbound(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        resolved_fiber = self._resolve_fiber_id(fiber_id)
        async with self._rpc_lock:
            inbound_rec, _ = self._require_started_records()
            if sequence <= self._inbound_high_water:
                return
            for seq in range(self._inbound_high_water + 1, sequence + 1):
                if seq in self._registered_inbound_seqs:
                    continue
                idem_key = self._consumption_key(
                    inbound_rec.channel_id,
                    seq,
                    resolved_fiber,
                )
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
                lambda c: c.ack(
                    inbound_rec.channel_id,
                    inbound_rec.generation,
                    sequence,
                ),
            )
            self._inbound_high_water = sequence
            self._prune_registered_inbound_seqs()

            if (
                sequence - self._last_inbound_compact_seq
            ) >= self._effective_compact_every_n():
                await self._call_with_retry_locked(
                    "compact[inbound]",
                    lambda c: c.compact(
                        inbound_rec.channel_id,
                        inbound_rec.generation,
                    ),
                )
                self._last_inbound_compact_seq = sequence

    async def publish_outbound(self, reply: OutboundReply) -> int:
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

    async def poll_outbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, OutboundReply]]:
        if limit <= 0:
            return []
        async with self._rpc_lock:
            _, outbound_rec = self._require_started_records()
            entries = await self._call_with_retry_locked(
                "receive[outbound]",
                lambda c: c.receive(outbound_rec.channel_id, limit=limit),
            )
        return [
            (int(entry.sequence), OutboundReply.from_json_bytes(bytes(entry.payload)))
            for entry in entries
        ]

    async def ack_outbound(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        resolved_fiber = self._resolve_fiber_id(fiber_id)
        async with self._rpc_lock:
            _, outbound_rec = self._require_started_records()
            if sequence <= self._outbound_high_water:
                return
            for seq in range(self._outbound_high_water + 1, sequence + 1):
                if seq in self._registered_outbound_seqs:
                    continue
                idem_key = self._consumption_key(
                    outbound_rec.channel_id,
                    seq,
                    resolved_fiber,
                )
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
                lambda c: c.ack(
                    outbound_rec.channel_id,
                    outbound_rec.generation,
                    sequence,
                ),
            )
            self._outbound_high_water = sequence
            self._prune_registered_outbound_seqs()

            if (
                sequence - self._last_outbound_compact_seq
            ) >= self._effective_compact_every_n():
                await self._call_with_retry_locked(
                    "compact[outbound]",
                    lambda c: c.compact(
                        outbound_rec.channel_id,
                        outbound_rec.generation,
                    ),
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


class AckCoordinator:
    """跨多个并发 SessionActor 与后台子 Agent 的连续高水位延迟确认协调器。"""

    def __init__(
        self,
        bus: MessageBus,
        *,
        initial_high_water: int | None = None,
    ) -> None:
        self._bus = bus
        if initial_high_water is not None:
            hw = max(0, int(initial_high_water))
            self._contiguous_high_water = hw
            self._expected_next: int | None = hw + 1
        else:
            bus_hw = int(
                getattr(
                    bus,
                    "inbound_high_water",
                    getattr(bus, "_inbound_high_water", 0),
                )
            )
            self._contiguous_high_water = max(0, bus_hw)
            self._expected_next = (
                self._contiguous_high_water + 1
                if self._contiguous_high_water > 0
                else None
            )

        self._pending: set[int] = set()
        self._completed: set[int] = set()
        self._fibers: dict[int, bytes] = {}
        self._lock = asyncio.Lock()

    @property
    def contiguous_high_water(self) -> int:
        return self._contiguous_high_water

    @property
    def inflight_count(self) -> int:
        return len(self._pending)

    def is_tracked(self, sequence: int) -> bool:
        return (
            sequence <= self._contiguous_high_water
            or sequence in self._pending
            or sequence in self._completed
        )

    @staticmethod
    def _resolve_track_fiber(
        session_key: str,
        fiber_id: bytes | None,
    ) -> bytes:
        if fiber_id is not None:
            return _validate_nonzero_id16(fiber_id, "fiber_id")
        clean_session = session_key.strip()
        if clean_session:
            return fiber_id_for_session(clean_session)
        return DEFAULT_CONSUMER_FIBER_ID

    def track(
        self,
        sequence: int,
        session_key: str = "",
        *,
        fiber_id: bytes | None = None,
    ) -> None:
        if sequence <= 0:
            raise ValueError("sequence must be >= 1")
        if sequence <= self._contiguous_high_water or sequence in self._completed:
            return
        resolved_fiber = self._resolve_track_fiber(session_key, fiber_id)
        if sequence in self._pending:
            if fiber_id is not None:
                self._fibers[sequence] = resolved_fiber
            return

        if self._expected_next is None or (
            sequence < self._expected_next and sequence > self._contiguous_high_water
        ):
            self._expected_next = sequence
        self._pending.add(sequence)
        self._fibers[sequence] = resolved_fiber

    async def mark_done(
        self,
        sequence: int,
        *,
        fiber_id: bytes | None = None,
    ) -> int:
        if sequence <= 0:
            raise ValueError("sequence must be >= 1")
        async with self._lock:
            if sequence <= self._contiguous_high_water:
                return self._contiguous_high_water

            if fiber_id is not None:
                resolved_fiber = _validate_nonzero_id16(fiber_id, "fiber_id")
                self._fibers[sequence] = resolved_fiber
            else:
                resolved_fiber = self._fibers.get(sequence, DEFAULT_CONSUMER_FIBER_ID)

            if self._expected_next is None:
                self._expected_next = sequence

            register_fn = getattr(self._bus, "register_inbound_consumption", None)
            if callable(register_fn):
                await register_fn(sequence, fiber_id=resolved_fiber)

            self._pending.discard(sequence)
            self._completed.add(sequence)

            target_hw = self._contiguous_high_water
            cursor = self._expected_next
            advanced = False
            while cursor in self._completed:
                target_hw = cursor
                cursor += 1
                advanced = True

            if advanced and target_hw > self._contiguous_high_water:
                await self._bus.ack_inbound(target_hw)
                for acked_seq in range(self._expected_next, cursor):
                    self._completed.discard(acked_seq)
                    self._fibers.pop(acked_seq, None)
                self._contiguous_high_water = target_hw
                self._expected_next = cursor

            return self._contiguous_high_water


class AdaptivePoller:
    """无 `WaitControl` SABI 推送面（L-009）期间的指数退避与活跃瞬时复位轮询器。"""

    __slots__ = (
        "_backoff_factor",
        "_current_s",
        "_idle_streak",
        "_max_interval_s",
        "_min_interval_s",
    )

    def __init__(
        self,
        min_interval_s: float = 0.015,
        max_interval_s: float = 0.20,
        backoff_factor: float = 1.6,
    ) -> None:
        low = max(0.0005, float(min_interval_s))
        high = max(low, float(max_interval_s))
        factor = max(1.0, float(backoff_factor))
        self._min_interval_s = low
        self._max_interval_s = high
        self._backoff_factor = factor
        self._current_s = low
        self._idle_streak = 0

    @property
    def min_interval_s(self) -> float:
        return self._min_interval_s

    @property
    def max_interval_s(self) -> float:
        return self._max_interval_s

    @property
    def backoff_factor(self) -> float:
        return self._backoff_factor

    @property
    def current_interval_s(self) -> float:
        return self._current_s

    def reset(self) -> None:
        self._idle_streak = 0
        self._current_s = self._min_interval_s

    def next_delay(self, had_items: bool) -> float:
        if had_items:
            self.reset()
            return self._min_interval_s
        if self._idle_streak == 0:
            self._idle_streak = 1
            self._current_s = self._min_interval_s
            return self._min_interval_s
        self._idle_streak += 1
        self._current_s = min(
            self._max_interval_s,
            self._current_s * self._backoff_factor,
        )
        return self._current_s


class NapCatWsAdapter:
    """连接 NapCat OB11 正向 WebSocket 的双向桥接器，将消息解耦推拉至 MessageBus。"""

    def __init__(
        self,
        ws_url: str,
        bus: MessageBus,
        *,
        bot_qq: int | None = None,
        bot_name: str = "阿橘",
        access_token: str | None = None,
        auto_approve_friend_ids: set[int] | None = None,
        auto_approve_group_ids: set[int] | None = None,
        stickers: StickerRegistry | None = None,
        sticker_delay_s: float = 0.25,
        poll_interval_s: float = 0.05,
        reconnect_delay_s: float = 1.0,
        rpc_timeout_s: float = 1.5,
    ) -> None:
        self._ws_url = ws_url
        self._bus = bus
        self._bot_qq = bot_qq
        self._bot_name: str = str(bot_name).strip() or "阿橘"
        self._access_token = access_token
        self._auto_approve_friend_ids: set[int] = (
            set(auto_approve_friend_ids) if auto_approve_friend_ids else set()
        )
        self._auto_approve_group_ids: set[int] = (
            set(auto_approve_group_ids) if auto_approve_group_ids else set()
        )
        self._stickers = stickers
        self._sticker_delay_s = max(0.0, float(sticker_delay_s))
        self._seen_request_flags: deque[str] = deque(maxlen=128)
        self._poll_interval_s = max(0.001, float(poll_interval_s))
        if self._poll_interval_s < 0.05:
            self._outbound_poller = AdaptivePoller(
                min_interval_s=self._poll_interval_s,
                max_interval_s=max(
                    self._poll_interval_s * 4.0,
                    self._poll_interval_s,
                ),
            )
        else:
            self._outbound_poller = AdaptivePoller(
                min_interval_s=0.015,
                max_interval_s=max(0.20, self._poll_interval_s),
            )
        self._reconnect_delay_s = max(0.0, float(reconnect_delay_s))
        self._rpc_timeout_s = max(0.05, float(rpc_timeout_s))

        self._msg_cache: OrderedDict[int, MsgCacheEntry] = OrderedDict()
        self._outbound_echo_texts: OrderedDict[str, str] = OrderedDict()
        self._pending_rpc: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._rpc_seq: int = 0
        self._outbound_seq: int = 0
        self._last_self_id: int | None = None
        self._ws_send_lock = asyncio.Lock()

    @property
    def ws_url(self) -> str:
        return self._ws_url

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def bot_name(self) -> str:
        return self._bot_name

    @property
    def outbound_poller(self) -> AdaptivePoller:
        return self._outbound_poller

    def _cache_put(self, msg_id: int, entry: MsgCacheEntry) -> None:
        if msg_id in self._msg_cache:
            self._msg_cache.move_to_end(msg_id)
        self._msg_cache[msg_id] = entry
        while len(self._msg_cache) > MSG_CACHE_CAPACITY:
            self._msg_cache.popitem(last=False)

    def _cache_get(self, msg_id: int) -> MsgCacheEntry | None:
        entry = self._msg_cache.get(msg_id)
        if entry is not None:
            self._msg_cache.move_to_end(msg_id)
        return entry

    def _record_outbound_echo(self, echo: str, text: str) -> None:
        if not echo:
            return
        if echo in self._outbound_echo_texts:
            self._outbound_echo_texts.move_to_end(echo)
        self._outbound_echo_texts[echo] = text
        while len(self._outbound_echo_texts) > OUTBOUND_ECHO_CACHE_CAPACITY:
            self._outbound_echo_texts.popitem(last=False)

    async def _ws_send_json(self, ws: Any, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False)
        async with self._ws_send_lock:
            await ws.send(encoded)

    async def _call_ob11_api(
        self,
        ws: Any,
        action: str,
        params: dict[str, Any],
        *,
        timeout_s: float = 1.5,
    ) -> dict[str, Any] | None:
        """非阻塞发起 OB11 WS 动作查询；超时、失败回执或断连安全返回 None。"""
        self._rpc_seq += 1
        echo = f"rpc-{action}-{self._rpc_seq}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_rpc[echo] = fut
        frame: dict[str, Any] = {
            "action": action,
            "params": params,
            "echo": echo,
        }
        try:
            await self._ws_send_json(ws, frame)
            resp = await asyncio.wait_for(fut, timeout=max(0.05, float(timeout_s)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("OB11 WS RPC %s failed or timed out: %s", action, exc)
            return None
        finally:
            self._pending_rpc.pop(echo, None)

        if not isinstance(resp, dict):
            return None
        status = str(resp.get("status") or "").lower()
        retcode = resp.get("retcode")
        if status == "failed" or (isinstance(retcode, int) and retcode != 0):
            return None
        return resp

    async def _fetch_forward_nodes(self, ws: Any, forward_id: str) -> list[Any] | None:
        resp = await self._call_ob11_api(
            ws,
            "get_forward_msg",
            {"message_id": forward_id, "id": forward_id},
            timeout_s=self._rpc_timeout_s,
        )
        if not isinstance(resp, dict):
            return None
        payload = resp.get("data")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("messages", "message", "content"):
                val = payload.get(key)
                if isinstance(val, list):
                    return val
        return None

    async def _resolve_reply_via_rpc(
        self,
        ws: Any,
        reply_msg_id: int,
    ) -> MsgCacheEntry | None:
        cached = self._cache_get(reply_msg_id)
        if cached is not None:
            return cached
        resp = await self._call_ob11_api(
            ws,
            "get_msg",
            {"message_id": reply_msg_id},
            timeout_s=self._rpc_timeout_s,
        )
        if not isinstance(resp, dict):
            return None
        raw_data = resp.get("data")
        if not isinstance(raw_data, dict):
            return None
        if not any(k in raw_data for k in ("message", "raw_message", "sender", "user_id")):
            return None

        nested_fids = extract_unresolved_forward_ids(raw_data)
        nested_resolved: dict[str, list[Any]] = {}
        if nested_fids:
            nested_results = await asyncio.gather(
                *(self._fetch_forward_nodes(ws, nfid) for nfid in nested_fids),
                return_exceptions=True,
            )
            for nfid, res in zip(nested_fids, nested_results):
                if isinstance(res, list) and res:
                    nested_resolved[nfid] = res

        synthetic_event = dict(raw_data)
        synthetic_event["post_type"] = "message"
        if synthetic_event.get("user_id") is None:
            raw_sender = synthetic_event.get("sender")
            if isinstance(raw_sender, dict) and raw_sender.get("user_id") is not None:
                synthetic_event["user_id"] = raw_sender["user_id"]
        if self._last_self_id is not None and synthetic_event.get("self_id") is None:
            synthetic_event["self_id"] = self._last_self_id

        parsed = event_to_inbound(
            synthetic_event,
            bot_qq=self._bot_qq,
            resolved_forwards=nested_resolved or None,
        )
        if parsed is None:
            return None

        entry: MsgCacheEntry = (
            parsed.user_id,
            parsed.sender_name,
            parsed.text,
            parsed.image_urls,
        )
        self._cache_put(reply_msg_id, entry)
        return entry

    async def _enrich_and_publish_inbound(
        self,
        *,
        ws: Any,
        raw_event: dict[str, Any],
        inbound: InboundMessage,
        unresolved_fids: list[str],
        cached_reply: MsgCacheEntry | None,
        need_reply_rpc: bool,
    ) -> None:
        fwd_coros = [self._fetch_forward_nodes(ws, fid) for fid in unresolved_fids]
        reply_coro = (
            self._resolve_reply_via_rpc(ws, inbound.reply_to_id)
            if (need_reply_rpc and inbound.reply_to_id is not None)
            else None
        )

        resolved_forwards: dict[str, list[Any]] = {}
        reply_entry: MsgCacheEntry | None = cached_reply

        if reply_coro is not None and fwd_coros:
            results = await asyncio.gather(*fwd_coros, reply_coro, return_exceptions=True)
            fwd_results = results[:-1]
            rep_res = results[-1]
            for fid, res in zip(unresolved_fids, fwd_results):
                if isinstance(res, list) and res:
                    resolved_forwards[fid] = res
            if isinstance(rep_res, tuple):
                reply_entry = rep_res
        elif fwd_coros:
            fwd_results = await asyncio.gather(*fwd_coros, return_exceptions=True)
            for fid, res in zip(unresolved_fids, fwd_results):
                if isinstance(res, list) and res:
                    resolved_forwards[fid] = res
        elif reply_coro is not None:
            rep_res = await reply_coro
            if rep_res is not None:
                reply_entry = rep_res

        if resolved_forwards:
            reparsed = event_to_inbound(
                raw_event,
                bot_qq=self._bot_qq,
                resolved_forwards=resolved_forwards,
            )
            if reparsed is not None:
                inbound = dataclasses.replace(
                    inbound,
                    text=reparsed.text,
                    image_urls=reparsed.image_urls,
                )

        msg_id = _try_parse_int(inbound.event_id)
        if msg_id is not None:
            self._cache_put(
                msg_id,
                (inbound.user_id, inbound.sender_name, inbound.text, inbound.image_urls),
            )

        if reply_entry is not None:
            r_uid, r_name, r_text, r_imgs = reply_entry
            merged_imgs = inbound.image_urls if inbound.image_urls else r_imgs
            inbound = dataclasses.replace(
                inbound,
                reply_to_user_id=r_uid,
                reply_to_sender_name=r_name,
                reply_to_text=r_text,
                image_urls=merged_imgs,
            )

        await self._bus.publish_inbound(inbound)

    def _handle_action_response(self, data: dict[str, Any]) -> None:
        echo_raw = data.get("echo")
        if echo_raw is None:
            return
        echo = str(echo_raw)
        if not echo:
            return

        fut = self._pending_rpc.get(echo)
        if fut is not None and not fut.done():
            fut.set_result(data)

        if echo in self._outbound_echo_texts:
            sent_text = self._outbound_echo_texts.pop(echo)
            resp_data = data.get("data")
            if isinstance(resp_data, dict):
                msg_id = _try_parse_int(resp_data.get("message_id"))
                if msg_id is not None:
                    bot_uid = (
                        self._bot_qq
                        if self._bot_qq is not None
                        else (self._last_self_id or 0)
                    )
                    self._cache_put(msg_id, (bot_uid, self._bot_name, sent_text, ()))

    async def _maybe_auto_approve_request(self, ws: Any, data: dict[str, Any]) -> None:
        flag = data.get("flag")
        if not isinstance(flag, str) or not flag or flag in self._seen_request_flags:
            return
        req_type = data.get("request_type")
        try:
            uid = int(data.get("user_id") or 0)
        except (TypeError, ValueError):
            uid = 0

        if req_type == "friend" and uid in self._auto_approve_friend_ids:
            self._seen_request_flags.append(flag)
            action = {
                "action": "set_friend_add_request",
                "params": {"flag": flag, "approve": True},
                "echo": f"auto-friend-{flag}",
            }
            await self._ws_send_json(ws, action)
            logger.info("Auto-approved whitelisted admin friend request")
            return

        if req_type == "group" and data.get("sub_type") == "invite":
            try:
                gid = int(data.get("group_id") or 0)
            except (TypeError, ValueError):
                gid = 0
            if gid in self._auto_approve_group_ids or uid in self._auto_approve_friend_ids:
                self._seen_request_flags.append(flag)
                action = {
                    "action": "set_group_add_request",
                    "params": {"flag": flag, "sub_type": "invite", "approve": True},
                    "echo": f"auto-group-{flag}",
                }
                await self._ws_send_json(ws, action)
                logger.info("Auto-approved whitelisted group invite")

    async def _recv_loop(
        self,
        ws: Any,
        session_stop: asyncio.Event,
        session_tasks: set[asyncio.Task[Any]],
    ) -> None:
        async for raw in ws:
            if session_stop.is_set():
                break
            text = (
                raw.decode("utf-8", errors="replace")
                if isinstance(raw, (bytes, bytearray))
                else str(raw)
            )
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue

            self_id_val = _try_parse_int(data.get("self_id"))
            if self_id_val is not None:
                self._last_self_id = self_id_val

            post_type = data.get("post_type")
            if post_type is None:
                self._handle_action_response(data)
                continue
            if post_type == "request":
                await self._maybe_auto_approve_request(ws, data)
                continue
            if post_type != "message":
                continue

            inbound = event_to_inbound(data, bot_qq=self._bot_qq)
            if inbound is None:
                continue

            unresolved_fids = extract_unresolved_forward_ids(data)
            cached_reply: MsgCacheEntry | None = None
            need_reply_rpc = False
            if inbound.reply_to_id is not None:
                cached_reply = self._cache_get(inbound.reply_to_id)
                if cached_reply is None:
                    need_reply_rpc = True

            msg_id = _try_parse_int(inbound.event_id)
            if msg_id is not None:
                self._cache_put(
                    msg_id,
                    (inbound.user_id, inbound.sender_name, inbound.text, inbound.image_urls),
                )

            if not unresolved_fids and not need_reply_rpc:
                if cached_reply is not None:
                    r_uid, r_name, r_text, r_imgs = cached_reply
                    merged_imgs = inbound.image_urls if inbound.image_urls else r_imgs
                    inbound = dataclasses.replace(
                        inbound,
                        reply_to_user_id=r_uid,
                        reply_to_sender_name=r_name,
                        reply_to_text=r_text,
                        image_urls=merged_imgs,
                    )
                await self._bus.publish_inbound(inbound)
                continue

            # Slow path: never await RPC directly inside _recv_loop (single WS reader)
            enrich_task = asyncio.create_task(
                self._enrich_and_publish_inbound(
                    ws=ws,
                    raw_event=data,
                    inbound=inbound,
                    unresolved_fids=unresolved_fids,
                    cached_reply=cached_reply,
                    need_reply_rpc=need_reply_rpc,
                )
            )
            session_tasks.add(enrich_task)
            enrich_task.add_done_callback(session_tasks.discard)

    async def _send_loop(self, ws: Any, session_stop: asyncio.Event) -> None:
        self._outbound_poller.reset()
        while not session_stop.is_set():
            batch = await self._bus.poll_outbound()
            if batch:
                self._outbound_poller.next_delay(True)
                for seq, reply in batch:
                    if session_stop.is_set():
                        return
                    if reply.correlation_id:
                        echo = reply.correlation_id
                    else:
                        self._outbound_seq += 1
                        echo = f"out-{self._outbound_seq}"

                    sticker_uri = (
                        self._stickers.read_base64_uri(reply.sticker_id)
                        if (reply.sticker_id and self._stickers is not None)
                        else None
                    )
                    actions = build_send_actions(
                        reply,
                        sticker_file=sticker_uri,
                        echo=echo,
                    )
                    for idx, action in enumerate(actions):
                        if idx > 0 and self._sticker_delay_s > 0:
                            await asyncio.sleep(self._sticker_delay_s)
                        act_echo = action.get("echo")
                        if isinstance(act_echo, str) and act_echo:
                            cached_text = (
                                reply.text
                                if (idx == 0 and reply.text.strip())
                                else "[动画表情]"
                            )
                            self._record_outbound_echo(act_echo, cached_text)
                        await self._ws_send_json(ws, action)
                    await self._bus.ack_outbound(seq)
            else:
                delay_s = self._outbound_poller.next_delay(False)
                try:
                    await asyncio.wait_for(
                        session_stop.wait(),
                        timeout=delay_s,
                    )
                    return
                except TimeoutError:
                    pass

    async def _execute_session(
        self,
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        """运行单次 WS 会话，返回是否成功完成过连接握手。"""
        connect_kwargs: dict[str, Any] = {"max_size": 16 * 1024 * 1024}
        if self._access_token:
            connect_kwargs["additional_headers"] = {
                "Authorization": f"Bearer {self._access_token}"
            }

        connected = False
        async with websockets.connect(self._ws_url, **connect_kwargs) as ws:
            connected = True
            session_stop = asyncio.Event()
            session_tasks: set[asyncio.Task[Any]] = set()
            recv_task = asyncio.create_task(
                self._recv_loop(ws, session_stop, session_tasks)
            )
            send_task = asyncio.create_task(self._send_loop(ws, session_stop))
            wait_tasks: set[asyncio.Task[Any]] = {recv_task, send_task}
            stop_wait_task: asyncio.Task[Any] | None = None

            if stop_event is not None:
                if stop_event.is_set():
                    session_stop.set()
                stop_wait_task = asyncio.create_task(stop_event.wait())
                wait_tasks.add(stop_wait_task)

            done: set[asyncio.Task[Any]] = set()
            pending: set[asyncio.Task[Any]] = set()
            try:
                done, pending = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                session_stop.set()
                # Unblock any in-flight RPC futures without raising unretrieved-exception warnings
                for fut in list(self._pending_rpc.values()):
                    if not fut.done():
                        fut.set_result({"status": "failed", "retcode": -1, "data": None})
                self._pending_rpc.clear()

                # If server closed cleanly without an external stop request, allow completed
                # enrichment tasks one event-loop turn to flush publish_inbound before cancelling.
                if session_tasks and not (stop_event is not None and stop_event.is_set()):
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*list(session_tasks), return_exceptions=True),
                            timeout=0.2,
                        )
                    except TimeoutError:
                        pass

                to_cancel = [
                    t for t in (pending | set(session_tasks)) if not t.done()
                ]
                for task in to_cancel:
                    task.cancel()
                if to_cancel:
                    await asyncio.gather(*to_cancel, return_exceptions=True)

            for task in done:
                if task is stop_wait_task or task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    if isinstance(exc, websockets.ConnectionClosedOK):
                        continue
                    raise exc
        return connected

    async def run_once_session(
        self,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """建立并运行单次 WS 会话，便于测试或受控生命周期停止。"""
        await self._execute_session(stop_event=stop_event)

    async def run_forever(
        self,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """持久运行并在断线/重连失败时按指数退避自动重连，直至 stop_event 置位。"""
        effective_stop = stop_event if stop_event is not None else asyncio.Event()
        base_delay = self._reconnect_delay_s
        max_delay = max(base_delay, min(base_delay * 32.0, 30.0))
        current_delay = base_delay

        while not effective_stop.is_set():
            connected = False
            try:
                connected = await self._execute_session(stop_event=effective_stop)
            except (OSError, TimeoutError, websockets.WebSocketException):
                pass

            if effective_stop.is_set():
                break

            wait_s = base_delay if connected else current_delay
            if connected:
                current_delay = base_delay
            else:
                current_delay = (
                    min(current_delay * 2.0, max_delay) if current_delay > 0 else 0.0
                )

            if wait_s > 0:
                try:
                    await asyncio.wait_for(effective_stop.wait(), timeout=wait_s)
                    break
                except TimeoutError:
                    pass


__all__ = [
    "AckCoordinator",
    "AdaptivePoller",
    "BusQueueMetrics",
    "CONSUMER_FIBER_GENERATION",
    "DEFAULT_CONSUMER_FIBER_ID",
    "DirectMemoryBus",
    "LlmosChannelBus",
    "MAX_REGISTERED_SEQ_CACHE",
    "MSG_CACHE_CAPACITY",
    "MessageBus",
    "NapCatWsAdapter",
    "OUTBOUND_ECHO_CACHE_CAPACITY",
    "ResilientLlmosBus",
]
