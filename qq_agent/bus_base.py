# qq_agent/bus_base.py
"""双队列消息总线协议、内存双队列总线、连续高水位延迟确认协调器与自适应轮询器。"""

from __future__ import annotations

import asyncio
from collections import deque
import dataclasses
from itertools import islice
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .llmos_sdk import fiber_id_for_session
from .types import InboundMessage, OutboundReply

T = TypeVar("T")

DEFAULT_CONSUMER_FIBER_ID: bytes = b"qq-agent-bus-fib"  # 16 non-zero bytes


def _validate_nonzero_id16(raw: bytes, label: str) -> bytes:
    val = bytes(raw)
    if len(val) != 16 or val == b"\x00" * 16:
        raise ValueError(f"{label} must be 16 non-zero bytes")
    return val


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


__all__ = [
    "AckCoordinator",
    "AdaptivePoller",
    "BusQueueMetrics",
    "DEFAULT_CONSUMER_FIBER_ID",
    "DirectMemoryBus",
    "MessageBus",
]
