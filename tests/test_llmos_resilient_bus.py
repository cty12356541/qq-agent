# tests/test_llmos_resilient_bus.py
"""Wave W4-01 resilient llmos ChannelControl spine tests:
- Deterministic channel creation replay across adapter/brain roles & crash recovery via inspect_queue
- Bidirectional ADR-0012 ExecutionFiber registration (enqueue_registered + register_inbound_consumption)
- Out-of-order concurrent completion with contiguous high-water deferred ACK (AckCoordinator)
- Transient IpcError exponential-backoff bulkhead, ChannelControlError pass-through, batched compact_every_n
- AdaptivePoller idle backoff / active reset and real system-control-daemon cross-process restart recovery
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import pytest

from qq_agent.channel_bridge import (
    AckCoordinator,
    AdaptivePoller,
    BusQueueMetrics,
    DirectMemoryBus,
    LlmosChannelBus,
    NapCatWsAdapter,
    ResilientLlmosBus,
)
from qq_agent.llmos_sdk import (
    INBOUND_CHANNEL_CREATE_KEY,
    OUTBOUND_CHANNEL_CREATE_KEY,
    derive_deterministic_id16,
    fiber_id_for_session,
    fiber_id_for_subagent,
    load_nlos_sdk,
    process_id_for_role,
)
from qq_agent.types import InboundMessage, OutboundReply

LLMOS_ROOT = Path(r"D:\llmos")
DAEMON_EXE = LLMOS_ROOT / "target" / "debug" / "system-control-daemon.exe"

_PIPE_SEQ = 0


def _unique_endpoint(label: str) -> str:
    global _PIPE_SEQ
    _PIPE_SEQ += 1
    token = f"{os.getpid():x}-{time.time_ns():x}-{_PIPE_SEQ:x}-{label}"
    if sys.platform == "win32":
        return rf"\\.\pipe\qq-agent-w4-{token}"
    return str(Path(tempfile.gettempdir()) / f"qq-agent-w4-{token}.sock")


def _make_inbound(
    idx: int,
    *,
    chat_type: str = "group",
    event_id: str | None = None,
    group_id: int | None = 70001,
    user_id: int | None = None,
) -> InboundMessage:
    uid = (10000 + idx) if user_id is None else user_id
    gid = group_id if chat_type == "group" else None
    return InboundMessage(
        event_id=f"ev-{idx}" if event_id is None else event_id,
        chat_type="group" if chat_type == "group" else "private",
        user_id=uid,
        group_id=gid,
        sender_name=f"User{idx}",
        text=f"msg-{idx}",
        raw_message=f"msg-{idx}",
        mentioned_bot=True,
        self_id=999,
        timestamp=1710000000 + idx,
    )


def _make_outbound(
    idx: int,
    *,
    chat_type: str = "group",
    target_id: int = 70001,
    correlation_id: str | None = None,
    text: str | None = None,
) -> OutboundReply:
    return OutboundReply(
        chat_type="group" if chat_type == "group" else "private",
        target_id=target_id,
        text=f"reply-{idx}" if text is None else text,
        reply_to_message_id=100 + idx,
        at_user_id=10000 + idx,
        correlation_id=f"corr-{idx}" if correlation_id is None else correlation_id,
    )


# ---------------------------------------------------------------------------
# Shared in-memory fake ChannelAuthority backing multiple client instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeChannelRec:
    channel_id: bytes
    generation: int
    fencing_token: bytes
    capacity_bytes: int
    idempotency_key: bytes
    replayed: bool = False


@dataclass(frozen=True)
class _FakeQueueEntry:
    channel_id: bytes
    generation: int
    sequence: int
    payload: bytes
    idempotency_key: bytes
    binding: bytes = b""
    fiber_generation: int = 0
    replayed: bool = False


@dataclass(frozen=True)
class _FakeQueueState:
    channel_id: bytes
    capacity_bytes: int
    consume_high_water: int
    trim_high_water: int
    backlog_bytes: int
    retained_bytes: int
    max_sequence: int

    @property
    def acknowledged_through_sequence(self) -> int:
        return self.consume_high_water


class IpcError(Exception):
    """Fake nlos_sdk.IpcError for fault-injection tests without daemon."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ChannelControlError(Exception):
    """Fake nlos_sdk.ChannelControlError business error (must NOT be retried)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _SharedFakeDaemonState:
    """Authoritative persistent queue state shared across client reconnects and bus restarts."""

    def __init__(self) -> None:
        self.channels_by_key: dict[bytes, _FakeChannelRec] = {}
        self.channels_by_id: dict[bytes, _FakeChannelRec] = {}
        self.entries: dict[bytes, list[_FakeQueueEntry]] = {}
        self.entries_by_idem: dict[bytes, _FakeQueueEntry] = {}
        self.consumptions: dict[tuple[bytes, int, bytes], bytes] = {}
        self.consume_hw: dict[bytes, int] = {}
        self.trim_hw: dict[bytes, int] = {}
        self.max_seq: dict[bytes, int] = {}
        self.rpc_log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.fault_plan: list[BaseException] = []
        self.clients_created: int = 0
        self.clients_closed: int = 0


class _ResilientFakeChannelClient:
    def __init__(self, endpoint: str, backend: _SharedFakeDaemonState) -> None:
        self.endpoint = endpoint
        self._b = backend
        self._b.clients_created += 1
        self.closed = False

    def _inject_fault_if_any(self) -> None:
        if self._b.fault_plan:
            exc = self._b.fault_plan.pop(0)
            raise exc

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._b.clients_closed += 1

    async def create_channel(
        self,
        capacity_bytes: int,
        policy_digest: bytes = b"\x00" * 32,
        *,
        idempotency_key: bytes | None = None,
    ) -> _FakeChannelRec:
        self._inject_fault_if_any()
        assert idempotency_key is not None and len(idempotency_key) == 16
        self._b.rpc_log.append(
            ("create_channel", (capacity_bytes,), {"idempotency_key": idempotency_key})
        )
        if idempotency_key in self._b.channels_by_key:
            orig = self._b.channels_by_key[idempotency_key]
            return _FakeChannelRec(
                channel_id=orig.channel_id,
                generation=orig.generation,
                fencing_token=orig.fencing_token,
                capacity_bytes=orig.capacity_bytes,
                idempotency_key=orig.idempotency_key,
                replayed=True,
            )
        idx = len(self._b.channels_by_key) + 1
        cid = bytes([idx]) * 16
        rec = _FakeChannelRec(
            channel_id=cid,
            generation=1,
            fencing_token=bytes([0xB0 + idx]) * 32,
            capacity_bytes=capacity_bytes,
            idempotency_key=idempotency_key,
            replayed=False,
        )
        self._b.channels_by_key[idempotency_key] = rec
        self._b.channels_by_id[cid] = rec
        self._b.entries[cid] = []
        self._b.consume_hw[cid] = 0
        self._b.trim_hw[cid] = 0
        self._b.max_seq[cid] = 0
        return rec

    async def inspect_queue(self, channel_id: bytes) -> _FakeQueueState:
        self._inject_fault_if_any()
        self._b.rpc_log.append(("inspect_queue", (channel_id,), {}))
        rec = self._b.channels_by_id[channel_id]
        chw = self._b.consume_hw[channel_id]
        thw = self._b.trim_hw[channel_id]
        live = self._b.entries[channel_id]
        backlog = sum(len(e.payload) for e in live if e.sequence > chw)
        retained = sum(len(e.payload) for e in live)
        return _FakeQueueState(
            channel_id=channel_id,
            capacity_bytes=rec.capacity_bytes,
            consume_high_water=chw,
            trim_high_water=thw,
            backlog_bytes=backlog,
            retained_bytes=retained,
            max_sequence=self._b.max_seq[channel_id],
        )

    async def enqueue_registered(
        self,
        channel_id: bytes,
        expected_generation: int,
        expected_fencing_token: bytes,
        payload: bytes,
        producer_binding: bytes,
        producer_fiber_generation: int,
        *,
        idempotency_key: bytes | None = None,
    ) -> _FakeQueueEntry:
        self._inject_fault_if_any()
        assert idempotency_key is not None and len(idempotency_key) == 16
        assert len(producer_binding) == 16 and producer_binding != b"\x00" * 16
        assert producer_fiber_generation == 1
        self._b.rpc_log.append(
            (
                "enqueue_registered",
                (
                    channel_id,
                    expected_generation,
                    expected_fencing_token,
                    payload,
                    producer_binding,
                    producer_fiber_generation,
                ),
                {"idempotency_key": idempotency_key},
            )
        )
        if idempotency_key in self._b.entries_by_idem:
            prev = self._b.entries_by_idem[idempotency_key]
            if (
                prev.channel_id != channel_id
                or prev.payload != payload
                or prev.binding != producer_binding
            ):
                raise ChannelControlError("CONFLICT", "idempotency key conflict")
            return _FakeQueueEntry(
                channel_id=prev.channel_id,
                generation=prev.generation,
                sequence=prev.sequence,
                payload=prev.payload,
                idempotency_key=prev.idempotency_key,
                binding=prev.binding,
                fiber_generation=prev.fiber_generation,
                replayed=True,
            )

        seq = self._b.max_seq[channel_id] + 1
        self._b.max_seq[channel_id] = seq
        entry = _FakeQueueEntry(
            channel_id=channel_id,
            generation=expected_generation,
            sequence=seq,
            payload=payload,
            idempotency_key=idempotency_key,
            binding=producer_binding,
            fiber_generation=producer_fiber_generation,
            replayed=False,
        )
        self._b.entries[channel_id].append(entry)
        self._b.entries_by_idem[idempotency_key] = entry
        return entry

    async def receive(
        self,
        channel_id: bytes,
        limit: int = 16,
    ) -> list[_FakeQueueEntry]:
        self._inject_fault_if_any()
        self._b.rpc_log.append(("receive", (channel_id,), {"limit": limit}))
        chw = self._b.consume_hw[channel_id]
        return [e for e in self._b.entries[channel_id] if e.sequence > chw][:limit]

    async def register_queue_consumption(
        self,
        channel_id: bytes,
        expected_generation: int,
        sequence: int,
        consumer_fiber_id: bytes,
        consumer_fiber_generation: int,
        *,
        idempotency_key: bytes | None = None,
    ) -> dict[str, Any]:
        self._inject_fault_if_any()
        assert len(consumer_fiber_id) == 16 and consumer_fiber_id != b"\x00" * 16
        assert idempotency_key is not None and len(idempotency_key) == 16
        self._b.rpc_log.append(
            (
                "register_queue_consumption",
                (
                    channel_id,
                    expected_generation,
                    sequence,
                    consumer_fiber_id,
                    consumer_fiber_generation,
                ),
                {"idempotency_key": idempotency_key},
            )
        )
        live_seqs = {e.sequence for e in self._b.entries[channel_id]}
        if sequence not in live_seqs:
            raise ChannelControlError(
                "NOT_FOUND",
                f"entry sequence={sequence} missing or already compacted",
            )
        ident = (channel_id, sequence, consumer_fiber_id)
        if ident in self._b.consumptions and self._b.consumptions[ident] != idempotency_key:
            raise ChannelControlError("CONFLICT", "consumption registration key mismatch")
        self._b.consumptions[ident] = idempotency_key
        return {"sequence": sequence}

    async def ack(
        self,
        channel_id: bytes,
        expected_generation: int,
        sequence: int,
    ) -> dict[str, Any]:
        self._inject_fault_if_any()
        self._b.rpc_log.append(("ack", (channel_id, expected_generation, sequence), {}))
        if sequence < self._b.consume_hw[channel_id]:
            raise ChannelControlError("INVALID_ARGUMENT", "ack regresses below high-water")
        self._b.consume_hw[channel_id] = sequence
        return {"consume_high_water": sequence}

    async def compact(
        self,
        channel_id: bytes,
        expected_generation: int = 1,
    ) -> dict[str, Any]:
        self._inject_fault_if_any()
        self._b.rpc_log.append(("compact", (channel_id, expected_generation), {}))
        chw = self._b.consume_hw[channel_id]
        removed = [e for e in self._b.entries[channel_id] if e.sequence <= chw]
        for entry in removed:
            self._b.entries_by_idem.pop(entry.idempotency_key, None)
        self._b.entries[channel_id] = [
            e for e in self._b.entries[channel_id] if e.sequence > chw
        ]
        self._b.trim_hw[channel_id] = chw
        return {"trim_high_water": chw}


# ---------------------------------------------------------------------------
# 1) Deterministic channel replay across adapter/brain + crash high-water recovery
# ---------------------------------------------------------------------------


def test_deterministic_create_keys_replay_across_roles_and_recover_high_water():
    async def _run() -> None:
        assert ResilientLlmosBus is LlmosChannelBus
        backend = _SharedFakeDaemonState()
        factory = lambda ep: _ResilientFakeChannelClient(ep, backend)

        adapter_bus = ResilientLlmosBus(
            r"\\.\pipe\w4-shared",
            role="adapter",
            compact_every_n=2,
            client_factory=factory,
        )
        brain_bus = ResilientLlmosBus(
            r"\\.\pipe\w4-shared",
            role="brain",
            compact_every_n=2,
            client_factory=factory,
        )
        assert adapter_bus.process_id == process_id_for_role("adapter")
        assert brain_bus.process_id == process_id_for_role("brain")
        assert adapter_bus.process_id != brain_bus.process_id

        await adapter_bus.start()
        await brain_bus.start()

        # Both independent instances share the exact same inbound/outbound channel_id
        assert adapter_bus.inbound_channel_id == brain_bus.inbound_channel_id
        assert adapter_bus.outbound_channel_id == brain_bus.outbound_channel_id
        assert adapter_bus.inbound_channel_id != adapter_bus.outbound_channel_id

        m_brain = await brain_bus.inspect_health()
        assert isinstance(m_brain, BusQueueMetrics)
        assert m_brain.inbound_replayed is True
        assert m_brain.outbound_replayed is True
        assert m_brain.inbound_acked_seq == 0

        # Adapter publishes 3 inbound messages
        m1, m2, m3 = _make_inbound(1), _make_inbound(2), _make_inbound(3)
        assert await adapter_bus.publish_inbound(m1) == 1
        assert await adapter_bus.publish_inbound(m2) == 2
        assert await adapter_bus.publish_inbound(m3) == 3

        # Brain consumes and ACKs 1..2 (triggering compact at seq=2 since compact_every_n=2),
        # then crashes BEFORE ACKing seq=3!
        polled = await brain_bus.poll_inbound()
        assert [s for s, _ in polled] == [1, 2, 3]
        await brain_bus.ack_inbound(2, fiber_id=fiber_id_for_session(m2.session_key))
        await brain_bus.close()

        # Compact removed entries 1 and 2 from durable table! If restarted brain did NOT sync
        # _inbound_high_water=2 via inspect_queue, ack_inbound(3) would try registering seq=1
        # and fail closed with NOT_FOUND!
        backend.rpc_log.clear()
        restarted_brain = ResilientLlmosBus(
            r"\\.\pipe\w4-shared",
            role="brain",
            compact_every_n=2,
            client_factory=factory,
        )
        await restarted_brain.start()
        assert restarted_brain.inbound_high_water == 2

        health = await restarted_brain.inspect_health()
        assert health.inbound_acked_seq == 2
        assert health.inbound_backlog_bytes == len(m3.to_json_bytes())
        assert health.inbound_replayed is True

        recovered = await restarted_brain.poll_inbound()
        assert recovered == [(3, m3)]

        backend.rpc_log.clear()
        await restarted_brain.ack_inbound(3, fiber_id=fiber_id_for_session(m3.session_key))
        reg_seqs = [
            call[1][2]
            for call in backend.rpc_log
            if call[0] == "register_queue_consumption"
        ]
        # Only seq=3 was registered — seq=1 and seq=2 were NOT re-registered after restart!
        assert reg_seqs == [3]
        assert await restarted_brain.poll_inbound() == []

        await restarted_brain.close()
        await adapter_bus.close()

    asyncio.run(_run())


def test_direct_memory_bus_inspect_health_and_register_consumption_compat():
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        m1 = _make_inbound(1)
        r1 = _make_outbound(1)
        await bus.publish_inbound(m1)
        await bus.publish_outbound(r1)

        h1 = await bus.inspect_health()
        assert h1.inbound_backlog_bytes == len(m1.to_json_bytes())
        assert h1.outbound_backlog_bytes == len(r1.to_json_bytes())
        assert h1.inbound_acked_seq == 0
        assert h1.outbound_acked_seq == 0

        await bus.register_inbound_consumption(1, fiber_id=fiber_id_for_session(m1.session_key))
        await bus.ack_inbound(1, fiber_id=fiber_id_for_session(m1.session_key))
        await bus.ack_outbound(1)

        h2 = await bus.inspect_health()
        assert h2 == BusQueueMetrics(
            inbound_backlog_bytes=0,
            inbound_acked_seq=1,
            outbound_backlog_bytes=0,
            outbound_acked_seq=1,
            inbound_replayed=False,
            outbound_replayed=False,
        )
        await bus.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2) enqueue_registered producer Fiber binding & deterministic idempotency_key
# ---------------------------------------------------------------------------


def test_enqueue_registered_fiber_binding_and_idempotent_deduplication():
    async def _run() -> None:
        backend = _SharedFakeDaemonState()
        bus = ResilientLlmosBus(
            r"\\.\pipe\w4-idem",
            client_factory=lambda ep: _ResilientFakeChannelClient(ep, backend),
        )
        await bus.start()
        backend.rpc_log.clear()

        msg_grp = _make_inbound(10, chat_type="group", group_id=88001)
        seq1 = await bus.publish_inbound(msg_grp)
        seq1_dup = await bus.publish_inbound(msg_grp)
        assert seq1 == 1
        assert seq1_dup == 1  # deduplicated via deterministic idempotency_key!

        enq_calls = [c for c in backend.rpc_log if c[0] == "enqueue_registered"]
        assert len(enq_calls) == 2
        expected_in_fiber = fiber_id_for_session("group:88001")
        expected_in_idem = derive_deterministic_id16(
            "qq-agent/enqueue/in/v1",
            msg_grp.session_key,
            msg_grp.event_id,
            msg_grp.timestamp,
        )
        assert enq_calls[0][1][4] == expected_in_fiber
        assert enq_calls[0][1][5] == 1
        assert enq_calls[0][2]["idempotency_key"] == expected_in_idem
        assert enq_calls[1][2]["idempotency_key"] == expected_in_idem

        # Empty event_id falls back to random 16B keys so distinct ID-less messages never collide
        msg_no_id_a = _make_inbound(11, event_id="   ")
        msg_no_id_b = _make_inbound(12, event_id="")
        assert await bus.publish_inbound(msg_no_id_a) == 2
        assert await bus.publish_inbound(msg_no_id_b) == 3

        # Outbound producer fiber & correlation_id idempotency
        backend.rpc_log.clear()
        rep_priv = _make_outbound(1, chat_type="private", target_id=30009, correlation_id="ev-10")
        out_seq1 = await bus.publish_outbound(rep_priv)
        out_seq1_dup = await bus.publish_outbound(rep_priv)
        assert out_seq1 == 1
        assert out_seq1_dup == 1

        # Second distinct outbound reply sharing correlation_id (e.g. background sub-agent follow-up)
        # must NOT collide with CONFLICT because payload is distinct
        rep_priv_followup = _make_outbound(
            2,
            chat_type="private",
            target_id=30009,
            correlation_id="ev-10",
            text="查完回来了：结果如下",
        )
        assert await bus.publish_outbound(rep_priv_followup) == 2

        out_calls = [c for c in backend.rpc_log if c[0] == "enqueue_registered"]
        assert out_calls[0][1][4] == fiber_id_for_session("private:30009")

        await bus.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3) AckCoordinator out-of-order completion & contiguous high-water deferred ACK
# ---------------------------------------------------------------------------


def test_ack_coordinator_out_of_order_completion_never_prematurely_acks_inflight():
    async def _run() -> None:
        backend = _SharedFakeDaemonState()
        bus = ResilientLlmosBus(
            r"\\.\pipe\w4-ack-coord",
            compact_every_n=32,
            client_factory=lambda ep: _ResilientFakeChannelClient(ep, backend),
        )
        await bus.start()

        m1 = _make_inbound(1, chat_type="group", group_id=70001)
        m2 = _make_inbound(2, chat_type="private", user_id=20002)
        m3 = _make_inbound(3, chat_type="private", user_id=20003)
        await bus.publish_inbound(m1)
        await bus.publish_inbound(m2)
        await bus.publish_inbound(m3)

        coord = AckCoordinator(bus)
        sub_fiber_1 = fiber_id_for_subagent(m1.session_key, "task-web-01")
        coord.track(1, m1.session_key, fiber_id=sub_fiber_1)
        coord.track(2, m2.session_key)
        coord.track(3, m3.session_key)
        assert coord.inflight_count == 3

        backend.rpc_log.clear()

        # seq=2 (fast private chat) finishes FIRST while seq=1 (slow web search) is still running
        hw_after_2 = await coord.mark_done(2)
        assert hw_after_2 == 0
        assert coord.inflight_count == 2
        assert bus.inbound_high_water == 0
        # Consumption for seq=2 WAS immediately registered with m2's session fiber,
        # while ZERO ack RPCs were sent!
        assert [c[0] for c in backend.rpc_log] == ["register_queue_consumption"]
        assert backend.rpc_log[0][1][2] == 2
        assert backend.rpc_log[0][1][3] == fiber_id_for_session("private:20002")

        # seq=3 finishes SECOND while seq=1 is STILL running -> still MUST NOT ack!
        backend.rpc_log.clear()
        hw_after_3 = await coord.mark_done(3)
        assert hw_after_3 == 0
        assert coord.inflight_count == 1
        assert bus.inbound_high_water == 0
        assert [c[0] for c in backend.rpc_log] == ["register_queue_consumption"]
        assert backend.rpc_log[0][1][2] == 3
        assert backend.rpc_log[0][1][3] == fiber_id_for_session("private:20003")

        # If process crashed right here, poll_inbound still sees 1, 2, 3 unacked!
        assert len(await bus.poll_inbound()) == 3

        # Now seq=1 finishes! AckCoordinator registers seq=1 with sub_fiber_1 and
        # advances contiguous high-water straight to 3 in ONE ack(3) call (zero duplicate registrations!)
        backend.rpc_log.clear()
        hw_after_1 = await coord.mark_done(1)
        assert hw_after_1 == 3
        assert coord.inflight_count == 0
        assert bus.inbound_high_water == 3
        assert [c[0] for c in backend.rpc_log] == [
            "register_queue_consumption",
            "ack",
        ]
        assert backend.rpc_log[0][1][2] == 1
        assert backend.rpc_log[0][1][3] == sub_fiber_1
        assert backend.rpc_log[1][1][2] == 3
        # Bounded set pruned cleanly after ACK advance
        assert bus._registered_inbound_seqs == set()
        assert await bus.poll_inbound() == []

        await bus.close()

    asyncio.run(_run())


def test_ack_coordinator_arbitrary_starting_sequence_10_and_11():
    async def _run() -> None:
        acked: list[int] = []
        registered: list[tuple[int, bytes | None]] = []

        class _SpyBus(DirectMemoryBus):
            async def register_inbound_consumption(
                self,
                sequence: int,
                *,
                fiber_id: bytes | None = None,
            ) -> None:
                registered.append((sequence, fiber_id))

            async def ack_inbound(
                self,
                sequence: int,
                *,
                fiber_id: bytes | None = None,
            ) -> None:
                _ = fiber_id
                acked.append(sequence)

        spy = _SpyBus()
        coord = AckCoordinator(spy)
        coord.track(10, "group:70001")
        coord.track(11, "private:10001")
        assert coord.inflight_count == 2

        assert await coord.mark_done(11) == 0
        assert acked == []
        assert coord.inflight_count == 1

        assert await coord.mark_done(10) == 11
        assert acked == [11]
        assert coord.inflight_count == 0
        assert [s for s, _ in registered] == [11, 10]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4) IPC fault bulkhead, batched compact_every_n, AdaptivePoller & NapCatWsAdapter
# ---------------------------------------------------------------------------


def test_ipc_timeout_exponential_backoff_reconnect_and_business_error_passthrough():
    async def _run() -> None:
        backend = _SharedFakeDaemonState()
        bus = ResilientLlmosBus(
            r"\\.\pipe\w4-retry",
            max_ipc_retries=3,
            retry_base_delay_s=0.005,
            client_factory=lambda ep: _ResilientFakeChannelClient(ep, backend),
        )
        await bus.start()
        assert backend.clients_created == 1

        # Inject 2 transient failures before publish_inbound succeeds on attempt 3
        backend.fault_plan.extend(
            [
                IpcError("TIMEOUT", "IPC exchange timed out"),
                TimeoutError("pipe connect timeout"),
            ]
        )
        msg = _make_inbound(1)
        seq = await bus.publish_inbound(msg)
        assert seq == 1
        assert backend.clients_created == 3
        assert backend.clients_closed == 2

        # Business error ChannelControlError MUST NOT be swallowed or retried
        clients_before = backend.clients_created
        backend.fault_plan.append(ChannelControlError("FENCED", "stale fencing token"))
        with pytest.raises(ChannelControlError, match="FENCED"):
            await bus.publish_inbound(_make_inbound(2))
        assert backend.clients_created == clients_before

        # Exceeding max_ipc_retries propagates the final IpcError
        backend.fault_plan.extend(
            [IpcError("TIMEOUT", "IPC exchange timed out") for _ in range(4)]
        )
        with pytest.raises(IpcError, match="IPC exchange timed out"):
            await bus.poll_inbound()

        await bus.close()

    asyncio.run(_run())


def test_batched_compact_every_n_eliminates_per_message_compaction_storm():
    async def _run() -> None:
        backend = _SharedFakeDaemonState()
        bus = ResilientLlmosBus(
            r"\\.\pipe\w4-compact",
            compact_every_n=4,
            client_factory=lambda ep: _ResilientFakeChannelClient(ep, backend),
        )
        await bus.start()

        for i in range(1, 9):
            await bus.publish_inbound(_make_inbound(i))
            await bus.publish_outbound(_make_outbound(i))

        backend.rpc_log.clear()
        for i in range(1, 4):
            await bus.ack_inbound(i)
            await bus.ack_outbound(i)
        assert [c for c in backend.rpc_log if c[0] == "compact"] == []

        # 4th ACK crosses compact_every_n=4 threshold -> triggers 1 inbound + 1 outbound compact
        await bus.ack_inbound(4)
        await bus.ack_outbound(4)
        compact_calls_at_4 = [c for c in backend.rpc_log if c[0] == "compact"]
        assert len(compact_calls_at_4) == 2

        for i in range(5, 8):
            await bus.ack_inbound(i)
            await bus.ack_outbound(i)
        assert len([c for c in backend.rpc_log if c[0] == "compact"]) == 2

        await bus.ack_inbound(8)
        await bus.ack_outbound(8)
        assert len([c for c in backend.rpc_log if c[0] == "compact"]) == 4

        await bus.close()

    asyncio.run(_run())


def test_adaptive_poller_idle_backoff_and_active_reset_and_adapter_wiring():
    poller = AdaptivePoller(min_interval_s=0.015, max_interval_s=0.20, backoff_factor=1.6)
    d1 = poller.next_delay(False)
    d2 = poller.next_delay(False)
    d3 = poller.next_delay(False)
    assert d1 == pytest.approx(0.015)
    assert d2 == pytest.approx(0.024)
    assert d3 == pytest.approx(0.0384)

    # Idle saturation clamps strictly at max_interval_s=0.20
    last = d3
    for _ in range(15):
        nxt = poller.next_delay(False)
        assert nxt >= last
        assert nxt <= 0.20
        last = nxt
    assert last == pytest.approx(0.20)

    # Active poll immediately resets interval back to min_interval_s
    assert poller.next_delay(True) == pytest.approx(0.015)
    assert poller.next_delay(False) == pytest.approx(0.015)
    assert poller.next_delay(False) == pytest.approx(0.024)

    # NapCatWsAdapter fast unit-test poll_interval_s < 0.05 scaling & bot_name
    mem_bus = DirectMemoryBus()
    fast_adapter = NapCatWsAdapter("ws://127.0.0.1:3001", mem_bus, poll_interval_s=0.01)
    assert fast_adapter.bot_name == "阿橘"
    assert fast_adapter.outbound_poller.min_interval_s == pytest.approx(0.01)
    assert fast_adapter.outbound_poller.max_interval_s == pytest.approx(0.04)

    ame_adapter = NapCatWsAdapter("ws://127.0.0.1:3001", mem_bus, bot_name="AME")
    assert ame_adapter.bot_name == "AME"
    assert ame_adapter.outbound_poller.min_interval_s == pytest.approx(0.015)
    assert ame_adapter.outbound_poller.max_interval_s == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# 5) Real D:\llmos system-control-daemon cross-process crash/restart recovery
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DAEMON_EXE.exists(),
    reason=f"system-control-daemon not built at {DAEMON_EXE}",
)
def test_real_llmos_daemon_cross_process_crash_restart_unacked_recovery():
    async def _run() -> None:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        root_dir = (
            Path(tempfile.gettempdir())
            / f"qq-agent-w4-real-{os.getpid()}-{time.time_ns()}"
        )

        async def _spawn_daemon() -> tuple[asyncio.subprocess.Process, str]:
            auth_ep = _unique_endpoint("auth")
            plain_ep = _unique_endpoint("plain")
            proc = await asyncio.create_subprocess_exec(
                str(DAEMON_EXE),
                "--root",
                str(root_dir),
                "--auth-socket",
                auth_ep,
                "--plain-socket",
                plain_ep,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert proc.stdout is not None
            ready_raw = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            ready_line = ready_raw.decode("utf-8", errors="replace").strip()
            assert ready_line.startswith("READY ") and "plain_socket=" in ready_line
            return proc, plain_ep

        proc1: asyncio.subprocess.Process | None = None
        proc2: asyncio.subprocess.Process | None = None
        try:
            proc1, plain_ep_1 = await _spawn_daemon()

            # Phase 1: Adapter & Brain connect independently using default deterministic keys
            adapter_bus = ResilientLlmosBus(
                plain_ep_1,
                llmos_root=LLMOS_ROOT,
                role="adapter",
                compact_every_n=1,
            )
            brain_bus_1 = ResilientLlmosBus(
                plain_ep_1,
                llmos_root=LLMOS_ROOT,
                role="brain",
                compact_every_n=1,
            )
            await adapter_bus.start()
            await brain_bus_1.start()

            assert adapter_bus.inbound_channel_id == brain_bus_1.inbound_channel_id
            assert adapter_bus.outbound_channel_id == brain_bus_1.outbound_channel_id
            orig_in_cid = adapter_bus.inbound_channel_id

            m1 = _make_inbound(101, chat_type="group", group_id=91001)
            m2 = _make_inbound(102, chat_type="private", user_id=51002)
            m3 = _make_inbound(103, chat_type="group", group_id=91001)

            assert await adapter_bus.publish_inbound(m1) == 1
            # Duplicate publish_inbound with same deterministic idempotency_key replays seq=1
            assert await adapter_bus.publish_inbound(m1) == 1
            assert await adapter_bus.publish_inbound(m2) == 2
            assert await adapter_bus.publish_inbound(m3) == 3

            # Brain ACKs & compacts seq=1, leaving seq=2 and seq=3 unacknowledged, then crashes!
            batch_1 = await brain_bus_1.poll_inbound()
            assert [s for s, _ in batch_1] == [1, 2, 3]
            await brain_bus_1.register_inbound_consumption(
                1,
                fiber_id=fiber_id_for_session(m1.session_key),
            )
            await brain_bus_1.ack_inbound(1)
            await adapter_bus.close()
            await brain_bus_1.close()

            # Kill daemon proc1 and restart proc2 on the SAME persistent SQLite root_dir!
            proc1.kill()
            await proc1.wait()
            proc1 = None

            proc2, plain_ep_2 = await _spawn_daemon()
            brain_bus_2 = ResilientLlmosBus(
                plain_ep_2,
                llmos_root=LLMOS_ROOT,
                role="brain",
                compact_every_n=2,
            )
            await brain_bus_2.start()
            try:
                assert brain_bus_2.inbound_channel_id == orig_in_cid
                assert brain_bus_2.inbound_high_water == 1

                health = await brain_bus_2.inspect_health()
                assert health.inbound_acked_seq == 1
                assert health.inbound_backlog_bytes > 0
                assert health.inbound_replayed is True

                recovered = await brain_bus_2.poll_inbound()
                assert recovered == [(2, m2), (3, m3)]

                # Out-of-order completion via AckCoordinator against real Rust ChannelAuthority!
                coord = AckCoordinator(brain_bus_2)
                coord.track(2, m2.session_key)
                coord.track(3, m3.session_key)
                assert await coord.mark_done(3) == 1  # seq=2 still in-flight!
                assert len(await brain_bus_2.poll_inbound()) == 2
                assert await coord.mark_done(2) == 3  # contiguous advance 2..3!
                assert await brain_bus_2.poll_inbound() == []

                final_health = await brain_bus_2.inspect_health()
                assert final_health.inbound_acked_seq == 3
                assert final_health.inbound_backlog_bytes == 0
            finally:
                await brain_bus_2.close()

            # Verify D:\llmos remains 100% unpolluted (no .egg-info created)
            assert not (LLMOS_ROOT / "sdk" / "python" / "nlos_sdk.egg-info").exists()
        finally:
            for proc in (proc1, proc2):
                if proc is not None and proc.returncode is None:
                    proc.kill()
                    await proc.wait()
            shutil.rmtree(root_dir, ignore_errors=True)

    asyncio.run(_run())
