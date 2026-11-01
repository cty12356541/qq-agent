# tests/test_channel_bridge.py
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import websockets

from qq_agent.channel_bridge import (
    DirectMemoryBus,
    LlmosChannelBus,
    MessageBus,
    NapCatWsAdapter,
    _load_channel_client_cls,
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
        return rf"\\.\pipe\qq-agent-test-{token}"
    return str(Path(tempfile.gettempdir()) / f"qq-agent-test-{token}.sock")


def _sample_inbound(idx: int) -> InboundMessage:
    return InboundMessage(
        event_id=f"ev-{idx}",
        chat_type="group",
        user_id=10000 + idx,
        group_id=70000,
        sender_name=f"User{idx}",
        text=f"hello-{idx}",
        raw_message=f"hello-{idx}",
        mentioned_bot=(idx % 2 == 1),
        reply_to_id=None,
        self_id=999,
        timestamp=1700000000 + idx,
    )


def _sample_outbound(idx: int) -> OutboundReply:
    return OutboundReply(
        chat_type="group",
        target_id=70000,
        text=f"reply-{idx}",
        reply_to_message_id=100 + idx,
        at_user_id=10000 + idx,
        correlation_id=f"corr-{idx}",
    )


def test_direct_memory_bus_nondestructive_poll_and_monotonic_idempotent_ack():
    async def _run() -> None:
        bus = DirectMemoryBus()
        assert isinstance(bus, MessageBus)
        await bus.start()

        m1, m2, m3, m4 = (_sample_inbound(i) for i in (1, 2, 3, 4))
        assert await bus.publish_inbound(m1) == 1
        assert await bus.publish_inbound(m2) == 2
        assert await bus.publish_inbound(m3) == 3

        # Non-destructive poll + limit window
        first_two = await bus.poll_inbound(limit=2)
        assert first_two == [(1, m1), (2, m2)]
        reread_all = await bus.poll_inbound(limit=16)
        assert reread_all == [(1, m1), (2, m2), (3, m3)]

        # High-water ack trims <= 2; duplicate and regressing acks are idempotent no-ops
        await bus.ack_inbound(2)
        assert await bus.poll_inbound() == [(3, m3)]
        await bus.ack_inbound(2)
        await bus.ack_inbound(1)
        assert await bus.poll_inbound() == [(3, m3)]

        # Sequence keeps increasing monotonically after trim
        assert await bus.publish_inbound(m4) == 4
        assert await bus.poll_inbound() == [(3, m3), (4, m4)]
        await bus.ack_inbound(4)
        assert await bus.poll_inbound() == []

        # Outbound queue is independent and obeys identical semantics
        r1, r2 = _sample_outbound(1), _sample_outbound(2)
        assert await bus.publish_outbound(r1) == 1
        assert await bus.publish_outbound(r2) == 2
        assert await bus.poll_outbound() == [(1, r1), (2, r2)]
        assert await bus.poll_outbound() == [(1, r1), (2, r2)]
        await bus.ack_outbound(1)
        await bus.ack_outbound(1)
        assert await bus.poll_outbound() == [(2, r2)]
        await bus.ack_outbound(2)
        assert await bus.poll_outbound() == []

        await bus.close()

    asyncio.run(_run())


@dataclass(frozen=True)
class _FakeChannelRecord:
    channel_id: bytes
    generation: int
    fencing_token: bytes
    capacity_bytes: int
    idempotency_key: bytes


@dataclass(frozen=True)
class _FakeQueueEntry:
    channel_id: bytes
    generation: int
    sequence: int
    payload: bytes


class _StubChannelClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._channels_by_key: dict[bytes, _FakeChannelRecord] = {}
        self._queues: dict[bytes, list[_FakeQueueEntry]] = {}
        self._next_seq: dict[bytes, int] = {}
        self._high_water: dict[bytes, int] = {}

    async def create_channel(
        self,
        capacity_bytes: int,
        policy_digest: bytes = b"\x00" * 32,
        *,
        idempotency_key: bytes | None = None,
        created_at_ms: int = 0,
    ) -> _FakeChannelRecord:
        assert idempotency_key is not None and len(idempotency_key) == 16
        self.calls.append(
            (
                "create_channel",
                (capacity_bytes,),
                {"idempotency_key": idempotency_key, "policy_digest": policy_digest},
            )
        )
        if idempotency_key not in self._channels_by_key:
            idx = len(self._channels_by_key) + 1
            cid = bytes([idx]) * 16
            rec = _FakeChannelRecord(
                channel_id=cid,
                generation=1,
                fencing_token=bytes([0xA0 + idx]) * 32,
                capacity_bytes=capacity_bytes,
                idempotency_key=idempotency_key,
            )
            self._channels_by_key[idempotency_key] = rec
            self._queues[cid] = []
            self._next_seq[cid] = 1
            self._high_water[cid] = 0
        return self._channels_by_key[idempotency_key]

    async def enqueue(
        self,
        channel_id: bytes,
        expected_generation: int,
        expected_fencing_token: bytes,
        payload: bytes,
        *,
        idempotency_key: bytes | None = None,
        enqueued_at_ms: int = 0,
    ) -> _FakeQueueEntry:
        self.calls.append(
            (
                "enqueue",
                (channel_id, expected_generation, expected_fencing_token, payload),
                {},
            )
        )
        seq = self._next_seq[channel_id]
        self._next_seq[channel_id] = seq + 1
        entry = _FakeQueueEntry(
            channel_id=channel_id,
            generation=expected_generation,
            sequence=seq,
            payload=payload,
        )
        self._queues[channel_id].append(entry)
        return entry

    async def receive(
        self,
        channel_id: bytes,
        limit: int = 16,
    ) -> list[_FakeQueueEntry]:
        self.calls.append(("receive", (channel_id,), {"limit": limit}))
        hw = self._high_water[channel_id]
        unacked = [e for e in self._queues[channel_id] if e.sequence > hw]
        return unacked[:limit]

    async def register_queue_consumption(
        self,
        channel_id: bytes,
        expected_generation: int,
        sequence: int,
        consumer_fiber_id: bytes,
        consumer_fiber_generation: int,
        *,
        idempotency_key: bytes | None = None,
        registered_at_ms: int = 0,
    ) -> dict[str, Any]:
        assert len(consumer_fiber_id) == 16
        assert any(b != 0 for b in consumer_fiber_id)
        assert consumer_fiber_generation == 1
        self.calls.append(
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
        return {"sequence": sequence}

    async def ack(
        self,
        channel_id: bytes,
        expected_generation: int,
        sequence: int,
        *,
        acked_at_ms: int = 0,
        idempotency_key: bytes | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("ack", (channel_id, expected_generation, sequence), {}))
        self._high_water[channel_id] = max(self._high_water[channel_id], sequence)
        return {"consume_high_water": self._high_water[channel_id]}

    async def compact(
        self,
        channel_id: bytes,
        expected_generation: int = 1,
        *,
        trim_to_sequence: int | None = None,
        idempotency_key: bytes | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("compact", (channel_id, expected_generation), {}))
        hw = self._high_water[channel_id]
        self._queues[channel_id] = [
            e for e in self._queues[channel_id] if e.sequence > hw
        ]
        return {"trim_high_water": hw}


def test_llmos_channel_bus_stub_client_protocol_assertions():
    async def _run() -> None:
        created_clients: list[_StubChannelClient] = []

        def factory(ep: str) -> _StubChannelClient:
            c = _StubChannelClient(ep)
            created_clients.append(c)
            return c

        bus = LlmosChannelBus(
            r"\\.\pipe\fake-llmos",
            capacity_bytes=32768,
            client_factory=factory,
        )
        assert isinstance(bus, MessageBus)
        await bus.start()
        assert len(created_clients) == 1
        stub = created_clients[0]
        assert stub.endpoint == r"\\.\pipe\fake-llmos"

        create_calls = [c for c in stub.calls if c[0] == "create_channel"]
        assert len(create_calls) == 2
        in_key = create_calls[0][2]["idempotency_key"]
        out_key = create_calls[1][2]["idempotency_key"]
        assert in_key != out_key
        assert create_calls[0][1] == (32768,)
        assert create_calls[1][1] == (32768,)

        # Restarting same bus instance replays exact same instance-level idempotency_keys
        await bus.start()
        create_calls_after = [c for c in stub.calls if c[0] == "create_channel"]
        assert len(create_calls_after) == 4
        assert create_calls_after[2][2]["idempotency_key"] == in_key
        assert create_calls_after[3][2]["idempotency_key"] == out_key

        in_cid = bytes([1]) * 16
        out_cid = bytes([2]) * 16

        msg = _sample_inbound(7)
        seq_in = await bus.publish_inbound(msg)
        assert seq_in == 1
        polled_in = await bus.poll_inbound(limit=8)
        assert polled_in == [(1, msg)]

        stub.calls.clear()
        await bus.ack_inbound(1)
        assert [c[0] for c in stub.calls] == [
            "register_queue_consumption",
            "ack",
            "compact",
        ]
        assert stub.calls[0][1][:3] == (in_cid, 1, 1)
        assert len(stub.calls[0][1][3]) == 16
        assert stub.calls[0][1][4] == 1
        assert stub.calls[1][1] == (in_cid, 1, 1)
        assert stub.calls[2][1] == (in_cid, 1)
        assert await bus.poll_inbound() == []

        # Duplicate ack is idempotent
        stub.calls.clear()
        await bus.ack_inbound(1)
        assert stub.calls == []

        # Outbound path uses distinct outbound channel
        reply = _sample_outbound(9)
        seq_out = await bus.publish_outbound(reply)
        assert seq_out == 1
        assert await bus.poll_outbound(limit=4) == [(1, reply)]

        stub.calls.clear()
        await bus.ack_outbound(1)
        assert [c[0] for c in stub.calls] == [
            "register_queue_consumption",
            "ack",
            "compact",
        ]
        assert stub.calls[0][1][:3] == (out_cid, 1, 1)
        assert await bus.poll_outbound() == []

        await bus.close()

    asyncio.run(_run())


@pytest.mark.skipif(
    not (LLMOS_ROOT / "sdk" / "python").exists(),
    reason=f"llmos Python SDK checkout not found at {LLMOS_ROOT}",
)
def test_llmos_channel_bus_dynamic_sdk_import():
    channel_client_cls = _load_channel_client_cls(LLMOS_ROOT)
    assert channel_client_cls.__name__ == "ChannelClient"
    client = channel_client_cls(r"\\.\pipe\qq-agent-verify-only")
    assert client is not None


@pytest.mark.skipif(
    not DAEMON_EXE.exists(),
    reason=f"system-control-daemon not built at {DAEMON_EXE}",
)
def test_llmos_channel_bus_real_daemon_named_pipe_roundtrip():
    async def _run() -> None:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        root_dir = (
            Path(tempfile.gettempdir())
            / f"qq-agent-llmos-test-{os.getpid()}-{time.time_ns()}"
        )
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
        try:
            assert proc.stdout is not None
            ready_raw = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            ready_line = ready_raw.decode("utf-8", errors="replace").strip()
            assert ready_line.startswith("READY ") and "plain_socket=" in ready_line, (
                f"Unexpected daemon banner: {ready_line!r}"
            )

            bus = LlmosChannelBus(plain_ep, llmos_root=LLMOS_ROOT, capacity_bytes=65536)
            await bus.start()
            try:
                msg = _sample_inbound(42)
                seq_in = await bus.publish_inbound(msg)
                assert seq_in == 1
                # Non-destructive receive before ack
                assert await bus.poll_inbound() == [(1, msg)]
                assert await bus.poll_inbound() == [(1, msg)]
                await bus.ack_inbound(seq_in)
                assert await bus.poll_inbound() == []

                reply = _sample_outbound(99)
                seq_out = await bus.publish_outbound(reply)
                assert seq_out == 1
                assert await bus.poll_outbound() == [(1, reply)]
                await bus.ack_outbound(seq_out)
                assert await bus.poll_outbound() == []
            finally:
                await bus.close()
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            shutil.rmtree(root_dir, ignore_errors=True)

    asyncio.run(_run())


def test_napcat_ws_adapter_group_roundtrip_and_auth_with_fake_server():
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()

        seen_auth_headers: list[str | None] = []
        received_actions: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            seen_auth_headers.append(ws.request.headers.get("Authorization"))
            # 1. Heartbeat meta_event -> ignored by adapter
            await ws.send(
                json.dumps(
                    {
                        "post_type": "meta_event",
                        "meta_event_type": "heartbeat",
                        "time": 1700002000,
                    }
                )
            )
            # 2. Action echo receipt -> ignored by adapter
            await ws.send(
                json.dumps(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"message_id": 111},
                        "echo": "prev-echo",
                    }
                )
            )
            # 3. Group message mentioning bot -> pushed to bus inbound
            await ws.send(
                json.dumps(
                    {
                        "post_type": "message",
                        "message_type": "group",
                        "message_id": 321,
                        "time": 1700002005,
                        "self_id": 999,
                        "user_id": 10000,
                        "group_id": 70000,
                        "message": [
                            {"type": "reply", "data": {"id": "300"}},
                            {"type": "at", "data": {"qq": "999"}},
                            {"type": "text", "data": {"text": " 帮我总结一下 "}},
                        ],
                        "raw_message": "[CQ:reply,id=300][CQ:at,qq=999] 帮我总结一下 ",
                        "sender": {"card": "测试群友", "nickname": "Test"},
                    },
                    ensure_ascii=False,
                )
            )

            async for raw in ws:
                action_data = json.loads(raw)
                await received_actions.put(action_data)
                await ws.send(
                    json.dumps(
                        {
                            "status": "ok",
                            "retcode": 0,
                            "data": {"message_id": 654},
                            "echo": action_data.get("echo", ""),
                        }
                    )
                )

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                bot_qq=999,
                access_token="secret-napcat-token",
                poll_interval_s=0.01,
                reconnect_delay_s=0.05,
            )
            stop_event = asyncio.Event()
            adapter_task = asyncio.create_task(
                adapter.run_once_session(stop_event=stop_event)
            )
            try:
                # Poll inbound until the single group message arrives
                deadline = time.monotonic() + 5.0
                inbound_batch: list[tuple[int, InboundMessage]] = []
                while time.monotonic() < deadline:
                    inbound_batch = await bus.poll_inbound()
                    if inbound_batch:
                        break
                    await asyncio.sleep(0.01)

                assert len(inbound_batch) == 1
                in_seq, inbound_msg = inbound_batch[0]
                assert inbound_msg.event_id == "321"
                assert inbound_msg.chat_type == "group"
                assert inbound_msg.group_id == 70000
                assert inbound_msg.user_id == 10000
                assert inbound_msg.sender_name == "测试群友"
                assert inbound_msg.mentioned_bot is True
                assert inbound_msg.reply_to_id == 300
                assert inbound_msg.text == "帮我总结一下"
                await bus.ack_inbound(in_seq)

                # Publish an OutboundReply onto bus and verify adapter sends OB11 action back
                await bus.publish_outbound(
                    OutboundReply(
                        chat_type="group",
                        target_id=70000,
                        text="总结完成",
                        reply_to_message_id=321,
                        at_user_id=10000,
                        correlation_id="corr-321",
                    )
                )
                sent_action = await asyncio.wait_for(received_actions.get(), timeout=5.0)
                assert sent_action == {
                    "action": "send_group_msg",
                    "params": {
                        "group_id": 70000,
                        "message": [
                            {"type": "reply", "data": {"id": "321"}},
                            {"type": "at", "data": {"qq": "10000"}},
                            {"type": "text", "data": {"text": " 总结完成"}},
                        ],
                    },
                    "echo": "corr-321",
                }
                assert seen_auth_headers == ["Bearer secret-napcat-token"]

                # Wait briefly for ack_outbound to complete
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and await bus.poll_outbound():
                    await asyncio.sleep(0.01)
                assert await bus.poll_outbound() == []
            finally:
                stop_event.set()
                await asyncio.wait_for(adapter_task, timeout=5.0)
                await bus.close()

    asyncio.run(_run())


def test_napcat_ws_adapter_run_forever_reconnects_after_drop():
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        conn_count = 0

        async def flaky_napcat(ws: websockets.ServerConnection) -> None:
            nonlocal conn_count
            conn_count += 1
            if conn_count == 1:
                await ws.close(code=1011, reason="simulated drop")
                return
            await ws.send(
                json.dumps(
                    {
                        "post_type": "message",
                        "message_type": "private",
                        "message_id": 777,
                        "user_id": 20000,
                        "raw_message": "重连成功",
                    },
                    ensure_ascii=False,
                )
            )
            await ws.wait_closed()

        async with websockets.serve(flaky_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                poll_interval_s=0.01,
                reconnect_delay_s=0.02,
            )
            stop_event = asyncio.Event()
            runner = asyncio.create_task(adapter.run_forever(stop_event=stop_event))
            try:
                deadline = time.monotonic() + 5.0
                got: list[tuple[int, InboundMessage]] = []
                while time.monotonic() < deadline:
                    got = await bus.poll_inbound()
                    if got:
                        break
                    await asyncio.sleep(0.01)
                assert conn_count >= 2
                assert len(got) == 1
                assert got[0][1].text == "重连成功"
                assert got[0][1].chat_type == "private"
            finally:
                stop_event.set()
                await asyncio.wait_for(runner, timeout=5.0)
                await bus.close()

    asyncio.run(_run())


def test_napcat_ws_adapter_auto_approves_only_whitelisted_friend_and_group_requests():
    async def _run() -> None:
        bus = DirectMemoryBus()
        await bus.start()
        received_actions: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def fake_napcat(ws: websockets.ServerConnection) -> None:
            # 1) Stranger friend request -> MUST NOT auto-approve
            await ws.send(
                json.dumps(
                    {
                        "post_type": "request",
                        "request_type": "friend",
                        "user_id": 99999,
                        "flag": "stranger-flag",
                    }
                )
            )
            # 2) Whitelisted admin friend request -> MUST auto-approve
            await ws.send(
                json.dumps(
                    {
                        "post_type": "request",
                        "request_type": "friend",
                        "user_id": 10001,
                        "flag": "admin-friend-flag",
                    }
                )
            )
            # 3) Whitelisted group invite -> MUST auto-approve
            await ws.send(
                json.dumps(
                    {
                        "post_type": "request",
                        "request_type": "group",
                        "sub_type": "invite",
                        "group_id": 70001,
                        "user_id": 88888,
                        "flag": "group-invite-flag",
                    }
                )
            )
            async for raw in ws:
                await received_actions.put(json.loads(raw))

        async with websockets.serve(fake_napcat, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            adapter = NapCatWsAdapter(
                f"ws://127.0.0.1:{port}",
                bus,
                auto_approve_friend_ids={10001},
                auto_approve_group_ids={70001},
                poll_interval_s=0.01,
            )
            stop_event = asyncio.Event()
            task = asyncio.create_task(adapter.run_once_session(stop_event=stop_event))
            try:
                act1 = await asyncio.wait_for(received_actions.get(), timeout=3.0)
                act2 = await asyncio.wait_for(received_actions.get(), timeout=3.0)
                assert act1 == {
                    "action": "set_friend_add_request",
                    "params": {"flag": "admin-friend-flag", "approve": True},
                    "echo": "auto-friend-admin-friend-flag",
                }
                assert act2 == {
                    "action": "set_group_add_request",
                    "params": {"flag": "group-invite-flag", "sub_type": "invite", "approve": True},
                    "echo": "auto-group-group-invite-flag",
                }
                assert received_actions.empty()
            finally:
                stop_event.set()
                await asyncio.wait_for(task, timeout=3.0)
                await bus.close()

    asyncio.run(_run())
