# qq_agent/channel_bridge.py
"""双队列消息总线（DirectMemoryBus / LlmosChannelBus）与 NapCat OB11 正向 WS 适配器。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import secrets
import sys
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar, runtime_checkable

import websockets

from .ob11 import build_send_action, build_send_actions, event_to_inbound
from .stickers import StickerRegistry
from .types import InboundMessage, OutboundReply

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CONSUMER_FIBER_ID: bytes = b"qq-agent-bus-fib"  # 16 non-zero bytes
CONSUMER_FIBER_GENERATION: int = 1


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

    async def ack_inbound(self, sequence: int) -> None:
        self._inbound.ack(sequence)

    async def publish_outbound(self, reply: OutboundReply) -> int:
        return self._outbound.enqueue(reply)

    async def poll_outbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, OutboundReply]]:
        return self._outbound.receive(limit=limit)

    async def ack_outbound(self, sequence: int) -> None:
        self._outbound.ack(sequence)


def _load_channel_client_cls(llmos_root: Path) -> Any:
    """动态注入 `<llmos_root>/sdk/python` 并导入 `nlos_sdk.ChannelClient`。"""
    try:
        from nlos_sdk import ChannelClient

        return ChannelClient
    except ImportError:
        sdk_python = str(llmos_root / "sdk" / "python")
        if sdk_python not in sys.path:
            sys.path.insert(0, sdk_python)
        gen_python = str(llmos_root / "gen" / "python")
        if gen_python not in sys.path:
            sys.path.insert(0, gen_python)
        from nlos_sdk import ChannelClient

        return ChannelClient


class LlmosChannelBus:
    """基于 `llmos` `nlos.sabi.ChannelControl` 持久化队列的双通道总线（DOGFOOD-FULL 模式）。"""

    def __init__(
        self,
        endpoint: str,
        *,
        llmos_root: Path | str = r"D:\llmos",
        capacity_bytes: int = 65536,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._llmos_root = Path(llmos_root)
        self._capacity_bytes = capacity_bytes
        self._client_factory = client_factory

        self._inbound_create_key: bytes = secrets.token_bytes(16)
        self._outbound_create_key: bytes = secrets.token_bytes(16)
        while self._outbound_create_key == self._inbound_create_key:
            self._outbound_create_key = secrets.token_bytes(16)

        self._consumer_fiber_id: bytes = DEFAULT_CONSUMER_FIBER_ID
        assert len(self._consumer_fiber_id) == 16

        self._client: Any | None = None
        self._inbound_record: Any | None = None
        self._outbound_record: Any | None = None
        self._inbound_high_water: int = 0
        self._outbound_high_water: int = 0
        self._rpc_lock = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def start(self) -> None:
        async with self._rpc_lock:
            if self._client is None:
                if self._client_factory is not None:
                    self._client = self._client_factory(self._endpoint)
                else:
                    client_cls = _load_channel_client_cls(self._llmos_root)
                    self._client = client_cls(self._endpoint)

            self._inbound_record = await self._client.create_channel(
                self._capacity_bytes,
                idempotency_key=self._inbound_create_key,
            )
            self._outbound_record = await self._client.create_channel(
                self._capacity_bytes,
                idempotency_key=self._outbound_create_key,
            )

    async def close(self) -> None:
        async with self._rpc_lock:
            if self._client is not None:
                close_fn = getattr(self._client, "close", None)
                if callable(close_fn):
                    maybe_coro = close_fn()
                    if inspect.isawaitable(maybe_coro):
                        await maybe_coro
                self._client = None

    def _require_started(self) -> tuple[Any, Any, Any]:
        if (
            self._client is None
            or self._inbound_record is None
            or self._outbound_record is None
        ):
            raise RuntimeError(
                "LlmosChannelBus is not started; call `await bus.start()` first"
            )
        return self._client, self._inbound_record, self._outbound_record

    def _consumption_key(self, channel_id: bytes, sequence: int) -> bytes:
        return hashlib.blake2b(
            channel_id + sequence.to_bytes(8, "big") + self._consumer_fiber_id,
            digest_size=16,
        ).digest()

    async def publish_inbound(self, msg: InboundMessage) -> int:
        payload = msg.to_json_bytes()
        async with self._rpc_lock:
            client, inbound_rec, _ = self._require_started()
            entry = await client.enqueue(
                inbound_rec.channel_id,
                inbound_rec.generation,
                inbound_rec.fencing_token,
                payload,
            )
            return int(entry.sequence)

    async def poll_inbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, InboundMessage]]:
        if limit <= 0:
            return []
        async with self._rpc_lock:
            client, inbound_rec, _ = self._require_started()
            entries = await client.receive(inbound_rec.channel_id, limit=limit)
        return [
            (int(entry.sequence), InboundMessage.from_json_bytes(bytes(entry.payload)))
            for entry in entries
        ]

    async def ack_inbound(self, sequence: int) -> None:
        async with self._rpc_lock:
            client, inbound_rec, _ = self._require_started()
            if sequence <= self._inbound_high_water:
                return
            for seq in range(self._inbound_high_water + 1, sequence + 1):
                await client.register_queue_consumption(
                    inbound_rec.channel_id,
                    inbound_rec.generation,
                    seq,
                    self._consumer_fiber_id,
                    CONSUMER_FIBER_GENERATION,
                    idempotency_key=self._consumption_key(inbound_rec.channel_id, seq),
                )
            await client.ack(
                inbound_rec.channel_id,
                inbound_rec.generation,
                sequence,
            )
            await client.compact(
                inbound_rec.channel_id,
                inbound_rec.generation,
            )
            self._inbound_high_water = sequence

    async def publish_outbound(self, reply: OutboundReply) -> int:
        payload = reply.to_json_bytes()
        async with self._rpc_lock:
            client, _, outbound_rec = self._require_started()
            entry = await client.enqueue(
                outbound_rec.channel_id,
                outbound_rec.generation,
                outbound_rec.fencing_token,
                payload,
            )
            return int(entry.sequence)

    async def poll_outbound(
        self,
        limit: int = 16,
    ) -> list[tuple[int, OutboundReply]]:
        if limit <= 0:
            return []
        async with self._rpc_lock:
            client, _, outbound_rec = self._require_started()
            entries = await client.receive(outbound_rec.channel_id, limit=limit)
        return [
            (int(entry.sequence), OutboundReply.from_json_bytes(bytes(entry.payload)))
            for entry in entries
        ]

    async def ack_outbound(self, sequence: int) -> None:
        async with self._rpc_lock:
            client, _, outbound_rec = self._require_started()
            if sequence <= self._outbound_high_water:
                return
            for seq in range(self._outbound_high_water + 1, sequence + 1):
                await client.register_queue_consumption(
                    outbound_rec.channel_id,
                    outbound_rec.generation,
                    seq,
                    self._consumer_fiber_id,
                    CONSUMER_FIBER_GENERATION,
                    idempotency_key=self._consumption_key(outbound_rec.channel_id, seq),
                )
            await client.ack(
                outbound_rec.channel_id,
                outbound_rec.generation,
                sequence,
            )
            await client.compact(
                outbound_rec.channel_id,
                outbound_rec.generation,
            )
            self._outbound_high_water = sequence


class NapCatWsAdapter:
    """连接 NapCat OB11 正向 WebSocket 的双向桥接器，将消息解耦推拉至 MessageBus。"""

    def __init__(
        self,
        ws_url: str,
        bus: MessageBus,
        *,
        bot_qq: int | None = None,
        access_token: str | None = None,
        auto_approve_friend_ids: set[int] | None = None,
        auto_approve_group_ids: set[int] | None = None,
        stickers: StickerRegistry | None = None,
        sticker_delay_s: float = 0.25,
        poll_interval_s: float = 0.05,
        reconnect_delay_s: float = 1.0,
    ) -> None:
        self._ws_url = ws_url
        self._bus = bus
        self._bot_qq = bot_qq
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
        self._reconnect_delay_s = max(0.0, float(reconnect_delay_s))

    @property
    def ws_url(self) -> str:
        return self._ws_url

    @property
    def bus(self) -> MessageBus:
        return self._bus

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
            await ws.send(json.dumps(action, ensure_ascii=False))
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
                await ws.send(json.dumps(action, ensure_ascii=False))
                logger.info("Auto-approved whitelisted group invite")

    async def _recv_loop(self, ws: Any, session_stop: asyncio.Event) -> None:
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
            post_type = data.get("post_type")
            if post_type == "request":
                await self._maybe_auto_approve_request(ws, data)
                continue
            # Ignore action echo responses (no post_type) and meta_event heartbeats
            if post_type != "message":
                continue
            inbound = event_to_inbound(data, bot_qq=self._bot_qq)
            if inbound is not None:
                await self._bus.publish_inbound(inbound)

    async def _send_loop(self, ws: Any, session_stop: asyncio.Event) -> None:
        while not session_stop.is_set():
            batch = await self._bus.poll_outbound()
            if batch:
                for seq, reply in batch:
                    if session_stop.is_set():
                        return
                    echo = reply.correlation_id or None
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
                        await ws.send(json.dumps(action, ensure_ascii=False))
                    await self._bus.ack_outbound(seq)
            else:
                try:
                    await asyncio.wait_for(
                        session_stop.wait(),
                        timeout=self._poll_interval_s,
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
            recv_task = asyncio.create_task(self._recv_loop(ws, session_stop))
            send_task = asyncio.create_task(self._send_loop(ws, session_stop))
            wait_tasks: set[asyncio.Task[Any]] = {recv_task, send_task}
            stop_wait_task: asyncio.Task[Any] | None = None

            if stop_event is not None:
                if stop_event.is_set():
                    session_stop.set()
                stop_wait_task = asyncio.create_task(stop_event.wait())
                wait_tasks.add(stop_wait_task)

            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            session_stop.set()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                if task is stop_wait_task:
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
    "CONSUMER_FIBER_GENERATION",
    "DEFAULT_CONSUMER_FIBER_ID",
    "DirectMemoryBus",
    "LlmosChannelBus",
    "MessageBus",
    "NapCatWsAdapter",
]
