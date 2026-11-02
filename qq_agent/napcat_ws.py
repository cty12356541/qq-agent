# qq_agent/napcat_ws.py
"""连接 NapCat OneBot v11 正向 WebSocket 的双向桥接适配器与消息缓存组件。"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
import dataclasses
import json
import logging
from typing import Any

import websockets

from .bus_base import AdaptivePoller, MessageBus
from .llmos_bus import _try_parse_int
from .ob11 import (
    build_send_actions,
    event_to_inbound,
    extract_unresolved_forward_ids,
)
from .stickers import StickerRegistry
from .types import InboundMessage

logger = logging.getLogger(__name__)

MSG_CACHE_CAPACITY: int = 500
OUTBOUND_ECHO_CACHE_CAPACITY: int = 256

MsgCacheEntry = tuple[int, str, str, tuple[str, ...]]


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
                            is_voice = bool(reply.voice_uri or reply.ai_voice_character)
                            cached_text = (
                                f"[语音：{reply.text}]"
                                if (idx == 0 and is_voice)
                                else (
                                    reply.text
                                    if (idx == 0 and reply.text.strip())
                                    else "[动画表情]"
                                )
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
    "MSG_CACHE_CAPACITY",
    "MsgCacheEntry",
    "NapCatWsAdapter",
    "OUTBOUND_ECHO_CACHE_CAPACITY",
]
