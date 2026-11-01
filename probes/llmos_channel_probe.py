"""M0 / W1 跨进程探针：经 llmos ChannelControl SABI v1.0 驱动 InboundMessage / OutboundReply 端到端闭环。

验证范围（对应 docs/evidence/m0-channel-budget.md W73/W80 闭环复验）：
  1) 自举启动 llmos `system-control-daemon`（优先直启预构建二进制，回退 cargo run），监听唯一 Named Pipe / UDS；
  2) 使用 `nlos_sdk.ChannelClient` 创建 `inbound` 与 `outbound` 两条持久化 Channel（64 KiB 容量）并校验幂等重放；
  3) `InboundMessage`（群聊 @bot 消息）JSON 序列化 -> `enqueue` -> `receive` -> `from_json_bytes` 100% 字段断言
     -> `register_queue_consumption` -> `ack` -> `compact` -> `inspect_queue` 校验队列清空；
  4) `OutboundReply`（群聊回包）在 `outbound` Channel 完成同等投递/接收/消费登记/确认/压实闭环；
  5) 停机并清理临时持久化目录，退出码 0。

用法：
    uv run python probes/llmos_channel_probe.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_llmos_root() -> Path:
    """解析 LLMOS_ROOT：环境变量优先；未设置时自动探测本地或同级 ../llmos 目录。"""
    env_val = os.environ.get("LLMOS_ROOT", "").strip()
    if env_val:
        return Path(env_val)
    preferred = Path(r"D:\llmos")
    if (preferred / "sdk" / "python" / "nlos_sdk" / "channel.py").exists():
        return preferred
    return REPO_ROOT.parent / "llmos"


LLMOS_ROOT = resolve_llmos_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LLMOS_ROOT / "sdk" / "python"))
sys.path.insert(0, str(LLMOS_ROOT / "gen" / "python"))

import nlos_sdk  # noqa: E402
from nlos_sdk import (  # noqa: E402
    CHANNEL_CONTROL_SCHEMA,
    CHANNEL_CONTROL_SERVICE,
    ChannelClient,
    ChannelRecord,
    TransportConfig,
)
from qq_agent.types import InboundMessage, OutboundReply  # noqa: E402

_ENDPOINT_SEQ = 0
CHANNEL_CAPACITY_BYTES = 65536


def endpoint(label: str) -> str:
    """构造唯一本地 IPC 端点（PID + 完整 time_ns + 进程内递增序号 + label，规避 Windows 粗时钟粒度撞名）。"""
    global _ENDPOINT_SEQ
    _ENDPOINT_SEQ += 1
    unique = f"{os.getpid():x}-{time.time_ns()}-{_ENDPOINT_SEQ}-{label}"
    if sys.platform == "win32":
        return rf"\\.\pipe\qq-agent-cc-{unique}"
    return str(Path(tempfile.gettempdir()) / f"qq-agent-cc-{unique}.sock")


def resolve_daemon_command(
    root_dir: Path,
    auth_endpoint: str,
    plain_endpoint: str,
) -> tuple[list[str], str]:
    """优先使用 target/debug 预编译二进制秒级启动，缺失时回退 cargo run。"""
    exe_name = (
        "system-control-daemon.exe" if sys.platform == "win32" else "system-control-daemon"
    )
    prebuilt = LLMOS_ROOT / "target" / "debug" / exe_name
    daemon_args = [
        "--root",
        str(root_dir),
        "--auth-socket",
        auth_endpoint,
        "--plain-socket",
        plain_endpoint,
    ]
    if prebuilt.exists():
        return [str(prebuilt), *daemon_args], f"prebuilt ({prebuilt})"
    return [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "nlos-system-control",
        "--features",
        "daemon",
        "--bin",
        "system-control-daemon",
        "--",
        *daemon_args,
    ], "cargo run fallback"


async def start_daemon(
    root_dir: Path,
    auth_endpoint: str,
    plain_endpoint: str,
) -> asyncio.subprocess.Process:
    cmd, mode_desc = resolve_daemon_command(root_dir, auth_endpoint, plain_endpoint)
    print(f"[probe] starting system-control-daemon via {mode_desc}", flush=True)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=LLMOS_ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        ready = await asyncio.wait_for(process.stdout.readline(), timeout=120)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    ready_text = ready.decode("utf-8", errors="replace").strip()
    if not ready_text.startswith("READY ") or "plain_socket=" not in ready_text:
        assert process.stderr is not None
        stderr = await process.stderr.read()
        process.kill()
        await process.wait()
        raise AssertionError(
            f"system-control-daemon failed before READY: {ready_text!r} / {stderr!r}"
        )
    print(f"[probe] daemon {ready_text}", flush=True)
    return process


async def stop_daemon(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
        await process.wait()


async def create_and_verify_channel(
    client: ChannelClient,
    *,
    name: str,
    idempotency_key: bytes,
    policy_digest: bytes,
    created_at_ms: int,
) -> ChannelRecord:
    """创建持久化 Channel 并验证相同幂等键重放返回一致的 channel_id / generation / fencing_token。"""
    created = await client.create_channel(
        CHANNEL_CAPACITY_BYTES,
        policy_digest=policy_digest,
        idempotency_key=idempotency_key,
        created_at_ms=created_at_ms,
    )
    assert not created.replayed, f"{name} 首次创建不应标记 replayed"
    assert len(created.channel_id) == 16
    assert created.generation == 1
    assert len(created.fencing_token) == 32
    assert created.capacity_bytes == CHANNEL_CAPACITY_BYTES

    replayed = await client.create_channel(
        CHANNEL_CAPACITY_BYTES,
        policy_digest=policy_digest,
        idempotency_key=idempotency_key,
        created_at_ms=created_at_ms,
    )
    assert replayed.replayed, f"{name} 重复创建应命中幂等重放"
    assert replayed.channel_id == created.channel_id
    assert replayed.generation == created.generation
    assert replayed.fencing_token == created.fencing_token
    print(
        f"[probe] channel '{name}' created & idempotent replay OK: "
        f"id={created.channel_id.hex()[:16]}... gen={created.generation} "
        f"cap={created.capacity_bytes}B",
        flush=True,
    )
    return created


async def verify_inbound_roundtrip(
    client: ChannelClient,
    channel: ChannelRecord,
) -> None:
    """在 inbound channel 上验证 InboundMessage 完整投递 -> 接收 -> 消费登记 -> ACK -> 压实闭环。"""
    print("\n[probe] === Phase 2: InboundMessage 跨进程队列闭环 ===", flush=True)
    inbound_msg = InboundMessage(
        event_id="evt-m0-probe-001",
        chat_type="group",
        user_id=10001,
        group_id=70001,
        sender_name="测试群友",
        text="你好，验证一下 ChannelControl 跨进程收发",
        raw_message="[CQ:at,qq=90001] 你好，验证一下 ChannelControl 跨进程收发",
        mentioned_bot=True,
        reply_to_id=42,
        self_id=90001,
        timestamp=1_730_000_001,
    )
    payload = inbound_msg.to_json_bytes()

    enqueued = await client.enqueue(
        channel.channel_id,
        channel.generation,
        channel.fencing_token,
        payload,
        idempotency_key=b"\x21" * 16,
        enqueued_at_ms=2_100,
    )
    assert not enqueued.replayed
    assert enqueued.sequence == 1
    assert enqueued.payload == payload
    assert enqueued.payload_len == len(payload)
    print(
        f"[probe] inbound enqueue OK: seq={enqueued.sequence} bytes={enqueued.payload_len}",
        flush=True,
    )

    entries = await client.receive(channel.channel_id, limit=16)
    assert len(entries) == 1
    assert entries[0].sequence == 1
    decoded = InboundMessage.from_json_bytes(entries[0].payload)
    assert decoded == inbound_msg, f"InboundMessage 字段不一致: {decoded!r} != {inbound_msg!r}"
    assert decoded.session_key == "group:70001"
    print(
        f"[probe] inbound receive & decode 100% match: event_id={decoded.event_id} "
        f"session={decoded.session_key} mentioned_bot={decoded.mentioned_bot}",
        flush=True,
    )

    consumption = await client.register_queue_consumption(
        channel.channel_id,
        channel.generation,
        entries[0].sequence,
        b"\xa1" * 16,
        1,
        idempotency_key=b"\x31" * 16,
        registered_at_ms=2_200,
    )
    assert not consumption.replayed
    assert consumption.sequence == 1
    assert len(consumption.registration_id) == 16

    ack_res = await client.ack(
        channel.channel_id,
        channel.generation,
        entries[0].sequence,
        acked_at_ms=2_300,
        idempotency_key=b"\x41" * 16,
    )
    assert ack_res.acked
    assert ack_res.consume_high_water == 1

    compacted = await client.compact(
        channel.channel_id,
        channel.generation,
        idempotency_key=b"\x51" * 16,
    )
    assert compacted.removed_entries == 1
    assert compacted.reclaimed_bytes == len(payload)

    qstate = await client.inspect_queue(channel.channel_id)
    assert qstate.outstanding_entries == 0
    assert qstate.unacked_entries == 0
    assert qstate.acked_uncompacted_entries == 0
    assert qstate.outstanding_bytes == 0
    assert qstate.consume_high_water == 1
    assert qstate.trim_high_water == 1
    assert await client.receive(channel.channel_id, limit=16) == []
    print(
        f"[probe] inbound register -> ack -> compact -> inspect_queue OK: "
        f"removed={compacted.removed_entries} reclaimed={compacted.reclaimed_bytes}B "
        f"outstanding={qstate.outstanding_entries}",
        flush=True,
    )


async def verify_outbound_roundtrip(
    client: ChannelClient,
    channel: ChannelRecord,
) -> None:
    """在 outbound channel 上验证 OutboundReply 完整投递 -> 接收 -> 消费登记 -> ACK -> 压实闭环。"""
    print("\n[probe] === Phase 3: OutboundReply 跨进程队列闭环 ===", flush=True)
    outbound_reply = OutboundReply(
        chat_type="group",
        target_id=70001,
        text="收到！llmos ChannelControl SABI v1.0 跨进程收发正常。",
        reply_to_message_id=42,
        at_user_id=10001,
        correlation_id="evt-m0-probe-001",
    )
    payload = outbound_reply.to_json_bytes()

    enqueued = await client.enqueue(
        channel.channel_id,
        channel.generation,
        channel.fencing_token,
        payload,
        idempotency_key=b"\x22" * 16,
        enqueued_at_ms=3_100,
    )
    assert not enqueued.replayed
    assert enqueued.sequence == 1
    assert enqueued.payload == payload
    assert enqueued.payload_len == len(payload)
    print(
        f"[probe] outbound enqueue OK: seq={enqueued.sequence} bytes={enqueued.payload_len}",
        flush=True,
    )

    entries = await client.receive(channel.channel_id, limit=16)
    assert len(entries) == 1
    assert entries[0].sequence == 1
    decoded = OutboundReply.from_json_bytes(entries[0].payload)
    assert decoded == outbound_reply, (
        f"OutboundReply 字段不一致: {decoded!r} != {outbound_reply!r}"
    )
    print(
        f"[probe] outbound receive & decode 100% match: target={decoded.target_id} "
        f"at_user={decoded.at_user_id} correlation_id={decoded.correlation_id}",
        flush=True,
    )

    consumption = await client.register_queue_consumption(
        channel.channel_id,
        channel.generation,
        entries[0].sequence,
        b"\xa2" * 16,
        1,
        idempotency_key=b"\x32" * 16,
        registered_at_ms=3_200,
    )
    assert not consumption.replayed
    assert consumption.sequence == 1
    assert len(consumption.registration_id) == 16

    ack_res = await client.ack(
        channel.channel_id,
        channel.generation,
        entries[0].sequence,
        acked_at_ms=3_300,
        idempotency_key=b"\x42" * 16,
    )
    assert ack_res.acked
    assert ack_res.consume_high_water == 1

    compacted = await client.compact(
        channel.channel_id,
        channel.generation,
        idempotency_key=b"\x52" * 16,
    )
    assert compacted.removed_entries == 1
    assert compacted.reclaimed_bytes == len(payload)

    qstate = await client.inspect_queue(channel.channel_id)
    assert qstate.outstanding_entries == 0
    assert qstate.unacked_entries == 0
    assert qstate.acked_uncompacted_entries == 0
    assert qstate.outstanding_bytes == 0
    assert qstate.consume_high_water == 1
    assert qstate.trim_high_water == 1
    assert await client.receive(channel.channel_id, limit=16) == []
    print(
        f"[probe] outbound register -> ack -> compact -> inspect_queue OK: "
        f"removed={compacted.removed_entries} reclaimed={compacted.reclaimed_bytes}B "
        f"outstanding={qstate.outstanding_entries}",
        flush=True,
    )


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"[probe] llmos root: {LLMOS_ROOT}", flush=True)
    print(
        f"[probe] nlos_sdk v{getattr(nlos_sdk, '__version__', 'unknown')} import OK "
        f"(service={CHANNEL_CONTROL_SERVICE}, schema={CHANNEL_CONTROL_SCHEMA})",
        flush=True,
    )

    req = ChannelClient.service_requirement()
    assert req.service == CHANNEL_CONTROL_SERVICE
    assert req.schema_name == CHANNEL_CONTROL_SCHEMA
    assert req.major == 1
    assert req.minimum_minor == 0

    root_dir = (
        Path(tempfile.gettempdir())
        / f"qq-agent-cc-probe-{os.getpid()}-{time.time_ns()}"
    )
    auth_endpoint = endpoint("auth")
    plain_endpoint = endpoint("plain")

    daemon = await start_daemon(root_dir, auth_endpoint, plain_endpoint)
    try:
        config = TransportConfig(
            connect_timeout=5.0,
            read_timeout=5.0,
            write_timeout=5.0,
        )
        client = ChannelClient(plain_endpoint, config=config)

        print("\n[probe] === Phase 1: 创建 inbound / outbound 持久化 Channel 与幂等重放 ===", flush=True)
        inbound_ch = await create_and_verify_channel(
            client,
            name="inbound",
            idempotency_key=b"\x11" * 16,
            policy_digest=b"\xd1" * 32,
            created_at_ms=1_000,
        )
        outbound_ch = await create_and_verify_channel(
            client,
            name="outbound",
            idempotency_key=b"\x12" * 16,
            policy_digest=b"\xd2" * 32,
            created_at_ms=1_001,
        )
        assert inbound_ch.channel_id != outbound_ch.channel_id

        await verify_inbound_roundtrip(client, inbound_ch)
        await verify_outbound_roundtrip(client, outbound_ch)
    finally:
        await stop_daemon(daemon)
        if sys.platform != "win32":
            for sock in (auth_endpoint, plain_endpoint):
                try:
                    Path(sock).unlink()
                except FileNotFoundError:
                    pass
        shutil.rmtree(root_dir, ignore_errors=True)
        print("\n[probe] daemon stopped & temporary state root cleaned", flush=True)

    print(
        "[probe] ALL PASS: InboundMessage & OutboundReply 经 llmos ChannelControl SABI v1.0 跨进程收发闭环通过",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
