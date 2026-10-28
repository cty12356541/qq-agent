"""M0 探针：用 llmos 的 Python SDK 对本机 conformance server 做真实 IPC 往返。

验证两件事（对应 m0-llmos-notes.md §1 的 API 面）：
  1) 直连链：起 nlos-ipc-echo -> LocalRpcClient.connect -> exchange -> 校验回显
  2) 目录链：起 nlos-directory-chain -> ServiceDirectoryClient.negotiate_and_connect
     -> exchange(side-effecting) -> validate_response_context 校验回执

前置：
  - D:\\AI\\llmos 仓库已构建（cargo run 增量启动，脚本自动起 server，无需另开终端）
  - 依赖 protobuf（已入 qq-agent dev 组）：uv run python probes/llmos_ipc_ping.py

本机坑（m0-llmos-notes.md §3 + 实测补充）：
  - Windows named pipe 客户端要求 Proactor 事件循环：asyncio.run 默认满足，勿自选 loop
  - LocalRpcClient 单连接单 in-flight：本探针串行调用
  - pipe 名唯一性：本机 time.time_ns() 连续调用 ~99.99% 返回同值（时钟粒度粗），
    官方 directory_chain.py 的 `& 0xFFFFFFF` 掩码 + label 首字符截断会导致两个 pipe
    同名而 bind 失败（ACCESS_DENIED）；本探针用完整 time_ns + 进程内计数器规避
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

LLMOS_ROOT = Path(os.environ.get("LLMOS_ROOT", r"D:\AI\llmos"))
sys.path.insert(0, str(LLMOS_ROOT / "sdk" / "python"))
sys.path.insert(0, str(LLMOS_ROOT / "gen" / "python"))

from nlos.sabi.v1.envelope_pb2 import ExchangeRequest  # noqa: E402
from nlos_sdk import (  # noqa: E402
    LocalRpcClient,
    MethodSemantics,
    ServiceDirectoryClient,
    ServiceRequirement,
    TransportConfig,
    validate_response_context,
)

_ENDPOINT_COUNTER = 0


def endpoint(label: str) -> str:
    """生成唯一 pipe 名：完整 time_ns + 进程内计数器 + 完整 label。"""
    global _ENDPOINT_COUNTER
    _ENDPOINT_COUNTER += 1
    unique = f"{os.getpid():x}-{time.time_ns()}-{_ENDPOINT_COUNTER}-{label}"
    if sys.platform == "win32":
        return rf"\\.\pipe\nlos-ping-{unique}"
    return str(Path(tempfile.gettempdir()) / f"nlos-ping-{unique}.sock")


async def start_server(binary: str, *args: str) -> asyncio.subprocess.Process:
    """起 conformance server（feature-gated bin），等 stdout 出现 READY。"""
    process = await asyncio.create_subprocess_exec(
        "cargo",
        "run",
        "--quiet",
        "-p",
        "nlos-ipc",
        "--features",
        "conformance-server",
        "--bin",
        binary,
        "--",
        *args,
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
    if ready.strip() != b"READY":
        assert process.stderr is not None
        stderr = await process.stderr.read()
        raise AssertionError(f"{binary} failed before READY: {stderr!r}")
    print(f"[probe] {binary} READY ({binary} args={args[:2]}...)")
    return process


def echo_request(request_id_byte: int, payload: bytes) -> ExchangeRequest:
    """构造一个非 side-effecting 的最小合法 envelope（镜像 client.py request()）。"""
    value = ExchangeRequest()
    value.envelope.schema.name = "nlos.sabi.Envelope"
    value.envelope.schema.major = 1
    value.envelope.schema.minor = 0
    value.envelope.request_id = bytes([request_id_byte]) * 16
    value.envelope.service = "operation"
    value.envelope.method = "get"
    value.envelope.payload = payload
    return value


def business_request(request_id_byte: int, payload: bytes) -> ExchangeRequest:
    """构造 side-effecting + long-running 业务请求（镜像 directory_chain.py
    business_request()：caller 三元组 + idempotency_key + deadline + capability）。
    """
    request = ExchangeRequest()
    request.envelope.schema.name = "nlos.sabi.Envelope"
    request.envelope.schema.major = 1
    request.envelope.schema.minor = 1
    request.envelope.request_id = bytes([request_id_byte]) * 16
    request.envelope.service = "operation"
    request.envelope.method = "cancel"
    context = request.envelope.request_context
    context.caller.principal_id = bytes([1]) * 16
    context.caller.application_id = bytes([2]) * 16
    context.caller.process_id = bytes([3]) * 16
    context.caller.process_generation = 7
    context.correlation_id = bytes([5]) * 16
    context.idempotency_key = bytes([6]) * 16
    context.deadline_monotonic_ns = 123_456
    capability = context.capability_handles.add()
    capability.slot = 11
    capability.generation = 2
    request.envelope.payload = payload
    return request


async def ping_echo() -> None:
    """链路 1：直连 echo server 的最小往返。"""
    print("\n[probe] === 链路 1：LocalRpcClient -> nlos-ipc-echo ===")
    address = endpoint("echo")
    payload = b"qq-agent-m0-ping"
    server = await start_server("nlos-ipc-echo", address, "0")
    try:
        client = await LocalRpcClient.connect(address, TransportConfig())
        print(f"[probe] SDK connect OK: {type(client).__name__} -> {address}")
        request = echo_request(0x07, payload)
        response = await client.exchange(request)
        assert response.envelope.request_id == bytes([0x07]) * 16, "request_id 未回显"
        assert response.envelope.payload == payload, "payload 未回显"
        print(f"[probe] exchange OK: request_id 回显, payload 回显 {response.envelope.payload!r}")
        await client.close()
        exit_code = await asyncio.wait_for(server.wait(), timeout=30)
        assert exit_code == 0, f"echo server 退出码 {exit_code}"
        print("[probe] echo server 正常退出 (exit 0) —— 链路 1 PASS")
    except BaseException:
        server.kill()
        await server.wait()
        raise


async def ping_directory_chain() -> None:
    """链路 2：经 service_directory 协商后对业务服务的 side-effecting 往返。"""
    print("\n[probe] === 链路 2：ServiceDirectoryClient.negotiate_and_connect -> operation ===")
    directory_endpoint = endpoint("dir")
    business_endpoint = endpoint("biz")
    authority_path = str(
        Path(tempfile.gettempdir())
        / f"nlos-ping-{os.getpid()}-{time.time_ns()}-authority.sqlite3"
    )
    payload = b"qq-agent-m0-dir-ping"
    server = await start_server(
        "nlos-directory-chain",
        directory_endpoint,
        business_endpoint,
        authority_path,
        "recover",
    )
    try:
        connected = await ServiceDirectoryClient.negotiate_and_connect(
            directory_endpoint,
            ServiceRequirement(
                service="operation",
                schema_name="nlos.sabi.Envelope",
                major=1,
                minimum_minor=1,
            ),
            TransportConfig(),
        )
        print(f"[probe] negotiate OK: endpoint={connected.binding.endpoint.address}")
        assert connected.binding.endpoint.address == business_endpoint, "协商 endpoint 不符"
        assert connected.binding.candidate.generation == 7, "candidate generation 不符"
        print("[probe] binding 校验 OK: address/generation 与服务端注册一致")

        response = await connected.client.exchange(business_request(0x0A, payload))
        expected_payload = payload + b"\xd0"
        assert response.envelope.request_id == bytes([0x0A]) * 16, "request_id 未回显"
        assert response.envelope.payload == expected_payload, (
            f"payload 期望 {expected_payload!r}，实际 {response.envelope.payload!r}"
        )
        context = validate_response_context(
            response.envelope,
            MethodSemantics(side_effecting=True, long_running=True),
        )
        assert context.receipts, "side-effecting 响应缺少回执"
        assert context.receipts[0].receipt_id == bytes([0x99]) * 16, "回执 ID 不符"
        assert context.correlation_id == bytes([5]) * 16, "correlation_id 未回显"
        print(f"[probe] exchange OK: payload {response.envelope.payload!r}, receipt 0x99*16")
        await connected.client.close()
        print("[probe] 链路 2 PASS")
    finally:
        if server.returncode is None:
            server.kill()
            await server.wait()
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(authority_path + suffix)
            except FileNotFoundError:
                pass


async def main() -> int:
    print(f"[probe] llmos root: {LLMOS_ROOT}")
    print("[probe] SDK import OK:", LocalRpcClient.__module__, ServiceDirectoryClient.__module__)
    await ping_echo()
    await ping_directory_chain()
    print("\n[probe] ALL PASS: Python 编程使用 llmos SDK 完成两条真实 IPC 往返")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
