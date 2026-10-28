# M0 证据：llmos Python IPC 连通（Task 5）

> 探针脚本：`probes/llmos_ipc_ping.py`（自写，非官方脚本复跑）。
> 环境：Windows 11 Pro 10.0.28000；llmos `d4a9d34`（只读，未改任何文件）；Python 3.12.10（qq-agent venv，uv 管理，protobuf 7.36.2 已入 dev 依赖组）；cargo workspace 37 crates 已预构建（dev profile）。
> API 依据：`docs/evidence/m0-llmos-notes.md`（Task 4，均已核实）。

## IPC 结论：**PASS**

conformance 绿（简报 Step 1 指定的 `client.py` exit 0）+ 自写探针用 SDK 编程完成两条真实 IPC 往返（直连 echo + 经 service_directory 协商的 side-effecting 调用，exit 0）。

G0 判据输入：**Python 可以编程使用 llmos SDK 做真实 IPC 往返**（Windows named pipe，Proactor 事件循环），不是只会跑官方脚本。

## Step 1：官方 conformance

### 1.1 `tests/conformance/ipc/client.py`（简报指定项）—— **PASS**

```bash
cd /d/AI/llmos
uv run --no-project --index-url https://pypi.tuna.tsinghua.edu.cn/simple --with protobuf \
  python tests/conformance/ipc/client.py
# 无输出（断言型脚本，静默即全绿）
echo $?   # → 0
```

**exit 0**。覆盖：INVALID_CONFIG 负例 ×3、正常 echo 往返（request_id/payload 回显）、COMPATIBILITY（schema major=2 拒收）、BACKPRESSURE（并发第二 exchange）、CONNECT/TIMEOUT（不存在的 endpoint 连接失败）。注：直连 PyPI 超时（本机网络），换清华镜像装 protobuf。

### 1.2 `tests/conformance/ipc/directory_chain.py`（额外跑的目录链套件）—— 本机 FAIL，根因为上游脚本的 pipe 命名缺陷，非 IPC 故障

原样运行 exit 1：

```
AssertionError: directory chain failed before ready: b'Error: Io { operation: Accept,
source: Os { code: 5, kind: PermissionDenied, message: "拒绝访问。" } }'
```

**根因（实测定位，非猜测）**：该脚本 `endpoint()` 用 `time.time_ns() & 0xFFFFFFF`（28 位掩码）+ label 首字符（"bootstrap"/"business" 均为 'b'）构造唯一名。本机 `time.time_ns()` 连续调用 **99990/100000 返回同值**（时钟更新粒度粗于两次函数调用间隔，实测脚本验证），连续两次 `endpoint()` 生成**完全相同**的 pipe 名 → 服务端对目录/业务两个 listener bind 同名 pipe，第二个 bind 带 `first_pipe_instance`（`crates/nlos-ipc/src/windows.rs:177`）→ `ERROR_ACCESS_DENIED`（code 5，Windows 对已存在同名 pipe 首实例标志的文档化行为）。`client.py` 不快速连续生成两个并存 pipe 名，故不受影响。

**验证性对照**：把官方脚本复制到仓库外，仅改两处（ROOT 钉到 `D:\AI\llmos`；endpoint 用完整 time_ns + 计数器 + 完整 label），其余逐字节不变：

```bash
uv run --no-project --index-url https://pypi.tuna.tsinghua.edu.cn/simple --with protobuf \
  python "$TEMP/directory_chain_namefix.py"
echo $?   # → 0
```

**exit 0**——同一 conformance（commit/recover 两阶段、~20 组断言：协商 binding 校验、UNCERTAIN/CONFLICT/DEADLINE/CANCELLED/PARTIAL/EFFECT_UNKNOWN 语义矩阵、operation_control query/cancel 幂等重放、deadline worker）全部通过。证明目录协商→业务调用全链路 IPC 在本机健康，唯一红项是上游测试脚本的命名生成缺陷（本机时钟粒度触发），与 IPC 机制无关。可后续给 llmos 提 issue/PR（属上游仓库，本任务未改动）。

## Step 2：自写探针 `probes/llmos_ipc_ping.py` —— **PASS**

```bash
PYTHONIOENCODING=utf-8 uv run python probes/llmos_ipc_ping.py
echo $?   # → 0
```

单脚本自举（asyncio 内 subprocess 起 conformance server 等 READY，无需另开终端；pipe 名用完整 time_ns + 进程内计数器规避上述撞名坑），按 m0-llmos-notes.md 的真实 API 完成**两条**真实往返：

```
[probe] SDK import OK: nlos_sdk.local_rpc nlos_sdk.service_directory

[probe] === 链路 1：LocalRpcClient -> nlos-ipc-echo ===
[probe] nlos-ipc-echo READY (...)
[probe] SDK connect OK: LocalRpcClient -> \\.\pipe\nlos-ping-7a78-...-echo
[probe] exchange OK: request_id 回显, payload 回显 b'qq-agent-m0-ping'
[probe] echo server 正常退出 (exit 0) —— 链路 1 PASS

[probe] === 链路 2：ServiceDirectoryClient.negotiate_and_connect -> operation ===
[probe] nlos-directory-chain READY (...)
[probe] negotiate OK: endpoint=\\.\pipe\nlos-ping-7a78-...-biz
[probe] binding 校验 OK: address/generation 与服务端注册一致
[probe] exchange OK: payload b'qq-agent-m0-dir-ping\xd0', receipt 0x99*16
[probe] 链路 2 PASS

[probe] ALL PASS: Python 编程使用 llmos SDK 完成两条真实 IPC 往返
```

- 链路 1（直连）：`LocalRpcClient.connect(pipe名)` → `exchange()`，断言 request_id/payload 均回显，echo server 干净退出（exit 0）
- 链路 2（目录协商）：`ServiceDirectoryClient.negotiate_and_connect` 以 `ServiceRequirement("operation", "nlos.sabi.Envelope", major=1, minimum_minor=1)` 协商 → 断言返回 endpoint/generation 与服务端注册一致 → 构造 side-effecting + long-running 请求（caller 三元组/idempotency_key/deadline/capability）→ `exchange()` → 断言 payload=`原文+\xd0`、`validate_response_context` 过、receipt=`0x99*16`、correlation_id 回显
- 复跑第二次 exit 0（稳定）

## 遇到的问题

1. **PyPI 直连超时**：`uv` 装 protobuf 走 files.pythonhosted.org 5 次重试全超时 → 换清华镜像（`--index-url https://pypi.tuna.tsinghua.edu.cn/simple`）；qq-agent 侧已 `uv add --dev protobuf` 固化，之后 `uv run python probes/llmos_ipc_ping.py` 即可复现。
2. **官方 directory_chain.py 本机必挂**：见 §1.2 根因（上游 endpoint() 命名在粗时钟粒度机器上撞名）；上游仓库保持只读未改动，用仓库外副本对照验证。
3. git-bash 会改写 `\\.\pipe\...` 参数与 heredoc 反斜杠，排查期手动传参测试被污染过一次（InvalidFilename code 123 是假信号）；探针内全部由 Python 构造 pipe 名，不受影响。

## 对 G0 的输入

- Windows named pipe + Proactor loop + 4 字节长度前缀 + protobuf envelope 的 Python⇄Rust IPC 在本机**编程可用**（非仅官方脚本可跑）
- 目录协商（service_directory→negotiate→endpoint→直连）全链编程可用；但按 m0-llmos-notes.md Q1，**目录里现只有 conformance 注册的服务**（`operation`/`operation_control`），生产 system_control daemon 不走目录，channel 跨进程不可达——qq-agent 真实接入时的宿主问题仍按笔记结论处理
