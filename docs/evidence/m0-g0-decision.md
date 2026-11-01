# M0 决策门（G0）正式评审与架构裁定（Task 8）

> **G0 结论：`DOGFOOD-FULL`（双模运行：默认经 `llmos` `ChannelControl` SABI v1.0 跨进程队列收发，同时提供 `--direct-ws` 单进程内存总线直连开关用于无 daemon 快速联调）**
>
> - **评审日期**：2026-11-01
> - **判据基准**：`docs/superpowers/plans/2026-10-28-m0-probe.md` Task 8 预置判据 + `docs/2026-10-28-qq-agent-计划书.md` §5 M0
> - **验证环境**：Windows 11 Pro 10.0.28000 · Python 3.12.10 · `<LLMOS_ROOT>` `48964eb5`（只读引用，含 W73 dogfood 闭环与 W80 演练）

---

## 1. 预置 G0 判据逐项核对总表

对照实施计划 Task 8 预置判据（不放宽任何核心技术门槛），各项证据汇总如下：

| 判据维度 | `DOGFOOD-FULL` 预置要求 | 实测判定 | 权威证据锚点 |
|---|---|---|---|
| **Python ⇄ Rust IPC** | PASS（官方 conformance + 自写 SDK 探针真实往返） | **PASS** | `docs/evidence/m0-ipc.md`、`probes/llmos_ipc_ping.py` |
| **Channel 跨进程 E2E** | PASS（外部 Python 进程经 SABI 驱动队列投递/接收/确认） | **PASS** | `docs/evidence/m0-channel-budget.md` §4、`probes/llmos_channel_probe.py` |
| **Budget 预算机制** | 服务端测试全绿（SDK 跨进程可编程为加分项非必需） | **PASS** | `docs/evidence/m0-channel-budget.md` §3 & §4.3（`nlos-resource` v8 五维表已落地；M1 同口径本地 SQLite 护栏先行） |
| **Dogfood 上游问题台账** | 阻塞项具备明确闭环 | **7/7 CLOSED** | `docs/evidence/dogfood-issues.md`（`L-001`~`L-007` 全部于 `llmos` W73 修复闭环） |
| **NapCat 接入与宿主隔离** | 接入方案与监听探针就绪（受 `E-004` 约束须纯容器隔离） | **CONTAINER-READY**（待人工装妥 WSL2/Docker 后小号扫码回填） | `docs/evidence/m0-napcat.md`、`deploy/napcat/docker-compose.yml`、`deploy/napcat/README.md`、`probes/ob11_listen.py` |

**裁定理由**：
1. M0 初探阶段曾导致 `Channel E2E = BLOCKED` 的核心阻塞项 `L-002` 已在 `llmos` W73 波次通过正规架构路径彻底解决（冻结 `nlos.sabi.ChannelControl` `v1.0`、挂载至 `system-control-daemon`、导出 `nlos_sdk.ChannelClient`），且本仓 `probes/llmos_channel_probe.py` 已在 Windows Named Pipe 上完成 `InboundMessage` 与 `OutboundReply` 跨进程持久化收发闭环（exit 0）。
2. 因此无需退化至 `DOGFOOD-DEFERRED`，正式裁定进入 **`DOGFOOD-FULL`**。同时为保障无 Rust daemon 环境下的单机快速联调与自动化测试效率，M1 运行时内置 `--direct-ws` 单进程内存队列直连开关，与默认 `ChannelControl` IPC 模式共享完全一致的 `InboundMessage` / `OutboundReply` 契约。

---

## 2. 分项证据与命令输出摘要

### 2.1 Python IPC 连通（`docs/evidence/m0-ipc.md` —— **PASS**）

- 官方 `tests/conformance/ipc/client.py` 与修复粗时钟粒度 pipe 派生后的 `directory_chain.py` 全部 exit 0。
- 本仓自写 `probes/llmos_ipc_ping.py` 同时跑通 `LocalRpcClient` 直连回显与 `ServiceDirectoryClient.negotiate_and_connect` 目录协商 side-effecting 调用：

```powershell
uv run python probes/llmos_ipc_ping.py
```

```text
[probe] === 链路 1：LocalRpcClient -> nlos-ipc-echo ===
[probe] exchange OK: request_id 回显, payload 回显 b'qq-agent-m0-ping'
[probe] echo server 正常退出 (exit 0) —— 链路 1 PASS

[probe] === 链路 2：ServiceDirectoryClient.negotiate_and_connect -> operation ===
[probe] negotiate OK: endpoint=\\.\pipe\nlos-ping-7a78-...-biz
[probe] exchange OK: payload b'qq-agent-m0-dir-ping\xd0', receipt 0x99*16
[probe] 链路 2 PASS

[probe] ALL PASS: Python 编程使用 llmos SDK 完成两条真实 IPC 往返
```

### 2.2 ChannelControl SABI v1.0 跨进程闭环（`docs/evidence/m0-channel-budget.md` —— **PASS**）

- 本仓 `probes/llmos_channel_probe.py` 自举启动 `<LLMOS_ROOT>/target/debug/system-control-daemon.exe`，经 `nlos_sdk.ChannelClient` 创建 `inbound` / `outbound` 两条 64 KiB 持久化队列，完整校验幂等重放、`InboundMessage`（群聊 `@bot`）与 `OutboundReply` 的序列化入队、非破坏接收、100% 字段还原、`register_queue_consumption`、`ack`、`compact` 与 `inspect_queue` 清空断言：

```powershell
uv run python probes/llmos_channel_probe.py
```

```text
[probe] llmos root: <LLMOS_ROOT>
[probe] nlos_sdk v0.1.0 import OK (service=channel, schema=nlos.sabi.ChannelControl)
[probe] starting system-control-daemon via prebuilt (<LLMOS_ROOT>/target/debug/system-control-daemon.exe)
[probe] channel 'inbound' created & idempotent replay OK: id=a7db9e77d61cab57... gen=1 cap=65536B
[probe] channel 'outbound' created & idempotent replay OK: id=fc33693a42c5c98c... gen=1 cap=65536B
[probe] inbound receive & decode 100% match: event_id=evt-m0-probe-001 session=group:70001 mentioned_bot=True
[probe] inbound register -> ack -> compact -> inspect_queue OK: removed=1 reclaimed=338B outstanding=0
[probe] outbound receive & decode 100% match: target=70001 at_user=10001 correlation_id=evt-m0-probe-001
[probe] outbound register -> ack -> compact -> inspect_queue OK: removed=1 reclaimed=193B outstanding=0
[probe] ALL PASS: InboundMessage & OutboundReply 经 llmos ChannelControl SABI v1.0 跨进程收发闭环通过
EXIT_CODE=0
```

### 2.3 Budget 机制与 M1 同口径护栏策略（`docs/evidence/m0-channel-budget.md` —— **PASS**）

- 服务端扣减核心语义（`semantic_consume_ledger` 3/3、`semantic_budget_raise` 6/6、`lifecycle_meter_aggregate` 6/6）与 W73 新增的 `nlos-resource` v8 五维 LLM Token 账本（`llm_token_budgets` 表：`max_prompt_tokens` / `max_completion_tokens` / `max_total_tokens` / `max_api_calls` / `max_cost_micros_usd`，`llm_token_budget.rs` 7/7）全部通过。
- 因跨进程 SABI 预算服务面尚未暴露至 Python SDK，M1 阶段在 `qq-agent` 内采用**同口径（调用次数 / Token / 微美元 `cost_micros_usd`）本地 SQLite 预算护栏**落地按群/按日频率限制、软限切便宜模型与硬限拒答，保持数据模型与 `nlos-resource` v8 一一对齐。

### 2.4 Dogfood 问题台账闭环汇总（`docs/evidence/dogfood-issues.md` —— **7/7 已修复**）

| 编号 | M0 发现问题 | 上游 W73 修复提交 | 当前状态 |
|---|---|---|---|
| `L-001` | `directory_chain.py` Windows 粗时钟粒度下 pipe 名碰撞 | `ee6c42d4` | **已修复**（6/6 阶段全绿） |
| `L-002` | `channel` 缺 SABI 跨进程服务面、宿主分派与 Python SDK 绑定 | `5e409b0f` + `f8b1687f` | **已修复**（`ChannelControl` v1.0 9 方法全通） |
| `L-003` | Python SDK 无 `pyproject.toml` 打包入口与版本号 | `ee6c42d4` + `f8b1687f` | **已修复**（`nlos_sdk` v0.1.0 + 自动挂载 `gen/python`） |
| `L-004` | Windows Named Pipe 非 `ProactorEventLoop` 缺少明确报错诊断 | `ee6c42d4` | **已修复**（`UNSUPPORTED_EVENT_LOOP` fail-fast） |
| `L-005` | `nlos-resource` 预算体系缺少 LLM Token 与模型 API 成本维度 | `1c796212` + `2c6c9859` | **已修复**（`SCHEMA_VERSION = 8` 五维表 + 7/7 测试） |
| `L-006` | `client.py` 断言型静默、无逐用例进度输出 | `ee6c42d4` | **已修复**（4/4 实时进度输出） |
| `L-007` | `nlos-channel` 集成测试零结构化轨迹输出 | `98b9c3ac` | **已修复**（`[nlos-channel/<suite>]` 轨迹可观测） |

### 2.5 NapCat 纯容器化隔离方案就绪（`docs/evidence/m0-napcat.md` & `E-004`）

- 针对宿主环境约束 `E-004`（宿主 Windows 日常登录个人主 QQ 号，严禁在宿主直接安装 Bot 专用 QQ 客户端），已弃用宿主 OneKey 方案，交付纯容器化编排：
  - `deploy/napcat/docker-compose.yml`：基于 `mlikiowa/napcat-docker:latest`，端口严格仅绑定回环 `127.0.0.1:3001:3001`（OB11 正向 WS）与 `127.0.0.1:6099:6099`（WebUI），通过 `napcat_qq_data` 与 `napcat_config` 双命名卷分离持久化登录态与网络配置（重启免重扫）。
  - `deploy/napcat/README.md` + `.env.example`：提供从 WSL2/Docker 安装、扫码获取 WebUI Token、开启 `:3001` 正向 WS 到 `probes/ob11_listen.py` 实机收发验收与启动 `qq_agent` 的完整操作手册。
- 待用户在本机装妥 WSL2 / Docker Desktop 并用小号扫码后，直接按手册执行 `probes/ob11_listen.py` 回填 `docs/evidence/m0-napcat.md` 抹敏实测记录即可，不阻塞 M1 内核与适配器并行开发。

---

## 3. M1 实施架构与前置执行清单

依据 **`DOGFOOD-FULL`（双模运行）** 裁定，M1 按以下契约推进：

1. **双模传输总线**：
   - **默认模式（`llmos` IPC）**：连接或自举 `system-control-daemon`，使用 `nlos_sdk.ChannelClient` 创建/复用 `inbound` 与 `outbound` 持久化队列，OB11 适配器负责入站 `enqueue` 与出站 `receive/ack/compact`，Agent 内核负责入站 `receive/ack/compact` 与出站 `enqueue`。
   - **直连开关（`--direct-ws`）**：使用进程内异步内存队列实现同一套总线接口，供无 Rust daemon 环境快速联调与测试。
2. **OB11 适配器与安全边界**：对接 `ws://127.0.0.1:3001`（支持 `NAPCAT_ACCESS_TOKEN` Bearer 鉴权与断线重连），将 QQ 协议细节严格封闭在适配层。
3. **触发过滤与白名单管理命令**：实现 `@bot` / 私聊 / 群白名单规则触发，以及不经 LLM 上下文的管理员白名单命令（状态/额度查询、会话清理、人设重载）。
4. **同口径预算护栏与记忆会话层**：SQLite 持久化会话滑动窗口、冻结人设前缀热加载、OpenAI-Compatible（DeepSeek / GLM）Provider，以及按调用数 / Token / 微美元计量的软限降级与硬限拒答护栏。
5. **人工实机闭环（并行项）**：用户完成 WSL2 + Docker Desktop 部署后，按 `deploy/napcat/README.md` 拉起容器扫码，跑通 `probes/ob11_listen.py` 并回填 `docs/evidence/m0-napcat.md`。
