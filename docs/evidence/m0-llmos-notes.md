# llmos（NLOS）M0 现状盘点笔记

> Task 4 产出。后续 T5/T6 探针的 API 依据：本文所有类名/函数签名/常量均摘自源码原文并带 `文件路径:行号`，未凭记忆改写。
> 所有路径相对 `D:\AI\llmos`（HEAD `d4a9d34`，2026-10-28，`feat(w71): complete D-OMNIBAR-MULTIMODAL, D-TRUSTED-UI, and C-NET-SYNC-MIGRATE`；origin = github.com/cty12356541/llmos）。

## 0. 构建与测试基线（本机 Windows 11，dev profile）

| 命令 | 结果 |
|---|---|
| `cargo build --workspace` | exit 0（编排者首跑 39.5s；本次增量复核 0.30s "Finished dev profile"） |
| `cargo test -p nlos-channel` | **31 passed / 0 failed**，5 个 suite：unit lib 0、`channel_authority` 3、`channel_fault_injection` 13、`queue_consumption_registration` 4、`queue_delivery` 11（doc-tests 0） |
| `cargo test -p nlos-ipc --all-features` | **27 passed / 0 failed**：unit lib 0、bin `nlos-directory-chain` 0、bin `nlos-ipc-echo` 0、`framing` 10、`handshake` 12、`handshake_transport` 0、`handshake_transport_windows` 1、`platform` 2、`windows_named_pipe` 2（doc-tests 0） |

无红项。Windows named pipe 收发与握手在本机全绿（`platform.rs` / `windows_named_pipe.rs`）。

## 1. Python SDK 真实 API 面（`sdk/python/nlos_sdk/`，共 4 个文件）

导出面（`sdk/python/nlos_sdk/__init__.py:1-29`）：
`ConnectedService, CommonSemanticsError, DirectoryNegotiationError, IpcError, LocalRpcClient, MethodSemantics, ServiceDirectoryClient, ServiceRequirement, TransportConfig, validate_request_context, validate_response_context`。

**无 channel 相关导出**（`grep -r channel sdk/python/` 零命中）。

### 1.1 `local_rpc.py` — 本地 RPC 客户端

- `class IpcError(Exception)`（`local_rpc.py:19-24`）：`__init__(self, code: str, message: str)`，属性 `self.code`。已见错误码：`INVALID_CONFIG / TIMEOUT / CONNECT / BACKPRESSURE / CONNECTION_UNUSABLE / REQUEST_ID_MISMATCH / READ / WRITE / COMPATIBILITY / FRAME_TOO_LARGE / UNSUPPORTED_PLATFORM`。
- `@dataclass(frozen=True, slots=True) class TransportConfig`（`local_rpc.py:27-58`）：
  `maximum_frame_bytes: int = 1_048_576`（上限即此值）、`connect_timeout: float = 5.0`、`read_timeout: float = 5.0`、`write_timeout: float = 5.0`。
- `class LocalRpcClient`（`local_rpc.py:61`）：
  - `@classmethod async def connect(cls, endpoint: str, config: TransportConfig | None = None) -> LocalRpcClient`（`local_rpc.py:76-96`）
  - `async def exchange(self, request: ExchangeRequest) -> ExchangeResponse`（`local_rpc.py:98`）——**单连接、单 in-flight**：并发第二个调用抛 `BACKPRESSURE`（`local_rpc.py:104-107`）。
  - `async def close(self)`（`local_rpc.py:150`）。
- **endpoint 形态**（`local_rpc.py:164-186` `_open_platform_stream`）：非 Windows = Unix socket 路径（`asyncio.open_unix_connection(endpoint)`）；Windows = **named pipe 名**（如 `\\.\pipe\nlos-xxx`），要求事件循环有 `loop.create_pipe_connection`（即 **ProactorEventLoop**，否则抛 `UNSUPPORTED_PLATFORM`，`local_rpc.py:171-176`）。Python≥3.8 Windows 默认就是 Proactor，安全；探针若自选 SelectorEventLoop 会直接炸。
- **线协议**：4 字节大端长度前缀 + `ExchangeRequest` protobuf（`local_rpc.py:112, 197-201`）。envelope 校验：schema 名必须是 `"nlos.sabi.Envelope"`、major==1、`request_id` 16 字节、service/method 非空（`local_rpc.py:204-228`）。
- protobuf 模块在 `gen/python/nlos/sabi/v1/`：`envelope_pb2, service_directory_pb2, operation_control_pb2, system_control_pb2, wait_control_pb2, takeover_control_pb2, driver_provider_pb2, principal_handshake_pb2`——**没有 channel proto**。SDK 本体无 pyproject/依赖声明，conformance 脚本靠 `sys.path.insert` 同时挂 `sdk/python` 与 `gen/python`（`tests/conformance/ipc/client.py:12-14`）。

### 1.2 `service_directory.py` — 服务发现

- `@dataclass(frozen=True, slots=True) class ServiceRequirement`（`service_directory.py:29-51`）：
  `service: str, schema_name: str, major: int, minimum_minor: int = 0, required_feature_ids: tuple[int, ...] = ()`。
- `@dataclass(frozen=True, slots=True) class ConnectedService`（`service_directory.py:54-59`）：`binding: ServiceBinding`，`client: LocalRpcClient`。
- `class DirectoryNegotiationError(Exception)`（`service_directory.py:62-68`）：`.code`（int 枚举）、`.service`。
- `class ServiceDirectoryClient`（`service_directory.py:71`）：
  - `@classmethod async def connect(cls, trusted_bootstrap_endpoint: str, config: TransportConfig | None = None) -> ServiceDirectoryClient`（`:77-87`）
  - `@classmethod async def negotiate_and_connect(cls, trusted_bootstrap_endpoint: str, requirement: ServiceRequirement, config=None) -> ConnectedService`（`:89-105`）——一步到位：连 bootstrap → negotiate → 关目录连接 → 用返回的 endpoint 直连服务。
  - `async def negotiate(self, requirement: ServiceRequirement) -> ServiceBinding`（`:107`）；`async def close(self)`（`:177`）。
- 线上形态：envelope `service="service_directory"`、`method="negotiate"`、payload=`NegotiateServiceRequest`（`service_directory.py:23-26, 131-133`）。返回 `ServiceBinding{candidate, endpoint}`，`endpoint` 是 `LocalEndpoint{kind: LocalTransportKind, address: str}`（proto `schema/nlos/sabi/v1/service_directory.proto:76-84`）。
- 校验（`service_directory.py:202-232` `_require_compatible_binding`）：win32 上 `endpoint.kind` 必须等于 `LOCAL_TRANSPORT_KIND_WINDOWS_NAMED_PIPE`（`:223` + `_platform_transport()` `:235-238`），address 非空 ≤4096 无 NUL。**endpoint 地址只从 negotiate 响应来，SDK 不提供任何推导/枚举**。

### 1.3 `common.py` — envelope 公共语义（fail-closed 校验器）

- `@dataclass class MethodSemantics`（`common.py:17-22`）：`side_effecting: bool = False, long_running: bool = False`。
- `class CommonSemanticsError(Exception)`（`common.py:25-30`）：`.code: str`。
- `def validate_request_context(envelope, semantics: MethodSemantics, now_monotonic_ns: int) -> SabiRequestContext`（`common.py:33-37`）：要求 caller 三元组 ID（principal/application/process，各 16 字节）、process_generation>0、correlation_id 16 字节；side-effecting 必须带 idempotency_key（16B）；long_running 必须带 deadline 且未过期；capability_handles ≤64 无重复；digest 要么空要么 32B。
- `def validate_response_context(envelope, semantics: MethodSemantics) -> SabiResponseContext`（`common.py:89-93`）：错误码/重试指令合法（非 UNSPECIFIED），uncertain/effect_unknown 必须配 `QUERY_OPERATION_OR_RETRY_SAME_IDEMPOTENCY_KEY`，partial 必须带 Receipt，safe_message ≤512B 无 NUL。

### 1.4 参考 conformance 客户端写法（探针可直接抄）

- `tests/conformance/ipc/client.py`：pipe 命名 `rf"\\.\pipe\nlos-ipc-{label}-{pid}-{ns}"`（`:20-24`）；起 Rust echo 服务 `cargo run -q -p nlos-ipc --features conformance-server --bin nlos-ipc-echo -- <addr> <delay_ms>`，等 stdout `READY`（`:39-68`）。
- `tests/conformance/ipc/directory_chain.py`：完整示范 `ServiceDirectoryClient.negotiate_and_connect(...)`（`:185-197`）→ 断言 `connected.binding.endpoint.address == business_endpoint`、`candidate.generation == 7` → `connected.client.exchange(...)`。服务端 `nlos-directory-chain <dir-addr> <biz-addr> <authority.sqlite3> <phase>`。目录里注册的服务名只有 `"operation"`（schema `nlos.sabi.Envelope`）与 `"operation_control"`（`nlos.sabi.OperationControl`）（`crates/nlos-ipc/src/bin/nlos-directory-chain.rs:102-124`）。

## 2. 三个必答问题

### Q1：Python 侧当前能否经 service_directory 拿到 channel 服务的 endpoint？

**不能——机制可用，但目录里没有 channel 这个服务，channel 也没有任何 RPC/proto 面。**

依据：
1. 全仓库唯一 `ServiceDirectory` 实现是 `crates/nlos-service-directory/src/lib.rs:82` `SnapshotDirectory`（确定性内存快照，≤256 注册，`lib.rs:27`）。grep 全 crates 非测试代码，`SnapshotDirectory` 只出现在：`nlos-ipc/src/bin/nlos-directory-chain.rs`（conformance 专用 bin，`:31,102-124` 只注册 `operation` + `operation_control` 两个服务）、`nlos-ipc` 自身 transport/schema、以及 `nlos-service-directory` 自己。**没有任何生产 host/daemon 把自己注册进目录**。
2. `schema/nlos/sabi/v1/` 8 个 proto（driver_provider/envelope/operation_control/principal_handshake/service_directory/system_control/takeover_control/wait_control）**无 channel**；`gen/python` 无 `channel_pb2`；`sdk/python` 零 channel 引用。
3. channel 的真实形态是**进程内 Rust 库 API**：`ChannelAuthority`（`crates/nlos-channel/src/lib.rs:365`），由 `nlos-slice-k`（`crates/nlos-slice-k/src/runtime.rs:205`）与 `nlos-system-control` daemon（`crates/nlos-system-control/src/daemon.rs:1068`）以 `ChannelAuthority::open(root.join("channel"))` 直接打开——不跨进程。
4. 生产 IPC host 是 `system_control` daemon（服务名常量 `SYSTEM_CONTROL_SERVICE: &str = "system_control"`，`crates/nlos-system-control/src/lib.rs:41`），其 endpoint **按约定推导而非目录协商**：Unix `<root>/system-control-{auth,plain}.sock`；Windows `\\.\pipe\llmos-system-control-{auth|plain}-<sha256(root路径)前16hex>`（`crates/nlos-system-control/src/daemon.rs:222-265`）。
5. Python 链路本身已被 conformance 证明可用（`directory_chain.py:185-197` 成功 negotiate 出 endpoint 并完成业务调用）。

**T5/T6 探针落点**：要验 Python↔Rust IPC，只能 (a) 起 `nlos-directory-chain` conformance server 验 SDK→目录→服务全链（服务名用 `operation`），或 (b) 直连 system-control daemon 推导出的 pipe 名调 `system_control`。"channel 收发"现阶段跨进程不可达，只能进程内 Rust 调用（或等上游把 channel 挂成 SABI 服务）。

### Q2：channel 的收发/队列 API 叫什么？

**纠正前提：`crates/nlos-channel/src/schema.rs` 是 SQLite DDL 迁移文件（v1 建表 / v2 队列前缀 / v3 消费注册，`schema.rs:7,97,202`），里面没有方法名。** 真实 API 全在 `crates/nlos-channel/src/lib.rs` 的 `ChannelAuthority` 上：

| 方法（`lib.rs` 行号） | 签名要点 |
|---|---|
| `open` (`:376`) | `pub fn open(root: impl AsRef<Path>) -> Result<Self, ChannelAuthorityError>` — 打开/落 `<root>` 下 SQLite（WAL+FULL，`:285-288`） |
| `create_channel` (`:429`) | `(request: CreateChannelRequest) -> Result<ChannelDecision, _>`；`CreateChannelRequest{capacity_bytes: u64, policy_digest: [u8;32], idempotency_key, created_at_ms}`（`:36-41`）→ `Created/Replayed(ChannelRecord{channel_id, generation, fencing_token, capacity_bytes, ...})`（`:44-58`） |
| `enqueue` (`:664`) | `(request: EnqueueRequest) -> Result<EnqueueDecision, _>`；`EnqueueRequest{channel_id, expected_generation, expected_fencing_token: FencingToken, payload: Vec<u8>, idempotency_key, enqueued_at_ms}`（`:109-116`）。CAS 检查 generation+fencing_token（stale → `StaleChannel`）；**字节容量 admission**：`backlog_bytes + payload > capacity_bytes` → `QueueFull`，且拒绝前零持久写 |
| `enqueue_registered` (`:686`) | `(request: EnqueueRequest, producer: ProducerRegistration)`，`ProducerRegistration{binding: ExecutionFiberId, fiber_generation: Generation}`（`:143-146`）— ADR-0012 产出纤维身份随行写入 |
| `receive` (`:806-810`) | `(channel_id: ChannelId, limit: usize) -> Result<Vec<QueueEntryRecord>, _>` — **非破坏读**：`sequence > consume_high_water AND > trim_high_water` 按 sequence 升序（`:816-824`） |
| `ack` (`:868`) | `(request: AckRequest) -> Result<AckDecision, _>`；`AckRequest{channel_id, up_to_sequence: u64, acked_at_ms}`（`:208-212`）— 单调推进消费高水位，重放幂等，越界/回退 → `InvalidSequence` |
| `compact` (`:974-978`) | `(channel_id: ChannelId, trim_to_sequence: u64) -> Result<CompactDecision, _>` — `min(trim_to, consume_high_water)`，只能裁已消费前缀 |
| `inspect_queue` (`:1071`) | `(channel_id) -> Result<QueueState, _>`；`QueueState{capacity_bytes, consume_high_water, trim_high_water, backlog_bytes, retained_bytes, max_sequence}`（`:262-279`） |
| `register_queue_consumption` (`:1125`) | **注册消费者**：`(request: RegisterQueueConsumptionRequest) -> Result<ConsumptionRegistrationDecision, _>`；`RegisterQueueConsumptionRequest{channel_id, sequence: u64, binding: ExecutionFiberId, fiber_generation: Generation, idempotency_key, registered_at_ms}`（`:168-176`）。六道 fail-closed 门（`:1096-1115`）：binding 非全零、sequence≠0、channel 存在、幂等键 replay/冲突、**entry 必须仍在库里**（compacted 的 sequence 拒绝）、`(channel,sequence,binding)` 唯一。产出不可变 `QueueConsumptionRecord{registration_id: ReceiptId, ...}`（`:181-189`），在 entry 被 compact 后仍存活 |
| 其他 | `rotate_channel`(`:483`)、`inspect_channel`(`:574`)、`inspect_endpoint_proof`(`:601`)、`list_consumptions_for_binding`(`:1214`) |

消息类型汇总：`QueueEntryRecord{sequence, payload, payload_bytes, idempotency_key, enqueued_at_ms, binding: Option<ExecutionFiberId>, fiber_generation: Option<Generation>}`（`:125-136`）；decision 均为 `Xxx / Replayed(Record)` 双臂（enqueue `:149-152`、ack `:223-226`、compact `:245-248`、consumption `:192-195`）。错误枚举 `ChannelAuthorityError`（`:282-303`）：`QueueFull / StaleChannel / IdempotencyConflict / InvalidPayload / InvalidSequence(&'static str) / InvalidBindingRegistration / ChannelNotFound(ChannelId) / GenerationExhausted / CorruptRecord(&'static str) / DurabilityUnavailable / SchemaVersionUnsupported(i64)` 等。

持久层（`schema.rs`）：v1 channels/generations/rotations/endpoint-proofs（`:10-87`）；v2 `channel_queue_entries`（不可变，trigger 禁改）+ `channel_queue_cursors`（consume/trim 高水位，禁删）+ `channel_queue_bytes`（`:128-190`）；v3 entry 增 `binding_id/fiber_generation` 列 + `channel_queue_consumptions` 消费注册表（`:242-268`）。`SCHEMA_VERSION: i64 = 3`（`:5`）。

### Q3：budget/meter 相关测试在哪个 crate？

实测命令：`cargo test --workspace -- --list 2>&1 | grep -iE "meter|budget"`（exit 0，24 行命中；其中 `invalid_initiation_parameters_and_wrong_target_generation_fail_closed` 是 "para**meter**s" 的假阳性，已剔除）。canonical 命中（测试名: test）：

```
delegated_budget_is_independent_not_pooled_with_parent            # nlos-capability
converge_step_budget_returns_typed_error_and_resumes_from_prefix  # nlos-commit-coordinator
mixed_semantic_infra_and_artifact_plan_failures_use_separate_budgets  # nlos-commit-coordinator
terminal_metering_keeps_active_cpu_bounded_by_elapsed_wall_under_stress  # nlos-runtime-tokio
one_hundred_thousand_activation_meter_fibers_on_two_workers       # nlos-runtime-tokio
ten_thousand_activation_meter_fibers_on_two_workers              # nlos-runtime-tokio
birth_rejection_budget_when_deadline_already_exceeded            # nlos-runtime-tokio
kill_receipt_consumption_drives_w27c_cancel_and_meters           # nlos-runtime-tokio
kill_receipt_consumption_feeds_activation_meter_dimensions       # nlos-runtime-tokio
exhausted_budget_raise_restores_the_bridge_with_the_raised_budget # nlos-slice-k
raise_can_decrease_the_budget                                     # nlos-slice-k
bridge_success_charges_the_budget_exactly_once                    # nlos-slice-k
exhausted_budget_fails_closed_typed_with_zero_semantic_writes     # nlos-slice-k
bulkhead_failure_budget_poisons_faulty_task_group_and_evicts_waiters_while_preserving_peers_and_reset  # nlos-task
cascade_budget_spends_exactly_and_exhaustion_fails_closed         # nlos-topic
crash_window_budget_spent_pending_replay_supplements_once         # nlos-topic
multi_level_chain_budgets_levels_and_restart_durability           # nlos-topic
republish_charges_child_payer_and_insufficiency_spends_no_budget  # nlos-topic
budget_exhaustion_flips_quarantined_at_declared_attempts          # nlos-topic
quarantined_subscriber_lag_holds_no_retention_budget              # nlos-topic
republish_child_enqueue_enforces_retention_without_spending_parent_budget  # nlos-topic
```

另有两个**名字不含 meter/budget** 但主题就是计量的 suite（`--list` 的 grep 抓不到，按文件名补录）：`nlos-runtime-tokio/tests/lifecycle_meter_aggregate.rs`（4 测：`aggregate_sums_backpressure_and_suspended_across_live_fibers:52`、`aggregate_on_empty_registry_is_zero:134`、`open_metrics_text_matches_expected_format_snapshot:146`、`open_metrics_text_on_zero_aggregate_reports_zero_values:173`，行号为该文件内 `fn` 位置）。

分布（按命中密度）：

| crate | suite / 主题 | 代表测试（源码 file:line） |
|---|---|---|
| **nlos-runtime-tokio** | `tests/activation_meter.rs`、`tests/activation_meter_scale.rs`、`tests/lifecycle_meter_aggregate.rs` — 纤维 CPU 激活/生命周期计量 | `terminal_metering_keeps_active_cpu_bounded_by_elapsed_wall_under_stress`（`activation_meter.rs:150`）、`ten_thousand_activation_meter_fibers_on_two_workers` / `one_hundred_thousand_...`（`activation_meter_scale.rs:249,255`）；`kill_receipt_consumption_feeds_activation_meter_dimensions`（`kill_receipt.rs:451`）；`aggregate_sums_backpressure_and_suspended_across_live_fibers`（`lifecycle_meter_aggregate.rs:52`） |
| **nlos-slice-k** | `tests/semantic_budget_raise.rs`、`tests/semantic_consume_ledger.rs` — 语义写者预算 | `exhausted_budget_raise_restores_the_bridge_with_the_raised_budget`（`semantic_budget_raise.rs:203`）、`exhausted_budget_fails_closed_typed_with_zero_semantic_writes`（`semantic_consume_ledger.rs:307`） |
| **nlos-topic** | cascade/delivery 预算 | `cascade_budget_spends_exactly_and_exhaustion_fails_closed`（`topic_cascade.rs:206`）、`budget_exhaustion_flips_quarantined_at_declared_attempts`（`topic_delivery_attempts.rs:272`） |
| **nlos-capability** | `tests/call_limit_ledger.rs` — 能力调用限额（衰减委托） | `delegated_budget_is_independent_not_pooled_with_parent`（`call_limit_ledger.rs:451`）；运行时 API：`CallLimitExhausted`（`src/lib.rs:92`） |
| **nlos-commit-coordinator** | 收敛步预算 | `converge_step_budget_returns_typed_error_and_resumes_from_prefix`（`convergence_guards.rs:235`） |
| **nlos-task** | 故障组 bulkhead 预算 | `bulkhead_failure_budget_poisons_faulty_task_group_...`（`c_fanout_hierarchical_bulkhead.rs:1074`） |
| **nlos-runtime-tokio（lib）** | 计量读 API | `pub fn inspect_lifecycle_meter_aggregate(&self) -> LifecycleMeterAggregate`（`src/lib.rs:1116`，含 `to_open_metrics_text` OpenMetrics 渲染，`:126`） |

**没有** 名为 budget/meter 的独立 crate；也没有 LLM token 预算这类领域计量——现有 meter 均是运行时 CPU/生命周期维度，budget 均是队列/级联/调用限额维度。

### 应用如何被安装/运行（nlos-package，`examples/sample-app/`）

- `package.manifest` 行式格式（`examples/sample-app/package/package.manifest:1-18`）：`package-id = <hex32>`、`version = 1.0.0`、`entry = <name> <role> <path>`（role ∈ executable/data）、`task = <template-id-hex32> <kind> <binding> <inputs> <outputs> <policy> <ceiling> [deps=<template-id>]`；五个 task 文件字段取文件内容（`package/tasks/*.txt`，如 ceiling 写 `resource-ceiling: 1 fiber, 1 operation, 1 artifact revision`）。
- 纯 CLI 闭环（`README.md:97-105`）：
  `nlos-package install app-v1.nlospkg --root $STATE` → `run <package-id> <entry-name> --root $STATE` → `update app-v2.nlospkg --root $STATE --compat same-major` → `uninstall <package-id> --root $STATE`；前置 `keygen --seed <hex64>`、`build <dir> --key <devkey>`、`conformance`、`verify --store --identity`（verify 输出 `VERIFIED <receipt-id>`）。
- `lifecycle.sh`（226 行）一键演练 10 步：keygen→build→conformance→verify→install→run（载荷执行回执 `payload_operation_id/payload_generation/payload_digest`）→update→gen2 复验→uninstall（活动门+teardown）→负门；机械化测试 `cargo test -p nlos-package --test sample_app_cli_lifecycle`（README:36）。
- 诚实边界（README:107-113）：载荷执行走 `nlos_slice_k::execute_application_payload` + `nlos-driver-mock` register→dispatch→complete 三边界回执，**不是任意 Wasm/ELF 沙箱 VM**；`background-service` 的 Os 替身 Unix 下是真实 `sleep 600` 子进程，Windows 走 Noop kill 适配器。

## 3. 探针（T5/T6）落地备忘/坑

1. **Windows 事件循环**：named pipe 客户端要求 Proactor loop（`local_rpc.py:170-176`）。`asyncio.run` 默认满足；别手工 `SelectorEventLoop`。
2. **单 in-flight**：一个 `LocalRpcClient` 同时只能有一个 `exchange`（`BACKPRESSURE`）；并发探针请每连接一个 client 或自旋等待。
3. **pipe 名唯一性**：conformance 惯例 `{pid}-{time_ns}` 后缀（`client.py:20-24`），避免机器级 pipe 命名空间碰撞。
4. **服务端启动握手**：`cargo run -q -p nlos-ipc --features conformance-server --bin nlos-ipc-echo|nlos-directory-chain --`，以 stdout 行 `READY` 为就绪信号（`client.py:59-67`）；首次 `cargo run` 会编译，conformance 里给了 60s ready 超时。
5. **daemon endpoint 推导**（绕过目录直连 system_control 时）：pipe 名 = `\\.\pipe\llmos-system-control-{auth|plain}-` + sha256(`"llmos/system-control-daemon/pipe-name/v1"` ‖ root 路径字节) 前 16 hex（`daemon.rs:244-265`）；Unix 是 `<root>/system-control-{auth,plain}.sock`。daemon 需 `daemon` feature 且另有 ADR-0011 challenge-response 握手（auth 端点）。
6. **envelope 必填**：schema `{name:"nlos.sabi.Envelope", major:1, minor:N}`、16B request_id、非空 service/method；side-effecting 方法还要 `request_context`（caller 三 ID + idempotency_key，见 §1.3）——探针造请求时直接照抄 `directory_chain.py:88-114` 的构造函数。
7. **channel 跨进程不可达**是当前最重要的 G0 事实：任何"QQ 消息收发走 llmos channel"的探针必须先解决宿主问题（进程内 Rust sidecar，或推动上游暴露 SABI 服务），纯 Python 侧无从发起。

## 附录：环境

- Windows 11 Pro 10.0.28000；git-bash；cargo workspace 37 crates；测试均在 dev profile。
- llmos 仓库保持只读（未改文件、未切分支、未拉取），复核 `git log -1` 与编排者记录一致（`d4a9d34 2026-10-28`）。
