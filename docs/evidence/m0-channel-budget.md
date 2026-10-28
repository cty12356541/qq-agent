# M0 证据：Channel 端到端 + Budget 扣减探针（Task 6）

> 交付物：本文件一个（简报 Step 2 的 BLOCKED 分支——T4 盘点确认 channel 是进程内 Rust 库、无 host 注册、Python 跨进程不可达，故**不写 `probes/llmos_channel_probe.py`**，缺口清单占满 Step 2 部分）。
> 环境：Windows 11 Pro 10.0.28000；llmos `d4a9d34`（只读，全程 `git status --porcelain` 为空）；cargo workspace 37 crates dev profile（增量构建秒级）。
> API 依据：`docs/evidence/m0-llmos-notes.md`（Task 4，行号已逐一复核仍然准确）。

## 总判定

| G0 判据 | 结论 |
|---|---|
| Channel 服务端能力（进程内） | **成立**——31 个权威侧测试全绿（本任务复跑 15 个指定 suite 用例 + T4 全量 31/0） |
| Channel E2E（Python/跨进程收发） | **BLOCKED**——llmos 缺 proto、缺服务宿主注册、缺 SDK 绑定、缺能力签发设计；缺口清单见 §2 |
| Budget 服务端机制（进程内） | **成立**——扣减恰好一次、耗尽 fail-closed 零写、重放零重复扣费、meter 有 OpenMetrics 渲染（§3） |
| Budget SDK 侧可编程 | **否**——Python SDK 无任何 budget/meter API（T4 结论，本任务复核）；且无 LLM token 计量维度 |

---

## 1. Channel 服务端证据（简报 Step 1）

### 1.1 `queue_delivery` suite —— 11/11 PASS

```bash
cd /d/AI/llmos
cargo test -p nlos-channel --test queue_delivery -- --nocapture
```

```text
running 11 tests
test queue_full_admission_rejects_before_any_durable_write ... ok
test ack_is_monotonic_idempotent_and_bounded_by_durable_maximum ... ok
test enqueue_idempotency_replays_and_conflicts_on_payload_or_fence_drift ... ok
test compact_never_trims_unconsumed_prefix_and_is_idempotent ... ok
test enqueue_receive_ack_cycle_is_ordered_and_drains ... ok
test ddl_guards_reject_entry_mutation_outside_compaction ... ok
test stale_fence_rejects_enqueue_but_old_generation_entries_stay_receivable ... ok
test unknown_schema_version_fails_closed ... ok
test restart_replays_entries_cursors_bookkeeping_and_decisions ... ok
test tampered_queue_state_fails_closed_as_corrupt_record ... ok
test v1_database_migrates_to_v2_preserving_channel_data ... ok

test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.06s
```

**投递链摘录**（`enqueue_receive_ack_cycle_is_ordered_and_drains`，`crates/nlos-channel/tests/queue_delivery.rs:92-110`）：三条 payload 依次 `enqueue` 得 sequence 1/2/3（单调、不可变），`receive` 非破坏读返回同序窗口，重复 receive 返回同一前缀（不推进游标、零写）：

```rust
let first = enqueue(&authority, &head, 1, b"first", 1_100);
let second = enqueue(&authority, &head, 2, b"second", 1_101);
let third = enqueue(&authority, &head, 3, b"third", 1_102);
assert_eq!(first.sequence, 1);  // ... 2, 3
let window = authority.receive(head.channel_id, 10).expect("receive");
assert_eq!(window, vec![first.clone(), second.clone(), third.clone()]);
// Receive advances no cursor and writes nothing: repeating it (even with
// a smaller limit) returns the same prefix of the same window.
```

注：该 suite 测试为断言型、无运行时日志输出（全 tests 目录仅 `channel_fault_injection.rs:458` 有一处 `println!`），故摘录以 runner 输出 + 源码片段为准。

### 1.2 `queue_consumption_registration` suite —— 4/4 PASS

```bash
cargo test -p nlos-channel --test queue_consumption_registration -- --nocapture
```

```text
running 4 tests
test registration_gates_fail_closed_without_side_effects ... ok
test consume_registration_mirrors_register_wait_and_replays ... ok
test projection_read_isolates_bindings_in_registration_order ... ok
test schema_v3_migration_is_idempotent_and_legacy_rows_decode_none ... ok
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
```

**注册链摘录**（`consume_registration_mirrors_register_wait_and_replays`，`queue_consumption_registration.rs:111-125`）：`enqueue_registered` 把产出纤维身份（ADR-0012）随行写入 entry：

```rust
let enqueued = match authority.enqueue_registered(
    request(&head, 1),
    ProducerRegistration {
        binding: binding(0x11),
        fiber_generation: Generation::INITIAL,
    },
) {
    Ok(EnqueueDecision::Enqueued(entry)) => entry,
    ...
};
assert_eq!(enqueued.binding, Some(binding(0x11)));
```

**结论**：channel 队列投递与消费注册在服务端（进程内 Rust，SQLite WAL/FULL 持久层）能力完备且测试证据充分。API 面即笔记 Q2 的 `ChannelAuthority` 七方法（`crates/nlos-channel/src/lib.rs:365` 起步：`open:376 / create_channel:429 / enqueue:664 / enqueue_registered:686 / receive:806 / ack:868 / compact:974 / inspect_queue:1071 / register_queue_consumption:1125`——本任务 grep 复核行号全部一致）。

---

## 2. Channel E2E = **BLOCKED** + 缺口清单（简报 Step 2 BLOCKED 分支）

**判定**：Python（或任何跨进程客户端）当前**无法**调用 channel 收发。不是 SDK 有 bug，而是 llmos 从 schema 到宿主到 SDK 整条服务面都不存在——channel 是纯进程内库 API（`ChannelAuthority` 结构体方法，非 RPC）。逐条缺口如下（每条附上游文件:行号；"该改哪"均为**建议**，本任务未动 llmos 任何文件）。

### 缺口 1：proto/schema 定义缺失

- `schema/nlos/sabi/v1/` 只有 8 个 proto：`driver_provider / envelope / operation_control / principal_handshake / service_directory / system_control / takeover_control / wait_control`——**无 channel**；`gen/python/nlos/sabi/v1/` 对应 8 个 `*_pb2.py`，无 `channel_pb2`（目录实列复核）。
- channel 的"请求/应答"是 Rust 类型（`CreateChannelRequest`/`EnqueueRequest`/`AckRequest` 等，`crates/nlos-channel/src/lib.rs:36-41, 109-116, 208-212`），无任何 wire schema。`enqueue` 的 payload 是裸 `Vec<u8>`（`:113`）——跨进程需要定义 bytes 上限与分帧语义。
- codegen 管线已就绪：`npm run schema:generate` = `buf generate && node tools/normalize-generated-schema.mjs`（`package.json:9-10`），`buf.gen.yaml` 从 `schema/` 目录产出 python（protocolbuffers:v33.4）+ typescript 两份生成物。补一个 `schema/nlos/sabi/v1/channel.proto` 后再生成即可，管线本身零改动。

### 缺口 2：服务宿主注册缺失（该挂哪个 crate）

- channel 的生产宿主只有两个，且都是**进程内直接打开**，不经任何 RPC：
  - `crates/nlos-slice-k/src/runtime.rs:205`：`ChannelAuthority::open(root.join("channel"))`；
  - `crates/nlos-system-control/src/daemon.rs:1067-1068`：daemon 打开 `root.join("channel")`，字段 `pub channel: Arc<ChannelAuthority>`（`daemon.rs:827`）。**但 daemon 只把它喂给 TopicAuthority/WaitAuthority 内部用（`daemon.rs:1071, 1075`），control 面没有任何 channel 方法分派**（grep `nlos-system-control/src` 无 enqueue/receive/register_queue_consumption 处理器；channel 命中全是 topic 的 `channel_id` 字段）。
- 服务目录侧：全仓唯一 `ServiceDirectory` 实现是 `SnapshotDirectory`（`crates/nlos-service-directory/src/lib.rs:82`，内存快照、≤256 注册 `lib.rs:27`），**只有 conformance bin 在用**——`crates/nlos-ipc/src/bin/nlos-directory-chain.rs:31, 101-135` 以 `ServiceRegistration{candidate, endpoint}` 注册 `operation` + `operation_control` 两个服务。生产上没有任何 host 把自己注册进目录；`system_control` 自己的 endpoint 是约定推导（Windows pipe 名 = `\\.\pipe\llmos-system-control-{auth|plain}-<sha256 前缀16hex>`，`daemon.rs:226-268`），不走目录。
- **建议挂载点**（二选一）：
  - (a) `nlos-system-control` daemon——它已持有 `Arc<ChannelAuthority>`（`daemon.rs:827`），在 `SystemControlAuthorizer`（`lib.rs:222-233`）体系里加 channel 方法分派最顺，复用其 serve 循环（`serve_plain_endpoint`，`daemon.rs:1243`；`serve_authenticated_endpoint`，`:1178`）与 `nlos_ipc::serve_one`（`crates/nlos-ipc/src/lib.rs:493`）；
  - (b) 独立 channel host bin（照 `nlos-directory-chain.rs` 模板）+ `SnapshotDirectory` 注册——与 daemon 解耦，但要自己解决目录实例的常驻问题（SnapshotDirectory 是构造期定死快照，无动态注册 API）。

### 缺口 3：SDK 侧生成与绑定缺失

- `sdk/python/nlos_sdk/` 共 4 文件（`__init__.py / common.py / local_rpc.py / service_directory.py`），导出面（`__init__.py:1-29`）无 channel；`grep -r channel sdk/python/` **零命中**（复核）。
- 缺三件：`channel_pb2`（缺口 1 的 buf 再生成产物）；一个 `channel.py` 客户端模块（模板即 `service_directory.py`——envelope 组装 + 方法封装 + `MethodSemantics` 声明，side-effecting 方法带 idempotency_key，Python 侧校验器 `common.py:33-37` 已就绪）；`__init__.py` 导出。
- 底座没问题：`LocalRpcClient` 的 named-pipe 往返 T5 已实测通过（`docs/evidence/m0-ipc.md`，PASS）。

### 缺口 4：鉴权/能力怎么过（设计缺口，非纯代码缺口）

现有两条通道的先例：
- plain endpoint 靠本地信任域：Unix owner-only 0600 socket / Windows machine-local named pipe，in-transport peer gate 直接放行（`daemon.rs:621-626` 注释 + `AllowPeer`，`:627-633`）；
- auth endpoint 走 ADR-0011 principal handshake，之后每请求过 `ControlCapabilityPolicy`：要求 `capability_handles == [{slot: CONTROL_CAPABILITY_SLOT, generation: ...}]` 固定槽位（`daemon.rs:589-619`）。

channel 暴露为 SABI 服务需回答（上游无现成答案）：
- 谁给外部客户端签发 channel 能力 handle？`nlos-capability` 有衰减委托与调用限额台账（`crates/nlos-capability/tests/call_limit_ledger.rs:451` 委托预算不与父池合并；运行时错误 `CallLimitExhausted`，`src/lib.rs:92`），但把"channel 操作"接进该能力体系是新的设计项；
- 能力与 `channel_id` / 消费者 `binding`（`ExecutionFiberId`）的绑定关系——`register_queue_consumption` 六道 fail-closed 门（`lib.rs:1096-1115`）目前信任进程内调用者，跨进程后 binding 身份需要与 principal 对应；
- `enqueue`/`ack`/`register` 均为 side-effecting，SABI 语义强制 idempotency_key（Rust `validate_sabi_request_context` 与 Python `common.py:33-37` 同款约束），服务面要为每个方法声明语义（`MethodSemantics`）。

### 两条补齐路径取舍（供 owner 决策 ≤4 周与否）

| | A：上游补 SABI 服务面 | B：qq-agent 自带 Rust sidecar |
|---|---|---|
| 做法 | 新 `channel.proto` + buf 再生成 → daemon（或独立 host）挂 channel 方法 → 目录注册 → Python `channel.py` 客户端 → 能力策略设计 | qq-agent 仓库内写小 bin：依赖 `nlos-channel` crate，进程内 `ChannelAuthority::open` + `nlos_ipc::serve_one`（`nlos-ipc/src/lib.rs:493`，`PeerAuthorizer`，`:270`）暴露 create/enqueue/receive/ack/register 五方法；Python 直连 pipe（endpoint 自约定） |
| 优点 | canonical；所有客户端受益；能力/审计/目录体系统一；不产生私有协议债 | 完全自控、不等上游排期；`nlos-ipc` serve 件与 `nlos-directory-chain.rs` 模板现成；llmos 只读依赖不 fork |
| 缺点 | 跨 schema/host/directory/sdk 四层，且缺口 4 是设计决策——llmos 维护者约定创造性工作须 brainstorming/spec 先行（`CLAUDE.md` §9），排期不可控；等落地期间 M0 完全停滞 | 私有协议面，上游真出 SABI channel 面后需迁移；绕开目录与能力体系，安全面（pipe ACL）自负；qq-agent 需维护 Rust 构建链 |
| 粗估工作量 | proto+codegen 1-2 天；host 挂载与方法分派（≈9 方法 × envelope 语义/幂等/错误映射 + 测试）3-5 天；目录/发现 1-2 天；SDK 模块 + conformance 2-3 天；**小计 2-4 周纯开发**，另有能力策略设计与上游评审（+1-2 周）及排期等待 | 3-5 天（serve 模板现成；Python 侧抄 `service_directory.py` 结构；风险点仅 Windows pipe 命名与并发——T5 已趟过 Proactor/单 in-flight 的坑） |
| ≤4 周判据 | 总量贴线（2-4 周开发）但**排期与设计评审不可控**，按上游节奏大概率超 | **≤1 周**，稳达标 |

**建议**：M0 判据只要求"channel 收发链路可被外部进程驱动"→ 选 B 先过 G0，同时把缺口 1-4 清单提给 llmos owner 立上游 lane；若 M0 定义坚持"走 canonical 服务面"→ 只能 A，接受排期风险。

---

## 3. Budget 证据（简报 Step 3）

按笔记 Q3 定位跑三个代表性 suite（`--nocapture`，逐个跑）：

### 3.1 `nlos-runtime-tokio --test lifecycle_meter_aggregate` —— 6/6 PASS

```text
running 6 tests
test open_metrics_text_on_zero_aggregate_reports_zero_values ... ok
test open_metrics_text_converts_durations_to_fractional_seconds ... ok
test open_metrics_text_preserves_nanosecond_and_large_second_precision ... ok
test open_metrics_text_matches_expected_format_snapshot ... ok
test aggregate_on_empty_registry_is_zero ... ok
test aggregate_sums_backpressure_and_suspended_across_live_fibers ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s
```

要点：生命周期 meter 有聚合读 API 与 OpenMetrics 文本渲染（`crates/nlos-runtime-tokio/src/lib.rs:1116` `inspect_lifecycle_meter_aggregate`，`:126` `to_open_metrics_text`），空注册表归零、跨活跃纤维聚合 backpressure/suspended、格式快照稳定。

### 3.2 `nlos-slice-k --test semantic_budget_raise` —— 6/6 PASS

```text
running 6 tests
test same_limit_raise_replays_as_a_typed_noop ... ok
test raise_can_decrease_the_budget ... ok
test midwindow_no_active_root_fails_closed_and_recovers ... ok
test runtime_raise_persists_across_reopen ... ok
test old_root_handle_is_generation_fenced_but_the_bridge_is_unaffected ... ok
test exhausted_budget_raise_restores_the_bridge_with_the_raised_budget ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.32s
```

### 3.3 `nlos-slice-k --test semantic_consume_ledger`（扣减核心）—— 3/3 PASS

```text
running 3 tests
test bridge_success_charges_the_budget_exactly_once ... ok
test exhausted_budget_fails_closed_typed_with_zero_semantic_writes ... ok
test rerun_and_reopen_replays_charge_nothing ... ok
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.73s
```

三个测试名即扣减语义的完整陈述：**成功恰好扣一次**；**耗尽 fail-closed 且零语义写**（`semantic_consume_ledger.rs:307` 起）；**重放/重启重跑零重复扣费**。

### 3.4 两档判定（如实记录）

| 档 | 判定 | 依据 |
|---|---|---|
| 服务端机制成立？ | **成立**（进程内） | §3.1-3.3 全绿 + 笔记 Q3 的 21 个 budget/meter 测试分布（nlos-topic 级联预算、nlos-capability 调用限额台账等）均为服务端 Rust 测试证据 |
| SDK 侧可编程？ | **否** | Python SDK 无任何 budget/meter/reserve/consume API（`sdk/python/nlos_sdk/` 4 文件零命中）；budget 全部在进程内由各权威（slice-k 桥、topic、capability 台账）自行扣减，无跨进程扣减接口——与 channel 同病（无服务面）。另：**无 LLM token 计量维度**（现有 meter=CPU 激活/生命周期，budget=队列字节/级联/调用限额），QQ 机器人要按 token 限额需上游新增维度或 qq-agent 自建计量 |

---

## 复核记录

- llmos 只读自查：任务前后 `git -C D:\AI\llmos status --porcelain` 均为空；HEAD `d4a9d34` 未动。
- 笔记行号抽查（ChannelAuthority 方法 11 处、daemon.rs 端点推导/能力策略、service_directory.proto、buf.gen.yaml）全部与 `m0-llmos-notes.md` 一致或已在文中更新为实测行号。
