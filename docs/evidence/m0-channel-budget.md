# M0 证据：Channel 端到端 + Budget 扣减探针（Task 6 · 含 W73/W80 闭环复验）

> **W73/W80 闭环复验更新（2026-11-01）**：上游 `llmos` 已在 W73 波次（`5e409b0f` / `f8b1687f` / `1c796212` / `2c6c9859`，当前只读复验快照 `48964eb5`）全量闭环 `L-001`~`L-007`，正式冻结第 9 个 SABI 通道 `nlos.sabi.ChannelControl` (`v1.0`)、挂载至 `system-control-daemon` 并交付 Python SDK `nlos_sdk.ChannelClient` (`v0.1.0`)。本仓已新增跨进程探针 `probes/llmos_channel_probe.py` 并实测 `EXIT_CODE=0`，**`Channel E2E（Python/跨进程收发）` 正式更新为 `PASS`**（见 §4）。
>
> **初探历史背景保留说明**：下文 §1–§3 完整保留 2026-10-28 基于 `llmos` `d4a9d34` 的 M0 初探记录与四项缺口分析（彼时 channel 尚无跨进程服务面，标记为 `BLOCKED`），以便对照审计上游 dogfood 反哺前后的演进闭环。
> 环境：Windows 11 Pro 10.0.28000；Python 3.12.10（`qq-agent` `.venv`）；`llmos` 全程只读引用（`git status --porcelain` 为空）。
> API 依据：`docs/evidence/m0-llmos-notes.md`（初探）+ `D:\llmos\sdk\python\nlos_sdk\channel.py`（W73/W80 复验）。

## 总判定

| G0 判据 | M0 初探基线（`d4a9d34`） | W73/W80 闭环复验结论（`48964eb5`） |
|---|---|---|
| Channel 服务端能力（进程内） | **成立**——31 个权威侧测试全绿（复跑 15 个指定 suite 用例 + T4 全量 31/0） | **PASS（成立）**——31/31 全绿，且已补齐 `[nlos-channel/<suite>]` 结构化轨迹输出（`L-007`） |
| Channel E2E（Python/跨进程收发） | **BLOCKED**——缺 proto、缺服务宿主注册、缺 SDK 绑定、缺能力签发设计（见 §2） | **PASS**——`ChannelControl` SABI v1.0 已挂载 `system-control-daemon`，`probes/llmos_channel_probe.py` 跨进程驱动 `InboundMessage` / `OutboundReply` 收发闭环 exit 0（见 §4.2） |
| Budget 服务端机制（进程内） | **成立**——扣减恰好一次、耗尽 fail-closed 零写、重放零重复扣费、meter 有 OpenMetrics 渲染（§3） | **PASS（成立）**——`nlos-resource` 升至 v8，新增五维 `llm_token_budgets` 表与 7/7 测试全绿（`L-005`，见 §4.3） |
| Budget SDK 侧可编程 | **否**——Python SDK 无任何 budget/meter API；且无 LLM token 计量维度 | **M1 同口径本地护栏先行**——上游五维账本已在 `nlos-resource` v8 落地，跨进程 SABI 预算服务面暴露前，M1 采用同口径（调用数/Token/微美元）本地 SQLite 护栏执行（见 §4.3） |

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

---

## 4. W73/W80 闭环复验（Channel E2E = **PASS** & 五维 LLM Token 预算对齐）

### 4.1 上游缺口闭环核对（对照 §2 四缺口）

§2 登记的四项缺口（归档于 `docs/evidence/dogfood-issues.md` `L-002`）已由 `llmos` W73 波次采用 canonical 路径 A 全量落地（复验工作树 `D:\llmos` HEAD `48964eb5`）：

1. **Schema 冻结**：新增 `schema/nlos/sabi/v1/channel_control.proto`（`nlos.sabi.ChannelControl` `v1.0`，单帧上限 `64 KiB`）及生成物 `gen/python/nlos/sabi/v1/channel_control_pb2.py`（提交 `5e409b0f`）。
2. **宿主挂载与目录注册**：`crates/nlos-system-control/src/channel_control.rs` 实现全量 9 方法（`create_channel`、`rotate_channel`、`enqueue`、`receive`、`ack`、`compact`、`inspect_channel`、`inspect_queue`、`register_queue_consumption`），挂载至 `system-control-daemon` 的 plain/auth 端点，并以 `service = "channel"` 注册入内置 `ServiceDirectory`。
3. **Python SDK 导出与打包**：新增 `sdk/python/nlos_sdk/channel.py`（`ChannelClient` 及类型化记录/异常）与 `sdk/python/pyproject.toml`（`nlos_sdk.__version__ == "0.1.0"`，提交 `f8b1687f`）。
4. **本地信任域与能力门禁**：`plain_socket` 依托本机 Named Pipe / `0600` UDS 本地信任域直接承载 SABI Envelope v1.1 请求（强制校验 caller 三元组、幂等键与回执）；上游跨语言一致性测试 `tests/conformance/ipc/channel_control.py` 6/6 全绿。

### 4.2 本仓跨进程探针 `probes/llmos_channel_probe.py` 实测 —— **PASS**

为验证 `qq-agent` 自身业务域类型在 `llmos` 持久化队列上的端到端表现，本仓编写并运行 `probes/llmos_channel_probe.py`：自举拉起 `D:\llmos\target\debug\system-control-daemon.exe`，通过 `nlos_sdk.ChannelClient` 创建 `inbound` 与 `outbound` 两条持久化 Channel（`capacity_bytes=65536`），分别将 `qq_agent.types.InboundMessage`（群聊 `@bot` 消息）与 `OutboundReply`（群聊回包）序列化入队、跨进程接收反序列化（100% 字段等价断言），并走完 `register_queue_consumption` → `ack` → `compact` → `inspect_queue` 队列清空闭环。

**实测命令**：

```powershell
D:\AI\qq-agent\.venv\Scripts\python.exe probes\llmos_channel_probe.py
```

**真实运行输出摘录**：

```text
[probe] llmos root: D:\llmos
[probe] nlos_sdk v0.1.0 import OK (service=channel, schema=nlos.sabi.ChannelControl)
[probe] starting system-control-daemon via prebuilt (D:\llmos\target\debug\system-control-daemon.exe)
[probe] daemon READY service=system_control root=C:\Users\ADMINI~1\AppData\Local\Temp\qq-agent-cc-probe-17024-1793531856587014900 auth_socket=\\.\pipe\qq-agent-cc-4280-1793531856587014900-1-auth plain_socket=\\.\pipe\qq-agent-cc-4280-1793531856587014900-2-plain principal=none operator_root_receipt=none

[probe] === Phase 1: 创建 inbound / outbound 持久化 Channel 与幂等重放 ===
[probe] channel 'inbound' created & idempotent replay OK: id=a7db9e77d61cab57... gen=1 cap=65536B
[probe] channel 'outbound' created & idempotent replay OK: id=fc33693a42c5c98c... gen=1 cap=65536B

[probe] === Phase 2: InboundMessage 跨进程队列闭环 ===
[probe] inbound enqueue OK: seq=1 bytes=338
[probe] inbound receive & decode 100% match: event_id=evt-m0-probe-001 session=group:70001 mentioned_bot=True
[probe] inbound register -> ack -> compact -> inspect_queue OK: removed=1 reclaimed=338B outstanding=0

[probe] === Phase 3: OutboundReply 跨进程队列闭环 ===
[probe] outbound enqueue OK: seq=1 bytes=193
[probe] outbound receive & decode 100% match: target=70001 at_user=10001 correlation_id=evt-m0-probe-001
[probe] outbound register -> ack -> compact -> inspect_queue OK: removed=1 reclaimed=193B outstanding=0

[probe] daemon stopped & temporary state root cleaned
[probe] ALL PASS: InboundMessage & OutboundReply 经 llmos ChannelControl SABI v1.0 跨进程收发闭环通过
EXIT_CODE=0
```

### 4.3 Budget 侧上游进展与 M1 护栏执行口径

- **上游 `nlos-resource` v8 已落地五维 LLM Token 账本（`L-005` 闭环）**：`llmos` W73（提交 `1c796212` + `2c6c9859`）已将 `crates/nlos-resource` 升级至 `SCHEMA_VERSION = 8`（`migrate_v8`），新增 `llm_token_budgets`、`llm_token_consumptions`、`llm_token_budget_revisions` 三张 `STRICT` 表及 SQLite 触发器，支持 `max_prompt_tokens`、`max_completion_tokens`、`max_total_tokens`、`max_api_calls`、`max_cost_micros_usd` 五维配额、幂等扣减（`consume_llm_tokens`）、在线调额与 OpenMetrics 导出（`crates/nlos-resource/tests/llm_token_budget.rs` 7 项测试全绿）。
- **跨进程服务面现状与 M1 护栏方案**：当前 `nlos_sdk` Python 客户端尚未暴露跨进程 SABI 预算扣减/查询 RPC 服务面（`sdk/python/nlos_sdk/` 导出面现为 `LocalRpcClient`、`ServiceDirectoryClient`、`ChannelClient`）。因此 M1 阶段 `qq-agent` 采用**与上游 `nlos-resource` v8 完全同口径（API 调用次数 / Prompt & Completion & Total Tokens / 微美元 `cost_micros_usd`）的本地 SQLite 预算护栏**执行频率限制、软限降级与硬限拒答，后续上游暴露跨进程预算 SABI 服务面后即可零模型变更平滑切换。
