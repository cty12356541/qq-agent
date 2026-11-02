# llmos Dogfood 问题台账

> **记录规则**（用户指令 2026-10-28）：dogfood 过程中出现的**每一个**问题都要记录在本文件，发现即追加，不筛选严重度、不因为"是我们自己环境的问题"而省略。
> 分类：`L` = llmos 侧问题（反馈上游）｜`E` = 环境问题（非 llmos 责任，记录备查）
> 每条必须带证据锚点（证据文件或命令输出）；状态跟踪到"已反馈/已修复"为止。
> 本文件是给 llmos 提 issue/立项的第一手素材，M0 结束时随 G0 一起review。

---

## L-001 | directory_chain.py 在 Windows 粗时钟粒度下确定性失败

- **日期**：2026-10-28（T5 IPC 探针）
- **场景**：跑官方 `tests/conformance/ipc/directory_chain.py`
- **症状**：bind 报 ACCESS_DENIED；测量 `time.time_ns()` 连续调用 99990/100000 次同值 → 两个 pipe 同名
- **根因**：`endpoint()` 用 `time_ns & 0xFFFFFFF`（28 位掩码）+ `label[0]`（标签首字符截断）构造 pipe 名，本机时钟粒度下熵不足撞名；服务端 `windows.rs create_server` 带 `first_pipe_instance`，同名第二个实例直接拒绝
- **上游位置**：`tests/conformance/ipc/directory_chain.py:47-48`（推导域串 `llmos/system-control-daemon/pipe-name/v1` + sha256 前 8 字节）
- **影响**：阻塞目录链 conformance 用例；IPC 本身不受影响（仓库外副本仅改 ROOT 与命名两处即 exit 0）
- **证据**：`m0-ipc.md` §1.2、`task-5-report.md`
- **建议**：掩码加宽或引入随机熵（pid/计数器）；label 全量参与
- **严重度**：中｜**状态**：**已修复（llmos W73 `ee6c42d4`）**——`tests/conformance/ipc/directory_chain.py` 引入进程内递增计数器 `_PIPE_SEQ` + PID + 全量 `label` + `secrets.token_bytes(8)` 构造 pipe 名，6/6 阶段实测全绿（上游证据：`docs/evidence/stage-c/w73-qq-agent-dogfood-closure.md`）

## L-002 | channel 无跨进程服务面——daemon 持有 Arc<ChannelAuthority> 但零方法分派

- **日期**：2026-10-28（T4 盘点 + T6 探针）
- **场景**：外部应用（Python SDK）尝试经 SABI 调用 channel 收发
- **症状**：ServiceDirectory 中无 channel 服务可协商；channel 为进程内 Rust 库 API
- **根因**：`nlos-system-control/daemon.rs:827` 持有 `pub channel: Arc<ChannelAuthority>` 但对 enqueue/receive/register_queue_consumption 零分派（全 crate grep 确证）；proto/pb2 全目录无 channel；SDK 导出面零命中
- **影响**：**dogfood 最大阻塞项**——消息通道这一核心服务对应用层不可达；qq-agent 被迫在"等上游补面（≈2-4 周）"与"自养 Rust sidecar（≈3-5 天）"间选择
- **证据**：`m0-channel-budget.md` §2 四缺口清单、`m0-llmos-notes.md` Q1
- **建议**：上游立项 channel SABI 服务面（proto + daemon 分派 + SDK 生成 + 能力签发），或官方提供 sidecar 模板
- **严重度**：高（对 dogfood 用户）｜**状态**：**已修复（llmos W73 `5e409b0f` + `f8b1687f`）**——冻结第 9 个 SABI 通道 `nlos.sabi.ChannelControl` (`v1.0`)，新增 `schema/nlos/sabi/v1/channel_control.proto`、`crates/nlos-system-control/src/channel_control.rs`（9 方法全量挂载至 `SystemControlDaemon` plain/auth 端点并注册入 `ServiceDirectory`）、Python SDK `nlos_sdk.ChannelClient` (`sdk/python/nlos_sdk/channel.py`) 及跨进程一致性测试 `tests/conformance/ipc/channel_control.py`（6/6 用例全绿）

## L-003 | Python SDK 无打包——source-in-tree，靠 sys.path 挂载

- **日期**：2026-10-28（T4 盘点）
- **症状**：应用侧必须 `sys.path.insert(0, r"<LLMOS_ROOT>/sdk/python")` 才能导入；无 PyPI 包、无版本号、无安装文档（上游 evidence 自述 SDK-2 CANDIDATE 无发布承诺）
- **影响**：应用与 llmos checkout 路径耦合；版本漂移不可追踪；CI/部署（VPS）复杂化
- **证据**：`m0-llmos-notes.md` Python SDK 节、`probes/llmos_ipc_ping.py` 头部
- **建议**：哪怕 dev 阶段也给个 `pip install -e sdk/python` 入口（pyproject 已有基础）+ 版本标记
- **严重度**：中｜**状态**：**已修复（llmos W73 `ee6c42d4` + `f8b1687f`）**——新增 `sdk/python/pyproject.toml` 与 `sdk/python/README.md`（支持 `pip install -e sdk/python`），内置 `_ensure_gen_python_path()` 自动解析仓内 `gen/python`，并在 `nlos_sdk.__init__` 导出 `__version__ = "0.1.0"`

## L-004 | Python Windows named pipe 强依赖 Proactor loop（非稳定公共 API）

- **日期**：2026-10-28（T4 盘点；上游 evidence B-SCHEMA-006 自述）
- **症状**：`create_pipe_connection` 是 CPython concrete loop 能力，不在 `AbstractEventLoop` 稳定表面；应用若用 SelectorEventLoop（如某些框架默认）直接连不上
- **影响**：Windows 应用侧隐性地雷；错误信息不会提示是 loop 类型问题
- **证据**：`m0-llmos-notes.md`（local_rpc.py Proactor 要求）、上游 `b-schema-006-typescript-python-ipc-clients.md` §5
- **建议**：SDK 侧启动时显式检测并报错提示（fail-fast with clear message）
- **严重度**：低-中｜**状态**：**已修复（llmos W73 `ee6c42d4`）**——`sdk/python/nlos_sdk/local_rpc.py` 在 Windows Named Pipe 连接前显式检查 `hasattr(loop, "create_pipe_connection")`，非 Proactor loop 下立即抛出 `IpcError("UNSUPPORTED_EVENT_LOOP", ...)` 并提示 `asyncio.WindowsProactorEventLoopPolicy()`，已由 `tests/conformance/ipc/client.py` case 4/4 验证

## L-005 | budget 六维资源守恒不含 LLM token/API 调用计量

- **日期**：2026-10-28（T4 Q3 + T6 §3.4）
- **症状**：meter/budget 全部是 CPU/队列/限额维度（activation/lifecycle meter、semantic budget、call_limit_ledger）；无 token 计量、无模型调用费维度
- **影响**：llmos 定位"管理 LLM Agent 的操作系统"，但 Agent 最核心的成本维度（模型调用）不在预算体系内——AI 应用宿主场景下预算执行不完整
- **证据**：`m0-llmos-notes.md` Q3、`m0-channel-budget.md` §3.4
- **建议**：作为 service family 或 meter 维度补 LLM 调用计量（ qq-agent 的"花费护栏"目前只能自建）
- **严重度**：中（对定位承诺）｜**状态**：**已修复（llmos W73 `1c796212` + `2c6c9859`）**——`crates/nlos-resource` 升级至 `SCHEMA_VERSION = 8`（`migrate_v8`），新增 `llm_token_budgets` / `llm_token_consumptions` / `llm_token_budget_revisions` 三张 `STRICT` 表及触发器，支持 prompt/completion/total tokens、API 调用次数与微美元成本五维配额执行、幂等扣减、在线调额与 OpenMetrics 导出（`crates/nlos-resource/tests/llm_token_budget.rs` 7 项测试全绿）

## L-006 | conformance 脚本断言型静默——无任何进度输出

- **日期**：2026-10-28（T5 评审复核发现）
- **症状**：`client.py` 无 print，跑完只有退出码；哪个用例、跑了多少、卡在哪全不可见
- **影响**：调试体验差；失败时定位成本高（T5 为此额外做了输出考古）
- **证据**：`m0-ipc.md` §1.1（"断言型静默脚本"注）、上游 `client.py:72-125`
- **建议**：加最小 per-case 输出（一行一用例）
- **严重度**：低｜**状态**：**已修复（llmos W73 `ee6c42d4`）**——`tests/conformance/ipc/client.py` 与 `directory_chain.py` 均已补齐逐用例/逐阶段 `PASS` 实时进度输出与汇总行

## L-007 | channel 集成测试断言型零日志

- **日期**：2026-10-28（T6 评审发现）
- **症状**：`nlos-channel/tests` 全目录仅 `channel_fault_injection.rs:458` 一处 `println!`；`--nocapture` 摘录不到投递/注册日志，只能贴测试源码替代
- **影响**：外部 dogfood 者无法从测试输出观察服务行为，证据链被迫绕路
- **证据**：`m0-channel-budget.md` §1 注
- **建议**：关键路径加 trace 日志（或测试输出钩子）
- **严重度**：低｜**状态**：**已修复（llmos W73 `98b9c3ac`）**——`crates/nlos-channel/tests/{queue_delivery,queue_consumption_registration,channel_authority}.rs` 全部补齐 `[nlos-channel/<suite>]` 结构化轨迹输出，支持 `-- --nocapture` 直接观测投递、消费登记、ACK、压实与世代轮转全生命周期

---

## E-001 | 本机 pypi.org 不可达 → 清华镜像进了 pyproject 默认源

- **日期**：2026-10-28（T5）
- **症状**：`uv add` 时 pypi 直连超时/极慢；清华镜像被写进 `pyproject.toml:17-19` default index
- **影响**：机器特定网络偏好固化进项目配置（他人/CI 会被带偏）；uv.lock 哈希锁保完整性
- **处置**：已留痕，网络良好时对 pypi.org 重锁（deferred minor，最终评审triage）
- **严重度**：低

## E-002 | 本机原无 uv / git 身份未配

- **日期**：2026-10-28（T1）
- **处置**：pip 装 uv 0.13.6；qq-agent 仓库本地配置身份（未动全局）；llmos 克隆走直连（gh-proxy 仅 git 推送需要）
- **严重度**：无（已解决，备查）

## E-003 | QQ NT 未安装 / NapCat 未部署

- **日期**：2026-10-28（T3 人工步骤）
- **状态**：OneKey 包已就位 `tools/napcat`，等用户选择安装方式并扫码
- **严重度**：阻塞 T3 实测（非 llmos 侧）

## E-004 | 宿主系统约束：主号需登录 QQ → NapCat 必须纯容器化

- **日期**：2026-10-28（用户提出）
- **约束**：主号要在宿主 Windows 登录 QQ 客户端，宿主不得安装 bot 专用 QQ（避免多实例/账号冲突）
- **处置**：弃用 OneKey（会装 QQ 进宿主）；改 WSL2 + Docker + napcat 容器（机器无 docker/WSL 发行版，需先打底）
- **影响**：T3 部署路径变长（系统级安装 + 可能重启一次）
- **严重度**：部署约束（非 llmos 侧）


---

## L-008 | system-control-daemon 单连接单次交换（serve_one 主动关流）且串行 accept 导致并发轮询 Named Pipe 建连抖动与超时

- **日期**：2026-11-01（Wave W4 多进程双通道压测与源码审计）
- **场景**：独立 `adapter` 进程与 `brain` 进程高频并发轮询同一 `system-control-daemon` 的 `inbound` / `outbound` 双持久化队列
- **症状**：客户端无法复用 `LocalRpcClient` 长连接；高并发收发与频繁 `compact` 下出现 Windows Named Pipe 建连抖动与瞬时 `IpcError("TIMEOUT", "IPC exchange timed out")`
- **根因**：`serve_plain_endpoint` 与 `serve_authenticated_endpoint` / `authenticated_serve_one_control_with_channel` 采用单连接单次交换模型（`serve_one` 完成一次请求/响应后立即主动关闭底层流），且 `accept_plain` 循环内串行 `.await` 处理单次交换；同时 `ChannelClient.compact()` 单次调用内部串行发起 3 次短连接 RPC（`inspect_queue` → `compact` → `inspect_queue`），导致过去每处理 1 条消息产生 5 次短连接风暴
- **上游位置**：`crates/nlos-system-control/src/daemon.rs:1200-1333`（重点 `1252-1332`）、`sdk/python/nlos_sdk/channel.py:632-676, 839-849`
- **影响**：高频轮询与双进程并发下 IPC 尾延迟抖动显著，瞬时建连/交换超时若不加舱壁会直接中断宿主循环
- **证据**：`<LLMOS_ROOT>/crates/nlos-system-control/src/daemon.rs:1252-1332`、`qq_agent/channel_bridge.py`、`tests/test_llmos_resilient_bus.py`
- **建议**：上游支持同一 `LocalRpcClient` 长连接多路/连续 `Exchange` 复用与并发 accept task 派发，并在 `CompactChannelResultWire` 直接内联返回压实前后字节差以消除额外 2 次 `inspect_queue`
- **严重度**：高｜**状态**：已定位并建立客户端过渡舱壁（`LlmosChannelBus._call_with_retry` 指数退避重连 + `compact_every_n=32` 批量压实，待上游长连接交换就绪后收敛）

## L-009 | WaitAuthority 已在 daemon 打开但未挂载 nlos.sabi.WaitControl v1.0 服务面与 Python SDK

- **日期**：2026-11-01（Wave W4 通道唤醒机制审计）
- **场景**：`ChannelClient` 消费端希望基于事件通知阻塞等待新入站/出站消息，而非空转轮询 `receive`
- **症状**：daemon 进程内部已实例化 `WaitAuthority`，但 IPC 侧无 `WaitControl` SABI 服务可调，Python `nlos_sdk` 亦无对应客户端，消费端被迫轮询 `ChannelClient.receive`
- **根因**：`SystemControlDaemon` 在启动时已打开 `WaitAuthority::open(options.root.join("waits"), Arc::clone(&channel))`，但 `serve_validated_with_channel` 与 `authenticated_serve_one_control_with_channel` 尚未挂载 `nlos.sabi.WaitControl v1.0` 服务分派，`sdk/python/nlos_sdk` 未导出 `WaitClient`
- **上游位置**：`crates/nlos-system-control/src/daemon.rs:1096-1099`、`crates/nlos-system-control/src/auth.rs:229-246`、`crates/nlos-wait`、`crates/nlos-wait-control`
- **影响**：空闲期仍需周期性发起 `receive` 查询，增加无效 IPC 往返与配合 `L-008` 的短连接建连开销
- **证据**：`<LLMOS_ROOT>/crates/nlos-system-control/src/daemon.rs:1096`、`qq_agent/channel_bridge.py`（`AdaptivePoller`）
- **建议**：在 `SystemControlDaemon` plain/auth 端点挂载 `nlos.sabi.WaitControl v1.0` 并在 `nlos_sdk` 导出异步 `WaitClient`
- **严重度**：中-高｜**状态**：已由 Python `AdaptivePoller(min_interval_s=0.015, max_interval_s=0.20, backoff_factor=1.6)` 过渡（待 `WaitControl` SABI 与 SDK 导出后彻底删除轮询器）

## L-010 | ResourceAuthority v8 五维 LLM Token 预算账本无跨进程 SABI 服务面与 Python SDK

- **日期**：2026-11-01（Wave W4 跨进程配额护栏审计）
- **场景**：外部 Python Agent 进程希望直接跨 IPC 调用 `llmos` `ResourceAuthority` v8 的 `llm_token_budgets` 五维账本执行预扣、实扣与 `/quota` 查询
- **症状**：`L-005`（W73）虽在 Rust `nlos-resource` 库内落地了五维配额表与触发器，且 `daemon.rs:1079` 已打开 `ResourceAuthority`，但跨进程 SABI 层无任何方法分派与 Python SDK 封装
- **根因**：`schema/nlos/sabi/v1/` 尚无 `resource_control.proto`（或 LLM 预算 SABI 方法），`serve_validated_with_channel` 零 `ResourceAuthority` 路由，`nlos_sdk` 无 `ResourceClient`
- **上游位置**：`crates/nlos-system-control/src/daemon.rs:1079-1081`、`crates/nlos-resource`、`crates/nlos-system-control/src/auth.rs:229-246`
- **影响**：外部多语言 Agent 无法消费 `llmos` 原生五维 Token/微美元预算权威账本，被迫维持应用侧本地过渡配额库
- **证据**：`<LLMOS_ROOT>/crates/nlos-system-control/src/daemon.rs:1079`、`qq_agent/guardrails.py`
- **建议**：暴露 `nlos.sabi.ResourceControl`（或随 `L-013` LLM Driver SABI 统一强制预扣/实扣）并提供 `nlos_sdk` 客户端
- **严重度**：中-高｜**状态**：待上游暴露 SABI 面（当前由 Python `GuardrailManager` 过渡，上游就绪后删除本地 `guardrails.sqlite3`）

## L-011 | system-control-daemon 未在 plain/auth 端点分派 ServiceDirectory，且 Python nlos_sdk 缺 PrincipalHandshake + Capability 客户端封装

- **日期**：2026-11-01（Wave W4 服务发现与身份握手审计）
- **场景**：外部多角色进程（`supervisor` / `adapter` / `brain`）启动时尝试通过 `ServiceDirectoryClient` 动态协商服务版本并经 `--auth-socket` 最小权限接入
- **症状**：对 daemon 端点调用 `ServiceDirectory` 返回未知服务失败；Python 侧无法直接连接 `--auth-socket` 完成 ADR-0011 `PrincipalHandshake` 与 Capability 令牌注入
- **根因**：`auth.rs:229-246` 的 `serve_validated_with_channel` 仅分派 `CHANNEL_CONTROL_SERVICE`（`"channel"`）与 `RecoverySystemControl`，未接入 `ServiceDirectory` 请求处理；同时 `sdk/python/nlos_sdk` 仅有 `LocalRpcClient` 明文帧交换与静态占位 `DEFAULT_PRINCIPAL_ID`，缺少 `--auth-socket` 挑战签名握手与 Capability 句柄协商封装
- **上游位置**：`crates/nlos-system-control/src/auth.rs:229-246`、`crates/nlos-ipc/src/handshake`、`sdk/python/nlos_sdk/service_directory.py`、`sdk/python/nlos_sdk/channel.py:77-82, 801-815`
- **影响**：Python 客户端只能走本机 `--plain-socket`（`AllowAllChannelControlAuthorizer`），无法在传输层强制分角色 Capability 最小权限隔离
- **证据**：`<LLMOS_ROOT>/crates/nlos-system-control/src/auth.rs:229-246`、`qq_agent/llmos_sdk.py`（`process_id_for_role`）
- **建议**：在 daemon 路由挂载 `ServiceDirectory`，并在 `nlos_sdk` 补齐 `AuthenticatedRpcClient`（`PrincipalHandshake` + Capability 槽位绑定）
- **严重度**：中｜**状态**：当前经 `--plain-socket` + 确定性角色 `process_id_for_role(role)` 区分身份过渡（待上游补齐后切入 `--auth-socket`）

## L-012 | sdk/python/pyproject.toml 漏打包 gen/python/nlos/sabi/v1/*_pb2.py，导致非 editable 安装后缺失 nlos 模块

- **日期**：2026-11-01（Wave W4 只读上游零污染接入实测）
- **场景**：在保持 `<LLMOS_ROOT>` 100% 严格只读（禁止向 `<LLMOS_ROOT>/sdk/python` 写入 `nlos_sdk.egg-info` 或 build 产物）的前提下安装或导入 `nlos_sdk`
- **症状**：非 editable 构建/安装 `sdk/python` 后，运行期报 `ModuleNotFoundError: No module named 'nlos'`；而 `pip install -e <LLMOS_ROOT>/sdk/python` 会试图在只读上游目录内创建 `.egg-info`
- **根因**：`sdk/python/pyproject.toml` 中 `[tool.setuptools.packages.find]` 仅配置 `where = ["."]` 与 `include = ["nlos_sdk*"]`，未将仓内 `../../gen/python/nlos/sabi/v1/*_pb2.py` 纳入 wheel 打包产物，`nlos_sdk/local_rpc.py` 的 `_ensure_gen_python_path()` 仅在源码树相对路径存在时生效
- **上游位置**：`sdk/python/pyproject.toml:15-17`、`sdk/python/nlos_sdk/local_rpc.py`
- **影响**：任何脱离 `llmos` 源码目录的 wheel 分发、容器镜像构建或严格只读源码仓挂载均无法通过标准 `pip install` 开箱即用
- **证据**：`<LLMOS_ROOT>/sdk/python/pyproject.toml:15-17`、`qq_agent/llmos_sdk.py`（`load_nlos_sdk` / `ensure_local_venv_pth`）
- **建议**：构建阶段将 `gen/python/nlos` 同步或映射进 `sdk/python` wheel 包（`include = ["nlos_sdk*", "nlos*"]`），确保独立 wheel 自包含全部 `*_pb2.py`
- **严重度**：中｜**状态**：已由 `qq_agent/llmos_sdk.py` 只读零污染加载器 + `.venv` 内 `.pth` 指针过渡（待上游修正 wheel 打包后移除路径注入）

## L-013 | crates/nlos-driver-llm-http（LlmEnforcementShim + glm-keypool + 五维预算预扣/实扣）未暴露跨进程 SABI 服务面

- **日期**：2026-11-01（Wave W4 模型驱动与密钥安全边界审计）
- **场景**：希望将外部 LLM HTTP 调用、多 Key 池轮转降级与五维 Token 预算强制执行完全下沉至 `llmos` 内核驱动层，使 Python 业务进程零明文持有 `LLM_API_KEY`
- **症状**：`crates/nlos-driver-llm-http` 已在 Rust 侧实现 W77-PRE-B `LlmEnforcementShim`、`glm-keypool` 调度及关联五维预算预扣/实扣闭环，但未暴露跨进程 SABI 服务面
- **根因**：`SystemControlDaemon` 未挂载 `nlos-driver-llm-http` SABI 服务，亦无对应 SABI proto 与 `nlos_sdk` 客户端，跨进程 Agent 无法通过 Capability 句柄委托内核发起受控 LLM 推理
- **上游位置**：`crates/nlos-driver-llm-http`、`crates/nlos-system-control/src/daemon.rs`
- **影响**：Python `brain` 进程仍需在运行期读取明文 `LLM_API_KEY` 并维护本地 `OpenAICompatProvider` 与 `guardrails.sqlite3`
- **证据**：`<LLMOS_ROOT>/crates/nlos-driver-llm-http`、`qq_agent/provider.py`、`qq_agent/guardrails.py`
- **建议**：在 `SystemControlDaemon` 暴露 `nlos.sabi.LlmDriver`（托管密钥池 + 强制 `LlmEnforcementShim` 五维预扣/实扣 + Tool Call 归一化透传）并提供 Python SDK
- **严重度**：高｜**状态**：当前由 Python `OpenAICompatProvider` + `GuardrailManager` 过渡（待 SABI 暴露后彻底删除本地 `guardrails.sqlite3` 与业务侧明文 `LLM_API_KEY`）

## L-014 | 不可信群消息触发外部工具（web_search / fetch_webpage）缺少跨进程 nlos-capability + nlos-wasm-sandbox 双层强沙箱委托执行 SABI 面

- **日期**：2026-11-01（Wave W4 联网工具与子 Agent 安全边界审计）
- **场景**：不可信群聊消息触发 Agent 调用联网搜索（`web_search`）与网页抓取（`fetch_webpage`）工具及后台研究子 Agent
- **症状**：`llmos` 虽已具备 `crates/nlos-capability` 与 W80 `crates/nlos-wasm-sandbox` 双层强沙箱能力，但缺少供外部进程跨 IPC 委托工具沙箱执行的 SABI 服务面
- **根因**：`SystemControlDaemon` 尚未导出受 Capability 衰减约束的跨进程沙箱工具执行 SABI 通道与 Python SDK，外部 Agent 无法将高风险外网抓取置于 `nlos-wasm-sandbox` 中隔离运行
- **上游位置**：`crates/nlos-capability`、`crates/nlos-wasm-sandbox`、`crates/nlos-system-control/src/daemon.rs`
- **影响**：在沙箱 SABI 面就绪前，Python 进程需自行承担外网 HTTP 抓取的 SSRF 防护、字节截断、超时熔断与子 Agent 递归深度隔离职责
- **证据**：`<LLMOS_ROOT>/crates/nlos-capability`、`<LLMOS_ROOT>/crates/nlos-wasm-sandbox`
- **建议**：暴露跨进程 `SandboxToolExecution` SABI 服务面（按会话/子 Agent `fiber_id` 绑定衰减后的出网域名/字节配额 Capability，在 `nlos-wasm-sandbox` 内执行并返回净化摘要）
- **严重度**：高｜**状态**：当前由 Python DNS 解析级 SSRF 防火墙、硬超时字节上限与 `depth=1` 只读子 Agent 护栏过渡（待沙箱 SABI 就绪后迁入 OS 级双层强沙箱）

## L-015 | channel 单帧 payload 上限（~64KB）迫使大语音 base64 出站走应用侧落盘中转

- **日期**：2026-11-02（TTS 语音管线联调）
- **场景**：`send_voice` 工具将 F5-TTS 合成 WAV 以 `base64://` URI 装入 `OutboundReply.voice_uri`，经 `nlos.sabi.ChannelControl` enqueue 跨进程发送
- **症状**：稍长的语音（合成 WAV base64 后 >16KB）会逼近乃至超过单帧 ~64KB payload 上限，存在帧超限发送失败或被截断的风险
- **根因**：`nlos-channel` 单条消息 payload 无分帧/大载荷传输原语，帧大小约束由应用侧自行保证
- **上游位置**：`crates/nlos-channel`（payload 大小上限与投递路径）
- **影响**：一切富媒体出站（语音、大图 base64）都必须在应用侧落盘再由 adapter 回灌，平添 IO、缓存目录清理与两端协议约定职责
- **证据**：`qq_agent/llmos_bus.py` `_spill_large_voice_uri` / `_rehydrate_voice_uri`（commit `dd4b8e2`）
- **建议**：channel 侧提供大载荷分帧重组传输，或基于 Capability 的文件句柄/共享路径传递原语
- **严重度**：低-中｜**状态**：应用侧已绕过——>16KB 语音 URI 落盘 `data/tts/cache`，adapter 收帧后回灌还原为 `base64://` 发送（用后即删）

---

## Wave W4 反夹生演进锚点对照表（Python 过渡层 → llmos 权威接管面）

> **反夹生纪律**：下表登记的每一项 Python 侧过渡实现均严守清晰协议边界，不与核心对话编排耦合；一旦对应 `llmos` 上游 crate 暴露跨进程 SABI 服务面与 `nlos_sdk` 绑定，即刻按「接管后删除动作」整块移除，绝不保留双轨夹生状态。

| 问题编号 | 当前 `qq-agent` Python 过渡实现 | 未来接管的 `llmos` 权威 Crate / SABI 面 | 接管后删除动作 |
| :--- | :--- | :--- | :--- |
| `L-008` | `LlmosChannelBus._call_with_retry` 瞬时 IPC 断连退避重建 + `compact_every_n=32` 批量压实（`qq_agent/channel_bridge.py`） | `crates/nlos-ipc` + `crates/nlos-system-control`（长连接多交换 `LocalRpcClient` 复用、并发 accept 与单次内联压实回执） | 切换为复用单一持久 `LocalRpcClient` 连接池，移除每次 RPC 建连/拆连容错补丁 |
| `L-009` | `AdaptivePoller` 空闲指数退避与活跃复位轮询器（`qq_agent/channel_bridge.py`） | `crates/nlos-wait` + `crates/nlos-wait-control`（`nlos.sabi.WaitControl v1.0` + `nlos_sdk.WaitClient`） | **彻底删除** `AdaptivePoller` 与 `receive` 轮询睡眠分支，改为 `WaitClient` 事件驱动阻塞唤醒 |
| `L-010` | `GuardrailManager` 本地日配额账本 `data/guardrails.sqlite3`（`qq_agent/guardrails.py`） | `crates/nlos-resource` v8（`llm_token_budgets` 五维账本跨进程 `nlos.sabi.ResourceControl` SABI 面） | **彻底删除**本地 `guardrails.sqlite3` 表结构与 SQLite 记账逻辑，改为 SABI 权威配额预扣/查询 |
| `L-011` | `--plain-socket` 本地连接 + `process_id_for_role(role)` 确定性角色身份标记（`qq_agent/llmos_sdk.py`） | `crates/nlos-system-control/src/auth.rs` + `crates/nlos-ipc/src/handshake`（`ServiceDirectory` 路由 + `--auth-socket` `PrincipalHandshake` + Capability SDK） | 停用业务进程 `--plain-socket` 直连，全面切入 `--auth-socket` 握手与按角色最小 Capability 句柄授权 |
| `L-012` | `load_nlos_sdk` 只读 `sys.path` 注入与 `ensure_local_venv_pth`（`qq_agent/llmos_sdk.py`） | `sdk/python/pyproject.toml`（自包含打包 `gen/python/nlos/sabi/v1/*_pb2.py` 的正式 `nlos-sdk` wheel） | 删除 `sys.path` 动态注入与 `.pth` 引导逻辑，改为 `pyproject.toml` 标准版本化依赖声明 |
| `L-013` | `OpenAICompatProvider` 直连 HTTP 模型推理 + 业务侧读取明文 `LLM_API_KEY`（`qq_agent/provider.py`） | `crates/nlos-driver-llm-http`（W77-PRE-B `LlmEnforcementShim` + `glm-keypool` + 五维预算预扣/实扣 SABI 服务面） | **彻底删除** Python 侧明文 `LLM_API_KEY` 持有、HTTP 重试传输层及本地 `guardrails.sqlite3` |
| `L-014` | Python 应用层 DNS SSRF 防火墙、抓取字节截断与 `depth=1` 只读子 Agent 护栏 | `crates/nlos-capability` + `crates/nlos-wasm-sandbox`（W80 双层强沙箱跨进程委托执行 SABI 服务面） | 删除 Python 进程内直接外网抓取客户端，将 `web_search` / `fetch_webpage` 全部委托至 OS/Wasm 强沙箱执行 |
| `L-015` | 出站前 `_spill_large_voice_uri` 落盘 + adapter `_rehydrate_voice_uri` 回灌（`qq_agent/llmos_bus.py`） | `crates/nlos-channel`（大载荷分帧重组或 Capability 文件句柄/共享路径传递原语） | 删除语音落盘缓存与回灌逻辑，`OutboundReply.voice_uri` 直接携带大载荷经 channel 原语传输 |
