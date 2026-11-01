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
- **症状**：应用侧必须 `sys.path.insert(0, r"D:\AI\llmos\sdk\python")` 才能导入；无 PyPI 包、无版本号、无安装文档（上游 evidence 自述 SDK-2 CANDIDATE 无发布承诺）
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
