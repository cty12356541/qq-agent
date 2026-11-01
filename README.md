# qq-agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

跑在 [`llmos`](https://github.com/cty12356541/llmos) 持久化通道之上的全异步小圈子 QQ 群友智能体，采用 **Pi-Agent 快慢双脑（`depth=1` SubAgent）**、**Use Chat as Tool 拟人多气泡交互** 与 **跨群「一魂多体」隐私隔离记忆中枢** 架构。

> **双模开箱即用**：默认支持 `llmos` 三进程故障隔离模式（`DOGFOOD-FULL`）；若本机未安装/编译 `llmos`，只需加上 `--direct-ws` 即可通过单进程内存总线零门槛启动全部 AI 群友能力。

---

## 核心特性

- **三进程故障隔离与 0 断连热更新（`DOGFOOD-FULL`）**
  - `supervisor` 自动编排并独立自愈 `llmos system-control-daemon`、`adapter`（OneBot v11 WS 桥接）与 `brain`（异步推理内核）三个操作系统进程。
  - 依托 `nlos.sabi.ChannelControl` v1.0 双持久化队列与连续高水位 `AckCoordinator`，热重启 `brain` 升级代码时 **WebSocket 零断连、在途消息零丢失**；同时内置 `--direct-ws` 单进程直连模式供轻量部署与快速联调。
- **Pi-Agent 异步内核与快慢双脑（Fast / Slow Dual-Brain）**
  - **快脑（社交人格层）**：默认关闭冗余思考链并开启高密度 `QQ_COMPACT_PROMPT`，通过 **Use Chat as Tool** 工具集驱动发言：`send_message`（拆 1–3 条短句气泡、指定引用回复、发送表情包）、`stay_silent`（话题硬插生硬时自然闭嘴）、`web_search`（秒级事实核查）、`remember_fact` / `recall_memories`。
  - **慢脑（`depth=1` 后台研究子 Agent）**：遇到需要多步搜索与抓页核实的复杂问题，快脑同轮先发一句自然口语垫话，同时调用 `spawn_subagent` 派生只读后台任务（支持全局并发上限与单群公平配额 `SUBAGENT_MAX_PER_SESSION`），查证完毕后自动异步追发结论。
- **多群 QoS 抢占调度与深度 OB11 协议感知**
  - **硬触发 0ms 抢占**：`SessionActor` 对私聊、`@bot`、引用回复与名字唤醒等硬触发赋予最高优先级，可在软触发（随机插话/兴趣词搭话）尚未发出气泡前无感抢占取消，彻底消除多群排队阻塞。
  - **富媒体与复杂消息解析**：原生支持 `[CQ:forward]` 合并转发递归展开、最近消息环缓 + `get_msg` 异步引用回溯、入站图片/动画表情后台线程缩放（默认 `512px` JPEG）多模态视觉识别，以及 `data/stickers/` 表情包热发现与斗图。
- **三层记忆（L1/L2/L3）与跨群「一魂多体」共享认知中枢**
  - **分层记忆**：L1 SQLite 会话滑窗 + L2 异步滚动摘要 + L3 作用域严格锁定的长期记忆与群友画像（`MemoryStore` WAL 线程安全）。
  - **一魂多体（Shared Cortex）与隐私防火墙**：跨群共享客观查证事实缓存（`SharedFact` TTL 缓存，同一热点多群免重复搜索）、跨群熟人公开兴趣印象与机器人自身脱敏近况；内置隐私防火墙硬性拦截任何群号、QQ 号、手机号、URL 或私密记忆跨群泄露。
- **纵深安全防御与成本护栏**
  - **防提示词注入**：所有群聊消息、潜水上文、搜索结果与抓取网页均以 `<untrusted_*>` XML 标签转义隔离。
  - **免 Key 竞速搜索与防 SSRF 抓页**：内置 Bing CN + DuckDuckGo HTML 并发竞速搜索（可选挂载 Tavily API），网页抓取强制经过 DNS 解析级私网/回环 IP 拦截（防 SSRF）。
  - **双协议模型与五维预算护栏**：同时支持 OpenAI `/chat/completions` 与 Anthropic `/v1/messages` 工具调用协议；按日统计调用次数、Token 与微美元花费，超软限自动降级 `LLM_CHEAP_MODEL`，超硬限走人设口吻拒答；管理员命令独立前置拦截，零 Token 消耗。

---

## 系统架构

```text
+-------------------------------+     OneBot v11 Forward WS (127.0.0.1:3001)
|  NapCat Docker (专用小号隔离)  | <=============================================> +------------------------+
+-------------------------------+                                                 |    NapCatWsAdapter     |
                                                                                  | ([CQ:forward] / 引用回溯 |
                                                                                  |  白名单审批 / 表情包编码)|
                                                                                  +-----------+------------+
                                                                                              |
                                           inbound / outbound 双队列解耦                        v
                           +------------------------------------------------------------------+---+
                           | MessageBus                                                           |
                           |  - 默认 DOGFOOD-FULL : ResilientLlmosBus <-> system-control-daemon    |
                           |                       (supervisor 守护 daemon + adapter + brain 三进程)|
                           |  - 轻量 --direct-ws  : DirectMemoryBus (单进程异步内存队列，免 Rust)   |
                           +------------------------------------------------------------------+---+
                                                                                              |
                                                                                              v
+----------------------------------------------------------------------------------------------------------+
| AgentKernel (全异步多会话协程池)                                                                          |
|  1. 防自环 & 三层人设 TOML 热同步 (default -> data/personas/<profile>.toml -> data/persona.local.toml)    |
|  2. 管理员白名单命令前置路由 (AdminCommandRouter: /status /stop /persona /quota 等 12 项指令，旁路免 LLM)  |
|  3. 高活跃群聊规则触发 + 输入防抖合批 + AmbientBuffer 潜水上文缓冲                                        |
|  4. 频率冷却与日预算护栏 (GuardrailManager: 调用数 / Token / 微美元账本，软限自动切 cheap_model)          |
|  5. SessionActor QoS 抢占调度 (硬触发 0ms 抢占未出声软触发)                                               |
|  6. 快脑 agent_loop (Use Chat as Tool: send_message / stay_silent / web_search / spawn_subagent ...)     |
|     ├── 慢脑 SubAgentRunner (depth=1 只读后台深查：web_search + SSRF 防护 fetch_webpage，单群公平配额)     |
|     └── MemoryStore 一魂多体中枢 (L1 滑窗 / L2 摘要 / L3 作用域记忆 + SharedFact 缓存 + 跨群隐私防火墙)   |
+----------------------------------------------------------------------------------------------------------+
```

---

## 快速上手

### 1. 环境准备

需要 **Python 3.12+** 与 [`uv`](https://docs.astral.sh/uv/)，克隆仓库并同步依赖：

```bash
git clone https://github.com/cty12356541/qq-agent.git
cd qq-agent
uv sync
```

### 2. 启动 NapCat 容器并扫码登录机器人小号

为避免与宿主机日常登录的个人主 QQ 冲突，`deploy/napcat/docker-compose.yml` 采用纯容器化隔离且**仅绑定本机回环 `127.0.0.1`**（详细图文步骤见 [`deploy/napcat/README.md`](deploy/napcat/README.md)）：

```bash
docker compose -f deploy/napcat/docker-compose.yml up -d
```

打开 `http://127.0.0.1:6099/webui`（Token 见 `docker logs qq-agent-napcat`），用**机器人专用小号**扫码登录，并在 **网络配置** 中新建 **WebSocket 服务器（正向 WS，端口 `3001`）**。

### 3. 配置 `.env`

复制环境变量模板并填入账号与模型密钥（`.env` 已在 `.gitignore` 中排除，切勿提交真实凭证）：

```bash
cp .env.example .env   # PowerShell: Copy-Item .env.example .env
```

核心必填项：
- `QQ_BOT_ID`：机器人小号 QQ 号
- `QQ_ADMIN_IDS`：管理员 QQ 号（多个用英文逗号分隔，如 `12345678,87654321`）
- `QQ_ALLOWED_GROUPS`：允许发言的群号白名单（逗号分隔；留空允许所在群）
- `NAPCAT_ACCESS_TOKEN`：NapCat 正向 WS Token（未设置可留空）
- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`：支持 DeepSeek、GLM 等任意 OpenAI `/chat/completions` 或 Anthropic `/v1/messages` 兼容接口（开启入站图片/表情包识别建议选用支持多模态的 Flash 模型）

### 4. 健康自检与启动

先运行无密钥健康自检（校验 `.env`、三层人设 TOML、表情包清单与 SQLite WAL 存储）：

```bash
# 单进程直连模式自检（无需本机 llmos）
uv run python -m qq_agent --direct-ws --dry-check

# llmos DOGFOOD-FULL 模式自检（自动探测同级 ../llmos 或 LLMOS_ROOT）
uv run python -m qq_agent --dry-check
```

正式启动机器人：

```bash
# 方式 A：轻量单进程直连模式（推荐初次体验，零外部守护进程依赖）
uv run python -m qq_agent --direct-ws

# 方式 B：llmos 三进程故障隔离守护模式（supervisor 托管 daemon + adapter + brain）
uv run python -m qq_agent
```

常用 CLI 参数：`--env-file <path>`、`--role {supervisor,adapter,brain,all-in-one}`、`--ws-url ws://127.0.0.1:3001`、`--persona config/persona.default.toml`。

---

## 自定义人设与表情包

### 三层人设热加载（免改开源跟踪文件）

`PersonaManager` 按以下优先级自动叠加并在文件变更时运行期热生效：

1. **基础模板层**：[`config/persona.default.toml`](config/persona.default.toml)（仓库内置通用群友模板「阿橘」）
2. **可选档案层**：`data/personas/<档案名>.toml`（可存放多套角色设定与 `[[few_shot_examples]]` 语料，群内或私聊发 `/persona use <档案名>` 即可秒级热切换）
3. **本地私有覆盖层**：`data/persona.local.toml`（持久化当前激活档案与本地微调项，已被 `.gitignore` 忽略）

### 表情包斗图图库

- 清单定义位于 [`config/stickers.toml`](config/stickers.toml)，对应图片文件放入本地 `data/stickers/` 目录（支持 `.png` / `.gif` / `.jpg` / `.webp`）。
- 即使不配置清单，直接将图片拖入 `data/stickers/` 也会被 `StickerRegistry` 自动发现并注入工具枚举供模型斗图使用。

---

## 管理员白名单命令速查

所有管理命令在进入触发器与大模型前由 `AdminCommandRouter` 独立拦截：**仅 `QQ_ADMIN_IDS` 白名单可执行，零 Token 消耗、绝不污染 LLM 会话上下文**（群聊可直接发 `/命令` 或 `@机器人 /命令`）：

| 命令 | 功能说明 |
|---|---|
| `/ping` | 检查机器人进程与消息链路在线状态（返回 `pong!`） |
| `/status` | 查看异步会话协程池、QoS 抢占计数、后台慢脑子任务、共享事实缓存与 ACK 水位快照 |
| `/stop [all]` | 立即中断当前会话（或 `all` 全部会话）的在途主推理与后台搜索子任务 |
| `/quota` | 查看当日已用调用次数、Token 用量与美元花费额度（含软限降级线与硬限阈值） |
| `/reset` | 清空当前群聊或私聊的 L1 会话上下文滑窗 |
| `/reload` | 立即热重载三层人设 TOML 配置并刷新唤醒词 |
| `/persona [use <档案名>]` | 查看当前角色与可选档案列表，或热切换至 `data/personas/<档案名>.toml` |
| `/remember [--public\|--private] <内容>` | 手动写入长期记忆（严格遵守群聊/管理员私密作用域隔离） |
| `/memories` | 查看当前会话可见的长期记忆条目与 ID |
| `/forget <ID>` | 按数字 ID 删除指定长期记忆 |
| `/profile <QQ号> <画像备注>` | 为指定群友设置跨会话画像备注 |
| `/help` | 查看管理员命令帮助说明 |

---

## 目录结构

```text
qq-agent/
├── config/                 # 默认人设模板 (persona.default.toml) 与表情包元数据清单 (stickers.toml)
├── deploy/napcat/          # 纯容器化 NapCat Docker Compose 编排（仅回环端口）与部署手册
├── docs/                   # 架构计划书与 llmos Dogfood 跨进程验证证据链
├── probes/                 # OB11 WebSocket 与 llmos IPC / ChannelControl 独立探针脚本
├── qq_agent/
│   ├── agent.py            # AgentKernel 主装配、防抖合批、快慢双脑编排与共享认知接线
│   ├── channel_bridge.py   # NapCatWsAdapter、DirectMemoryBus 与 ResilientLlmosBus 持久化通道桥
│   ├── cli.py              # CLI 入口与 supervisor / adapter / brain / all-in-one 多角色进程守护
│   ├── commands.py         # 前置管理员白名单命令路由器
│   ├── config.py           # 零第三方依赖 .env 与环境变量类型化配置加载器
│   ├── guardrails.py       # 频率冷却与五维同口径 SQLite 日预算软硬限护栏
│   ├── llmos_sdk.py        # 上游 llmos Python SDK 只读零污染加载器
│   ├── memory.py           # L2/L3 线程安全记忆库、SharedFact 缓存与跨群隐私防火墙
│   ├── ob11.py             # OneBot v11 事件解析、合并转发递归展开与出站编码器
│   ├── persona.py          # 三层人设 TOML 热加载、高密度 Compact 提示词与防注入包裹
│   ├── pi_core.py          # 异步 agent_loop、SessionActor QoS 抢占状态机与 depth=1 SubAgentRunner
│   ├── provider.py         # OpenAI / Anthropic 双协议多模态 Tool-Calling HTTP 客户端
│   ├── session.py          # L1 SQLite 会话滑动窗口存储
│   ├── stickers.py         # 表情包清单与本地图库热发现注册表
│   ├── trigger.py          # 词边界唤醒、引用免 @、追问窗口与活跃群聊规则触发器
│   ├── types.py            # 核心不可变数据契约
│   └── tools/              # Use Chat as Tool 工具套件、竞速联网搜索与防 SSRF 网页抓取
└── tests/                  # 全量单元与跨进程端到端集成测试套件
```

---

## 开发与测试

```bash
# 运行全量单元测试与端到端集成测试
uv run pytest -v
```

---

## 开源许可

本项目基于 [Apache License 2.0](LICENSE) 开源。
