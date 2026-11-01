# qq-agent

跑在 `llmos` 上的自用小圈子 QQ 群友智能体。完整设计方案见 `docs/2026-10-28-qq-agent-计划书.md`，M0 跨进程验证证据见 `docs/evidence/`。

## 里程碑状态

| 里程碑 | 状态 | 说明 |
|---|---|---|
| M0 前置验证 | 已完成（G0 = DOGFOOD-FULL） | `llmos` `nlos.sabi.ChannelControl` v1.0 跨进程收发闭环、本机容器隔离 NapCat 与五维 Token 预算口径对齐全部通过 |
| M1 最小可用内核 | 已完成（待本机 Docker NapCat 扫码联调） | 双模消息总线（`LlmosChannelBus` / `DirectMemoryBus`）、防注入人设热加载、L1 SQLite 会话滑窗、一级规则触发器、日预算护栏、白名单管理命令与 CLI 全链路测试就绪 |
| M2 群聊增强与记忆演进 | 规划中 | 二级语义搭话判断、长期记忆检索与工具扩展 |

---

## 系统架构简图

```text
+---------------------------+        OneBot v11 Forward WS (127.0.0.1:3001)
|  NapCat Docker (小号隔离)  | <=================================================> +-------------------+
+---------------------------+                                                     |  NapCatWsAdapter  |
                                                                                  +---------+---------+
                                                                                            |
                                   inbound / outbound 双队列解耦                             v
                     +----------------------------------------------------------------------+---+
                     | MessageBus                                                               |
                     |  - 默认 DOGFOOD-FULL : LlmosChannelBus <-> ManagedLlmosDaemon (SABI v1.0)|
                     |  - 调试 --direct-ws  : DirectMemoryBus (单进程内存单调序号队列)           |
                     +----------------------------------------------------------------------+---+
                                                                                            |
                                                                                            v
+-----------------------------------------------------------------------------------------------+
| AgentKernel (严格 7 级流水线)                                                                  |
|  1. 防自环过滤 (bot_qq / self_id)                                                              |
|  2. 人设 TOML 热同步 (PersonaManager: 角色唤醒词 + 卖萌限流/超额拒答文案)                       |
|  3. 群白名单前置门禁 (QQ_ALLOWED_GROUPS)                                                       |
|  4. 管理员白名单命令路由 (AdminCommandRouter: /ping /quota /reset /reload /help，旁路不进 LLM) |
|  5. 一级零成本规则触发器 (RuleTrigger: 私聊直达 / 群聊 @bot / 角色名与别名关键词)               |
|  6. 频率与日预算护栏 (GuardrailManager: 用户冷却 / 群滑窗限流 / SQLite 调用数·Token·微美元账本)|
|  7. 防注入上下文拼装 (<untrusted_chat_message> 标签转义 + SessionStore L1 滑窗) -> LLMProvider |
+-----------------------------------------------------------------------------------------------+
```

---

## 快速上手（4 步启动）

### ① 启动 NapCat 容器并扫码登录小号

宿主 Windows 可正常登录个人主号，机器人专用小号在纯 Docker 容器内隔离运行（端口仅绑定 `127.0.0.1`）：

```powershell
docker compose -f deploy/napcat/docker-compose.yml up -d
```

打开 `http://127.0.0.1:6099` 进入 NapCat WebUI，使用**机器人专用 QQ 小号**扫码登录，并确认开启 OneBot v11 **正向 WebSocket Server**（监听容器内 `3001` 端口，若配置了 Access Token 请记录下来）。

### ② 复制 `.env.example` 为 `.env` 并填写配置

```powershell
Copy-Item .env.example .env
```

编辑 `.env`（已被 `.gitignore` 保护，切勿提交入库），填入关键配置：
- `QQ_BOT_ID`：机器人小号 QQ 号
- `QQ_ADMIN_IDS`：管理员 QQ 号（多个用英文逗号分隔，如 `12345678,87654321`）
- `QQ_ALLOWED_GROUPS`：允许发言的目标测试群号（逗号分隔；留空表示允许所有所在群）
- `LLM_API_KEY`：OpenAI 兼容接口密钥（默认接入 DeepSeek `https://api.deepseek.com/v1`，模型 `deepseek-chat`）

### ③ 运行无密钥健康自检（`--dry-check`）

无需联网调用大模型，一键校验 `.env` 配置、人设 TOML 语法、SQLite WAL 存储以及 `llmos` `nlos_sdk.ChannelClient` 可用性：

```powershell
uv run python -m qq_agent --dry-check
```

### ④ 正式启动机器人

- **默认 `DOGFOOD-FULL` 模式**（自动拉起并托管 `llmos` `system-control-daemon` 持久化通道，或连接 `LLMOS_PLAIN_ENDPOINT`）：
  ```powershell
  uv run python -m qq_agent
  ```
- **单进程直连调试模式**（使用进程内 `DirectMemoryBus`，无需本机 `llmos` 守护进程）：
  ```powershell
  uv run python -m qq_agent --direct-ws
  ```

常用可选 CLI 参数：`--env-file <path>`、`--ws-url ws://127.0.0.1:3001`、`--persona config/persona.default.toml`。

---

## 管理员白名单命令速查表

所有管理命令在进入触发器与大模型前由 `AdminCommandRouter` 独立拦截处理：**仅 `QQ_ADMIN_IDS` 白名单内的 QQ 号可执行**（非管理员触发会直接收到权限不足提示），**零 Token 消耗、绝不写入 LLM 会话上下文**。群聊中可直接发送 `/命令` 或 `@机器人 /命令`：

| 命令 | 功能说明 |
|---|---|
| `/ping` | 检查机器人进程与消息链路在线状态（返回 `pong!`） |
| `/quota` | 查看当日已用调用次数、Token 用量与微美元花费（含软限降级线与硬限额度） |
| `/reset` | 清空当前群聊（`group:<群号>`）或当前私聊（`private:<QQ号>`）的 L1 会话上下文滑窗 |
| `/reload` | 立即热重载人设 TOML 配置，并同步刷新角色名/别名唤醒词与限流拒答文案 |
| `/help` | 查看管理员命令帮助说明 |

---

## 开发与测试

```powershell
# 运行全量单元与端到端集成测试（含 Fake NapCat OB11 WS 与真实 llmos ChannelControl 跨进程往返）
uv run pytest -v
```
