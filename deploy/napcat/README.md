# NapCat 纯容器化部署指南（Windows 宿主隔离版）

本目录提供 `qq-agent` 接入层的纯容器化 NapCat（无头 NTQQ + OneBot v11）编排配置，严格契合宿主约束 **`E-004`**（详见 `docs/evidence/dogfood-issues.md`）。

## 为什么必须纯容器隔离（约束 `E-004`）

- **主号与 Bot 严格物理隔离**：宿主 Windows 日常需要登录个人主 QQ 客户端，**严禁**在宿主机安装 OneKey 或第二套 Bot 专用 QQ 客户端，避免进程锁冲突、本地配置目录互相污染以及同设备大小号连带风控。
- **仅回环暴露（零局域网暴露面）**：`docker-compose.yml` 将 OneBot v11 正向 WebSocket（`3001`）与 WebUI（`6099`）严格绑定在 `127.0.0.1:3001:3001` 与 `127.0.0.1:6099:6099`，仅宿主机本机进程可访问。
- **双命名卷重启免重扫**：使用 `napcat_qq_data`（挂载 `/app/.config/QQ`）与 `napcat_config`（挂载 `/app/napcat/config`）两个独立命名卷分离保存 NTQQ 登录凭证与 NapCat 网络配置，容器重启或镜像更新后无需重新扫码。

---

## Step 1：WSL2 与 Docker Desktop 前置准备

若本机尚未安装 WSL2 与 Docker Desktop，请在**管理员 PowerShell** 中执行：

```powershell
# 1. 启用 WSL2 并安装默认 Linux 发行版（首次安装后按提示重启 Windows）
wsl --install

# 2. 重启后确认默认版本为 WSL 2
wsl --status
```

随后安装并启动 **Docker Desktop for Windows**（设置中勾选 `Use the WSL 2 based engine`），在普通 PowerShell 中确认引擎就绪：

```powershell
docker version
docker compose version
```

---

## Step 2：准备 `.env` 并拉起 NapCat 容器

在仓库根目录复制环境变量模板（`.env` 已被 `.gitignore` 忽略，**切勿将真实账号或密钥提交入库**）：

```powershell
Copy-Item .env.example .env
```

按需在 `.env` 中填入机器人专用小号 `QQ_BOT_ID`（也可留空，稍后直接扫码任意小号），然后拉起容器：

```powershell
docker compose -f deploy/napcat/docker-compose.yml up -d
docker compose -f deploy/napcat/docker-compose.yml ps
```

验证宿主机端口仅绑定在 `127.0.0.1` 回环地址：

```powershell
netstat -ano | findstr ":3001 :6099"
```

---

## Step 3：查看容器日志获取 WebUI Token 并扫码登录

首次启动后查看容器实时日志：

```powershell
docker logs -f qq-agent-napcat
```

1. 在日志中找到类似输出：
   `[WebUi] WebUi User Panel Url: http://127.0.0.1:6099/webui?token=<你的随机Token>`
2. 在宿主浏览器打开该 URL（或访问 `http://127.0.0.1:6099/webui` 并粘贴 Token）。
3. 使用手机上已登录**机器人专用 QQ 小号**的客户端扫描 WebUI 二维码完成登录。
4. 登录成功后，确认 WebUI 首页显示小号在线，并将该小号拉入专用**测试群**。

---

## Step 4：配置 OneBot v11 正向 WebSocket（`:3001`）

在 NapCat WebUI 左侧菜单进入 **网络配置（Network）** → **新建** → **WebSocket 服务器（正向 WS）**：

| 配置项 | 填写值 | 说明 |
|---|---|---|
| **启用（Enable）** | 开启 | 激活正向 WS 服务端 |
| **Host** | `0.0.0.0` | 容器内监听全部网卡；宿主侧已由 `127.0.0.1:3001:3001` 锁死在回环网卡 |
| **Port** | `3001` | 对应 `.env` 中的 `NAPCAT_WS_URL=ws://127.0.0.1:3001` |
| **消息格式** | `Array` | 推荐使用消息段数组（兼容 CQ 码与文本抽取） |
| **Token（可选，推荐）** | 自定义随机串（或留空） | 若设置 Access Token，请同步填入根目录 `.env` 的 `NAPCAT_ACCESS_TOKEN=` |
| **心跳间隔** | `30000` | 默认 30 秒心跳（毫秒） |

点击保存后，配置立即写入 `napcat_config` 命名卷持久化生效。

> **提示**：执行下方 Step 5 的 M0 最小探针 `probes/ob11_listen.py` 时，因该探针不携带 Authorization 头，可先将 Token 留空完成探针验收，验收通过后再设置强随机 Token 并写入 `.env` 的 `NAPCAT_ACCESS_TOKEN` 供 `qq_agent` 正式运行使用。

---

## Step 5：用 `probes/ob11_listen.py` 验收收发并回填证据

在仓库根目录执行监听探针：

```powershell
# 1. 启动监听（事件将落盘至 data/ob11_events.jsonl）
uv run python probes/ob11_listen.py
```

1. 用个人主号在测试群内 `@小号` 发送一条 `"M0 测试"`，确认终端实时打印 `[msg] group user=... group=... text=...`。
2. 按 `Ctrl+C` 停止监听后，测试主动向测试群发送一条验证消息：

```powershell
uv run python probes/ob11_listen.py --send-group <测试群号> --text "M0 回发验证"
```

3. 确认测试群收到 `"M0 回发验证"`，且 `data/ob11_events.jsonl` 中记录了 `echo: "m0-send-1"` 回执。
4. 按 `docs/evidence/m0-napcat.md` 中的**抹敏要求**（将所有真实 QQ 号、群号、昵称替换为 `***`）回填实测记录。

---

## Step 6：启动 `qq_agent`

确认根目录 `.env` 已配置 `QQ_BOT_ID`、`QQ_ADMIN_IDS`、`QQ_ALLOWED_GROUPS`、`NAPCAT_WS_URL`、`NAPCAT_ACCESS_TOKEN` 与 `LLM_API_KEY` 后：

```powershell
# 默认模式（DOGFOOD-FULL：经 llmos ChannelControl SABI v1.0 跨进程持久化队列收发）
uv run python -m qq_agent

# 快速联调模式（--direct-ws：单进程内存总线直连 NapCat WS，无需启动 llmos daemon）
uv run python -m qq_agent --direct-ws
```

---

## 常用容器运维命令

```powershell
# 重启容器（验证命名卷免重新扫码）
docker compose -f deploy/napcat/docker-compose.yml restart

# 停止并移除容器（保留登录态与配置命名卷）
docker compose -f deploy/napcat/docker-compose.yml down

# 拉取最新 NapCat 镜像并无缝重建容器
docker compose -f deploy/napcat/docker-compose.yml pull
docker compose -f deploy/napcat/docker-compose.yml up -d
```

> **谨慎操作**：切勿随意附带 `-v` 执行 `docker compose down -v`，否则会删除 `qq_agent_napcat_qq_data` 与 `qq_agent_napcat_config` 命名卷，导致需要重新扫码与重配网络！
