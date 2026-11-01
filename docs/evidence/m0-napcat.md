# M0 证据：NapCat 接入 + OB11 收发探针（Task 3）

> 状态：**已完成实机容器化接入与 OB11 握手/动作往返验证（2026-11-01）**。
>
> 探针脚本：`probes/ob11_listen.py`（WS 地址 `ws://127.0.0.1:3001`；事件落盘 `data/ob11_events.jsonl`，已入 `.gitignore`）。

## 实测值

| 项 | 值 |
|---|---|
| 部署形态 | Docker Desktop (`29.9.2` / WSL `2.7.14.0`, WSL2 后端命名卷隔离存储) + `deploy/napcat/docker-compose.yml` (`mlikiowa/napcat-docker:latest`) |
| NapCat 版本 | `NapCat.Core 4.18.42` (`protocol_version: v11`) |
| QQ 登录方式 | 机器人专用小号通过本机回环 WebUI (`127.0.0.1:6099`) 登录成功，并固化 `autoLoginAccount` 与持久化卷 (`qq_agent_napcat_qq_data` / `qq_agent_napcat_config`)；与宿主机 Windows 主号 QQ 完全隔离（契合 `E-004`） |
| OneBot 11 正向 WS 端口 | `127.0.0.1:3001`（`netstat -ano \| findstr :3001` 确认仅 `127.0.0.1:3001 LISTENING`，未暴露 `0.0.0.0`） |

## 事件原文（已抹敏）

> **抹敏说明**：所有账号、群号与昵称均已替换为 `"***"`，原始落盘 `data/ob11_events.jsonl` 已由 `.gitignore` 排除。

### 1. 连接握手（`meta_event.lifecycle.connect` & `get_status`）

```json
{"time":1793537487,"self_id":"***","post_type":"meta_event","meta_event_type":"lifecycle","sub_type":"connect"}
```

```json
{"status":"ok","retcode":0,"data":{"online":true,"good":true,"stat":{}},"message":"","wording":"","echo":"stat-1"}
```

### 2. 出站动作回执（`echo=m0-send-1` 协议通路验证）

首次对未入群目标群发 `send_group_msg`（`echo="m0-send-1"`）时，OneBot v11 -> NTQQ 内核 `NodeIKernelMsgService/sendMsg` 完整返回业务级回执（验证请求/回执双向 JSON 通路畅通；待管理员将机器人小号加好友并拉入白名单群后即可正常收发消息）：

```json
{"status":"failed","retcode":1200,"data":null,"message":"EventChecker Failed: NTEvent serviceAndMethod:NodeIKernelMsgService/sendMsg ... result: 110","echo":"m0-send-1","stream":"normal-action"}
```

## 遇到的问题与处置

1. **Docker Desktop 与 WSL2 数据目录自定义隔离**：安装 Docker Desktop 时支持显式指定 `--installation-dir=<自定义程序目录>` 与 `--wsl-default-data-root=<自定义数据卷目录>`，将镜像与 WSL2 虚拟盘置于独立数据分区，避免占用系统盘空间。
2. **无界面容器内机器人加好友/入群审批**：由于容器内无桌面 GUI 点击「同意」，已在 `NapCatWsAdapter` 中增加**仅针对白名单管理员 (`QQ_ADMIN_IDS`) 与白名单群 (`QQ_ALLOWED_GROUPS`)** 的好友申请/入群邀请自动审批，陌生人请求一律不自动放行。
3. **本机 3-Key GLM 聚合路由协议适配**：扩展 `OpenAICompatProvider` 原生识别 `/anthropic` (`/v1/messages`) 端点并自动过滤 `thinking` 推理块，仅将 `text` 回复块投递到 QQ 会话。
