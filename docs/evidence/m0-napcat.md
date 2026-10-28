# M0 证据：NapCat 接入 + OB11 收发探针（Task 3）

> 状态：**模板待填**。以下"实测值"均为 `【待人工实测】` 占位，待人工完成 Step 1（安装并启动 NapCat、小号扫码登录、开启 OB11 正向 WS 3001）与 Step 3（实机收发验证）后回填。
>
> 探针脚本：`probes/ob11_listen.py`（CLI 见脚本 docstring；WS 地址常量 `ws://127.0.0.1:3001`；事件全量落盘 `data/ob11_events.jsonl`）。

## 实测值

| 项 | 值 |
|---|---|
| NapCat 版本 | 【待人工实测】 |
| QQ 登录方式 | 【待人工实测】（预期：小号扫码登录，WebUI 可见登录态） |
| OneBot 11 正向 WS 端口 | 【待人工实测】（预期：3001；`netstat -ano \| findstr :3001` 有监听） |

## 事件原文

> **抹敏要求（简报原文）：粘贴事件 JSON 前必须抹去昵称/QQ 号**——`sender.nickname`、`sender.card`、`user_id`、`sender.user_id` 等字段替换为 `***`。落盘文件 `data/ob11_events.jsonl` 含真实号码与昵称，**不得整段原样粘贴**进本文件或任何提交物。

### 收（大号在测试群 @小号 发 "M0 测试"，探针收到并打印）

【待人工实测】（粘贴 `data/ob11_events.jsonl` 末行该 message 事件原文，抹敏后）

### 发（探针 `--send-group <测试群号>` 回发，`echo=m0-send-1` 的响应记录）

【待人工实测】（粘贴 `data/ob11_events.jsonl` 中 `echo` 为 `m0-send-1` 的响应原文，抹敏后）

## 遇到的问题

【待人工实测】（无则填"无"）
