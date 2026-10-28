# probes/ob11_listen.py
"""M0 探针：连接 NapCat 的 OB11 正向 WS，把所有事件落盘为 JSONL。

用法:
    uv run python probes/ob11_listen.py                # 仅监听落盘
    uv run python probes/ob11_listen.py --send-group 70000 "ping"
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qq_agent.ob11 import parse_event  # noqa: E402

WS_URL = "ws://127.0.0.1:3001"
OUT = Path(__file__).resolve().parents[1] / "data" / "ob11_events.jsonl"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send-group", type=int, default=None)
    parser.add_argument("--text", default="ping")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(WS_URL) as ws:
        print(f"[probe] connected {WS_URL}, logging -> {OUT}")
        if args.send_group:
            await ws.send(json.dumps({
                "action": "send_group_msg",
                "params": {"group_id": args.send_group, "message": [{"type": "text", "data": {"text": args.text}}]},
                "echo": "m0-send-1",
            }))
            print(f"[probe] sent to group {args.send_group}")
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with OUT.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                ev = parse_event(data)
                if ev:
                    print(f"[msg] {ev.message_type} user={ev.user_id} "
                          f"group={ev.group_id} text={ev.raw_message!r}")
        except websockets.ConnectionClosed as e:
            print(f"[probe] closed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
