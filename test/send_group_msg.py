"""
测试脚本：直接通过 OneBot11 WebSocket API 向指定群发送消息，
支持可选的 @全体成员。

用法：
  python test/send_group_msg.py                        # 使用脚本内默认参数
  python test/send_group_msg.py --at-all               # 开头加 @全体成员
  python test/send_group_msg.py --no-at-all            # 不加 @全体成员（默认）
  python test/send_group_msg.py --group 123456 --msg "内容" --at-all
"""

import argparse
import asyncio
import json
import sys
import os

import websockets

# NapCat WebSocket 连接配置（与 config.yaml 保持一致）
WS_URI   = "ws://127.0.0.1:3001"
WS_TOKEN = "NcatBot"

# 默认参数
DEFAULT_GROUP_ID = 1075786046
DEFAULT_MESSAGE  = "这是一条测试消息（来自 send_group_msg.py）"
DEFAULT_AT_ALL   = False


def build_message_segments(text: str, at_all: bool) -> list:
    """构造 OneBot11 消息段列表。"""
    segments = []
    if at_all:
        segments.append({"type": "at", "data": {"qq": "all"}})
        segments.append({"type": "text", "data": {"text": " "}})
    segments.append({"type": "text", "data": {"text": text}})
    return segments


async def send_group_msg(group_id: int, text: str, at_all: bool) -> None:
    headers = {"Authorization": f"Bearer {WS_TOKEN}"}

    print(f"[INFO] 连接 NapCat: {WS_URI}")
    async with websockets.connect(WS_URI, additional_headers=headers) as ws:
        payload = {
            "action": "send_group_msg",
            "params": {
                "group_id": group_id,
                "message": build_message_segments(text, at_all),
            },
            "echo": "test_send",
        }
        print(f"[INFO] 目标群: {group_id}")
        print(f"[INFO] at_all: {at_all}")
        print(f"[INFO] 消息内容: {'[CQ:at,qq=all] ' if at_all else ''}{text}")

        await ws.send(json.dumps(payload, ensure_ascii=False))

        # 等待响应（跳过心跳/其他事件）
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            resp = json.loads(raw)
            if resp.get("echo") == "test_send":
                if resp.get("status") == "ok":
                    print(f"[INFO] 发送成功，message_id={resp.get('data', {}).get('message_id')}")
                else:
                    print(f"[ERROR] 发送失败: {resp}")
                break


def main():
    parser = argparse.ArgumentParser(description="向指定 QQ 群发送消息")
    parser.add_argument("--group", type=int, default=DEFAULT_GROUP_ID, help="目标群号")
    parser.add_argument("--msg",   type=str, default=DEFAULT_MESSAGE,  help="消息内容")

    at_group = parser.add_mutually_exclusive_group()
    at_group.add_argument("--at-all",    dest="at_all", action="store_true",  help="在消息开头加 @全体成员")
    at_group.add_argument("--no-at-all", dest="at_all", action="store_false", help="不加 @全体成员（默认）")
    parser.set_defaults(at_all=DEFAULT_AT_ALL)

    args = parser.parse_args()
    asyncio.run(send_group_msg(args.group, args.msg, args.at_all))


if __name__ == "__main__":
    main()
