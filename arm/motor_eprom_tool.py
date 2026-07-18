#!/usr/bin/env python3
"""透過 arm agent /ws 設定馬達 EPROM:多圈模式 / 角度限位(持久)。

需要已部署『含 set_mode/set_limit op』的新版 arm agent。

用法:
  # ID 10、20 改多圈(無限旋轉,MODE=3):
  python3 arm/motor_eprom_tool.py --arm 192.168.0.123:8100 --multiturn 10 20
  # ID 23 硬體下限 tick=1300:
  python3 arm/motor_eprom_tool.py --arm 192.168.0.123:8100 --min 23:1300
  # 改回單圈:
  python3 arm/motor_eprom_tool.py --arm 192.168.0.123:8100 --singleturn 10
  # 讀回確認(MODE 暫存器 addr 33):
  python3 arm/motor_eprom_tool.py --arm 192.168.0.123:8100 --read-mode 10 20 23

⚠ 改多圈後該顆位置讀值語意會變(累加超過 0-4095),tick↔q 校正需重做。
"""
from __future__ import annotations
import argparse, asyncio, json, aiohttp


async def rpc(ws, op, **kw):
    rid = f"r{op}{kw.get('sid','')}"
    await ws.send_str(json.dumps({"op": op, "req_id": rid, **kw}))
    for _ in range(50):
        msg = await asyncio.wait_for(ws.receive(), timeout=5)
        if msg.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(msg.data)
            if d.get("req_id") == rid:
                return d
    return {"ok": False, "error": "no reply"}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="192.168.0.123:8100")
    ap.add_argument("--multiturn", nargs="*", type=int, default=[])
    ap.add_argument("--singleturn", nargs="*", type=int, default=[])
    ap.add_argument("--min", nargs="*", default=[], help="sid:tick,如 23:1300")
    ap.add_argument("--max", nargs="*", default=[], help="sid:tick")
    ap.add_argument("--read-mode", nargs="*", type=int, default=[])
    args = ap.parse_args()

    async with aiohttp.ClientSession() as s:
        async with s.ws_connect("http://" + args.arm + "/ws", timeout=8) as ws:
            for sid in args.multiturn:
                r = await rpc(ws, "set_mode", sid=sid, mode=3)
                print(f"  多圈 ID {sid}: {r}")
            for sid in args.singleturn:
                r = await rpc(ws, "set_mode", sid=sid, mode=0)
                print(f"  單圈 ID {sid}: {r}")
            for spec in args.min:
                sid, tick = spec.split(":");
                r = await rpc(ws, "set_limit", sid=int(sid), min=int(tick))
                print(f"  min ID {sid}={tick}: {r}")
            for spec in args.max:
                sid, tick = spec.split(":")
                r = await rpc(ws, "set_limit", sid=int(sid), max=int(tick))
                print(f"  max ID {sid}={tick}: {r}")
            for sid in args.read_mode:
                r = await rpc(ws, "read", sid=sid, addr=33, size=1)
                m = r.get("value")
                name = {0: "單圈位置", 1: "輪式", 3: "多圈"}.get(m, "?")
                print(f"  ID {sid} MODE={m} ({name})")


if __name__ == "__main__":
    asyncio.run(main())
