#!/usr/bin/env bash
# 鎖住頭部 sid 1/2/3 於當前位置,防止測試手臂時頭部亂動。
# 透過 head_rpi_agent 的 WebSocket 送 op:write 保持目前 tick。
set -e
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

c_cyan "==== 鎖住頭部 sid 1/2/3 於當前位置 ===="

# 確保 head agent 有跑
if ! curl -sf --max-time 3 "http://$RPI_HOST:8000/status" >/dev/null; then
    c_red "  head_agent 沒起來,無法鎖頭部"; exit 1
fi

python3 - << 'PY'
import asyncio, json, aiohttp

async def lock():
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect("ws://192.168.0.123:8000/ws") as ws:
            # 等一幀 tele 抓當前位置
            locked = {}
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT: continue
                d = json.loads(msg.data)
                if d.get("t") != "tele": continue
                for m in d["motors"]:
                    sid = m.get("sid")
                    pos = m.get("pos")
                    if sid in (1, 2, 3) and pos is not None:
                        # 送回同一個 tick 當 goal,低速 hold
                        await ws.send_str(json.dumps({
                            "op": "write", "sid": sid, "step": int(pos),
                            "speed": 100, "acc": 30
                        }))
                        locked[sid] = int(pos)
                if len(locked) >= 3:
                    break
            print(f"locked head positions: {locked}")

asyncio.run(lock())
PY

c_green "  完成 - 頭部固定於當前位置"
