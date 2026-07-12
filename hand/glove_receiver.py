#!/usr/bin/env python3
"""巧手 UDP JSON 手套接收 → FR01 右手驅動。

Sender:對方在 Windows 上用 `glove_send_gui.py` 送 JSON UDP,每包內容:
    {"seq": <int>, "t": <float>, "closure": {
        "Left":  {"Thumb": 0..1, "Index": 0..1, "Middle": 0..1, "Ring": 0..1, "Little": 0..1},
        "Right": {"Thumb": ..., "Index": ..., "Middle": ..., "Ring": ..., "Little": ...}
    }}

因為值已經是 0..1(0=張開、1=握起),直接餵 `config.tick_from_norm(finger, val)`
產出馬達 tick,不需 quaternion / 校準。

用法:
    python3 glove_receiver.py                     # 綁 0.0.0.0:6000,通馬達
    python3 glove_receiver.py --no-motor          # dry-run 只印,不動馬達
    python3 glove_receiver.py --port 6000 --ip 0.0.0.0

Keys:
    s = 印一次目前 closure 值    q = 退出   d = 開/關 debug
"""
import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, "/home/andykuo/FTServo_Python")
sys.path.insert(0, str(Path(__file__).parent))

from scservo_sdk import PortHandler, sms_sts, scscl, COMM_SUCCESS  # noqa: E402
import config as cfg  # noqa: E402

# ---- 網路 ----
BIND_IP = "0.0.0.0"
BIND_PORT = 6000
# 故意調小,強迫 kernel 只留最新 1~2 包,舊的丟掉,避免堆積延遲。
# (對方參考檔的關鍵設計:遠端控制延遲越跑越大的主因就是舊封包排隊)
RCVBUF_BYTES = 4096

# ---- 控制 (完全直通版:零軟體平滑)----
# 每收到新 packet → 每根手指 tick 都送。沒有 threshold、沒有 LPF、沒有 EMA。
# 唯一過濾:tick 若跟上次寫的完全一樣才跳過(避免重複封包)。
CONTROL_HZ = 500             # 拉高避免內部延遲(sender 有多快就吃多快)
STALE_TIMEOUT = 1.0

# ---- 馬達目標速度(硬體加速交給伺服本身)----
# SCS009 (model 1029) WritePos(id, pos, time_ms, speed):
#   time_ms=0 → 由 speed 決定;time_ms>0 → 強制走 time_ms(慢)
#   **speed 是 10-bit unsigned (0..1023)!超過會被截斷 → 反而變慢**
SCS_TIME_MS = 0
SCS_SPEED = 1023             # 10-bit max,全速

# STS3032 (model 521) WritePosEx(id, pos, speed, acc):
#   speed 15-bit;acc 8-bit unsigned max = 255
STS_SPEED = 3400
STS_ACC = 255                # 加速度拉到頂 (原本 150)

# ---- 對面用 "Little",我方 config 用 "pinky"。其他大小寫轉小寫。----
JSON_TO_FINGER = {
    "Thumb":  "thumb",
    "Index":  "index",
    "Middle": "middle",
    "Ring":   "ring",
    "Little": "pinky",
}


# =========================================================================
# 馬達層 — 沿用 vmc_receiver.py 的 HandBus + HandDriver
# =========================================================================
class HandBus:
    def __init__(self, port: str, baud: int):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            raise RuntimeError(f"open port {port} failed")
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(f"setBaudRate {baud} failed")
        self.pk_sts = sms_sts(self.ph)
        self.pk_scs = scscl(self.ph)
        # 預先分類:哪些 finger 是 SCS009、哪些是 STS3032
        self.scs_fingers = [(n, c) for n, c in cfg.FINGERS.items() if cfg.is_scscl(c["model"])]
        self.sts_fingers = [(n, c) for n, c in cfg.FINGERS.items() if not cfg.is_scscl(c["model"])]

    def _pk(self, model):
        return self.pk_scs if cfg.is_scscl(model) else self.pk_sts

    def torque(self, sid: int, model: int, on: bool):
        self._pk(model).write1ByteTxRx(sid, 40, 1 if on else 0)

    def write_pos(self, sid: int, model: int, tick: int):
        """單顆寫入(僅初始化 FIXED 用;控制迴圈請用 sync_write)。"""
        pk = self._pk(model)
        if cfg.is_scscl(model):
            pk.WritePos(sid, int(tick), SCS_TIME_MS, SCS_SPEED)
        else:
            pk.WritePosEx(sid, int(tick), STS_SPEED, STS_ACC)

    def sync_write_ticks(self, ticks_by_name: dict):
        """一次廣播:兩個 family 各發一個 sync 封包,吃 5 顆手指。
        ticks_by_name = {"pinky": 540, ...}。名字不在裡面就不動。"""
        # --- SCS009 group (WritePos 格式:pos_L, pos_H, time_L, time_H, speed_L, speed_H) ---
        gsw = self.pk_scs.groupSyncWrite
        gsw.clearParam()
        added = 0
        for name, c in self.scs_fingers:
            if name not in ticks_by_name:
                continue
            pos = int(ticks_by_name[name])
            data = [
                self.pk_scs.scs_lobyte(pos), self.pk_scs.scs_hibyte(pos),
                self.pk_scs.scs_lobyte(SCS_TIME_MS), self.pk_scs.scs_hibyte(SCS_TIME_MS),
                self.pk_scs.scs_lobyte(SCS_SPEED), self.pk_scs.scs_hibyte(SCS_SPEED),
            ]
            gsw.addParam(c["id"], data)
            added += 1
        if added:
            gsw.txPacket()

        # --- STS3032 group (WritePosEx 格式:acc, pos_L, pos_H, 0, 0, speed_L, speed_H) ---
        gsw = self.pk_sts.groupSyncWrite
        gsw.clearParam()
        added = 0
        for name, c in self.sts_fingers:
            if name not in ticks_by_name:
                continue
            pos = self.pk_sts.scs_toscs(int(ticks_by_name[name]), 15)  # STS 用 sign-magnitude
            data = [
                STS_ACC,
                self.pk_sts.scs_lobyte(pos), self.pk_sts.scs_hibyte(pos),
                0, 0,
                self.pk_sts.scs_lobyte(STS_SPEED), self.pk_sts.scs_hibyte(STS_SPEED),
            ]
            gsw.addParam(c["id"], data)
            added += 1
        if added:
            gsw.txPacket()

    def close(self):
        self.ph.closePort()


def enable_all(bus: HandBus):
    for c in cfg.FINGERS.values():
        bus.torque(c["id"], c["model"], True)
    for name, c in cfg.FIXED.items():
        bus.torque(c["id"], c["model"], True)
        bus.write_pos(c["id"], c["model"], c["hold_tick"])
        print(f"[fixed] {name} (id={c['id']}) → hold_tick={c['hold_tick']}")


def disable_all(bus: HandBus):
    for c in list(cfg.FINGERS.values()) + list(cfg.FIXED.values()):
        try:
            bus.torque(c["id"], c["model"], False)
        except Exception:
            pass


# =========================================================================
# 主 loop:單執行緒,非阻塞 recv + drain-to-latest + 200Hz 馬達更新
# =========================================================================
def run_loop(bus, args):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    s.setblocking(False)                # 非阻塞:才能 drain 到最新
    s.bind((args.ip, args.port))
    actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"[udp] 綁 {args.ip}:{args.port}  rcvbuf 要求 {RCVBUF_BYTES}B / 實得 {actual}B")

    # 狀態
    targets = {n: 0.0 for n in cfg.FINGERS}
    latest_closure = {}
    src_addr = None
    last_rx_t = 0.0

    # 計數
    got_per_sec = 0
    drained_per_sec = 0
    apply_per_sec = 0
    err_count = 0
    stat_t = time.perf_counter()
    last_t = time.perf_counter()

    period = 1.0 / CONTROL_HZ
    next_t = time.perf_counter() + period

    print(f"[loop] {CONTROL_HZ:.0f}Hz 主 loop 啟動,Ctrl+C 停止")
    try:
        while True:
            # ---- 1) drain socket 到最新 ----
            new_pkt = False
            while True:
                try:
                    data, addr = s.recvfrom(65535)
                except BlockingIOError:
                    break
                except OSError:
                    break
                drained_per_sec += 1
                try:
                    pkt = json.loads(data.decode("utf-8"))
                except Exception:
                    err_count += 1
                    continue
                latest_closure = pkt.get("closure", {}).get("Right", {})
                src_addr = addr
                last_rx_t = time.perf_counter()
                new_pkt = True
                got_per_sec += 1

            # ---- 2) 更新 targets from latest closure ----
            if new_pkt:
                for json_key, finger in JSON_TO_FINGER.items():
                    v = latest_closure.get(json_key)
                    if v is None:
                        continue
                    try:
                        targets[finger] = max(0.0, min(1.0, float(v)))
                    except Exception:
                        pass

            # ---- 3) 零平滑:每收到新 packet → 全部手指 tick 直接送 ----
            if new_pkt:
                changed = {name: cfg.tick_from_norm(name, targets[name])
                           for name in cfg.FINGERS}
                try:
                    bus.sync_write_ticks(changed)
                    apply_per_sec += len(changed)
                except Exception as e:
                    err_count += 1
                    if err_count % 50 == 1:
                        print(f"[motor err] sync_write: {e}")

            now = time.perf_counter()

            # ---- 4) 每秒印一次狀態 ----
            if now - stat_t >= 1.0:
                connected = (now - last_rx_t) < STALE_TIMEOUT and last_rx_t > 0
                tag = "✓連線" if connected else "✗待機"
                rvals = [round(latest_closure.get(f, 0.0), 2)
                         for f in ["Thumb", "Index", "Middle", "Ring", "Little"]]
                src_s = src_addr[0] if src_addr else "—"
                print(f"[{tag}] 來源={src_s}  收到={got_per_sec}/s  馬達寫={apply_per_sec}/s  "
                      f"右手 T/I/M/R/L={rvals}", flush=True)
                got_per_sec = drained_per_sec = apply_per_sec = 0
                stat_t = now

            # ---- 5) 定速 sleep ----
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_t += period
            # 若 loop 落後太多,重新對齊(避免積 debt)
            if next_t < time.perf_counter():
                next_t = time.perf_counter() + period

    except KeyboardInterrupt:
        pass
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=BIND_IP)
    ap.add_argument("--port", type=int, default=BIND_PORT)
    ap.add_argument("--no-motor", action="store_true", help="不通馬達,dry-run")
    ap.add_argument("--side", choices=["right", "left"], default="right",
                    help="機器人哪一隻手掌 (預設 right)")
    args = ap.parse_args()

    # 依 --side 切換 config 模組;class 內用 cfg.xxx 會抓 globals(),
    # 所以在建 HandBus 前把 cfg 換掉就會生效.
    global cfg
    if args.side == "left":
        import config_left as _cfg_left
        cfg = _cfg_left

    assert cfg.HAND_SIDE == args.side, \
        f"config mismatch: cfg.HAND_SIDE={cfg.HAND_SIDE!r} vs --side={args.side!r}"

    if not args.no_motor:
        bus = HandBus(cfg.SERVO_PORT, cfg.SERVO_BAUD)
        enable_all(bus)
    else:
        bus = _DryRunBus()

    side_zh = "左" if args.side == "left" else "右"
    print(f"🧤 Glove UDP receiver → {side_zh}手 {len(cfg.FINGERS)} 顆手指,"
          f"固定 {list(cfg.FIXED)}")

    try:
        run_loop(bus, args)
    finally:
        if not args.no_motor:
            disable_all(bus)
            bus.close()
        print("bye.")


class _DryRunBus:
    """--no-motor 用:忽略所有寫入,只讓 loop 跑起來。"""
    def torque(self, sid, model, on): pass
    def write_pos(self, sid, model, tick): pass
    def sync_write_ticks(self, ticks_by_name): pass
    def close(self): pass


if __name__ == "__main__":
    sys.exit(main())
