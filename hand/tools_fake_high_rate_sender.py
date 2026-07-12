#!/usr/bin/env python3
"""假高頻 VMC OSC sender — 模擬「幾千 pkt/s」流量,證明 receiver 端能吃下。

用途:
  - Loopback 測試:sender + receiver 都在同一台,只驗證軟體
  - LAN 測試:在別台 (如 Windows) 對 Linux 這台 IP 送

會送:
  - 5 根右手 finger bones × 3 段(Proximal/Intermediate/Distal) = 15 bones
  - 每 bone quaternion 用 sine 波掃 -30..+30 度彎曲角度
  - 頻率預設 3000 pkt/s

用法:
    python3 tools_fake_high_rate_sender.py                   # 送到 127.0.0.1:39539 @ 3000 Hz
    python3 tools_fake_high_rate_sender.py --ip 192.168.0.118 --rate 5000
"""
import argparse
import math
import time

from pythonosc.udp_client import SimpleUDPClient

BONES = []
for finger in ["Little", "Ring", "Middle", "Index", "Thumb"]:
    for seg in ["Proximal", "Intermediate", "Distal"]:
        BONES.append(f"Right{finger}{seg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--rate", type=int, default=3000, help="每秒送幾個封包")
    ap.add_argument("--dur", type=float, default=20.0)
    args = ap.parse_args()

    c = SimpleUDPClient(args.ip, args.port)
    period = 1.0 / args.rate
    print(f"→ {args.ip}:{args.port}  目標 {args.rate} pkt/s (每 {period*1000:.3f} ms 一發)")
    print(f"  骨骼池 {len(BONES)} 個,round-robin;quaternion 隨時間 sine 掃")

    t0 = time.time()
    n = 0
    next_send = t0
    last_report = t0
    last_n = 0
    try:
        while time.time() - t0 < args.dur:
            # 忙等到下一發時機
            while time.time() < next_send:
                pass
            bone = BONES[n % len(BONES)]
            t = time.time() - t0
            # 30° 彎曲的 sine 波,純 Z 軸 (符合真實 VMC finger 資料格式)
            ang_rad = math.radians(30.0) * math.sin(2 * math.pi * 0.5 * t)  # 0.5 Hz
            qz = math.sin(ang_rad / 2)
            qw = math.cos(ang_rad / 2)
            c.send_message("/VMC/Ext/Bone/Pos",
                           [bone, 0.0, 0.0, 0.0, 0.0, 0.0, qz, qw])
            n += 1
            next_send += period

            now = time.time()
            if now - last_report >= 2.0:
                dt = now - last_report
                dn = n - last_n
                last_report, last_n = now, n
                print(f"[t+{now-t0:5.1f}s] 送出 {dn/dt:6.0f} pkt/s")
    except KeyboardInterrupt:
        pass

    print(f"\n總送出 {n} 個封包 / {time.time()-t0:.1f} 秒 = {n/(time.time()-t0):.0f} pkt/s")


if __name__ == "__main__":
    main()
