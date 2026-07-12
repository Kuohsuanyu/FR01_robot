#!/usr/bin/env python3
"""VMC OSC 原資料解析器 — 只看目標手指骨骼的實際 quaternion。

用途:
  * 確認 sender 送過來的 raw quaternion 到底是什麼(不只 count)
  * 每收到一個目標骨骼就印出:qx, qy, qz, qw + 該骨骼距上次的秒差
  * 沒有目標骨骼時每 3 秒印一次心跳:整體 pkt/s + 有見過哪些骨骼

用法:
    python3 tools_dump_bones.py
    python3 tools_dump_bones.py --dump-all       # 連身體骨骼也印
    python3 tools_dump_bones.py --port 39539
"""
import argparse
import socket
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pythonosc.osc_packet import OscPacket
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from vmc_receiver import RIGHT_BONE_TO_FINGER, AXIS_LOCAL, twist_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--dump-all", action="store_true",
                    help="連非目標骨骼(身體 / 左手)也印")
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8*1024*1024)
    s.settimeout(0.1)
    s.bind((args.ip, args.port))
    print(f"[{args.ip}:{args.port}] raw OSC 解析中……  Ctrl-C 退出\n")

    targets = set(RIGHT_BONE_TO_FINGER.keys())
    last_seen: dict[str, float] = {}
    q_baseline: dict[str, np.ndarray] = {}   # 第一次收到當 baseline,算相對 twist
    pkt_count = 0
    msg_count = 0
    hit_count = 0
    seen_bones: Counter = Counter()
    last_heartbeat = time.time()

    try:
        while True:
            try:
                data, _ = s.recvfrom(65535)
            except socket.timeout:
                # heartbeat 每 3 秒
                now = time.time()
                if now - last_heartbeat >= 3.0:
                    dt = now - last_heartbeat
                    seen_target = [b for b in targets if b in seen_bones]
                    seen_other = [(b, n) for b, n in seen_bones.most_common()
                                  if b not in targets][:8]
                    print(f"\n[heartbeat] {pkt_count/dt:.1f} pkt/s,{msg_count/dt:.1f} msg/s,命中手指 {hit_count/dt:.1f}/s")
                    print(f"  目前看過的目標骨骼 ({len(seen_target)}/{len(targets)}): {seen_target}")
                    if args.dump_all and seen_other:
                        print(f"  非目標 (top 8): {seen_other}")
                    pkt_count = msg_count = hit_count = 0
                    last_heartbeat = now
                continue

            pkt_count += 1
            try:
                pkt = OscPacket(data)
            except Exception:
                continue

            for m in pkt.messages:
                msg_count += 1
                if m.message.address != "/VMC/Ext/Bone/Pos":
                    continue
                params = list(m.message)
                if len(params) != 8:
                    continue
                bone = str(params[0])
                seen_bones[bone] += 1

                is_target = bone in targets

                if not is_target and not args.dump_all:
                    continue

                q = np.array([float(params[4]), float(params[5]),
                              float(params[6]), float(params[7])])

                now = time.time()
                dt_bone = now - last_seen.get(bone, now)
                last_seen[bone] = now

                if is_target:
                    hit_count += 1
                    # 第一次收到 → 當 baseline
                    if bone not in q_baseline:
                        q_baseline[bone] = q.copy()
                        rel_ang = 0.0
                        note = "  ← baseline"
                    else:
                        r_rel = R.from_quat(q_baseline[bone]).inv() * R.from_quat(q)
                        rel_ang = twist_deg(r_rel.as_quat(), AXIS_LOCAL[bone])
                        note = ""
                    finger = RIGHT_BONE_TO_FINGER[bone]
                    print(f"[{bone:28s} → {finger:6s}] "
                          f"q=({q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f})  "
                          f"twist={rel_ang:+6.1f}°  Δt={dt_bone*1000:5.0f}ms{note}")
                else:
                    print(f"[{bone:28s} (skip)  ] q=({q[0]:+.3f} {q[1]:+.3f} {q[2]:+.3f} {q[3]:+.3f})")

    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        print("\n=== 總覽 ===")
        print(f"看過骨骼種類:{len(seen_bones)}")
        target_saw = sum(1 for b in targets if b in seen_bones)
        print(f"目標手指骨骼命中:{target_saw}/{len(targets)}")
        for b in targets:
            n = seen_bones.get(b, 0)
            print(f"  {'✓' if n else '✗'} {b:28s} 共 {n} 次")


if __name__ == "__main__":
    main()
