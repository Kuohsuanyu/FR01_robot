#!/usr/bin/env python3
"""送一個假的 VMC 手指骨骼封包過去,用來測試 receiver 本身通不通(排除 sender 問題)。"""
import argparse, time
from pythonosc.udp_client import SimpleUDPClient

BONES = [
    "RightLittleProximal", "RightRingProximal", "RightMiddleProximal",
    "RightIndexProximal", "RightThumbIntermediate",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--n", type=int, default=30, help="每根骨骼送幾次")
    args = ap.parse_args()

    c = SimpleUDPClient(args.ip, args.port)
    print(f"→ {args.ip}:{args.port}  送 {args.n} 輪 x {len(BONES)} 根骨骼")
    for i in range(args.n):
        for b in BONES:
            # 假 quaternion:單位四元數(w=1),等於「無旋轉」
            c.send_message("/VMC/Ext/Bone/Pos", [b, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        time.sleep(0.03)
    print("done")

if __name__ == "__main__":
    main()
