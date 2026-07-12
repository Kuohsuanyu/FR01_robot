#!/usr/bin/env python3
"""極限吞吐 raw UDP + OSC 完整解析,證明能吃下 sender 的最大負載。

架構:
  - Thread A (recv):raw socket + 32MB rcvbuf,recv 到 queue,不做任何解析
  - Thread B (parse):從 queue 取出、pythonosc.OscPacket 解,計 message 分佈
  - Main:每 2 秒印 throughput,總長 30 秒,結束後印全部骨骼分佈與 bytes

用法:
    python3 tools_max_throughput_sniff.py               # 30 秒,port 39539
    python3 tools_max_throughput_sniff.py --dur 60
"""
import argparse
import socket
import sys
import threading
import time
from collections import Counter, deque

from pythonosc.osc_packet import OscPacket

RCVBUF_BYTES = 32 * 1024 * 1024   # 32 MB
QUEUE_MAX = 1_000_000             # 1M packets buffered before drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=39539)
    ap.add_argument("--dur", type=float, default=30.0)
    args = ap.parse_args()

    q: deque = deque(maxlen=QUEUE_MAX)
    q_lock = threading.Lock()
    stop = threading.Event()

    # --- 統計計數 ---
    recv_pkts = 0
    recv_bytes = 0
    dropped = 0
    parsed_msgs = 0
    addr_cnt: Counter = Counter()
    bone_cnt: Counter = Counter()
    src_addrs: Counter = Counter()

    # --- Thread A: 只 recv,不解析 ---
    def recv_worker():
        nonlocal recv_pkts, recv_bytes, dropped
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
        actual = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        print(f"[recv] SO_RCVBUF requested {RCVBUF_BYTES/1024/1024:.0f}MB,"
              f"actual {actual/1024/1024:.1f}MB (kernel 可能有 max 限制)")
        s.settimeout(0.05)
        s.bind((args.ip, args.port))
        print(f"[recv] listening {args.ip}:{args.port}")
        while not stop.is_set():
            try:
                data, addr = s.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            recv_pkts += 1
            recv_bytes += len(data)
            src_addrs[addr[0]] += 1
            with q_lock:
                if len(q) >= QUEUE_MAX - 1:
                    dropped += 1
                    continue
                q.append(data)
        s.close()

    # --- Thread B: 從 queue 解析 ---
    def parse_worker():
        nonlocal parsed_msgs
        while not stop.is_set() or q:
            data = None
            with q_lock:
                if q:
                    data = q.popleft()
            if data is None:
                time.sleep(0.001)
                continue
            try:
                pkt = OscPacket(data)
            except Exception:
                continue
            for m in pkt.messages:
                parsed_msgs += 1
                a = m.message.address
                addr_cnt[a] += 1
                if a == "/VMC/Ext/Bone/Pos":
                    params = list(m.message)
                    if params:
                        bone_cnt[str(params[0])] += 1

    tA = threading.Thread(target=recv_worker, daemon=True)
    tB = threading.Thread(target=parse_worker, daemon=True)
    tA.start()
    tB.start()

    t0 = time.time()
    last_report = t0
    last_pkts = 0
    last_bytes = 0
    last_msgs = 0
    print(f"\n--- 開始收集 {args.dur:.0f} 秒 ---\n")
    try:
        while time.time() - t0 < args.dur:
            time.sleep(2.0)
            now = time.time()
            dt = now - last_report
            dp = recv_pkts - last_pkts
            db = recv_bytes - last_bytes
            dm = parsed_msgs - last_msgs
            last_pkts, last_bytes, last_msgs, last_report = recv_pkts, recv_bytes, parsed_msgs, now
            with q_lock:
                qlen = len(q)
            print(f"[t+{now-t0:5.1f}s] recv={dp/dt:6.0f}pkt/s  {db/dt/1024:6.1f}KB/s  "
                  f"parse={dm/dt:6.0f}msg/s  queue={qlen:>6}  drop={dropped}")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        tA.join(timeout=1)
        tB.join(timeout=2)

    total_t = time.time() - t0
    print(f"\n=== 總結 ({total_t:.1f} 秒) ===")
    print(f"UDP 封包    : {recv_pkts:>8}   ({recv_pkts/total_t:.0f} pkt/s)")
    print(f"UDP bytes  : {recv_bytes:>8}   ({recv_bytes/total_t/1024:.1f} KB/s)")
    print(f"OSC message: {parsed_msgs:>8}   ({parsed_msgs/total_t:.0f} msg/s)")
    print(f"每封包平均  : {recv_bytes/max(1,recv_pkts):.1f} bytes,{parsed_msgs/max(1,recv_pkts):.2f} msg")
    print(f"OS 端掉包(queue 溢出): {dropped}")
    print(f"來源 IP    : {dict(src_addrs)}")
    print(f"\n=== OSC address 分佈 (top 20) ===")
    for a, n in addr_cnt.most_common(20):
        print(f"  {a:36s} {n:>7} ({n/total_t:.0f} Hz)")
    print(f"\n=== /VMC/Ext/Bone/Pos 骨骼 (共 {len(bone_cnt)} 種) ===")
    right_fingers = [(b,n) for b,n in bone_cnt.items()
                     if b.startswith("Right") and any(f in b for f in
                     ["Thumb","Index","Middle","Ring","Little"])]
    left_fingers = [(b,n) for b,n in bone_cnt.items()
                    if b.startswith("Left") and any(f in b for f in
                    ["Thumb","Index","Middle","Ring","Little"])]
    body = [(b,n) for b,n in bone_cnt.items()
            if b not in dict(right_fingers) and b not in dict(left_fingers)]
    for label, grp in [("右手手指", right_fingers),
                       ("左手手指", left_fingers),
                       ("身體其他", body)]:
        s = sum(n for _, n in grp)
        print(f"  --- {label} ({s} msgs, {s/total_t:.0f} Hz) ---")
        for b, n in sorted(grp, key=lambda x: -x[1])[:20]:
            print(f"    {b:32s} {n:>7} ({n/total_t:.0f} Hz)")


if __name__ == "__main__":
    main()
