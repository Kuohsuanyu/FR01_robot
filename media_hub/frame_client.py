#!/usr/bin/env python3
"""示範:另一個 agent 進程如何從 Media Hub 的共享記憶體讀影格。

這證明「不再搶鏡頭」—— media_hub 唯一持有 /dev/video,其他 agent(手部視覺、
錄影、AI 推論…)不再各自 cv2.VideoCapture,而是 attach 同一塊 shared_memory,
零額外裝置開銷、零衝突。

用法:
    # 先跑 media_hub.py,再另開終端:
    python3 frame_client.py                 # 從 /status 自動取得 shm 名稱
    python3 frame_client.py --hub http://127.0.0.1:8090 --save out.jpg
"""
from __future__ import annotations
import argparse
import json
import time
import urllib.request

import numpy as np
import cv2
from multiprocessing import shared_memory


def get_meta(hub_url: str) -> dict:
    with urllib.request.urlopen(hub_url + "/status", timeout=3) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="http://127.0.0.1:8090")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    meta = get_meta(args.hub)
    h, w, c = meta["shm_shape"]
    shm = shared_memory.SharedMemory(name=meta["shm_name"])
    arr = np.ndarray((h, w, c), np.uint8, buffer=shm.buf)
    print(f"[client] attached shm='{meta['shm_name']}' shape={h}x{w}x{c} "
          f"(camera_error={meta['camera_error']!r})")

    t0 = time.time(); n = 0; last_mean = None
    while time.time() - t0 < args.seconds:
        frame = arr.copy()          # 拷一份,避免讀到寫一半
        mean = float(frame.mean())
        if mean != last_mean:       # 畫面有在更新
            n += 1; last_mean = mean
        time.sleep(1 / 30)
    dt = time.time() - t0
    print(f"[client] {n} 次畫面更新 / {dt:.1f}s  ≈ {n/dt:.1f} fps  "
          f"(最後一幀平均亮度={last_mean:.1f})")
    if args.save:
        cv2.imwrite(args.save, arr.copy())
        print(f"[client] 已存最後一幀 → {args.save}")
    shm.close()


if __name__ == "__main__":
    main()
