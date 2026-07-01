#!/usr/bin/env python3
"""Linux 攝影機測試 — 列出可用相機,然後用指定 index 開視窗。

用法:
  python3 test_camera.py              # 自動列出 + 開啟 index 0
  python3 test_camera.py 2            # 開啟 /dev/video2
  python3 test_camera.py 2 --rotate 90
按鍵:
  q / ESC = 離開
  s       = 截一張圖到本目錄 snap_*.png
  數字 0-9 = 切換到對應 /dev/videoN
"""
import argparse
import os
import sys
import time

import cv2


def list_devices():
    print("=== 可用攝影機 (Linux) ===")
    for i in range(10):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  index={i}  {path}  {w}x{h}  ✓")
        else:
            print(f"  index={i}  {path}  (open 失敗)")
        cap.release()


def open_camera(index: int, w: int, h: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
       180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=int, nargs="?", default=0)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    list_devices()
    idx = args.index
    cap = open_camera(idx, args.width, args.height)
    if cap is None:
        print(f"\n✗ /dev/video{idx} 開不起來")
        sys.exit(1)

    print(f"\n→ 開啟 /dev/video{idx}  q/ESC 離開  s 截圖  數字鍵切換 cam")

    rotate = args.rotate
    fps_t0 = time.time(); fps_n = 0; fps = 0.0
    last_log = time.time()
    win = "FR01 camera test (Linux)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, args.height)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  frame read 失敗, 重試…")
            time.sleep(0.1)
            continue
        if ROT.get(rotate) is not None:
            frame = cv2.rotate(frame, ROT[rotate])
        fps_n += 1
        now = time.time()
        if now - fps_t0 >= 0.5:
            fps = fps_n / (now - fps_t0)
            fps_t0 = now; fps_n = 0
        cv2.putText(frame, f"cam{idx}  {fps:.1f}fps  rot={rotate}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 50), 2)
        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k == ord('s'):
            fn = f"snap_{int(now)}.png"
            cv2.imwrite(fn, frame)
            print(f"  ↳ saved {fn}")
        if ord('0') <= k <= ord('9'):
            new_idx = k - ord('0')
            if os.path.exists(f"/dev/video{new_idx}") and new_idx != idx:
                cap.release()
                cap2 = open_camera(new_idx, args.width, args.height)
                if cap2:
                    cap = cap2; idx = new_idx
                    print(f"  ↳ 切到 cam{idx}")
                else:
                    print(f"  ↳ cam{new_idx} 開不起來,維持 cam{idx}")
                    cap = open_camera(idx, args.width, args.height)
        if k == ord('r'):
            rotate = (rotate + 90) % 360

    cap.release()
    cv2.destroyAllWindows()
    print("done.")


if __name__ == "__main__":
    main()
