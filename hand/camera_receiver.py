#!/usr/bin/env python3
"""相機 MediaPipe 手部追蹤 → FR01 右手驅動。

用 webcam 拍你的手,MediaPipe Hands 抓 21 個關鍵點,依每根手指的關節角度
算出 0..1 的 closure(0=張開、1=握起),再走 glove_receiver 同一組
sync_write 送給 5 顆手指馬達。拇指旋轉 (ID 35) 一樣固定在 config.FIXED 的 650。

依賴:
    pip install opencv-python mediapipe

用法:
    python3 camera_receiver.py                 # 預設 camera 0,無畫面
    python3 camera_receiver.py --show          # 顯示畫面 + 骨架 + 每指 closure bar
    python3 camera_receiver.py --camera 1
    python3 camera_receiver.py --no-motor      # dry-run,只印

熱鍵(--show 時):
    q = 退出

MediaPipe 21 個 landmark 索引參考:
    0=WRIST
    1-4=THUMB(CMC/MCP/IP/TIP)
    5-8=INDEX(MCP/PIP/DIP/TIP)
    9-12=MIDDLE, 13-16=RING, 17-20=PINKY
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, "/home/andykuo/FTServo_Python")
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg   # 預設右手 — main() 會依 --side left|right 切換
from glove_receiver import HandBus, enable_all, disable_all, _DryRunBus  # 共用馬達層


class _RemoteHeadAgentBus:
    """Send hand-motor writes to head_agent's WebSocket instead of local
    serial.  head_agent (>= scscl integration) dispatches SCS009 vs STS by
    model automatically, so we just send raw (sid, tick, speed, acc)."""

    def __init__(self, host_port: str, cfg_mod):
        try:
            import websocket
        except ImportError as e:
            raise RuntimeError(
                "websocket-client not installed — run `pip install "
                "websocket-client` on the laptop") from e
        self._websocket = websocket
        self._host = host_port if ":" in host_port else f"{host_port}:8000"
        self._ws = None
        self._connect()
        self._cfg = cfg_mod
        # per-family default speed/acc — mirrors glove_receiver.py
        self._spd_sts = 3400
        self._acc_sts = 255
        self._spd_scs = 1023
        self._acc_scs = 0
        # Reconnect state — long runs drop the WS silently (idle, wifi
        # flapping); we reconnect on next send.
        self._last_reconn_t = 0.0

    def _connect(self):
        self._ws = self._websocket.WebSocket()
        self._ws.settimeout(2.0)
        self._ws.connect(f"ws://{self._host}/ws", timeout=3)

    def _send(self, obj):
        payload = json.dumps(obj)
        try:
            self._ws.send(payload)
            return
        except Exception as e:
            # WS died — reconnect (rate-limited so we don't spin) then retry.
            now = time.monotonic()
            if now - self._last_reconn_t < 1.0:
                raise
            self._last_reconn_t = now
            try: self._ws.close()
            except Exception: pass
            print(f"[remote] WS lost ({e}); reconnecting to {self._host}...",
                  flush=True)
            self._connect()
            # Torque may have been dropped by the previous session close on
            # head_agent's end (or a full agent restart) — re-enable so the
            # motor obeys the retry.
            try: self.enable_all()
            except Exception: pass
            self._ws.send(payload)   # one retry; if still dead, caller sees it

    def enable_all(self):
        for c in self._cfg.FINGERS.values():
            self._send({"op": "torque", "sid": int(c["id"]), "on": True})
        for c in self._cfg.FIXED.values():
            self._send({"op": "torque", "sid": int(c["id"]), "on": True})
            self._send({"op": "write", "sid": int(c["id"]),
                        "step": int(c["hold_tick"]),
                        "speed": self._spd_for(c["model"]),
                        "acc":   self._acc_for(c["model"])})

    def disable_all(self):
        for c in list(self._cfg.FINGERS.values()) + list(self._cfg.FIXED.values()):
            try:
                self._send({"op": "torque", "sid": int(c["id"]), "on": False})
            except Exception:
                pass

    def _spd_for(self, model):
        return self._spd_scs if self._cfg.is_scscl(model) else self._spd_sts

    def _acc_for(self, model):
        return self._acc_scs if self._cfg.is_scscl(model) else self._acc_sts

    def sync_write_ticks(self, ticks_by_name: dict):
        # One broadcast packet (head_agent splits by handler → one SCS sync
        # + one STS sync), vs 5 per-sid writes.  Cuts serial round-trips
        # from 5 to 2 and removes per-finger settling jitter.
        cmds = [[int(self._cfg.FINGERS[n]["id"]), int(t)]
                for n, t in ticks_by_name.items()]
        # head_agent's `sync` op uses one (speed, acc) pair; use the STS
        # values because STS servos care about acc.  SCS009 branch on the
        # agent side ignores acc and caps speed to 10-bit.
        self._send({"op": "sync", "cmds": cmds,
                    "speed": self._spd_sts, "acc": self._acc_sts})

    def close(self):
        try: self._ws.close()
        except Exception: pass


# ---- 關鍵點索引 ----
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


def angle_at(a, b, c):
    """b 為頂點,回傳 a-b-c 的夾角 (度)。"""
    v1 = a - b
    v2 = c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
    cos = np.clip(np.dot(v1, v2) / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def norm_bend(angle_deg, straight_deg, folded_deg):
    """straight (張開) → 0,folded (握起) → 1,線性映射並 clip 到 [0,1]。"""
    span = straight_deg - folded_deg
    return float(np.clip((straight_deg - angle_deg) / span, 0.0, 1.0))


# 每個手指的「張開角」和「握起角」— 依 MediaPipe 觀察值調校.
# 和原本右手 hands_control 一致.
FINGER_ANGLE_RANGE = {
    "thumb":  (170.0, 130.0),
    "index":  (175.0,  55.0),
    "middle": (175.0,  55.0),
    "ring":   (175.0,  55.0),
    "pinky":  (175.0,  55.0),
}
CLOSURE_GAIN: dict[str, float] = {}


def compute_closures(lm: np.ndarray) -> dict:
    """lm: (21, 3) numpy array (MediaPipe landmark 的 x/y/z 正規化座標)。
    回傳每根手指 0..1 的 closure(0=張開、1=握起)。"""
    # 拇指:CMC-MCP-TIP 三點的 MCP 頂點夾角(和原本右手一致)
    ang_thumb  = angle_at(lm[THUMB_CMC],  lm[THUMB_MCP],  lm[THUMB_TIP])
    # 其他 4 指:MCP-PIP-TIP 的 PIP 頂點夾角(彎曲時角度變小)
    ang_index  = angle_at(lm[INDEX_MCP],  lm[INDEX_PIP],  lm[INDEX_TIP])
    ang_middle = angle_at(lm[MIDDLE_MCP], lm[MIDDLE_PIP], lm[MIDDLE_TIP])
    ang_ring   = angle_at(lm[RING_MCP],   lm[RING_PIP],   lm[RING_TIP])
    ang_pinky  = angle_at(lm[PINKY_MCP],  lm[PINKY_PIP],  lm[PINKY_TIP])
    out = {
        "thumb":  norm_bend(ang_thumb,  *FINGER_ANGLE_RANGE["thumb"]),
        "index":  norm_bend(ang_index,  *FINGER_ANGLE_RANGE["index"]),
        "middle": norm_bend(ang_middle, *FINGER_ANGLE_RANGE["middle"]),
        "ring":   norm_bend(ang_ring,   *FINGER_ANGLE_RANGE["ring"]),
        "pinky":  norm_bend(ang_pinky,  *FINGER_ANGLE_RANGE["pinky"]),
    }
    for k, g in CLOSURE_GAIN.items():
        out[k] = float(min(1.0, max(0.0, out[k] * g)))
    return out


def draw_overlay(frame, hand_lms, closures, mp_hands, mp_draw):
    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                           mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                           mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2))
    y = 30
    for name in ["thumb", "index", "middle", "ring", "pinky"]:
        v = closures[name]
        bar_len = int(v * 200)
        cv2.putText(frame, f"{name:>6}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.rectangle(frame, (90, y - 15), (90 + 200, y + 3), (60, 60, 60), 1)
        cv2.rectangle(frame, (90, y - 15), (90 + bar_len, y + 3),
                      (0, 255, 0) if v > 0.5 else (0, 200, 255), -1)
        cv2.putText(frame, f"{v:.2f}", (300, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--show", action="store_true", help="顯示畫面 + 骨架 debug")
    ap.add_argument("--no-motor", action="store_true", help="dry-run 不通馬達")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--hand", choices=["Right", "Left"], default="Right",
                    help="要追哪隻手(MediaPipe 標籤;鏡射後 Right = 使用者右手)")
    ap.add_argument("--side", choices=["right", "left"], default=None,
                    help="機器人哪一隻手掌 (預設跟隨 --hand;Right→right, Left→left)")
    ap.add_argument("--udp-to", default=None,
                    help="UDP JSON closure → host:port (給 glove_receiver 收)")
    ap.add_argument("--source", default=None,
                    help="Camera source override: cv2.VideoCapture 接得住的任意 URL"
                         "(例如 http://192.168.0.123:8000/mjpeg)")
    ap.add_argument("--remote-hand", default=None,
                    help="送馬達到 head_agent WebSocket (host:port);"
                         "例:192.168.0.123:8000 → ws://.../ws")
    args = ap.parse_args()

    # --side 未指定時,依 --hand 對映:MediaPipe Right 手 → 機器人 right 手
    side = args.side or args.hand.lower()

    # 依 --side 換到左手 config
    global cfg
    if side == "left":
        import importlib
        import config_left as _cfg_left
        cfg = _cfg_left
        # glove_receiver 模組層已 `import config`,也一起換掉
        sys.modules["config"] = _cfg_left
        importlib.reload(sys.modules["glove_receiver"])
        from glove_receiver import HandBus as _HB, enable_all as _EA, \
            disable_all as _DA, _DryRunBus as _DR  # noqa
        globals().update(HandBus=_HB, enable_all=_EA,
                         disable_all=_DA, _DryRunBus=_DR)

    assert cfg.HAND_SIDE == side, \
        f"config mismatch: cfg.HAND_SIDE={cfg.HAND_SIDE!r} vs --side={side!r}"

    # ---- Camera ----
    src = args.source if args.source else args.camera
    cap = cv2.VideoCapture(src)
    if not args.source:  # local V4L2 only — set explicit size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"[ERR] cannot open camera source: {src}")
        return 1
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[camera] source={src} @ {actual_w}x{actual_h}")

    # Reader thread for network sources: ffmpeg buffers up on http-mjpeg and
    # main-loop reads fall behind → freeze at fps 0.5.  Continuous reader
    # always overwrites the "latest" slot, main loop only consumes that.
    # Watchdog reopens the VideoCapture when reads stall (server hiccup /
    # ffmpeg internal deadlock), which used to freeze the whole GUI.
    import threading as _th
    STREAM_STALL_S = 3.0    # no new frame for this long → reopen
    reader_state = {"frame": None, "ts": 0.0, "stop": False, "err": 0,
                    "cap": cap, "reopen_count": 0}
    reader_thread = None
    is_stream = bool(args.source)

    def _open_stream():
        c = cv2.VideoCapture(src)
        try: c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        return c

    if is_stream:
        def _reader():
            last_ok_t = time.monotonic()
            fail_streak = 0
            while not reader_state["stop"]:
                c = reader_state["cap"]
                # Read one frame; guarded so a stuck read doesn't wedge us
                # for more than STREAM_STALL_S — after that the watchdog
                # branch reopens.
                if time.monotonic() - last_ok_t > STREAM_STALL_S:
                    reader_state["reopen_count"] += 1
                    print(f"[reader] stream stalled "
                          f"{time.monotonic() - last_ok_t:.1f}s → reopen "
                          f"#{reader_state['reopen_count']}", flush=True)
                    try: c.release()
                    except Exception: pass
                    try:
                        reader_state["cap"] = _open_stream()
                    except Exception as e:
                        print(f"[reader] reopen err: {e}", flush=True)
                        time.sleep(0.5); continue
                    last_ok_t = time.monotonic()
                    fail_streak = 0
                    continue
                ok, f = reader_state["cap"].read()
                if ok and f is not None:
                    reader_state["frame"] = f
                    reader_state["ts"] = time.monotonic()
                    last_ok_t = reader_state["ts"]
                    fail_streak = 0
                else:
                    reader_state["err"] += 1
                    fail_streak += 1
                    # 20 consecutive bad reads → force reopen even before
                    # the STALL_S timer fires.
                    if fail_streak >= 20:
                        last_ok_t = 0.0
                    time.sleep(0.02)
        reader_thread = _th.Thread(target=_reader, daemon=True)
        reader_thread.start()

    # ---- MediaPipe ----
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_conf,
        min_tracking_confidence=args.min_conf,
    )
    mp_draw = mp.solutions.drawing_utils if args.show else None

    # ---- 馬達 / UDP 送出 / 遠端 WS ----
    udp_sock = None
    udp_addr = None
    remote_bus = None
    if args.remote_hand:
        remote_bus = _RemoteHeadAgentBus(args.remote_hand, cfg)
        remote_bus.enable_all()
        bus = _DryRunBus()   # 本機不動馬達,由 head_agent 代寫
        print(f"[remote] motors via head_agent WS @ {args.remote_hand}")
    elif args.udp_to:
        host, _, port = args.udp_to.partition(":")
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_addr = (host, int(port or 6000))
        bus = _DryRunBus()
        print(f"[udp] closure → {udp_addr[0]}:{udp_addr[1]}")
    elif not args.no_motor:
        bus = HandBus(cfg.SERVO_PORT, cfg.SERVO_BAUD)
        enable_all(bus)
    else:
        bus = _DryRunBus()

    side_zh = "左" if cfg.HAND_SIDE == "left" else "右"
    print(f"📷 影像控制 → {side_zh}手 {len(cfg.FINGERS)} 顆手指,固定 {list(cfg.FIXED)}")
    print(f"    追蹤 {args.hand} 手(鏡射後 = 你的{'右' if args.hand=='Right' else '左'}手)")

    fps_n = 0
    last_stat = time.time()
    last_ticks = None
    lost_count = 0
    # Low-pass on closures — kills small jitter from MediaPipe landmark
    # noise.  α=0.35 → 65% previous + 35% new, ~3-frame settling at 15fps.
    CLOSURE_ALPHA = 0.35
    smoothed = {n: 0.0 for n in ("thumb", "index", "middle", "ring", "pinky")}

    last_ts_seen = 0.0
    try:
        while True:
            if is_stream:
                # Only process when the reader thread has a fresh frame.
                ts = reader_state["ts"]
                if ts == last_ts_seen or reader_state["frame"] is None:
                    time.sleep(0.005)
                    continue
                last_ts_seen = ts
                frame = reader_state["frame"].copy()
            else:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02); continue

            frame = cv2.flip(frame, 1)      # 鏡像:你舉右手就是畫面右手
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = hands.process(rgb)
            rgb.flags.writeable = True

            found = False
            closures = None
            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_lms, handedness in zip(result.multi_hand_landmarks,
                                                result.multi_handedness):
                    if handedness.classification[0].label != args.hand:
                        continue
                    lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark])
                    raw = compute_closures(lm)
                    for n in smoothed:
                        smoothed[n] += CLOSURE_ALPHA * (raw[n] - smoothed[n])
                    closures = dict(smoothed)
                    if remote_bus is not None:
                        ticks = {n: cfg.tick_from_norm(n, v)
                                 for n, v in closures.items()}
                        try:
                            remote_bus.sync_write_ticks(ticks)
                            last_ticks = ticks
                        except Exception as e:
                            print(f"[remote err] {e}")
                    elif udp_sock is not None:
                        # Match glove_receiver's JSON schema (Right slot only)
                        pkt = {"closure": {"Right": {
                            "Thumb":  closures["thumb"],
                            "Index":  closures["index"],
                            "Middle": closures["middle"],
                            "Ring":   closures["ring"],
                            "Little": closures["pinky"],
                        }}}
                        try:
                            udp_sock.sendto(json.dumps(pkt).encode(), udp_addr)
                        except Exception as e:
                            print(f"[udp err] {e}")
                    else:
                        ticks = {n: cfg.tick_from_norm(n, v)
                                 for n, v in closures.items()}
                        try:
                            bus.sync_write_ticks(ticks)
                            last_ticks = ticks
                        except Exception as e:
                            print(f"[motor err] {e}")
                    found = True
                    if args.show:
                        draw_overlay(frame, hand_lms, closures, mp_hands, mp_draw)
                    break

            if not found:
                lost_count += 1
                # 手看不到:維持最後一次的姿勢 (last_ticks 已經在馬達上,不用重送)
                if args.show:
                    cv2.putText(frame, f"NO {args.hand} HAND (lost {lost_count})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            fps_n += 1
            now = time.time()
            if now - last_stat >= 1.0:
                dt = now - last_stat
                cl = closures if closures else {}
                cl_str = "  ".join(f"{cl.get(k, 0):.2f}"
                                    for k in ["thumb","index","middle","ring","pinky"])
                tag = "✓偵測" if found else f"✗遺失({lost_count})"
                print(f"[stat] {fps_n/dt:.0f} fps  {tag}  T/I/M/R/L=[{cl_str}]", flush=True)
                fps_n = 0
                lost_count = 0
                last_stat = now

            if args.show:
                cv2.imshow(f"Hand Track ({args.hand})  q=quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        reader_state["stop"] = True
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
        # Release whichever cap the reader ended on (may be a reopened one).
        try: reader_state.get("cap", cap).release()
        except Exception: pass
        if args.show:
            cv2.destroyAllWindows()
        if remote_bus is not None:
            remote_bus.disable_all()
            remote_bus.close()
        elif udp_sock is not None:
            udp_sock.close()
        elif not args.no_motor:
            disable_all(bus)
            bus.close()
        hands.close()
        print("bye.")


if __name__ == "__main__":
    sys.exit(main())
