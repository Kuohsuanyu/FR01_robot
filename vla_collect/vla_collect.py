#!/usr/bin/env python3
"""FR01 VLA 資料採集 — 錄製「影像序列 + 對應手臂關節姿勢」的 episode。

每按一次錄製 = 一個 episode 資料夾,名稱自動遞增:
    <base><NN>/   例:設定 base="pick" → pick01, pick02, pick03 ...
      ├── meta.json          任務名 / index / 起訖時間 / fps / 幀數 / 關節名
      ├── frames/frame_00000.jpg ...   每幀影像
      └── trajectory.jsonl   每幀一行:{t, frame, q:{joint:rad}, ticks:{sid:tick}}

影像來源:snapshot URL(media_hub :8090/snapshot 或 head agent :8000/snapshot)
姿勢來源:arm agent :8100 /ws 遙測 → tick → q(用 arm/qbot_arm_calibration.json)

用法:
    python3 vla_collect/vla_collect.py
    python3 vla_collect/vla_collect.py --cam http://192.168.0.123:8000/snapshot \
                                       --arm 192.168.0.123:8100 --fps 15
"""
from __future__ import annotations
import argparse, json, os, re, sys, threading, time
import tkinter as tk
from tkinter import ttk
import urllib.request

import numpy as np
import cv2
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "arm"))
from qbot_ik_gui import RemoteBus, ARMS  # noqa: E402  reuse WS bus + joint spec

CAL_PATH = os.path.join(REPO, "arm", "qbot_arm_calibration.json")
DATA_DIR = os.path.join(HERE, "data")

C = {"bg": "#0e1620", "panel": "#172130", "card": "#1c2a3a", "text": "#d0e0f0",
     "dim": "#6080a0", "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
     "btn": "#1e4870", "rec": "#c0392b"}


# ── calibration tick → q ────────────────────────────────────────────────────
def load_cal():
    if not os.path.exists(CAL_PATH):
        return None
    d = json.load(open(CAL_PATH, encoding="utf-8"))
    return {a: (d.get(a) or {}).get("points") for a in ("L", "R")}


def tick_to_q(cal, arm, i, tick):
    if not cal or not cal.get(arm):
        return None
    p = cal[arm][i]
    den = p["tick_hi"] - p["tick_lo"]
    if abs(den) < 1e-9:
        return p["q_lo"]
    return p["q_lo"] + (tick - p["tick_lo"]) * (p["q_hi"] - p["q_lo"]) / den


# ── camera (snapshot polling) ───────────────────────────────────────────────
class Camera:
    def __init__(self, url, fps):
        self.url = url
        self.interval = 1.0 / max(1, fps)
        self.jpg: bytes | None = None
        self.frame = None
        self.error = ""
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            try:
                with urllib.request.urlopen(self.url, timeout=1.5) as r:
                    data = r.read()
                self.jpg = data
                self.frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                          cv2.IMREAD_COLOR)
                self.error = ""
            except Exception as e:
                self.error = str(e)[:40]
                self.frame = None
            time.sleep(self.interval)

    def stop(self):
        self._run = False


# ── pose (arm /ws telemetry → q) ────────────────────────────────────────────
class PoseReader:
    def __init__(self):
        self.bus = RemoteBus()
        self.cal = load_cal()
        self.sidmap = {}
        for arm in ("L", "R"):
            for i, (sid, _l, _g, _lo, _hi, jn) in enumerate(ARMS[arm]["joints"]):
                self.sidmap[sid] = (arm, i, jn)
        self.host = ""
        self.connected = False

    def connect(self, host):
        try:
            ok, msg = self.bus.open(host)
            self.host = host
            self.connected = bool(ok)
            return ok, msg
        except Exception as e:
            self.connected = False
            return False, str(e)

    def latest(self):
        """回傳 (q_by_joint, ticks_by_sid)。"""
        tele = getattr(self.bus, "latest_tele", {}) or {}
        q, ticks = {}, {}
        for sid, m in tele.items():
            pos = m.get("pos")
            if pos is None:
                continue
            ticks[str(int(sid))] = int(pos)
            info = self.sidmap.get(int(sid))
            if info:
                arm, i, jn = info
                if jn:
                    qv = tick_to_q(self.cal, arm, i, int(pos))
                    if qv is not None:
                        q[jn] = round(float(qv), 5)
        return q, ticks


def next_episode_name(base):
    os.makedirs(DATA_DIR, exist_ok=True)
    mx = 0
    for d in os.listdir(DATA_DIR):
        m = re.match(rf"^{re.escape(base)}(\d+)$", d)
        if m:
            mx = max(mx, int(m.group(1)))
    return f"{base}{mx + 1:02d}"


# ── GUI ─────────────────────────────────────────────────────────────────────
class VlaApp:
    def __init__(self, cam_url, arm_host, fps):
        self.fps = fps
        self.cam = Camera(cam_url, fps)
        self.pose = PoseReader()
        self.recording = False
        self._rec_thread = None
        self._last_ep = ""
        self._frame_count = 0

        self.root = tk.Tk()
        self.root.title("FR01 VLA 資料採集")
        self.root.configure(bg=C["bg"])
        self.root.geometry("760x680")

        top = tk.Frame(self.root, bg=C["panel"]); top.pack(fill="x", padx=6, pady=6)
        tk.Label(top, text="任務名稱", bg=C["panel"], fg=C["text"]).pack(side="left")
        self.name_var = tk.StringVar(value="pick")
        tk.Entry(top, textvariable=self.name_var, width=14, bg=C["card"],
                 fg=C["text"], insertbackground=C["text"]).pack(side="left", padx=4)
        tk.Label(top, text="fps", bg=C["panel"], fg=C["text"]).pack(side="left", padx=(10, 0))
        self.fps_var = tk.IntVar(value=fps)
        tk.Spinbox(top, from_=1, to=60, width=4, textvariable=self.fps_var,
                   bg=C["card"], fg=C["text"]).pack(side="left", padx=4)
        self.next_var = tk.StringVar(value=f"下一個:{next_episode_name('pick')}")
        tk.Label(top, textvariable=self.next_var, bg=C["panel"], fg=C["dim"]).pack(side="right", padx=6)
        self.name_var.trace_add("write", lambda *a: self._refresh_next())

        # 連線列
        conn = tk.Frame(self.root, bg=C["panel"]); conn.pack(fill="x", padx=6)
        tk.Label(conn, text="相機URL", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left")
        self.cam_var = tk.StringVar(value=cam_url)
        tk.Entry(conn, textvariable=self.cam_var, width=34, bg=C["card"], fg=C["text"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left", padx=3)
        tk.Button(conn, text="套用相機", bg=C["btn"], fg="white",
                  command=self._apply_cam).pack(side="left", padx=2)
        tk.Label(conn, text="手臂", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left", padx=(8, 0))
        self.arm_var = tk.StringVar(value=arm_host)
        tk.Entry(conn, textvariable=self.arm_var, width=18, bg=C["card"], fg=C["text"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left", padx=3)
        self.arm_btn = tk.Button(conn, text="連接手臂", bg=C["btn"], fg="white",
                                 command=self._connect_arm)
        self.arm_btn.pack(side="left", padx=2)

        # 預覽
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="#000",
                                highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self._img_id = None
        self._tk_img = None

        # 錄製列
        rec = tk.Frame(self.root, bg=C["panel"]); rec.pack(fill="x", padx=6, pady=4)
        self.rec_btn = tk.Button(rec, text="● 開始錄製", bg=C["rec"], fg="white",
                                 font=("DejaVu Sans", 13, "bold"), width=16,
                                 command=self._toggle_record)
        self.rec_btn.pack(side="left", padx=4, pady=4)
        self.status = tk.StringVar(value="就緒")
        tk.Label(rec, textvariable=self.status, bg=C["panel"], fg=C["text"],
                 font=("DejaVu Sans Mono", 11)).pack(side="left", padx=10)

        self.info = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.info, bg=C["bg"], fg=C["dim"],
                 anchor="w", font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=8)

        self._connect_arm()
        self.root.after(60, self._tick)

    def _refresh_next(self):
        self.next_var.set(f"下一個:{next_episode_name(self.name_var.get() or 'pick')}")

    def _apply_cam(self):
        self.cam.stop()
        self.cam = Camera(self.cam_var.get(), self.fps_var.get())
        self.info.set(f"相機 → {self.cam_var.get()}")

    def _connect_arm(self):
        ok, msg = self.pose.connect(self.arm_var.get())
        self.arm_btn.config(bg=(C["ok"] if ok else C["err"]))
        self.info.set(f"手臂 {'連上' if ok else '失敗'}:{msg}")

    def _tick(self):
        # 預覽
        f = self.cam.frame
        if f is not None:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != 640:
                rgb = cv2.resize(rgb, (640, 480))
            self._tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self._img_id is None:
                self._img_id = self.canvas.create_image(0, 0, image=self._tk_img, anchor="nw")
            else:
                self.canvas.itemconfig(self._img_id, image=self._tk_img)
        # 狀態
        q, ticks = self.pose.latest()
        cam_ok = self.cam.frame is not None
        self.info.set(f"相機:{'OK' if cam_ok else self.cam.error or '無'}   "
                      f"姿勢關節數:{len(q)}   "
                      f"{'錄製中 '+str(self._frame_count)+' 幀' if self.recording else ''}")
        self.root.after(60, self._tick)

    def _toggle_record(self):
        if self.recording:
            self.recording = False
            self.rec_btn.config(text="● 開始錄製", bg=C["rec"])
            self.status.set(f"已存 {self._last_ep}({self._frame_count} 幀)")
            self._refresh_next()
        else:
            if self.cam.frame is None:
                self.info.set("相機沒畫面,無法錄製"); return
            base = self.name_var.get().strip() or "pick"
            ep = next_episode_name(base)
            self._last_ep = ep
            self._frame_count = 0
            self.recording = True
            self.rec_btn.config(text="■ 停止", bg=C["warn"])
            self.status.set(f"● 錄製中:{ep}")
            self._rec_thread = threading.Thread(
                target=self._record_loop, args=(ep, base), daemon=True)
            self._rec_thread.start()

    def _record_loop(self, ep, base):
        ep_dir = os.path.join(DATA_DIR, ep)
        frames_dir = os.path.join(ep_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        traj = open(os.path.join(ep_dir, "trajectory.jsonl"), "w", encoding="utf-8")
        fps = max(1, self.fps_var.get())
        t0 = time.time(); n = 0
        while self.recording:
            jpg = self.cam.jpg
            q, ticks = self.pose.latest()
            if jpg is not None:
                with open(os.path.join(frames_dir, f"frame_{n:05d}.jpg"), "wb") as f:
                    f.write(jpg)
                traj.write(json.dumps({"t": round(time.time() - t0, 3), "frame": n,
                                       "q": q, "ticks": ticks}, ensure_ascii=False) + "\n")
                n += 1; self._frame_count = n
            time.sleep(1.0 / fps)
        traj.close()
        # metadata
        jn = [ARMS[a]["joints"][i][5] for a in ("L", "R") for i in range(5)
              if ARMS[a]["joints"][i][5]]
        json.dump({"task": base, "episode": ep, "fps": fps, "num_frames": n,
                   "camera": self.cam_var.get(), "arm_host": self.arm_var.get(),
                   "joint_names": jn, "duration_s": round(time.time() - t0, 2)},
                  open(os.path.join(ep_dir, "meta.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default="http://192.168.0.123:8000/snapshot")
    ap.add_argument("--arm", default="192.168.0.123:8100")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    VlaApp(args.cam, args.arm, args.fps).run()


if __name__ == "__main__":
    main()
