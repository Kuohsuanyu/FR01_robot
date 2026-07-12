#!/usr/bin/env python3
"""Q-BOT head PC client — talks to a Raspberry Pi running head_rpi_agent.

This is the 上位機 / visualisation side.  It:
  * connects to  ws://<pi>:8000/ws  for motor commands + telemetry
  * pulls          http://<pi>:8000/mjpeg  for the head camera stream
  * shows the same ghost robot as head_ghost_gui.py, with the head bit
    highlighted, so operator sees pose in 3-D
  * lets you drive the head with sliders (Live) or from the phone IMU
    server that head_ghost_gui.py already provides

Usage:
    python3 remote_gui.py                       # asks for host in the UI
    python3 remote_gui.py --host 192.168.0.42   # auto-connect on launch
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk

import numpy as np
import aiohttp
import urllib.request
from PIL import Image, ImageTk

# Reuse the MuJoCo World + neck joint table from head_ghost_gui so the ghost
# rendering is identical to the local-motor version.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "arm"))
from head_ghost_gui import (  # noqa: E402
    World, NECK, HEAD_BODIES, HEAD_BODY, RENDER_W, RENDER_H,
    TICKS_PER_DEG, CENTER, MJCF_PATH, C, SEND_THROTTLE_S, SMOOTH_ALPHA,
    SENDER_TICK_MS, SEND_EPS_DEG,
)

DEFAULT_HOST = "192.168.0.123"     # override with --host
CAM_W_DISPLAY = 320
CAM_H_DISPLAY = 240

# Remote-shutdown targets — user is on the same LAN as the FR01 RPis, so a
# quick ssh+poweroff is fine as an operator convenience.  Passwords are the
# initial ones set during first-boot and are hard-coded here on purpose:
# these RPis live on an isolated robot LAN with no internet exposure.
SHUTDOWN_TARGETS = {
    "頭部 (Head 192.168.0.123)":   {"user": "robot", "host": "192.168.0.123", "pw": "2919"},
    "外骨骼 (Exo 192.168.0.124)":  {"user": "robot", "host": "192.168.0.124", "pw": "2919"},
    "腿部 (Leg  192.168.0.???)":   {"user": "fr01",  "host": "192.168.0.???", "pw": "2919"},
}


# ── Remote WebSocket bus ─────────────────────────────────────────────────────
class RemoteBus:
    """Runs a private asyncio loop in a background thread.  Public methods
    are sync so they can be called from the Tk main thread; the async task
    picks them up via a thread-safe queue."""

    def __init__(self):
        self.host = ""
        self.connected = False
        self.latest_tele: dict[int, dict] = {}   # sid → {pos, volt, temp, ...}
        self.last_tele_t = 0.0
        self._out: "queue.Queue[dict]" = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws = None                 # live WebSocketClient
        self.status = "idle"

    # ── API used by GUI thread ─────────────────────────────────────────────
    def connect(self, host: str):
        if self._thread and self._thread.is_alive():
            self.disconnect()
        self.host = host.strip()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def disconnect(self):
        self._stop.set()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread = None
        self.connected = False
        self.status = "disconnected"

    def is_open(self):
        return self.connected

    def send(self, obj: dict):
        self._out.put(obj)

    # convenience:
    def write_sync(self, cmds, speed=0, acc=30):
        self.send({"op": "sync", "cmds": cmds, "speed": speed, "acc": acc})

    def write_pos(self, sid, step, speed=0, acc=30):
        self.send({"op": "write", "sid": sid, "step": step,
                   "speed": speed, "acc": acc})

    def torque(self, sid, on):
        self.send({"op": "torque", "sid": sid, "on": bool(on)})

    def torque_all(self, on):
        self.send({"op": "torque_all", "on": bool(on)})

    def request_scan(self):
        self.send({"op": "scan", "req_id": "scan", "range": [1, 30]})

    # ── async worker ───────────────────────────────────────────────────────
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self.status = f"loop err: {e}"
            print(f"[remote] loop error: {e}", flush=True)
        finally:
            self._loop.close()

    async def _main(self):
        url = f"ws://{self.host}/ws"
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(url, timeout=5,
                                           heartbeat=10) as ws:
                    self._ws = ws
                    self.connected = True
                    self.status = f"connected {self.host}"
                    print(f"[remote] connected {url}", flush=True)
                    send_task = asyncio.create_task(self._sender())
                    recv_task = asyncio.create_task(self._receiver())
                    stop_task = asyncio.create_task(self._wait_stop())
                    done, pending = await asyncio.wait(
                        [send_task, recv_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED)
                    for t in pending: t.cancel()
        except Exception as e:
            self.status = f"connect err: {e}"
            print(f"[remote] connect err: {e}", flush=True)
        finally:
            self.connected = False
            self._ws = None

    async def _sender(self):
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            # get from thread-safe queue with a timeout so we can check _stop
            try:
                obj = await loop.run_in_executor(
                    None, lambda: self._out.get(timeout=0.2))
            except queue.Empty:
                continue
            if obj is None: break
            try:
                await self._ws.send_str(json.dumps(obj))
            except Exception as e:
                print(f"[remote] send err: {e}", flush=True)
                return

    async def _receiver(self):
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT: continue
            try: d = json.loads(msg.data)
            except Exception: continue
            if d.get("t") == "tele":
                mdict = {m["sid"]: m for m in d.get("motors", [])}
                self.latest_tele = mdict
                self.last_tele_t = time.monotonic()

    async def _wait_stop(self):
        while not self._stop.is_set():
            await asyncio.sleep(0.1)


# ── Camera stream reader ─────────────────────────────────────────────────────
class MjpegReader:
    """Fetch an MJPEG multipart stream from the agent, decode each frame,
    keep the latest as a PIL Image. Very small; no aiohttp needed here since
    urllib handles multipart just fine for our purposes."""

    def __init__(self):
        self.host = ""
        self.latest: Image.Image | None = None
        self.frames = 0
        self.error = ""
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def start(self, host: str):
        self.stop()
        self.host = host
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        url = f"http://{self.host}/mjpeg"
        boundary = b"--frame"
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(url,
                                             headers={"User-Agent": "qbot-pc"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    self.error = ""
                    buf = b""
                    while not self._stop.is_set():
                        chunk = r.read(8192)
                        if not chunk: break
                        buf += chunk
                        # keep pulling complete frames as long as we have TWO
                        # boundary markers in the buffer
                        while True:
                            a = buf.find(boundary)
                            if a < 0:
                                # no boundary yet, but don't let buf grow forever
                                if len(buf) > 2_000_000: buf = b""
                                break
                            b = buf.find(boundary, a + len(boundary))
                            if b < 0:
                                # incomplete frame; keep buf FROM boundary a
                                # and wait for more bytes
                                buf = buf[a:]
                                break
                            frame = buf[a + len(boundary):b]
                            buf = buf[b:]                # leave marker b for next
                            hdr_end = frame.find(b"\r\n\r\n")
                            if hdr_end < 0: continue
                            body = frame[hdr_end + 4:].rstrip(b"\r\n")
                            if len(body) < 200: continue
                            try:
                                img = Image.open(io.BytesIO(body))
                                img.load()
                                self.latest = img
                                self.frames += 1
                            except Exception:
                                pass
            except Exception as e:
                self.error = str(e)
                time.sleep(2.0)


# ── App ──────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, host: str):
        self.world = World(MJCF_PATH)
        self.bus = RemoteBus()
        self.cam = MjpegReader()

        self.root = tk.Tk()
        self.root.title("Q-BOT head — PC 上位機(Remote)")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1240x740")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.q_deg = np.zeros(3)         # user command (deg)
        self.q_filt = np.zeros(3)        # low-pass filtered command → send
        self._q_last_sent = np.full(3, -9999.0)
        self._last_send_t = 0.0
        self._suppress = False
        # Phone-IMU takeover state: when the phone is connected + ACTIVE, the
        # ghost/sliders follow the phone (fetched from head_agent /status).
        self._phone_neck_q = None
        self._phone_last_t = 0.0
        self._phone_active = False

        self._build_ui()
        # apply full-body ghost + highlight head like local head_ghost_gui
        self._paint_ghost_and_head()
        self.world.set_neck(0, 0, 0)
        self._refresh_sliders()

        # animate
        self.root.after(33, self._render_tick)
        self.root.after(120, self._cam_tick)
        self.root.after(SENDER_TICK_MS, self._sender_tick)
        # background poll of head_agent /status for phone neck_q
        import threading as _th
        _th.Thread(target=self._phone_neck_poll, daemon=True).start()

        if host:
            self.host_var.set(host)
            self.root.after(500, self._toggle_connect)

    def _paint_ghost_and_head(self):
        m = self.world.model
        for i in range(m.ngeom):
            m.geom_rgba[i] = [0.40, 0.45, 0.52, 0.42]
        bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                for n in HEAD_BODIES}
        bids.discard(-1)
        for i in range(m.ngeom):
            if m.geom_bodyid[i] in bids:
                m.geom_rgba[i] = [0.95, 0.55, 0.20, 1.0]

    # ── UI ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#1a237e"); hdr.pack(fill="x")
        tk.Label(hdr, text="Q-BOT head — PC 上位機 (Remote → RPi agent)",
                 font=("DejaVu Sans", 15, "bold"), bg="#1a237e",
                 fg="white").pack(side="left", padx=12, pady=8)

        # connection bar
        bar = tk.Frame(self.root, bg=C["card"])
        bar.pack(fill="x", padx=8, pady=(6, 4))
        tk.Label(bar, text="RPi host:", bg=C["card"], fg=C["text"]
                 ).pack(side="left", padx=(6, 2))
        self.host_var = tk.StringVar(value=DEFAULT_HOST + ":8000")
        tk.Entry(bar, textvariable=self.host_var, width=22,
                 bg=C["bg"], fg=C["text"], insertbackground=C["text"]
                 ).pack(side="left")
        self.connect_btn = tk.Button(bar, text="▶ Connect", bg="#2e7d32",
                                     fg="white",
                                     font=("DejaVu Sans Mono", 10, "bold"),
                                     command=self._toggle_connect, width=12)
        self.connect_btn.pack(side="left", padx=8)
        self.live_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Live", variable=self.live_var,
                       bg=C["card"], fg=C["ok"], activebackground=C["card"],
                       activeforeground=C["ok"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 9, "bold")
                       ).pack(side="left", padx=4)
        # "手機→真實馬達": OFF = phone drives ghost only (calibration mode).
        # Sends {"op":"imu_active"} to head_agent — same flag the phone button
        # toggles, so either side can safely take control.
        self.phone_motor_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="手機→真實馬達",
                       variable=self.phone_motor_var,
                       bg=C["card"], fg=C["warn"], activebackground=C["card"],
                       activeforeground=C["warn"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 9, "bold"),
                       command=self._on_phone_motor_toggle
                       ).pack(side="left", padx=4)
        # ── Remote poweroff dropdown ───────────────────────────────────
        tk.Label(bar, text="關機:", bg=C["card"], fg=C["text"],
                 font=("DejaVu Sans Mono", 9)
                 ).pack(side="right", padx=(6, 2))
        self.shutdown_target_var = tk.StringVar(
            value=next(iter(SHUTDOWN_TARGETS)))
        tk.OptionMenu(bar, self.shutdown_target_var, *SHUTDOWN_TARGETS.keys()
                      ).pack(side="right")
        tk.Button(bar, text="⏻ 關", bg=C["btn_danger"], fg="white",
                  font=("DejaVu Sans Mono", 9, "bold"), width=5,
                  command=self._on_shutdown_click
                  ).pack(side="right", padx=(4, 8))

        self.conn_var = tk.StringVar(value="● 未連線")
        tk.Label(bar, textvariable=self.conn_var, bg=C["card"], fg=C["err"],
                 font=("DejaVu Sans Mono", 10)).pack(side="right", padx=8)

        # ── Mode bar: IMU vs IK ────────────────────────────────────────────
        # Two-button mode selector for demo — IMU 模式 lets the phone drive
        # the head, IK 模式 lets the operator drag the ghost / use the slider
        # sliders to command the neck joints directly (phone poll ignored).
        mode_bar = tk.Frame(self.root, bg=C["card"])
        mode_bar.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(mode_bar, text="模式:", bg=C["card"], fg=C["text"],
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(side="left", padx=(6, 4))
        self.mode_var = tk.StringVar(value="imu")
        tk.Radiobutton(mode_bar, text="▶ IMU 模式 (手機控制)",
                       variable=self.mode_var, value="imu",
                       command=self._on_mode_change,
                       bg=C["card"], fg=C["text"],
                       activebackground=C["card"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 10, "bold"), indicatoron=0,
                       width=22, pady=4,
                       ).pack(side="left", padx=4)
        tk.Radiobutton(mode_bar, text="▶ IK 拖曳模式",
                       variable=self.mode_var, value="ik",
                       command=self._on_mode_change,
                       bg=C["card"], fg=C["text"],
                       activebackground=C["card"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 10, "bold"), indicatoron=0,
                       width=18, pady=4,
                       ).pack(side="left", padx=4)
        # Pitch invert toggle — quick fix if phone pitch sign is opposite of
        # your motor wiring.  Applied client-side after phone poll.
        self.pitch_flip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(mode_bar, text="反轉 pitch",
                       variable=self.pitch_flip_var,
                       bg=C["card"], fg=C["text"],
                       activebackground=C["card"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 9)
                       ).pack(side="left", padx=8)

        # split: left = controls, middle = ghost, right = camera
        body = tk.Frame(self.root, bg=C["bg"]); body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=C["panel"], width=420)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        mid = tk.Frame(body, bg=C["bg"])
        mid.pack(side="left", fill="both", expand=True, padx=4, pady=8)
        self.ghost_canvas = tk.Canvas(mid, width=RENDER_W, height=RENDER_H,
                                      bg="#000", highlightthickness=0)
        self.ghost_canvas.pack(fill="both", expand=True)
        # Mouse-drag IK: only active in IK mode; handlers no-op otherwise.
        self._drag_ref = None
        self.ghost_canvas.bind("<Button-1>", self._on_ghost_press)
        self.ghost_canvas.bind("<B1-Motion>", self._on_ghost_drag)
        self._ghost_img_id = None
        self._ghost_tk = None

        right = tk.Frame(body, bg=C["panel"], width=CAM_W_DISPLAY + 20)
        right.pack(side="right", fill="y", padx=(4, 8), pady=8)
        right.pack_propagate(False)
        tk.Label(right, text="Camera (from RPi)", bg=C["panel"],
                 fg=C["bright"], font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(padx=4, pady=(4, 2))
        self.cam_canvas = tk.Canvas(right, width=CAM_W_DISPLAY,
                                    height=CAM_H_DISPLAY,
                                    bg="#000", highlightthickness=0)
        self.cam_canvas.pack(padx=4, pady=2)
        self.cam_status_var = tk.StringVar(value="waiting…")
        tk.Label(right, textvariable=self.cam_status_var, bg=C["panel"],
                 fg=C["dim"], font=("DejaVu Sans Mono", 9)
                 ).pack(padx=4, pady=(2, 6))
        # ── camera rotation buttons ─────────────────────────────────────
        self._cam_rot_deg = 90                # default matches head mount
        self.cam_rot_label = tk.StringVar(value=f"90°")
        rot_row = tk.Frame(right, bg=C["panel"])
        rot_row.pack(padx=4, pady=(0, 6))
        tk.Button(rot_row, text="↺ CCW",
                  bg=C["btn"], fg="white", width=6,
                  font=("DejaVu Sans Mono", 9, "bold"),
                  command=self._on_cam_rot_ccw
                  ).pack(side="left", padx=2)
        tk.Button(rot_row, text="0°",
                  bg=C["btn"], fg="white", width=4,
                  font=("DejaVu Sans Mono", 9, "bold"),
                  command=self._on_cam_rot_reset
                  ).pack(side="left", padx=2)
        tk.Button(rot_row, text="↻ CW",
                  bg=C["btn"], fg="white", width=6,
                  font=("DejaVu Sans Mono", 9, "bold"),
                  command=self._on_cam_rot_cw
                  ).pack(side="left", padx=2)
        tk.Label(rot_row, textvariable=self.cam_rot_label,
                 bg=C["panel"], fg=C["bright"], width=6,
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(side="left", padx=6)
        self._cam_img_id = None
        self._cam_tk = None

        # status
        self.status_var = tk.StringVar(value="ready")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 bg=C["bg"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=12, pady=(2, 4))

    def _build_controls(self, p):
        # 3 neck joint sliders (yaw/pitch/roll)
        title = tk.LabelFrame(p, text=" 頭部 3-DOF ",
                              bg=C["panel"], fg=C["text"],
                              font=("DejaVu Sans Mono", 10, "bold"))
        title.pack(fill="x", padx=6, pady=6)
        self.sliders = []; self.cmd_vars = []; self.id_lbls = []
        self.read_vars = []; self.volt_vars = []; self.temp_vars = []
        for idx, label, jname, motor_id, mdir, lo, hi in NECK:
            row = tk.Frame(title, bg=C["card"])
            row.pack(fill="x", padx=4, pady=2)
            top = tk.Frame(row, bg=C["card"]); top.pack(fill="x")
            tk.Label(top, text=f"ID {motor_id}", width=5, bg=C["card"],
                     fg=C["target"], font=("DejaVu Sans Mono", 10, "bold")
                     ).pack(side="left", padx=2)
            tk.Label(top, text=label, width=7, bg=C["card"],
                     fg=C["bright"], font=("DejaVu Sans Mono", 10, "bold")
                     ).pack(side="left")
            cv = tk.StringVar(value="0.0°"); self.cmd_vars.append(cv)
            tk.Label(top, textvariable=cv, width=8, bg=C["card"],
                     fg=C["warn"], font=("DejaVu Sans Mono", 10, "bold")
                     ).pack(side="left", padx=2)
            rv = tk.StringVar(value="—"); self.read_vars.append(rv)
            tk.Label(top, text="read:", bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=(8, 1))
            tk.Label(top, textvariable=rv, width=7, bg=C["card"],
                     fg=C["ok"], font=("DejaVu Sans Mono", 9)
                     ).pack(side="left")
            vv = tk.StringVar(value="—"); self.volt_vars.append(vv)
            tv = tk.StringVar(value="—"); self.temp_vars.append(tv)
            tk.Label(top, textvariable=vv, width=6, bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=4)
            tk.Label(top, textvariable=tv, width=5, bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            sld = tk.Scale(row, from_=lo, to=hi, resolution=0.1,
                           orient="horizontal", length=380,
                           bg=C["card"], fg=C["text"], showvalue=False,
                           highlightthickness=0, troughcolor=C["panel"],
                           command=lambda v, i=idx: self._on_slider(i, v))
            sld.pack(fill="x", padx=2, pady=(0, 2))
            self.sliders.append(sld)

        # buttons
        bp = tk.Frame(p, bg=C["panel"]); bp.pack(fill="x", padx=6, pady=6)
        tk.Button(bp, text="HOME (q=0)", bg=C["btn"], fg="white",
                  command=self._home, width=12).pack(side="left", padx=2)
        tk.Button(bp, text="E-STOP (torque off)", bg=C["btn_danger"],
                  fg="white", command=self._estop
                  ).pack(side="right", padx=2)

        # info
        self.tele_var = tk.StringVar(value="")
        tk.Label(p, textvariable=self.tele_var, anchor="w",
                 bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(4, 0))

    # ── slider / control ───────────────────────────────────────────────────
    def _refresh_sliders(self):
        self._suppress = True
        for i in range(3):
            self.sliders[i].set(round(self.q_deg[i], 1))
            self.cmd_vars[i].set(f"{self.q_deg[i]:+6.1f}°")
        self._suppress = False

    def _on_slider(self, i, val):
        if self._suppress: return
        try: deg = float(val)
        except ValueError: return
        self.q_deg[i] = deg
        self.cmd_vars[i].set(f"{deg:+6.1f}°")

    def _home(self):
        self.q_deg[:] = 0.0
        self._refresh_sliders()

    def _estop(self):
        if self.bus.is_open():
            self.bus.torque_all(False)
            self.status_var.set("E-STOP → RPi 已停 torque")
        else:
            self.status_var.set("尚未連線")

    def _on_shutdown_click(self):
        """Ship an ssh+poweroff to whichever RPi is picked in the dropdown.
        Confirmation is via a Tk messagebox so an accidental click on a
        real bench doesn't yank a running experiment."""
        import subprocess
        from tkinter import messagebox
        label = self.shutdown_target_var.get()
        tgt = SHUTDOWN_TARGETS.get(label)
        if not tgt:
            return
        if "???" in tgt["host"]:
            messagebox.showerror("目標 IP 未設定",
                                 f"{label}\n請先在 SHUTDOWN_TARGETS 設好 IP。")
            return
        if not messagebox.askyesno("確認遠端關機",
                                   f"要對「{label}」下 poweroff 嗎?\n"
                                   f"({tgt['user']}@{tgt['host']})"):
            return
        cmd = ["ssh",
               "-o", "StrictHostKeyChecking=no",
               "-o", "ConnectTimeout=3",
               f"{tgt['user']}@{tgt['host']}",
               f"echo {tgt['pw']} | sudo -S poweroff"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            ok = (r.returncode == 0)
            msg = (r.stderr or r.stdout or "").strip()[:180]
            self.status_var.set(
                f"[shutdown] {label} → "
                f"{'sent' if ok else f'err rc={r.returncode}'}  {msg}")
        except Exception as e:
            self.status_var.set(f"[shutdown] {label} → {e}")

    def _on_mode_change(self):
        """Switching modes — tell head_agent whether the phone should drive
        motors, and clear any stale phone_neck_q so ghost doesn't linger."""
        mode = self.mode_var.get()
        if self.bus.is_open():
            # IK mode implies "phone must not touch motors" regardless of the
            # phone_motor checkbox.  IMU mode restores the checkbox's state.
            on = (mode == "imu") and bool(self.phone_motor_var.get())
            try:
                self.bus.send({"op": "imu_active", "on": on})
            except Exception as e:
                print(f"[mode] imu_active send err: {e}", flush=True)
        if mode == "ik":
            # Wipe the phone override so sliders / drag input take over
            self._phone_neck_q = None
        print(f"[mode] → {mode}", flush=True)

    def _on_ghost_press(self, event):
        """Start of a click-drag on the ghost canvas — IK mode only.  Anchor
        the initial cursor position + current yaw/pitch so subsequent motion
        events can compute deltas relative to this reference."""
        if self.mode_var.get() != "ik":
            return
        self._drag_ref = (event.x, event.y,
                          float(self.q_deg[0]), float(self.q_deg[1]))

    def _on_ghost_drag(self, event):
        """Mouse drag in IK mode: x-delta → yaw, y-delta → pitch."""
        if self.mode_var.get() != "ik":
            return
        if not hasattr(self, "_drag_ref") or self._drag_ref is None:
            return
        x0, y0, y_deg0, p_deg0 = self._drag_ref
        # 0.25°/px yaw, 0.20°/px pitch — same feel as head_ghost_gui local.
        yaw   = y_deg0 + (event.x - x0) * 0.25
        pitch = p_deg0 - (event.y - y0) * 0.20
        yaw   = max(NECK[0][5], min(NECK[0][6], yaw))
        pitch = max(NECK[1][5], min(NECK[1][6], pitch))
        self.q_deg[0] = yaw
        self.q_deg[1] = pitch
        # Move sliders too so the UI stays in sync
        self._suppress = True
        try:
            self.sliders[0].set(round(yaw, 1))
            self.sliders[1].set(round(pitch, 1))
        except Exception:
            pass
        self._suppress = False

    def _on_phone_motor_toggle(self):
        """Ship the phone's IMU-active flag over the WS to head_agent so its
        pose_ws stops (or resumes) writing to the neck motors.  Ghost
        preview keeps working regardless — the ghost is driven by pose
        polling, not by this flag."""
        on = bool(self.phone_motor_var.get())
        if self.bus.is_open():
            try:
                self.bus.send({"op": "imu_active", "on": on})
                print(f"[phone→motor] {'ON' if on else 'OFF (ghost only)'}",
                      flush=True)
            except Exception as e:
                print(f"[phone→motor] send err: {e}", flush=True)
        else:
            # Not connected — nothing to send; leave the checkbox state as
            # user set it so it'll fire on next connect.
            pass

    # ── connect ────────────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.disconnect()
            self.cam.stop()
            self.connect_btn.config(text="▶ Connect", bg="#2e7d32")
            self.conn_var.set("● 未連線")
            return
        host = self.host_var.get().strip()
        if not host: return
        if ":" not in host: host += ":8000"
        self.bus.connect(host)
        self.cam.start(host)
        self.connect_btn.config(text="■ Disconnect", bg=C["btn_danger"])
        self.conn_var.set(f"● 連線 {host}…")
        # ask agent for a motor scan so telemetry starts populating
        self.root.after(400, lambda: self.bus.send({"op": "scan",
                                                    "range": [1, 10]}))
        # Push current "phone → real motor" state to head_agent so it matches
        # our checkbox (default OFF = ghost-only mode, safer for calibration).
        self.root.after(600,
                        lambda: self.bus.send({
                            "op": "imu_active",
                            "on": bool(self.phone_motor_var.get())}))

    def _phone_neck_poll(self):
        """Poll head_agent /status every 150 ms for the phone's neck_q so
        the ghost + sliders can mirror the operator's phone motion — even
        when the operator hasn't armed motor sends yet."""
        import urllib.request as _u, json as _j, time as _t
        host = None
        while True:
            try:
                # Refresh host from the entry each iteration in case operator
                # retypes / reconnects to a different RPi.
                h = (self.host_var.get() or "").strip()
                if ":" not in h: h = h + ":8000"
                if not h: _t.sleep(0.5); continue
                url = f"http://{h}/status"
                with _u.urlopen(url, timeout=1.0) as r:
                    d = _j.loads(r.read().decode())
                pc = int(d.get("phone_connected", 0) or 0)
                active = bool(d.get("phone_imu_active", False))
                age = int(d.get("phone_pose_age_ms", -1) or -1)
                q = d.get("neck_q") or {}
                if pc > 0 and 0 <= age < 500:
                    self._phone_neck_q = (float(q.get("yaw", 0)),
                                          float(q.get("pitch", 0)),
                                          float(q.get("roll", 0)))
                    self._phone_active = active
                    self._phone_last_t = _t.monotonic()
                else:
                    self._phone_active = False
            except Exception:
                pass
            _t.sleep(0.15)

    # ── sender: 50 Hz, low-pass, one sync packet ───────────────────────────
    def _sender_tick(self):
        # Mode gate: only IMU mode lets phone override sliders.  IK mode
        # ignores the phone entirely so operator drag / slider input drives.
        phone_driving = (
            self.mode_var.get() == "imu"
            and self._phone_neck_q is not None
            and time.monotonic() - self._phone_last_t < 1.0)
        if phone_driving:
            y, p, r = self._phone_neck_q
            if self.pitch_flip_var.get():
                p = -p                # match motor wiring direction
            vals = (y, p, r)
            # Only copy yaw + pitch (2-DOF control) — roll left at 0 so the
            # neck stays upright regardless of phone tilt.
            for j in range(2):
                lo = NECK[j][5]; hi = NECK[j][6]
                self.q_deg[j] = float(max(lo, min(hi, vals[j])))
            self.q_deg[2] = 0.0
            # move sliders visibly to match
            self._suppress = True
            try:
                for j in range(3):
                    self.sliders[j].set(round(self.q_deg[j], 1))
            except Exception:
                pass
            self._suppress = False
        self.q_filt += SMOOTH_ALPHA * (self.q_deg - self.q_filt)
        for i in range(3):
            if abs(self.q_deg[i] - self.q_filt[i]) < 0.02:
                self.q_filt[i] = self.q_deg[i]
        # ghost from filtered
        self.world.set_neck(*self.q_filt)
        # Motor gating rules (mutually exclusive with agent's motor writes):
        #   * IMU mode → client is a silent observer.  Motor motion only
        #     happens when the phone is armed (phone_motor_var → sends
        #     imu_active=True and head_agent writes motors directly).
        #     "手機→真實馬達 OFF" therefore truly stops motion because
        #     BOTH agent (imu_active=False) AND client (gate_motor=False)
        #     stop writing.
        #   * IK mode → client is sole writer (sliders / mouse drag).
        gate_motor = (self.mode_var.get() == "ik")
        if gate_motor and self.bus.is_open() and self.live_var.get():
            now = time.monotonic()
            max_d = float(np.max(np.abs(self.q_filt - self._q_last_sent)))
            if (now - self._last_send_t >= SEND_THROTTLE_S
                    and max_d > SEND_EPS_DEG):
                cmds = []
                for j, (idx, label, jname, motor_id, mdir, lo, hi) in enumerate(NECK):
                    step = int(round(self.q_filt[j] * mdir * TICKS_PER_DEG)) + CENTER
                    step = max(0, min(4095, step))
                    cmds.append([motor_id, step])
                self.bus.write_sync(cmds, speed=0, acc=30)
                self._q_last_sent[:] = self.q_filt
                self._last_send_t = now
        self.root.after(SENDER_TICK_MS, self._sender_tick)

    # ── render loop for ghost + telemetry text ─────────────────────────────
    def _render_tick(self):
        try:
            img = self.world.render()
            pil = Image.fromarray(img)
            self._ghost_tk = ImageTk.PhotoImage(pil)
            if self._ghost_img_id is None:
                self._ghost_img_id = self.ghost_canvas.create_image(
                    0, 0, anchor="nw", image=self._ghost_tk)
            else:
                self.ghost_canvas.itemconfig(self._ghost_img_id,
                                             image=self._ghost_tk)
        except Exception as e:
            self.status_var.set(f"render err: {e}")

        # update read from telemetry + connection status text
        if self.bus.connected:
            self.conn_var.set(f"● 已連線 {self.bus.host}")
        else:
            self.conn_var.set(f"● {self.bus.status}")
        # tele into read/volt/temp
        now = time.monotonic()
        age = (now - self.bus.last_tele_t) if self.bus.last_tele_t else 999
        for i, (idx, label, jname, motor_id, mdir, lo, hi) in enumerate(NECK):
            m = self.bus.latest_tele.get(motor_id)
            if m and age < 1.0:
                pos = m.get("pos") or 0
                deg = (pos - CENTER) / TICKS_PER_DEG * mdir
                self.read_vars[i].set(f"{deg:+6.1f}°")
                v = m.get("volt", 0)
                self.volt_vars[i].set(f"{v:4.1f}V" if v else "—")
                t = m.get("temp", 0)
                self.temp_vars[i].set(f"{t:3d}°C" if t else "—")
            else:
                self.read_vars[i].set("—")
                self.volt_vars[i].set("—"); self.temp_vars[i].set("—")
        self.tele_var.set(f"tele age {age*1000:.0f}ms   "
                          f"motors {len(self.bus.latest_tele)}")
        self.root.after(50, self._render_tick)

    def _on_cam_rot_ccw(self, _e=None):
        self._cam_rot_deg = (self._cam_rot_deg + 90) % 360
        self._apply_cam_rot()

    def _on_cam_rot_cw(self, _e=None):
        self._cam_rot_deg = (self._cam_rot_deg - 90) % 360
        self._apply_cam_rot()

    def _on_cam_rot_reset(self, _e=None):
        self._cam_rot_deg = 0
        self._apply_cam_rot()

    def _apply_cam_rot(self):
        """Update local label + push rotation to head_agent so every
        downstream (VR page, hand receiver, etc.) also sees rotated frames.
        Local display keeps its own rotation applied as immediate feedback
        while the server switch propagates through the buffer."""
        deg = self._cam_rot_deg
        self.cam_rot_label.set(f"{deg}°")
        import urllib.request, threading
        host = (self.host_var.get() or "").strip()
        if not host: return
        if ":" not in host: host += ":8000"
        url = f"http://{host}/camera/rotate?deg={deg}"
        def _post():
            try:
                req = urllib.request.Request(url, data=b"", method="POST")
                with urllib.request.urlopen(req, timeout=1.5) as r:
                    _ = r.read()
            except Exception as e:
                print(f"[cam] server rotate err: {e}", flush=True)
        threading.Thread(target=_post, daemon=True).start()

    def _cam_tick(self):
        try:
            img = self.cam.latest
            if img is not None:
                deg = getattr(self, "_cam_rot_deg", 90)
                pil = img.copy()
                if deg:
                    pil = pil.rotate(deg, expand=True)
                pil.thumbnail((CAM_W_DISPLAY, CAM_H_DISPLAY))
                self._cam_tk = ImageTk.PhotoImage(pil)
                if self._cam_img_id is None:
                    self._cam_img_id = self.cam_canvas.create_image(
                        CAM_W_DISPLAY // 2, CAM_H_DISPLAY // 2,
                        image=self._cam_tk)
                else:
                    self.cam_canvas.itemconfig(self._cam_img_id,
                                               image=self._cam_tk)
                self.cam_status_var.set(f"● {self.cam.frames} frames")
            elif self.cam.error:
                self.cam_status_var.set(f"cam err: {self.cam.error[:40]}")
            else:
                self.cam_status_var.set("waiting for stream…")
        except Exception as e:
            self.cam_status_var.set(f"cam ui err: {e}")
        self.root.after(33, self._cam_tick)   # ~30 Hz UI update

    def _on_close(self):
        try: self.cam.stop()
        except Exception: pass
        try: self.bus.disconnect()
        except Exception: pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# late import so head_ghost_gui module loads cleanly first
import mujoco  # noqa: E402


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="",
                    help="RPi ip[:port], e.g. 192.168.0.42 or 192.168.0.42:8000")
    args = ap.parse_args()
    App(args.host).run()
