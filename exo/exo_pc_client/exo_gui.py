#!/usr/bin/env python3
"""Exo dual-ghost GUI — full-fat build.

Two virtual robot ghosts, always visible, side by side.
Each side has its own connection: LEFT = real robot RPi (head_agent),
RIGHT = exoskeleton RPi (exo_agent).  Each side's ghost reflects that
side's real hardware in real time.

Bottom control panel:
  ▶ 對齊機器人 → 外骨骼   |  ★  starts CONVERGE (5s slew, ghost only)
  送馬達到當前 ghost      |  ★  one-shot sync write (needs bus + torque)
  扭力 ON / OFF           |  ★  toggle torque on/off for known sids
  ▶ 進入 LIVE            |  ★  50 Hz exo → motor (requires READY)
  ■ E-stop               |  ★  jump back to IDLE, motors freeze in place

State machine: IDLE → CONVERGE → READY → LIVE → IDLE
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
import mujoco
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
MJCF_PATH = os.path.join(REPO, "models", "QBOT_MJCF", "qbot.xml")

# Reuse RemoteBus + ARMS + MJ_SIGN from qbot_ik_gui
sys.path.insert(0, os.path.join(REPO, "arm"))
from qbot_ik_gui import RemoteBus, ARMS  # noqa: E402

CAL_PATH = os.path.join(REPO, "arm", "qbot_arm_calibration.json")

PANE_W, PANE_H = 440, 500
# Upper bound for the offscreen render buffer — a single pane at fullscreen on
# a 4K display never exceeds this, so the renderer can be resized freely up to
# it without reallocating the MjModel's offscreen buffer.
MAX_PANE_W, MAX_PANE_H = 2160, 2160
GAP_PX = 12
POLL_HZ_STATUS = 10
CONVERGE_SEC   = 5.0

ROBOT_COLOR = (0.30, 0.55, 0.85, 1.0)
EXO_COLOR   = (0.90, 0.55, 0.30, 1.0)

C = {
    "bg": "#0e1620", "panel": "#172130", "card": "#1c2a3a",
    "text": "#d0e0f0", "dim": "#6080a0", "bright": "#eef4fc",
    "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
    "btn": "#1e4870", "btn_green": "#2e7d32", "btn_red": "#7a2828",
}


# ────────────────────────────────────────────────────────────────────
# Ghost world
# ────────────────────────────────────────────────────────────────────
class GhostWorld:
    """One MJCF, one MjData, one Renderer — sized for a single pane.
    Two instances live inside ExoApp, one per side.  Because both worlds
    load the SAME xml + MjModel, writing the same q_map to both produces
    an identical rendering — this is the property the operator relies on
    when converging blue→orange."""
    def __init__(self, path: str, colour):
        self.model = mujoco.MjModel.from_xml_path(path)
        # Offscreen buffer sized to the largest pane we might ever render
        # (fullscreen on a 4K display).  The live renderer is (re)created at
        # the pane's actual pixel size via set_size(), never exceeding this.
        self.model.vis.global_.offwidth = MAX_PANE_W
        self.model.vis.global_.offheight = MAX_PANE_H
        self.data = mujoco.MjData(self.model)
        self.width, self.height = PANE_W, PANE_H
        self.renderer = mujoco.Renderer(self.model, height=PANE_H, width=PANE_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # Face-on view: qbot MJCF's front axis after the base's Z-90 quat
        # ends up along -Y in world frame, so azimuth=90 places the camera
        # on +Y looking at the robot's front chest.
        self.cam.azimuth = 90
        self.cam.elevation = -8
        self.cam.distance = 2.2
        self.cam.lookat[:] = [0.0, 0.0, 1.05]
        self.opt = mujoco.MjvOption()
        self.jqpos: dict[str, int] = {}
        self.jrange: dict[str, tuple[float, float]] = {}
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not n: continue
            self.jqpos[n] = self.model.jnt_qposadr[i]
            # Only cache range for joints that are actually limited.  Some
            # torso / free joints have no meaningful [lo, hi] so we skip them.
            if bool(self.model.jnt_limited[i]):
                lo = float(self.model.jnt_range[i][0])
                hi = float(self.model.jnt_range[i][1])
                self.jrange[n] = (lo, hi)
        for i in range(self.model.ngeom):
            self.model.geom_rgba[i] = colour
        mujoco.mj_forward(self.model, self.data)

    def write_q(self, q_map: dict[str, float]) -> dict[str, str]:
        """Write q values to MJCF qpos.  Joint-range clipping intentionally
        DISABLED for now — operator wants to identify the exo→ghost mapping
        first, then reinstate limits once alignment is confirmed.  The
        returned dict is kept for API compatibility but stays empty."""
        for name, val in q_map.items():
            adr = self.jqpos.get(name)
            if adr is None: continue
            self.data.qpos[adr] = float(val)
        mujoco.mj_forward(self.model, self.data)
        return {}

    def set_size(self, w: int, h: int):
        """Resize the live renderer to (w, h) pixels.  No-op if unchanged.
        Clamped to [160, MAX_PANE_*] so a stray 0/1-px Configure event or an
        oversized window can't crash the offscreen buffer."""
        w = max(160, min(int(w), MAX_PANE_W))
        h = max(160, min(int(h), MAX_PANE_H))
        if w == self.width and h == self.height:
            return
        try:
            self.renderer.close()
        except Exception:
            pass
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.width, self.height = w, h

    def render(self) -> Image.Image:
        self.renderer.update_scene(self.data, self.cam, self.opt)
        return Image.fromarray(self.renderer.render())


# ────────────────────────────────────────────────────────────────────
# Connections
# ────────────────────────────────────────────────────────────────────
class ExoConn:
    """HTTP-poll exo_agent's /status endpoint at 10 Hz.  q_map is a dict of
    MJCF joint names → radians (already mapped by exo_agent's config.py)."""
    def __init__(self):
        self.host = ""
        self.q: dict[str, float] = {}
        self.error = ""
        self.connected = False
        self.frames = 0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def connect(self, host: str):
        self.disconnect()
        self.host = host
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def disconnect(self):
        self._stop.set()
        self.connected = False
        self._th = None

    def _loop(self):
        import urllib.request, json as _j
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(
                        f"http://{self.host}/status", timeout=1.0) as r:
                    d = _j.loads(r.read().decode())
                self.q = dict(d.get("q") or {})
                self.error = d.get("motor_error") or ""
                self.connected = True
                self.frames += 1
            except Exception as e:
                self.error = str(e)
                self.connected = False
            time.sleep(1.0 / POLL_HZ_STATUS)


class RobotConn:
    """Connects to head_agent's WS (via RemoteBus for arm tele + motor writes)
    AND polls /status for the neck_q dict.  Aggregates into a single q_map of
    MJCF joint names → radians."""
    def __init__(self):
        self.host = ""
        self.bus = RemoteBus()
        self.cal: dict | None = None
        self.q: dict[str, float] = {}
        self.error = ""
        self.frames = 0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        # Cache arm channel → (jname, sid) list once
        self._arm_channels: list[tuple[str, str, int]] = []
        for arm in ("L", "R"):
            for ji, (sid, _lbl, _gear, _lo, _hi, jname) in enumerate(
                    ARMS[arm]["joints"]):
                if jname is not None:
                    self._arm_channels.append((arm, jname, sid))
        self._load_cal()

    def _load_cal(self):
        if not os.path.exists(CAL_PATH):
            return
        try:
            d = json.load(open(CAL_PATH))
            out = {}
            for arm in ("L", "R"):
                a = d.get(arm) or {}
                pts = a["points"] if isinstance(a, dict) and "points" in a else a
                if isinstance(pts, list):
                    out[arm] = [
                        {k: float(p.get(k, 0)) for k in
                         ("tick_lo", "q_lo", "tick_hi", "q_hi")}
                        for p in pts
                    ]
            self.cal = out or None
        except Exception as e:
            self.error = f"cal load err: {e}"

    def _tick_to_q(self, arm: str, ji: int, tick: int) -> float | None:
        if not self.cal or arm not in self.cal: return None
        p = self.cal[arm][ji]
        denom = p["tick_hi"] - p["tick_lo"]
        if abs(denom) < 1e-6: return p["q_lo"]
        return p["q_lo"] + (tick - p["tick_lo"]) * (p["q_hi"] - p["q_lo"]) / denom

    def connect(self, host: str):
        self.disconnect()
        self.host = host
        ok, msg = self.bus.open(host)
        self.error = "" if ok else msg
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return ok, msg

    def disconnect(self):
        self._stop.set()
        try: self.bus.close()
        except Exception: pass
        self._th = None

    @property
    def connected(self) -> bool: return self.bus.is_open()

    def _loop(self):
        import urllib.request, json as _j
        while not self._stop.is_set():
            # 1) neck angles from /status
            try:
                with urllib.request.urlopen(
                        f"http://{self.host}/status", timeout=1.0) as r:
                    d = _j.loads(r.read().decode())
                nq = d.get("neck_q") or {}
                self.q["ub_neck_yaw"]   = math.radians(float(nq.get("yaw",   0)))
                self.q["ub_neck_pitch"] = math.radians(float(nq.get("pitch", 0)))
                self.q["ub_neck_roll"]  = math.radians(float(nq.get("roll",  0)))
            except Exception as e:
                self.error = f"status err: {e}"
            # 2) arm joints via WS tele (bus.latest_tele filled by RemoteBus)
            tele = getattr(self.bus, "latest_tele", {}) or {}
            channel_idx: dict[str, int] = {}
            for arm in ("L", "R"):
                for ji, (sid, _l, _g, _lo, _hi, jname) in enumerate(
                        ARMS[arm]["joints"]):
                    if jname is None: continue
                    channel_idx[(arm, sid)] = ji
                    m = tele.get(int(sid))
                    if m is None: continue
                    pos = m.get("pos")
                    if pos is None: continue
                    q = self._tick_to_q(arm, ji, int(pos))
                    if q is not None:
                        self.q[jname] = float(q)
            self.frames += 1
            time.sleep(1.0 / POLL_HZ_STATUS)


# ────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────
class ExoApp:
    def __init__(self, exo_host: str, robot_host: str):
        self.robot_world = GhostWorld(MJCF_PATH, ROBOT_COLOR)
        self.exo_world   = GhostWorld(MJCF_PATH, EXO_COLOR)
        self.robot_conn  = RobotConn()
        self.exo_conn    = ExoConn()

        # Pre-fill hosts (from CLI defaults) but let user edit + connect
        self._exo_host_default   = exo_host
        self._robot_host_default = robot_host
        # Track ghost snapshot state for CONVERGE
        self._converge_t0 = 0.0
        self._converge_start_q: dict[str, float] = {}
        self.state = "IDLE"

        self.root = tk.Tk()
        self.root.title("FR01 外骨骼控制 — 雙 ghost")
        self.root.configure(bg=C["bg"])
        # Compute window size
        win_w = 2 * PANE_W + GAP_PX + 40
        win_h = PANE_H + 320
        self.root.geometry(f"{win_w}x{win_h}")
        self._build_ui()
        # Start maximized so the ghosts fill the screen; F11 toggles true
        # borderless fullscreen, Esc leaves it.
        self._fullscreen = False
        self.root.bind("<F11>", self._toggle_fullscreen)
        self.root.bind("<Escape>", lambda e: self._set_fullscreen(False))
        try:
            self.root.attributes("-zoomed", True)   # X11 maximize
        except tk.TclError:
            self.root.after(80, lambda: self.root.state("zoomed"))
        self.root.after(50, self._render_tick)
        self.root.after(50, self._ghost_tick)

    def _set_fullscreen(self, on: bool):
        self._fullscreen = on
        self.root.attributes("-fullscreen", on)

    def _toggle_fullscreen(self, _evt=None):
        self._set_fullscreen(not self._fullscreen)

    # ── build UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg="#1a237e"); hdr.pack(fill="x")
        tk.Label(hdr,
                 text="FR01 外骨骼控制 (左=機器人 藍青  右=外骨骼 橘紅)",
                 font=("DejaVu Sans", 14, "bold"),
                 bg="#1a237e", fg="white").pack(side="left", padx=10, pady=6)
        self.state_var = tk.StringVar(value="[ IDLE ]")
        tk.Label(hdr, textvariable=self.state_var,
                 bg="#1a237e", fg="#7ed3ff",
                 font=("DejaVu Sans Mono", 12, "bold")
                 ).pack(side="right", padx=10)

        # Two side-by-side panes, each with its own connection widgets
        body = tk.Frame(self.root, bg=C["bg"]); body.pack(fill="both", expand=True,
                                                          padx=6, pady=4)

        self.robot_pane, self.robot_ip_var, self.robot_status_var, \
            self.robot_btn = self._build_pane(
                body, "機器人 (Robot)", "#5aa2f0",
                self._robot_host_default, self._toggle_robot_conn)
        self.robot_pane.pack(side="left", padx=(0, GAP_PX // 2),
                             fill="both", expand=True)

        self.exo_pane, self.exo_ip_var, self.exo_status_var, \
            self.exo_btn = self._build_pane(
                body, "外骨骼 (Exo)", "#f0a05a",
                self._exo_host_default, self._toggle_exo_conn)
        self.exo_pane.pack(side="left", padx=(GAP_PX // 2, 0),
                           fill="both", expand=True)

        # Bottom control bar
        ctrl = tk.LabelFrame(self.root, text=" 控制 ",
                             bg=C["panel"], fg=C["text"],
                             font=("DejaVu Sans Mono", 10, "bold"))
        ctrl.pack(fill="x", padx=6, pady=(2, 4))
        row = tk.Frame(ctrl, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        tk.Button(row, text="[R] 對齊機器人 → 外骨骼",
                  bg=C["btn"], fg="white", width=22,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=self._start_converge
                  ).pack(side="left", padx=3)
        tk.Button(row, text="送馬達到當前 ghost",
                  bg=C["warn"], fg="white", width=20,
                  command=self._send_ghost_snapshot
                  ).pack(side="left", padx=3)
        self.torque_var = tk.BooleanVar(value=False)
        tk.Checkbutton(row, text="扭力 ON",
                       variable=self.torque_var,
                       bg=C["panel"], fg=C["ok"],
                       selectcolor=C["card"], activebackground=C["panel"],
                       font=("DejaVu Sans Mono", 10, "bold"),
                       command=self._on_torque_toggle
                       ).pack(side="left", padx=8)
        tk.Button(row, text="[>] 進入 LIVE",
                  bg=C["btn_green"], fg="white", width=14,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=self._enter_live
                  ).pack(side="left", padx=3)
        tk.Button(row, text="[X] E-stop",
                  bg=C["btn_red"], fg="white", width=12,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=self._e_stop
                  ).pack(side="right", padx=3)

        self.info = tk.StringVar(value="ready.")
        tk.Label(self.root, textvariable=self.info,
                 bg=C["bg"], fg=C["dim"], anchor="w", justify="left",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=6, pady=(0, 6))

    def _build_pane(self, parent, title, title_colour,
                    default_host, connect_cmd):
        pane = tk.Frame(parent, bg=C["panel"], bd=1, relief="solid")
        tk.Label(pane, text=title, bg=C["panel"], fg=title_colour,
                 font=("DejaVu Sans", 12, "bold")
                 ).pack(padx=6, pady=(4, 2))
        # Connection row
        row = tk.Frame(pane, bg=C["panel"]); row.pack(fill="x", padx=6)
        tk.Label(row, text="Host:", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        ip_var = tk.StringVar(value=default_host)
        tk.Entry(row, textvariable=ip_var, width=18,
                 bg=C["card"], fg=C["text"],
                 insertbackground=C["text"]
                 ).pack(side="left", padx=4)
        btn = tk.Button(row, text="Connect",
                        bg=C["btn_green"], fg="white", width=10,
                        command=connect_cmd)
        btn.pack(side="left", padx=4)
        # Status line
        status_var = tk.StringVar(value="idle")
        tk.Label(pane, textvariable=status_var, bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9), anchor="w"
                 ).pack(fill="x", padx=8, pady=(2, 4))
        # Canvas + optional trigger indicators (exo pane only) side by side
        cav_row = tk.Frame(pane, bg=C["panel"])
        cav_row.pack(padx=6, pady=6, fill="both", expand=True)
        canvas = tk.Canvas(cav_row, width=PANE_W, height=PANE_H,
                           bg="#000", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)
        # Save canvas ref via attribute name convention
        if title.startswith("機器人"):
            self.robot_canvas = canvas
            self._robot_img_id = None; self._robot_tk = None
            self._bind_camera_drag(canvas, self.robot_world)
        else:
            self.exo_canvas = canvas
            self._exo_img_id = None; self._exo_tk = None
            self._bind_camera_drag(canvas, self.exo_world)
            # Trigger LED strip: two circles (R grip, L grip) that turn green
            # when q > 0 (板機推向大值) or red when q < 0 (推向小值),
            # brightness scales with |q|.  Save-zero = both dark grey.
            leds = tk.Frame(cav_row, bg=C["panel"])
            leds.pack(side="left", padx=(10, 0), fill="y")
            def _mk(label):
                tk.Label(leds, text=label, bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold")
                         ).pack(pady=(6, 0))
                c = tk.Canvas(leds, width=60, height=60, bg=C["panel"],
                              highlightthickness=0)
                c.pack(pady=(2, 4))
                cid = c.create_oval(6, 6, 54, 54, fill="#333333", outline="#555")
                return c, cid
            self.led_r_c, self.led_r_id = _mk("R 板機")
            self.led_l_c, self.led_l_id = _mk("L 板機")
        return pane, ip_var, status_var, btn

    # ── camera drag ────────────────────────────────────────────────────
    def _bind_camera_drag(self, canvas: tk.Canvas, world: "GhostWorld"):
        """Left-drag = orbit (azimuth / elevation).
        Right-drag = pan lookat in the camera plane.
        Wheel      = zoom (cam.distance)."""
        state = {"x": 0, "y": 0}

        def on_press(e):
            canvas.focus_set()
            state["x"], state["y"] = e.x, e.y
        def on_drag(e):
            dx = e.x - state["x"]; dy = e.y - state["y"]
            state["x"], state["y"] = e.x, e.y
            world.cam.azimuth   = (world.cam.azimuth - dx * 0.6) % 360
            world.cam.elevation = max(-89.0, min(89.0,
                                       world.cam.elevation - dy * 0.6))

        def on_rpress(e):
            canvas.focus_set()
            state["x"], state["y"] = e.x, e.y
        def on_rdrag(e):
            dx = e.x - state["x"]; dy = e.y - state["y"]
            state["x"], state["y"] = e.x, e.y
            az = math.radians(world.cam.azimuth)
            k = world.cam.distance * 0.0015
            world.cam.lookat[0] -= (-math.sin(az)) * dx * k
            world.cam.lookat[1] -= ( math.cos(az)) * dx * k
            world.cam.lookat[2] += dy * k

        def on_mpress(e):
            canvas.focus_set()
            state["x"], state["y"] = e.x, e.y
        def on_mdrag(e):
            on_rdrag(e)

        def on_wheel_up(_e):
            world.cam.distance = max(0.3, world.cam.distance * 0.9)
        def on_wheel_dn(_e):
            world.cam.distance = min(20.0, world.cam.distance * 1.1)
        def on_wheel_win(e):
            (on_wheel_up if e.delta > 0 else on_wheel_dn)(e)

        canvas.bind("<ButtonPress-1>",  on_press)
        canvas.bind("<B1-Motion>",      on_drag)
        canvas.bind("<ButtonPress-2>",  on_mpress)
        canvas.bind("<B2-Motion>",      on_mdrag)
        canvas.bind("<ButtonPress-3>",  on_rpress)
        canvas.bind("<B3-Motion>",      on_rdrag)
        canvas.bind("<Button-4>",       on_wheel_up)
        canvas.bind("<Button-5>",       on_wheel_dn)
        canvas.bind("<MouseWheel>",     on_wheel_win)
        # Some window managers only deliver events after the widget takes
        # keyboard focus — force it up front so the first drag registers.
        canvas.config(cursor="fleur")
        canvas.bind("<Enter>", lambda e: canvas.focus_set())

    # ── connection toggles ─────────────────────────────────────────────
    def _toggle_robot_conn(self):
        if self.robot_conn.connected:
            self.robot_conn.disconnect()
            self.robot_btn.config(text="Connect", bg=C["btn_green"])
            self.robot_status_var.set("disconnected")
        else:
            host = (self.robot_ip_var.get() or "").strip()
            if ":" not in host: host += ":8000"
            ok, msg = self.robot_conn.connect(host)
            self.robot_status_var.set(msg)
            if ok:
                self.robot_btn.config(text="Disconnect", bg=C["btn_red"])

    def _toggle_exo_conn(self):
        if self.exo_conn.connected:
            self.exo_conn.disconnect()
            self.exo_btn.config(text="Connect", bg=C["btn_green"])
            self.exo_status_var.set("disconnected")
        else:
            host = (self.exo_ip_var.get() or "").strip()
            if ":" not in host: host += ":8200"
            self.exo_conn.connect(host)
            self.exo_status_var.set(f"polling {host}")
            self.exo_btn.config(text="Disconnect", bg=C["btn_red"])

    # ── control-panel handlers ─────────────────────────────────────────
    def _start_converge(self):
        if not self.exo_conn.q:
            self.info.set("外骨骼還沒送 pose 過來,先確認 Exo Connect 有連上")
            return
        self.state = "CONVERGE"
        self._converge_t0 = time.monotonic()
        self._converge_start_q = dict(self.robot_conn.q)
        self.state_var.set(f"[ CONVERGE ] slewing {CONVERGE_SEC:.0f}s")

    def _send_ghost_snapshot(self):
        if not self.robot_conn.bus.is_open():
            self.info.set("機器人 bus 未連,無法送馬達")
            return
        cmds = self._robot_q_to_tick_cmds()
        if cmds and hasattr(self.robot_conn.bus, "write_pos"):
            for sid, tick in cmds:
                self.robot_conn.bus.write_pos(int(sid), int(tick), 800, 30)
            self.info.set(f"已送 {len(cmds)} 顆 tick 一次(snapshot)")
        else:
            self.info.set("沒有 arm cal 或 bus 沒 write_pos")

    def _on_torque_toggle(self):
        on = bool(self.torque_var.get())
        if not self.robot_conn.bus.is_open():
            self.info.set("機器人 bus 未連,無法切扭力")
            self.torque_var.set(not on)   # revert
            return
        # torque_all covers all scanned sids on head_agent
        try:
            self.robot_conn.bus._out.put({
                "op": "torque_all", "on": bool(on)})
            self.info.set(f"扭力 → {'ON' if on else 'OFF'}")
        except Exception as e:
            self.info.set(f"torque_all send err: {e}")

    def _enter_live(self):
        if self.state != "READY":
            self.info.set("需先按對齊 → 等 CONVERGE 完成後才能進 LIVE")
            return
        self.state = "LIVE"
        self.state_var.set("[ LIVE ] exo → motors")

    def _e_stop(self):
        self.state = "IDLE"
        self.state_var.set("[ IDLE ]")
        self.info.set("E-stop:狀態回 IDLE,馬達停在當前 tick")

    # ── tick → cmd conversion ──────────────────────────────────────────
    def _robot_q_to_tick_cmds(self) -> list:
        """Convert the robot ghost's current arm qpos to (sid, tick) pairs.

        Cross-checked against ``qbot_ik_gui.py`` conventions:
          * cal file (``qbot_arm_calibration.json``) is the operator's own
            calibration — treated as authoritative.
          * ``MJ_SIGN`` is NOT applied here (learned from
            motion_replay_gui: MJ_SIGN only flips the ghost display in
            arm IK, motor-writes go through cal directly).
          * q is clamped to [min(q_lo,q_hi), max(q_lo,q_hi)] before
            interpolation so a runaway exo joint can't overshoot the
            motor's calibrated safe range.
          * Final tick is clamped to physical [0, 4095] as a hard
            guardrail.

        Head/neck is intentionally omitted — head_agent has its own
        NECK_MAPPING and IMU pipeline; sending redundant writes would
        conflict.  Wire it in only if you disable head_agent's direct
        IMU→motor path first.
        """
        if not self.robot_conn.cal:
            return []
        cmds = []
        # Skip head neck sids so we don't fight head_agent's IMU writer.
        HEAD_SIDS = {1, 2, 3}
        for (arm, jname, sid) in self.robot_conn._arm_channels:
            if sid in HEAD_SIDS: continue
            adr = self.robot_world.jqpos.get(jname)
            if adr is None: continue
            q_mjcf = float(self.robot_world.data.qpos[adr])
            ji = next(i for i, j in enumerate(ARMS[arm]["joints"])
                      if j[5] == jname)
            p = self.robot_conn.cal[arm][ji]
            q_min = min(p["q_lo"], p["q_hi"])
            q_max = max(p["q_lo"], p["q_hi"])
            q = max(q_min, min(q_max, q_mjcf))
            denom = p["q_hi"] - p["q_lo"]
            if abs(denom) < 1e-9:
                tick = p["tick_lo"]
            else:
                tick = p["tick_lo"] + (q - p["q_lo"]) * (
                    p["tick_hi"] - p["tick_lo"]) / denom
            cmds.append((int(sid), int(max(0, min(4095, tick)))))
        return cmds

    # ── main loops ─────────────────────────────────────────────────────
    def _ghost_tick(self):
        clipped_exo: dict[str, str] = {}
        clipped_robot: dict[str, str] = {}
        # Update exo ghost from its own conn
        if self.exo_conn.q:
            clipped_exo = self.exo_world.write_q(self.exo_conn.q)
        # Update robot ghost from real telemetry — unless we're in
        # CONVERGE/LIVE, in which case we drive it toward exo.
        if self.state in ("IDLE", "READY"):
            if self.robot_conn.q:
                clipped_robot = self.robot_world.write_q(self.robot_conn.q)
        elif self.state == "CONVERGE":
            t = time.monotonic() - self._converge_t0
            frac = min(1.0, t / CONVERGE_SEC)
            merged: dict[str, float] = {}
            for k, exo_q in self.exo_conn.q.items():
                start = self._converge_start_q.get(k,
                        self.robot_conn.q.get(k, 0.0))
                merged[k] = start + (exo_q - start) * frac
            clipped_robot = self.robot_world.write_q(merged)
            if frac >= 1.0:
                self.state = "READY"
                self.state_var.set(
                    "[ READY ] 檢查 ghost,再按送馬達 / 進入 LIVE")
        elif self.state == "LIVE":
            # Robot ghost = exo pose directly (BOTH ghosts see the SAME
            # q_map — writing to identical MjModel guarantees an identical
            # rendering so the operator's alignment is trustworthy).
            clipped_robot = self.robot_world.write_q(self.exo_conn.q)
            # Push to motors every tick
            if self.robot_conn.bus.is_open() and self.robot_conn.cal:
                cmds = self._robot_q_to_tick_cmds()
                if cmds and hasattr(self.robot_conn.bus, "write_pos"):
                    for sid, tick in cmds:
                        self.robot_conn.bus.write_pos(sid, tick, 1000, 50)

        # Status lines — include a compact `!clip` warning if either side
        # was over its MJCF limit, so the operator sees which joint is
        # saturating rather than silently going wrong.
        rob_flag = f"  !clip {','.join(clipped_robot.keys())[:30]}" if clipped_robot else ""
        exo_flag = f"  !clip {','.join(clipped_exo.keys())[:30]}" if clipped_exo else ""
        self.robot_status_var.set(
            f"{'● 連接' if self.robot_conn.connected else '○ 未連'} "
            f"  polls={self.robot_conn.frames}"
            f"  err={self.robot_conn.error[:30] if self.robot_conn.error else ''}"
            f"{rob_flag}")
        self.exo_status_var.set(
            f"{'● 連接' if self.exo_conn.connected else '○ 未連'} "
            f"  polls={self.exo_conn.frames}"
            f"  err={self.exo_conn.error[:30] if self.exo_conn.error else ''}"
            f"{exo_flag}")
        # Joint delta readout — sorted by |Δ| desc so the biggest mismatch
        # bubbles to the top (this is what the operator watches during
        # alignment / LIVE control).
        deltas = []
        for k in sorted(self.exo_conn.q):
            e = self.exo_conn.q.get(k, 0.0)
            r = self.robot_conn.q.get(k, 0.0)
            deltas.append((abs(e - r), k, e, r))
        deltas.sort(reverse=True)
        lines = []
        for _dabs, k, e, r in deltas[:5]:
            lines.append(f"{k[3:]:<24} exo={e:+.2f}  rob={r:+.2f}  Δ={e-r:+.2f}")
        self.info.set("\n".join(lines))
        # Trigger LEDs — recolour based on grip q sign + magnitude.
        # q = 0 → dark grey (idle).  q > 0 → green tint proportional to
        # min(1, |q|).  q < 0 → red tint.  Radius is fixed; only the fill
        # colour changes so the operator can eyeball trigger state from a
        # distance without reading numbers.
        def _grip_colour(q: float) -> str:
            mag = min(1.0, abs(q) / 1.2)     # saturate at ~1.2 rad
            base = int(60 + 195 * mag)       # 60..255 tint
            base = max(60, min(255, base))
            if abs(q) < 0.05: return "#333333"
            if q > 0:
                # green tint, brighter with magnitude
                return f"#00{base:02x}00"
            else:
                return f"#{base:02x}0000"    # red tint
        try:
            gr = float(self.exo_conn.q.get("exo_right_grip", 0.0))
            gl = float(self.exo_conn.q.get("exo_left_grip",  0.0))
            self.led_r_c.itemconfig(self.led_r_id, fill=_grip_colour(gr))
            self.led_l_c.itemconfig(self.led_l_id, fill=_grip_colour(gl))
        except Exception:
            pass
        self.root.after(50, self._ghost_tick)

    def _render_tick(self):
        # Match each renderer to its canvas's live pixel size so the ghost
        # scales to fill the pane (incl. fullscreen), not a fixed 440x500.
        try:
            self.robot_world.set_size(self.robot_canvas.winfo_width(),
                                      self.robot_canvas.winfo_height())
            self.exo_world.set_size(self.exo_canvas.winfo_width(),
                                    self.exo_canvas.winfo_height())
        except Exception as e:
            print(f"[resize] {e}", flush=True)
        try:
            img = self.robot_world.render()
            self._robot_tk = ImageTk.PhotoImage(img)
            if self._robot_img_id is None:
                self._robot_img_id = self.robot_canvas.create_image(
                    0, 0, image=self._robot_tk, anchor="nw")
            else:
                self.robot_canvas.itemconfig(self._robot_img_id,
                                             image=self._robot_tk)
        except Exception as e:
            print(f"[render robot] {e}", flush=True)
        try:
            img = self.exo_world.render()
            self._exo_tk = ImageTk.PhotoImage(img)
            if self._exo_img_id is None:
                self._exo_img_id = self.exo_canvas.create_image(
                    0, 0, image=self._exo_tk, anchor="nw")
            else:
                self.exo_canvas.itemconfig(self._exo_img_id,
                                           image=self._exo_tk)
        except Exception as e:
            print(f"[render exo] {e}", flush=True)
        self.root.after(50, self._render_tick)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo-host",   default="127.0.0.1:8200")
    ap.add_argument("--robot-host", default="192.168.0.123:8000")
    args = ap.parse_args()
    ExoApp(args.exo_host, args.robot_host).run()


if __name__ == "__main__":
    main()
