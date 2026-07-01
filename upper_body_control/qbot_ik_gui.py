#!/usr/bin/env python3
"""Q-BOT integrated IK GUI — single window, ghost robot + interactive IK.

Right side  : MuJoCo offscreen render of the full Q-BOT (translucent "ghost"
              materials).  Mouse:
                LMB drag on the wrist           ▸ drag the end-effector (IK)
                Shift + LMB drag on the wrist   ▸ drag along world-Z
                LMB drag elsewhere              ▸ orbit camera
                RMB drag                        ▸ pan camera (lookat)
                Scroll                          ▸ zoom

Left side   : 5 joint sliders, IK target XYZ readout, telemetry from real
              motors, Connect / Send-to-Robot / E-STOP.

Collisions  : Every IK solution is forward-simulated with mj_forward; if any
              ARM geom touches the rest of the robot (self-collision) the
              move is rejected and the wrist snaps back.  Drag indicator turns
              red.  Ground-foot contacts are ignored.

Joint indexing (5 per arm):
  J0  shoulder pitch  ← virtual / new motor (ID 10/20, NOT in URDF yet)
  J1  shoulder yaw    (MJCF: ub_*_shoulder)
  J2  lateral raise   (MJCF: ub_*_lateral_raise)
  J3  arm twist       (MJCF: ub_*_arm_twist)
  J4  elbow           (MJCF: ub_*_elbow)
IK uses J1-J4 only (J0 has no URDF link yet; manual override slider).
"""
from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np
from scipy import optimize
import mujoco
from PIL import Image, ImageTk

sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

# ── paths & constants ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.normpath(os.path.join(HERE, "..", "QBOT_MJCF", "qbot.xml"))

STEPS_PER_RAD = 2048.0 / math.pi
CENTER = 2048
DEFAULT_PORT = "/dev/ttyACM0"
# CH343 USB-serial serial number printed on the arm adapter; used to auto-find
# the right /dev/ttyACM* even after a reconnect bumps the device index.
ARM_USB_SERIAL = "5AE6082200"


def _find_arm_port(default=DEFAULT_PORT):
    """Return /dev/ttyACM* whose USB serial number matches ARM_USB_SERIAL."""
    try:
        import serial.tools.list_ports as _lp
        for p in _lp.comports():
            if (p.serial_number or "").upper() == ARM_USB_SERIAL.upper():
                return p.device
    except Exception:
        pass
    return default
DEFAULT_BAUD = 1_000_000
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63

RENDER_W = 900
RENDER_H = 680

C = {
    "bg": "#111820", "panel": "#172030", "card": "#1c2a3a",
    "text": "#d0e0f0", "dim": "#6080a0", "bright": "#eef4fc",
    "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
    "la": "#5898e8", "ra": "#e08848", "target": "#50c880",
    "collide": "#ff5066",
    "btn": "#1e4870", "btn_warn": "#7a4818", "btn_danger": "#7a2828",
}

ARMS = {
    "L": dict(
        joints=[
            (20, "Shoul Pitch",   4, -math.pi/2, math.pi/2, None),   # virtual
            (21, "Shoul Yaw",     4, -math.pi/2, math.pi/2, "ub_left_shoulder"),
            (22, "Lateral Raise", 4, -math.pi/2, math.pi/2, "ub_left_lateral_raise"),
            (23, "Arm Twist",     4, -math.pi/2, math.pi/2, "ub_left_arm_twist"),
            (24, "Elbow",         1,         0.0, 2.35619,  "ub_left_elbow"),
        ],
        ee_body="ub_115_2",
        # bodies considered part of this arm for collision filtering
        arm_bodies=("ub_l_4", "ub_l_3", "ub_link_7", "ub_l_2", "ub_113_2",
                    "ub_l_1", "ub_112_2", "ub_l", "ub_115_2", "ub_114_2"),
        col=C["la"],
    ),
    "R": dict(
        joints=[
            (10, "Shoul Pitch",   4, -math.pi/2, math.pi/2, None),   # virtual
            (11, "Shoul Yaw",     4, -math.pi/2, math.pi/2, "ub_right_shoulder"),
            (12, "Lateral Raise", 4, -math.pi/2, math.pi/2, "ub_right_lateral_raise"),
            (13, "Arm Twist",     4, -math.pi/2, math.pi/2, "ub_right_arm_twist"),
            (14, "Elbow",         1,         0.0, 2.35619,  "ub_right_elbow"),
        ],
        ee_body="ub_115",
        arm_bodies=("ub_link_2", "ub_link", "ub_link_6", "ub_2", "ub_113",
                    "ub_fake_motor_", "ub_112", "ub_1", "ub_115", "ub_114"),
        col=C["ra"],
    ),
}


# ── MuJoCo world + camera ────────────────────────────────────────────────────
class World:
    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(path)
        # offscreen framebuffer must be at least as large as our render size
        self.model.vis.global_.offwidth = RENDER_W
        self.model.vis.global_.offheight = RENDER_H
        self.data = mujoco.MjData(self.model)
        self._apply_ghost_palette()
        self._build_indices()
        self.renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = 135
        self.cam.elevation = -10
        self.cam.distance = 2.0
        self.cam.lookat[:] = [0.0, 0.0, 1.05]
        self.scene_option = mujoco.MjvOption()
        # baseline collision count (e.g. feet/floor) so we ignore those
        mujoco.mj_forward(self.model, self.data)
        self._baseline_ncon = self.data.ncon

    # ghost vs highlight colours (see apply_arm_highlight)
    _GHOST_RGBA  = [0.40, 0.45, 0.52, 0.42]   # translucent grey-blue
    _SOLID_RGBA  = [0.95, 0.55, 0.20, 1.00]   # warm orange (currently tested)

    def _apply_ghost_palette(self):
        m = self.model
        m.vis.headlight.ambient[:] = [0.55, 0.55, 0.6]
        m.vis.headlight.diffuse[:] = [0.9, 0.9, 0.92]
        m.vis.headlight.specular[:] = [0.3, 0.3, 0.35]
        for i in range(m.nmat):
            m.mat_rgba[i] = self._GHOST_RGBA
            m.mat_specular[i] = 0.35
            m.mat_shininess[i] = 0.3
        for i in range(m.ngeom):
            m.geom_rgba[i] = self._GHOST_RGBA

    def apply_arm_highlight(self, arm_key: str):
        """Paint the active arm's geoms solid orange; everything else ghost."""
        m = self.model
        # 1) reset everything to ghost
        for i in range(m.ngeom):
            m.geom_rgba[i] = self._GHOST_RGBA
        # 2) collect bodies of the active arm
        bids = {self.bid[n] for n in ARMS[arm_key]["arm_bodies"] if n in self.bid}
        # 3) repaint geoms that live in those bodies
        for i in range(m.ngeom):
            if m.geom_bodyid[i] in bids:
                m.geom_rgba[i] = self._SOLID_RGBA

    def _build_indices(self):
        m = self.model
        self.jqpos: dict[str, int] = {}
        for i in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n: self.jqpos[n] = m.jnt_qposadr[i]
        self.bid: dict[str, int] = {}
        for i in range(m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            if n: self.bid[n] = i

    def set_arm_qpos(self, arm_key: str, q5: np.ndarray):
        spec = ARMS[arm_key]
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            if jname is None: continue
            adr = self.jqpos.get(jname)
            if adr is None: continue
            self.data.qpos[adr] = float(np.clip(q5[i], lo, hi))
        mujoco.mj_forward(self.model, self.data)

    def ee_position(self, arm_key: str) -> np.ndarray:
        return self.data.xpos[self.bid[ARMS[arm_key]["ee_body"]]].copy()

    def check_self_collision(self, arm_key: str) -> bool:
        """Return True if any arm geom touches a non-arm geom right now."""
        m, d = self.model, self.data
        if d.ncon <= self._baseline_ncon:
            return False
        arm_bids = {self.bid[n] for n in ARMS[arm_key]["arm_bodies"] if n in self.bid}
        for i in range(d.ncon):
            c = d.contact[i]
            b1 = m.geom_bodyid[c.geom1]
            b2 = m.geom_bodyid[c.geom2]
            in1 = b1 in arm_bids
            in2 = b2 in arm_bids
            if in1 != in2:        # arm vs non-arm
                return True
            if in1 and in2 and abs(b1 - b2) > 2:  # adjacent links are fine
                return True
        return False

    # ── camera helpers ──────────────────────────────────────────────────────
    def cam_basis(self):
        """Return (forward, right, up) unit vectors of the free camera."""
        az = math.radians(self.cam.azimuth)
        el = math.radians(self.cam.elevation)
        # MuJoCo free camera: forward goes from camera→lookat
        fwd = np.array([math.cos(el)*math.cos(az),
                        math.cos(el)*math.sin(az),
                        math.sin(el)])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right) or 1.0
        up = np.cross(right, fwd)
        up /= np.linalg.norm(up) or 1.0
        return fwd, right, up

    def cam_pos(self):
        fwd, _, _ = self.cam_basis()
        return np.array(self.cam.lookat) - fwd * self.cam.distance

    def fovy_rad(self):
        # MuJoCo stores FOV in degrees at model.vis.global_.fovy
        return math.radians(float(self.model.vis.global_.fovy))

    def world_to_screen(self, p: np.ndarray, w: int, h: int):
        """Project a world point to pixel (x, y, depth)."""
        cpos = self.cam_pos()
        fwd, right, up = self.cam_basis()
        rel = p - cpos
        depth = float(np.dot(rel, fwd))
        if depth <= 1e-6:
            return None
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        fov = self.fovy_rad()
        h_world = 2.0 * depth * math.tan(fov / 2.0)
        w_world = h_world * (w / h)
        sx = w / 2.0 + (x_cam / w_world) * w
        sy = h / 2.0 - (y_cam / h_world) * h
        return sx, sy, depth

    def screen_delta_to_world(self, dx_pix: float, dy_pix: float, depth: float,
                              w: int, h: int) -> np.ndarray:
        fov = self.fovy_rad()
        h_world = 2.0 * depth * math.tan(fov / 2.0)
        scale = h_world / h
        _, right, up = self.cam_basis()
        return right * (dx_pix * scale) - up * (dy_pix * scale)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, self.cam, self.scene_option)
        return self.renderer.render()


# ── hardware bus ─────────────────────────────────────────────────────────────
class Bus:
    def __init__(self):
        self._ph = None
        self._pk: sms_sts | None = None
        self._lock = threading.Lock()

    def is_open(self): return self._pk is not None

    def open(self, port, baud):
        self.close()
        self._ph = PortHandler(port)
        if not self._ph.openPort(): return False, "openPort failed"
        if not self._ph.setBaudRate(baud):
            self._ph.closePort()
            return False, "setBaudRate failed"
        self._pk = sms_sts(self._ph)
        return True, "ok"

    def close(self):
        with self._lock:
            if self._ph is not None:
                try: self._ph.closePort()
                except Exception: pass
            self._ph = None; self._pk = None

    def write_pos(self, sid, step, speed, acc):
        with self._lock:
            if not self._pk: return False
            r, _ = self._pk.WritePosEx(sid, int(step), int(speed), int(acc))
            return r == COMM_SUCCESS

    def torque(self, sid, on):
        with self._lock:
            if not self._pk: return False
            r, _ = self._pk.write1ByteTxRx(sid, ADDR_TORQUE_ENABLE, 1 if on else 0)
            return r == COMM_SUCCESS

    def read_quick(self, sid):
        with self._lock:
            if not self._pk: return None
            pos, _, comm, _ = self._pk.ReadPosSpeed(sid)
            if comm != COMM_SUCCESS: return None
            volt, comm, _ = self._pk.read1ByteTxRx(sid, ADDR_PRESENT_VOLTAGE)
            v = volt * 0.1 if comm == COMM_SUCCESS else float("nan")
            temp, comm, _ = self._pk.read1ByteTxRx(sid, ADDR_PRESENT_TEMP)
            t = temp if comm == COMM_SUCCESS else 0
            return int(pos), v, int(t)


# ── IK with collision rejection ──────────────────────────────────────────────
def ik_solve(world: World, arm_key: str, q5_init: np.ndarray,
             target: np.ndarray):
    """Solve IK by minimising |EE - target|^2 over joints J1..J4."""
    spec = ARMS[arm_key]
    bounds = [(spec["joints"][i][3], spec["joints"][i][4]) for i in range(1, 5)]

    def cost(q4):
        q5 = q5_init.copy(); q5[1:5] = q4
        world.set_arm_qpos(arm_key, q5)
        d = world.ee_position(arm_key) - target
        return float(np.dot(d, d))

    res = optimize.minimize(cost, q5_init[1:5].copy(), method="SLSQP",
                            bounds=bounds,
                            options={"maxiter": 200, "ftol": 1e-9})
    q5_out = q5_init.copy(); q5_out[1:5] = res.x
    world.set_arm_qpos(arm_key, q5_out)
    return q5_out, math.sqrt(max(res.fun, 0.0))


# ── App ──────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.world = World(MJCF_PATH)

        self.root = tk.Tk()
        self.root.title("Q-BOT IK 整合控制台")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1480x780")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.bus = Bus()
        self.active = "R"     # right arm is the one currently wired up
        # store q per arm so switching doesn't lose pose
        self.q_by_arm = {"L": np.zeros(5), "R": np.zeros(5)}
        self.q = self.q_by_arm[self.active]
        # ── per-joint calibration: 2-point linear mapping
        # Each joint has (tick_lo, q_lo, tick_hi, q_hi).  Sending motor:
        #   step = tick_lo + (q - q_lo) * (tick_hi - tick_lo) / (q_hi - q_lo)
        # Reading motor:
        #   q_read = q_lo + (step - tick_lo) * (q_hi - q_lo) / (tick_hi - tick_lo)
        # Default is derived from the nominal gear ratio in ARMS[], so out of
        # the box it behaves identically to the old offset+dir+gear formula.
        # Users can override either end-point via "Set Lo"/"Set Hi" buttons
        # (captures the current motor tick + ghost q) or edit numerically.
        # ``cal_trim`` is an extra per-joint nudge that ONLY moves the ghost
        # (does not affect motor commands), for visually aligning ghost to real.
        self.cal_points = {arm: self._default_cal_points(arm)
                           for arm in ("L", "R")}
        self.cal_trim = {"L": np.zeros(5), "R": np.zeros(5)}
        self.cal_path = os.path.join(HERE, "qbot_arm_calibration.json")
        self._load_cal()
        # live-send throttle (per joint, 25 Hz)
        self._last_motor_send_t = [0.0] * 5
        self._suppress_slider = False
        # drag state
        self._drag = None        # 'ee' | 'orbit' | 'pan' | None
        self._drag_pix = (0, 0)
        self._drag_depth = 0.0
        self._drag_target0 = np.zeros(3)
        self._drag_cam0 = None
        self._drag_lookat0 = None
        self._drag_collide = False

        self._tele_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        self._build_ui()
        self.world.set_arm_qpos(self.active, self._ghost_q())
        self.world.apply_arm_highlight(self.active)  # start with active arm solid
        self._refresh_slider_meta()
        self._refresh_cal_ui()
        self._render_to_canvas()

    # ── calibration helpers ──────────────────────────────────────────────────
    def _default_cal_points(self, arm):
        """Single-turn-safe default: motor tick 0..4095 maps to joint range
        ±π/gear (i.e., one full motor revolution = ``2π / gear`` rad joint
        side, centred on tick 2048 = q 0).  Better than naively scaling the
        MJCF joint limits because for gear=4 that would put tick at -2048
        and 6144 (way outside 0-4095) — useless until the user calibrates."""
        pts = []
        for sid, label, gear, lo, hi, jname in ARMS[arm]["joints"]:
            half_q = math.pi / gear              # half joint range (rad)
            pts.append({
                "tick_lo": 0.0,
                "q_lo":    float(-half_q),
                "tick_hi": 4095.0,
                "q_hi":    float(+half_q),
            })
        return pts

    def _ghost_q(self):
        """Joint angles the ghost should display (rad). Trim adds visual nudge."""
        return self.q + self.cal_trim[self.active]

    def _q_to_step(self, i, q_val=None, arm=None):
        """Linear interp from command-frame q (rad) to motor tick."""
        arm = arm or self.active
        if q_val is None: q_val = self.q[i]
        p = self.cal_points[arm][i]
        denom = p["q_hi"] - p["q_lo"]
        if abs(denom) < 1e-9: return int(round(p["tick_lo"]))
        return int(round(p["tick_lo"]
                         + (q_val - p["q_lo"])
                         * (p["tick_hi"] - p["tick_lo"]) / denom))

    def _step_to_q(self, i, step, arm=None):
        """Inverse of _q_to_step — used when reading motor pos."""
        arm = arm or self.active
        p = self.cal_points[arm][i]
        denom = p["tick_hi"] - p["tick_lo"]
        if abs(denom) < 1e-9: return 0.0
        return (p["q_lo"]
                + (step - p["tick_lo"])
                * (p["q_hi"] - p["q_lo"]) / denom)

    def _load_cal(self):
        if not os.path.exists(self.cal_path): return
        try:
            d = json.load(open(self.cal_path))
            for arm in ("L", "R"):
                pts = d.get(arm)
                if not pts: continue
                if isinstance(pts, list) and len(pts) == 5 and isinstance(pts[0], dict):
                    # new 2-point format
                    self.cal_points[arm] = [
                        {k: float(p.get(k, 0)) for k in
                         ("tick_lo", "q_lo", "tick_hi", "q_hi")}
                        for p in pts
                    ]
        except Exception as e:
            print(f"[cal] load failed: {e}")

    def _save_cal(self):
        d = {arm: [{k: float(v) for k, v in p.items()}
                   for p in self.cal_points[arm]]
             for arm in ("L", "R")}
        try:
            with open(self.cal_path, "w") as f:
                json.dump(d, f, indent=2)
            self.status_var.set(f"校準存到 {self.cal_path}")
        except Exception as e:
            self.status_var.set(f"save cal failed: {e}")

        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.root.after(150, self._drain_tele)
        self.root.after(33, self._render_tick)

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        hdr = tk.Frame(self.root, bg="#1a237e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Q-BOT  IK  整合控制台",
                 font=("DejaVu Sans", 16, "bold"),
                 bg="#1a237e", fg="white").pack(side="left", padx=12, pady=8)
        tk.Label(hdr,
                 text="拖鬼影手腕 = IK | Shift = 拉 Z | LMB 拖背景 = 旋轉 | RMB = 平移 | 滾輪 = 縮放",
                 font=("DejaVu Sans Mono", 9),
                 bg="#1a237e", fg="#90caf9").pack(side="left", padx=10)

        body = tk.Frame(self.root, bg=C["bg"]); body.pack(fill="both", expand=True)

        # ── left column (controls)
        left = tk.Frame(body, bg=C["panel"], width=520)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        # ── right column (canvas)
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=8)
        self.canvas = tk.Canvas(right, width=RENDER_W, height=RENDER_H,
                                bg="#000000", highlightthickness=0,
                                cursor="hand2")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_img_id = None
        self._tk_img = None
        self._marker_id = None
        self._marker_ring_id = None

        # mouse bindings
        self.canvas.bind("<ButtonPress-1>",   self._on_lmb_press)
        self.canvas.bind("<B1-Motion>",       self._on_lmb_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_lmb_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self._on_lmb_press)
        self.canvas.bind("<Shift-B1-Motion>", self._on_lmb_motion)
        self.canvas.bind("<ButtonPress-3>",   self._on_rmb_press)
        self.canvas.bind("<B3-Motion>",       self._on_rmb_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_rmb_release)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)   # Windows / mac
        self.canvas.bind("<Button-4>",        lambda e: self._zoom(0.9, e))
        self.canvas.bind("<Button-5>",        lambda e: self._zoom(1/0.9, e))

    def _build_controls(self, p):
        # connection bar
        bar = tk.Frame(p, bg=C["card"]); bar.pack(fill="x", padx=6, pady=(6, 4))
        tk.Label(bar, text="Port:", bg=C["card"], fg=C["text"]
                 ).pack(side="left", padx=(6, 2))
        self.port_var = tk.StringVar(value=_find_arm_port())
        tk.Entry(bar, textvariable=self.port_var, width=14,
                 bg=C["bg"], fg=C["text"], insertbackground=C["text"]
                 ).pack(side="left")
        self.connect_btn = tk.Button(bar, text="Connect", bg=C["btn"], fg="white",
                                     command=self._toggle_connect, width=10)
        self.connect_btn.pack(side="left", padx=6)
        # Live: when ON, the joint sliders push their motor live (throttled).
        # OFF = old behaviour, use the explicit "Send to Robot" button.
        self.live_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Live (拖即送)",
                       variable=self.live_var,
                       bg=C["card"], fg=C["ok"], activebackground=C["card"],
                       activeforeground=C["ok"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 9, "bold")
                       ).pack(side="left", padx=4)
        self.conn_var = tk.StringVar(value="● 未連線")
        tk.Label(bar, textvariable=self.conn_var, bg=C["card"], fg=C["err"]
                 ).pack(side="right", padx=8)

        # arm selector
        sel = tk.Frame(p, bg=C["panel"]); sel.pack(fill="x", padx=6, pady=2)
        tk.Label(sel, text="Arm:", bg=C["panel"], fg=C["text"]).pack(side="left", padx=4)
        self.arm_var = tk.StringVar(value=self.active)
        for n, fg in (("L", C["la"]), ("R", C["ra"])):
            tk.Radiobutton(sel, text=n, value=n, variable=self.arm_var,
                           bg=C["panel"], fg=fg, selectcolor=C["card"],
                           activebackground=C["panel"], activeforeground=fg,
                           font=("DejaVu Sans Mono", 11, "bold"),
                           command=self._on_arm_change).pack(side="left", padx=4)

        # joint sliders
        jp = tk.LabelFrame(p, text=" 5-DOF Joint Sliders ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        jp.pack(fill="x", padx=6, pady=4)
        self.sliders = []; self.id_lbls = []
        self.cmd_vars = [tk.StringVar(value="0.0°") for _ in range(5)]
        self.read_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.step_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.volt_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.temp_vars = [tk.StringVar(value="—") for _ in range(5)]
        for i in range(5):
            row = tk.Frame(jp, bg=C["card"]); row.pack(fill="x", padx=4, pady=2)
            top = tk.Frame(row, bg=C["card"]); top.pack(fill="x")
            id_l = tk.Label(top, text=f"J{i}", width=5,
                            bg=C["card"], fg=C["bright"],
                            font=("DejaVu Sans Mono", 10, "bold"))
            id_l.pack(side="left", padx=2); self.id_lbls.append(id_l)
            tk.Label(top, textvariable=self.cmd_vars[i], width=8,
                     bg=C["card"], fg=C["warn"],
                     font=("DejaVu Sans Mono", 10, "bold")).pack(side="left")
            tk.Label(top, text="read:", bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=(8, 1))
            tk.Label(top, textvariable=self.read_vars[i], width=8,
                     bg=C["card"], fg=C["ok"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            tk.Label(top, textvariable=self.step_vars[i], width=6,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=2)
            tk.Label(top, textvariable=self.volt_vars[i], width=6,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            tk.Label(top, textvariable=self.temp_vars[i], width=6,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            sld = tk.Scale(row, from_=-90.0, to=90.0, resolution=0.5,
                           orient="horizontal", length=470,
                           bg=C["card"], fg=C["text"],
                           highlightthickness=0, troughcolor=C["panel"],
                           showvalue=False,
                           command=lambda v, i=i: self._on_slider(i, v))
            sld.pack(fill="x", padx=2, pady=(0, 2))
            self.sliders.append(sld)

        # EE / target readouts
        rp = tk.LabelFrame(p, text=" End-Effector / IK Target ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        rp.pack(fill="x", padx=6, pady=4)
        self.cur_ee_var = tk.StringVar(value="EE: —")
        tk.Label(rp, textvariable=self.cur_ee_var, bg=C["panel"], fg=C["target"],
                 font=("DejaVu Sans Mono", 10, "bold")).pack(anchor="w", padx=6, pady=1)
        self.tgt_var = tk.StringVar(value="Target: —")
        tk.Label(rp, textvariable=self.tgt_var, bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 10)).pack(anchor="w", padx=6)
        self.err_var = tk.StringVar(value="")
        self.err_lbl = tk.Label(rp, textvariable=self.err_var, bg=C["panel"],
                                fg=C["dim"], font=("DejaVu Sans Mono", 10, "bold"))
        self.err_lbl.pack(anchor="w", padx=6, pady=1)

        # speed/acc
        sa = tk.Frame(p, bg=C["panel"]); sa.pack(fill="x", padx=6, pady=2)
        tk.Label(sa, text="Speed:", bg=C["panel"], fg=C["text"]).pack(side="left", padx=2)
        self.speed_var = tk.IntVar(value=200)
        tk.Scale(sa, from_=50, to=2000, orient="horizontal", length=160,
                 variable=self.speed_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]).pack(side="left")
        tk.Label(sa, text="Acc:", bg=C["panel"], fg=C["text"]).pack(side="left", padx=(8, 2))
        self.acc_var = tk.IntVar(value=20)
        tk.Scale(sa, from_=1, to=150, orient="horizontal", length=120,
                 variable=self.acc_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]).pack(side="left")

        # ── calibration panel ─────────────────────────────────────────────
        # Workflow: open Live → drag joint slider until real arm hits its
        # physical limit → press [Lo] (or [Hi]) → that motor tick + ghost-side
        # angle become this end-point of the 2-point linear mapping.
        cal = tk.LabelFrame(p,
                            text=" 校準 (拖到實機極限 → 點 Lo/Hi 紀錄) ",
                            bg=C["panel"], fg=C["text"],
                            font=("DejaVu Sans Mono", 10, "bold"))
        cal.pack(fill="x", padx=6, pady=4)
        self.cal_labels = []   # per joint: StringVar showing "Lo: tick @ q  /  Hi: tick @ q"
        for i in range(5):
            row = tk.Frame(cal, bg=C["card"]); row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=f"J{i}", width=3, bg=C["card"],
                     fg=C["bright"], font=("DejaVu Sans Mono", 10, "bold")
                     ).pack(side="left", padx=1)
            tk.Button(row, text="←Lo (記錄此處為下限)", width=22,
                      bg=C["btn"], fg="white",
                      font=("DejaVu Sans Mono", 9),
                      command=lambda i=i: self._capture_lo(i)
                      ).pack(side="left", padx=2)
            tk.Button(row, text="←Hi (記錄此處為上限)", width=22,
                      bg=C["btn"], fg="white",
                      font=("DejaVu Sans Mono", 9),
                      command=lambda i=i: self._capture_hi(i)
                      ).pack(side="left", padx=2)
            lv = tk.StringVar(value="—")
            self.cal_labels.append(lv)
            tk.Label(row, textvariable=lv, bg=C["card"], fg=C["target"],
                     font=("DejaVu Sans Mono", 9), anchor="w"
                     ).pack(side="left", fill="x", expand=True, padx=4)

        # cal action row
        cb = tk.Frame(cal, bg=C["panel"]); cb.pack(fill="x", padx=4, pady=4)
        tk.Button(cb, text="Reset to default", bg=C["btn"], fg="white",
                  command=self._reset_cal, width=16
                  ).pack(side="left", padx=2)
        tk.Button(cb, text="Save", bg=C["btn"], fg="white",
                  command=self._save_cal, width=8).pack(side="right", padx=2)
        tk.Button(cb, text="Reload", bg=C["btn"], fg="white",
                  command=self._reload_cal, width=8).pack(side="right", padx=2)

        # actions
        ab = tk.Frame(p, bg=C["panel"]); ab.pack(fill="x", padx=6, pady=4)
        tk.Button(ab, text="HOME (q=0)", bg=C["btn"], fg="white",
                  command=self._home, width=12).pack(side="left", padx=2)
        tk.Button(ab, text="Send to Robot", bg=C["btn_warn"], fg="white",
                  command=self._send_to_robot, width=14
                  ).pack(side="left", padx=2)
        tk.Button(ab, text="E-STOP", bg=C["btn_danger"], fg="white",
                  command=self._estop, width=10).pack(side="right", padx=2)
        tk.Button(ab, text="↻ 重啟", bg=C["btn"], fg="white",
                  command=self._restart, width=8).pack(side="right", padx=2)

        # status
        self.status_var = tk.StringVar(value="ready")
        tk.Label(p, textvariable=self.status_var, anchor="w",
                 bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=6, pady=(4, 4))

    def _refresh_slider_meta(self):
        spec = ARMS[self.active]
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            self.sliders[i].config(from_=math.degrees(lo), to=math.degrees(hi))
            self.id_lbls[i].config(text=f"{sid}")
        self._suppress_slider = True
        for i in range(5):
            self.sliders[i].set(round(math.degrees(self.q[i]), 1))
            self.cmd_vars[i].set(f"{math.degrees(self.q[i]):+6.1f}°")
        self._suppress_slider = False

    # ── slider / arm logic ───────────────────────────────────────────────────
    def _on_slider(self, i, val_str):
        if self._suppress_slider: return
        try: deg = float(val_str)
        except ValueError: return
        self.q[i] = math.radians(deg)
        self.q_by_arm[self.active] = self.q
        self.cmd_vars[i].set(f"{deg:+6.1f}°")
        self.world.set_arm_qpos(self.active, self._ghost_q())
        # if Live: also send the matching motor (throttled per joint)
        self._maybe_send_motor(i)
        # also flag collision in status if any
        if self.world.check_self_collision(self.active):
            self.status_var.set("⚠ self-collision (slider move)")
            self.err_lbl.config(fg=C["collide"])
            self.err_var.set("collision!")
        else:
            self.err_lbl.config(fg=C["ok"])
            self.err_var.set("ok")

    def _maybe_send_motor(self, i: int):
        """Push joint i to its motor live (if Live toggle on)."""
        if not (self.bus.is_open() and getattr(self, "live_var", None)
                and self.live_var.get()):
            return
        now = time.monotonic()
        if now - self._last_motor_send_t[i] < 0.04:   # 25 Hz per joint
            return
        self._last_motor_send_t[i] = now
        sid = ARMS[self.active]["joints"][i][0]
        step = max(0, min(4095, self._q_to_step(i)))
        self.bus.write_pos(sid, step,
                           int(self.speed_var.get()),
                           int(self.acc_var.get()))

    def _on_arm_change(self):
        # preserve outgoing pose under its key
        self.q_by_arm[self.active] = self.q.copy()
        self.active = self.arm_var.get()
        self.q = self.q_by_arm[self.active]      # restore (or start at zeros)
        self.world.set_arm_qpos(self.active, self._ghost_q())
        # repaint: the arm we just switched TO becomes solid; the other goes ghost
        self.world.apply_arm_highlight(self.active)
        self._refresh_slider_meta()
        self._refresh_cal_ui()
        self.status_var.set(f"切換到 {self.active} arm")

    def _home(self):
        self.q[:] = 0.0
        self.q_by_arm[self.active] = self.q
        self.world.set_arm_qpos(self.active, self._ghost_q())
        self._refresh_slider_meta()
        self.err_var.set(""); self.status_var.set("HOME (虛擬,未送馬達)")

    # ── canvas rendering ─────────────────────────────────────────────────────
    def _render_to_canvas(self):
        img = self.world.render()
        pil = Image.fromarray(img)
        self._tk_img = ImageTk.PhotoImage(pil)
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._tk_img)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self._tk_img)
        # overlay EE marker
        ee = self.world.ee_position(self.active)
        cw = int(self.canvas.winfo_width()) or RENDER_W
        ch = int(self.canvas.winfo_height()) or RENDER_H
        proj = self.world.world_to_screen(ee, cw, ch)
        col = C["collide"] if self._drag_collide else C["target"]
        if proj is not None:
            sx, sy, _ = proj
            R = 14
            if self._marker_id is None:
                self._marker_id = self.canvas.create_oval(sx-4, sy-4, sx+4, sy+4,
                                                          fill=col, outline="")
                self._marker_ring_id = self.canvas.create_oval(sx-R, sy-R, sx+R, sy+R,
                                                                outline=col, width=2)
            else:
                self.canvas.coords(self._marker_id, sx-4, sy-4, sx+4, sy+4)
                self.canvas.coords(self._marker_ring_id, sx-R, sy-R, sx+R, sy+R)
                self.canvas.itemconfig(self._marker_id, fill=col)
                self.canvas.itemconfig(self._marker_ring_id, outline=col)
            self.canvas.tag_raise(self._marker_ring_id)
            self.canvas.tag_raise(self._marker_id)
        else:
            if self._marker_id is not None:
                self.canvas.itemconfig(self._marker_id, fill="")
                self.canvas.itemconfig(self._marker_ring_id, outline="")
        # update text readouts
        self.cur_ee_var.set(f"EE: ({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})")

    def _render_tick(self):
        try:
            self._render_to_canvas()
        except Exception as e:
            self.status_var.set(f"render err: {e}")
        self.root.after(33, self._render_tick)

    # ── mouse handlers ───────────────────────────────────────────────────────
    def _ee_pix(self):
        ee = self.world.ee_position(self.active)
        cw = int(self.canvas.winfo_width()) or RENDER_W
        ch = int(self.canvas.winfo_height()) or RENDER_H
        return self.world.world_to_screen(ee, cw, ch), ee

    def _on_lmb_press(self, ev):
        # Decide: drag EE (if near) or orbit camera
        proj_ee = self._ee_pix()
        if proj_ee[0] is not None:
            sx, sy, depth = proj_ee[0]
            if math.hypot(ev.x - sx, ev.y - sy) < 24:
                self._drag = "ee"
                self._drag_pix = (ev.x, ev.y)
                self._drag_depth = depth
                self._drag_target0 = proj_ee[1].copy()
                self._drag_shift = bool(ev.state & 0x0001)
                return
        # otherwise orbit
        self._drag = "orbit"
        self._drag_pix = (ev.x, ev.y)
        self._drag_cam0 = (self.world.cam.azimuth, self.world.cam.elevation)

    def _on_lmb_motion(self, ev):
        if self._drag == "ee":
            dx_pix = ev.x - self._drag_pix[0]
            dy_pix = ev.y - self._drag_pix[1]
            cw = int(self.canvas.winfo_width()) or RENDER_W
            ch = int(self.canvas.winfo_height()) or RENDER_H
            if not (ev.state & 0x0001):
                # normal drag in camera plane
                dv = self.world.screen_delta_to_world(dx_pix, dy_pix,
                                                     self._drag_depth, cw, ch)
                tgt = self._drag_target0 + dv
            else:
                # shift = world Z drag
                tgt = self._drag_target0.copy()
                fov = self.world.fovy_rad()
                h_world = 2 * self._drag_depth * math.tan(fov/2)
                scale = h_world / ch
                tgt[2] -= dy_pix * scale
            self._attempt_target(tgt)
            return
        if self._drag == "orbit":
            dx_pix = ev.x - self._drag_pix[0]
            dy_pix = ev.y - self._drag_pix[1]
            az0, el0 = self._drag_cam0
            self.world.cam.azimuth = az0 - dx_pix * 0.4
            self.world.cam.elevation = max(-89.0, min(89.0, el0 - dy_pix * 0.4))
            return

    def _on_lmb_release(self, ev):
        if self._drag == "ee" and self._drag_collide:
            # snap back if collided
            self.world.set_arm_qpos(self.active, self._ghost_q())
        self._drag = None
        self._drag_collide = False

    def _on_rmb_press(self, ev):
        self._drag = "pan"
        self._drag_pix = (ev.x, ev.y)
        self._drag_lookat0 = np.array(self.world.cam.lookat).copy()

    def _on_rmb_motion(self, ev):
        if self._drag != "pan": return
        dx_pix = ev.x - self._drag_pix[0]
        dy_pix = ev.y - self._drag_pix[1]
        cw = int(self.canvas.winfo_width()) or RENDER_W
        ch = int(self.canvas.winfo_height()) or RENDER_H
        dv = self.world.screen_delta_to_world(dx_pix, dy_pix,
                                              self.world.cam.distance, cw, ch)
        self.world.cam.lookat[:] = (self._drag_lookat0 - dv)

    def _on_rmb_release(self, ev):
        self._drag = None

    def _on_scroll(self, ev):
        factor = 0.9 if ev.delta > 0 else 1/0.9
        self._zoom(factor, ev)

    def _zoom(self, factor, ev):
        self.world.cam.distance = max(0.3, min(8.0, self.world.cam.distance * factor))

    # ── IK with collision rejection ──────────────────────────────────────────
    def _attempt_target(self, target: np.ndarray):
        q_before = self.q.copy()
        # was the starting pose already in collision? (e.g. dragged into body
        # previously). If so, we still allow IK to move it — we only block
        # poses that *worsen* the collision state.
        self.world.set_arm_qpos(self.active, q_before)
        collide_before = self.world.check_self_collision(self.active)
        q5, err = ik_solve(self.world, self.active, q_before.copy(), target)
        self.world.set_arm_qpos(self.active, q5)
        collide_after = self.world.check_self_collision(self.active)
        if collide_after and not collide_before:
            self.world.set_arm_qpos(self.active, q_before)
            self._drag_collide = True
            self.status_var.set("⚠ collision — reverted")
            self.err_lbl.config(fg=C["collide"])
            self.err_var.set("blocked: collision")
            self.tgt_var.set(f"Target (blocked): ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")
            return
        self._drag_collide = collide_after
        self.q = q5
        self.q_by_arm[self.active] = self.q
        self._suppress_slider = True
        for i in range(5):
            self.sliders[i].set(round(math.degrees(self.q[i]), 1))
            self.cmd_vars[i].set(f"{math.degrees(self.q[i]):+6.1f}°")
        self._suppress_slider = False
        if err > 0.05:
            self.err_lbl.config(fg=C["err"])
            self.err_var.set(f"reach err={err*1000:.1f}mm (out of range)")
        elif err > 0.01:
            self.err_lbl.config(fg=C["warn"])
            self.err_var.set(f"err={err*1000:.1f}mm")
        else:
            self.err_lbl.config(fg=C["ok"])
            self.err_var.set(f"err={err*1000:.1f}mm")
        self.tgt_var.set(f"Target: ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")

    # ── hardware ─────────────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.connect_btn.config(text="Connect", bg=C["btn"])
            self.conn_var.set("● 未連線")
            return
        port = self.port_var.get().strip()
        ok, msg = self.bus.open(port, DEFAULT_BAUD)
        if not ok:
            # device name may have bumped (ttyACM0 → ttyACM1) after a USB
            # reconnect glitch — auto-retry using the USB serial number
            auto = _find_arm_port(default=port)
            if auto != port:
                ok2, msg2 = self.bus.open(auto, DEFAULT_BAUD)
                if ok2:
                    self.port_var.set(auto)
                    ok, msg = True, f"retry@{auto}"
        if ok:
            self.connect_btn.config(text="Disconnect", bg=C["btn_danger"])
            self.conn_var.set("● 已連線")
            self.status_var.set(f"connected {self.port_var.get()}")
        else:
            self.status_var.set(f"connect failed: {msg}")

    def _send_to_robot(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線"); return
        if self.world.check_self_collision(self.active):
            self.status_var.set("⚠ 拒送 — 目前姿態 self-collision")
            return
        spec = ARMS[self.active]
        spd = int(self.speed_var.get()); acc = int(self.acc_var.get())
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            step = max(0, min(4095, self._q_to_step(i)))
            self.bus.write_pos(sid, step, spd, acc)
        self.status_var.set(f"sent {self.active}  q={np.degrees(self.q).round(1).tolist()}")

    def _estop(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — E-STOP 無作用"); return
        for spec in ARMS.values():
            for sid, *_ in spec["joints"]:
                self.bus.torque(sid, False)
        self.status_var.set("E-STOP — all torque off")

    # ── calibration handlers ─────────────────────────────────────────────────
    def _capture_lo(self, i): self._capture(i, "lo")
    def _capture_hi(self, i): self._capture(i, "hi")

    def _capture(self, i, side):
        sid = ARMS[self.active]["joints"][i][0]
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — 無法擷取 tick"); return
        pos, _, comm, _ = self.bus._pk.ReadPosSpeed(sid)
        if comm != COMM_SUCCESS:
            self.status_var.set(f"J{i} 馬達無回應"); return
        ghost = self.q[i]   # current slider q (ghost-side, rad)
        p = self.cal_points[self.active][i]
        if side == "lo":
            p["tick_lo"] = float(pos); p["q_lo"] = float(ghost)
        else:
            p["tick_hi"] = float(pos); p["q_hi"] = float(ghost)
        self._refresh_cal_ui()
        self.status_var.set(
            f"J{i} {side} ← tick={int(pos)}  鬼影 q={math.degrees(ghost):+.1f}°")

    def _reset_cal(self):
        self.cal_points[self.active] = self._default_cal_points(self.active)
        self._refresh_cal_ui()
        self.status_var.set("校準回到預設(依 gear 自動推算)")

    def _reload_cal(self):
        self._load_cal()
        self._refresh_cal_ui()
        self.status_var.set("校準已從 JSON 重新載入")

    def _refresh_cal_ui(self):
        """Show each joint's current 2-point mapping in its status label."""
        arm = self.active
        for i in range(5):
            p = self.cal_points[arm][i]
            self.cal_labels[i].set(
                f"Lo {p['tick_lo']:.0f}@{math.degrees(p['q_lo']):+5.1f}°    "
                f"Hi {p['tick_hi']:.0f}@{math.degrees(p['q_hi']):+5.1f}°")

    # ── telemetry ────────────────────────────────────────────────────────────
    def _poll_loop(self):
        while not self._stop.is_set():
            if self.bus.is_open():
                spec = ARMS[self.active]
                for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
                    if self._stop.is_set(): break
                    r = self.bus.read_quick(sid)
                    self._tele_q.put((i, gear, r))
            time.sleep(0.3)

    def _drain_tele(self):
        try:
            while True:
                i, gear, r = self._tele_q.get_nowait()
                if r is None:
                    self.read_vars[i].set("—"); self.step_vars[i].set("—")
                    self.volt_vars[i].set("—"); self.temp_vars[i].set("—")
                else:
                    pos, v, t = r
                    # invert 2-point linear mapping → command-frame q
                    rad_q = self._step_to_q(i, pos)
                    self.read_vars[i].set(f"{math.degrees(rad_q):+6.1f}°")
                    self.step_vars[i].set(f"{pos:5d}")
                    self.volt_vars[i].set(f"{v:4.1f}V" if v == v else "—")
                    self.temp_vars[i].set(f"{t:3d}°C")
        except queue.Empty:
            pass
        self.root.after(150, self._drain_tele)

    def _on_close(self):
        self._stop.set()
        try:
            if self.bus.is_open():
                for spec in ARMS.values():
                    for sid, *_ in spec["joints"]:
                        self.bus.torque(sid, False)
                self.bus.close()
        finally:
            self.root.destroy()

    def _restart(self):
        """Re-exec self.  Leaves motors holding their last target (no E-STOP).
        Use the E-STOP button beforehand if you want torque off across restart."""
        self._stop.set()
        try:
            if self.bus.is_open(): self.bus.close()
            self.root.destroy()
        except Exception:
            pass
        os.execvp(sys.executable, [sys.executable, sys.argv[0]] + sys.argv[1:])

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
