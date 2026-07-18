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

import asyncio
import json
import math
import os
import queue
import socket
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

# OpenCV is optional here — only used for the built-in camera panel that
# streams from the RPi head_agent /mjpeg endpoint.  Import lazily so a
# broken install doesn't kill the GUI.
try:
    import cv2  # noqa: E402
except Exception:
    cv2 = None

# ── paths & constants ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.normpath(os.path.join(HERE, "..", "models", "QBOT_MJCF", "qbot.xml"))

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
# Default RPi arm agent endpoint (matches deployment/arm_rpi_agent/agent.py).
REMOTE_DEFAULT_HOST = "192.168.0.123:8100"
# UDP port pose_imitator streams to.  Bound on 127.0.0.1 only.
POSE_UDP_PORT = 9999
# If a pose packet is older than this, treat imitation as stale (freeze).
POSE_STALE_MS = 500
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
            # Direct-drive (gear=1) for now — the planetary reducer chain isn't
            # wired in yet, so tick 0..4095 maps 1:1 to motor angle.  Motor
            # home is at tick 2047 (physical mid), so slider q=0 corresponds
            # to tick 2048 (≈2047, off by ½ tick which is <0.09°).
            # J4 (wrist rotation) has no MJCF joint — slider moves the motor,
            # ghost stays put.
            (20, "Shoul Rot",  1, -math.pi, math.pi, "ub_left_shoulder"),
            (21, "Shoul Lift", 1, -math.pi, math.pi, "ub_left_lateral_raise"),
            (22, "Arm Twist",  1, -math.pi, math.pi, "ub_left_arm_twist"),
            (23, "Elbow",      1, -math.pi, math.pi, "ub_left_elbow"),
            (24, "Wrist",      1, -math.pi, math.pi, None),
        ],
        ee_body="ub_115_2",
        # bodies considered part of this arm for collision filtering
        arm_bodies=("ub_l_4", "ub_l_3", "ub_link_7", "ub_l_2", "ub_113_2",
                    "ub_l_1", "ub_112_2", "ub_l", "ub_115_2", "ub_114_2"),
        col=C["la"],
    ),
    "R": dict(
        joints=[
            # See L-arm note above; same layout for R arm.
            (10, "Shoul Rot",  1, -math.pi, math.pi, "ub_right_shoulder"),
            (11, "Shoul Lift", 1, -math.pi, math.pi, "ub_right_lateral_raise"),
            (12, "Arm Twist",  1, -math.pi, math.pi, "ub_right_arm_twist"),
            (13, "Elbow",      1, -math.pi, math.pi, "ub_right_elbow"),
            (14, "Wrist",      1, -math.pi, math.pi, None),
        ],
        ee_body="ub_115",
        arm_bodies=("ub_link_2", "ub_link", "ub_link_6", "ub_2", "ub_113",
                    "ub_fake_motor_", "ub_112", "ub_1", "ub_115", "ub_114"),
        col=C["ra"],
    ),
}

# Per-arm sign flip applied only when writing to the ghost's MJCF joint.
# Motor commands are unaffected — use this to align the ghost's rotation
# direction with the physical robot when the MJCF axis convention differs.
MJ_SIGN = {
    # R was calibrated by hand; L mirrors R.  Shoulder Z-axis is already
    # opposite in MJCF (L: -Z, R: +Z), so J0 uses the SAME sign on both.
    # Y-axis joints (lat_raise / twist / elbow) share the same axis on both
    # arms, so L uses the OPPOSITE sign of R for those to mirror.
    "L": [-1, -1, -1, +1, +1],
    "R": [-1, +1, -1, -1, +1],
}

# Per-arm-per-joint sign flip applied to the pose→ghost delta.
# Needed because operator's cal_points can be either direction (e.g. L J1
# has HIGH q = hanging, R J1 has HIGH q = hanging too), while pose retarget
# always outputs "higher = arm more raised".  Adjust here if a joint moves
# the opposite way in ghost compared to your body.
POSE_SIGN = {
    "L": [+1, -1, +1, +1, +1],   # J1 (lift) inverted for L
    "R": [+1, -1, +1, +1, +1],   # J1 (lift) inverted for R (cal is same shape)
}

# Default "home" q per joint (rad).  Slider zero maps to this q, and the
# default cal_points puts q=0 ↔ tick 2048 — so with HOME_Q=0 everywhere,
# the slider's zero position matches the motor's factory home at tick 2047.
# Per-joint adjustments (e.g., "at tick 2047 this joint physically sits at
# +30°") will be added later by the operator using the "設為零位" button.
HOME_Q = {
    "L": [0.0, 0.0, 0.0, 0.0, 0.0],
    "R": [0.0, 0.0, 0.0, 0.0, 0.0],
}


# ── MuJoCo world + camera ────────────────────────────────────────────────────
class World:
    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(path)
        # offscreen framebuffer must be at least as large as our render size
        self.model.vis.global_.offwidth = RENDER_W
        self.model.vis.global_.offheight = RENDER_H
        self.data = mujoco.MjData(self.model)
        # Scratch data for IK — SLSQP mutates qpos across many cost evals per
        # drag; running those on a separate MjData keeps the rendered `data`
        # stable so the ghost never flashes to intermediate iterations.
        self.data_ik = mujoco.MjData(self.model)
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
        for i in range(m.ngeom):
            m.geom_rgba[i] = self._GHOST_RGBA
        bids = {self.bid[n] for n in ARMS[arm_key]["arm_bodies"] if n in self.bid}
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

    def _write_arm_qpos(self, data, arm_key: str, q5: np.ndarray):
        spec = ARMS[arm_key]
        signs = MJ_SIGN[arm_key]
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            if jname is None: continue
            adr = self.jqpos.get(jname)
            if adr is None: continue
            data.qpos[adr] = float(np.clip(q5[i], lo, hi)) * signs[i]
        mujoco.mj_forward(self.model, data)

    def set_arm_qpos(self, arm_key: str, q5: np.ndarray):
        self._write_arm_qpos(self.data, arm_key, q5)

    def set_arm_qpos_ik(self, arm_key: str, q5: np.ndarray):
        """Same as set_arm_qpos but writes to the scratch IK MjData."""
        self._write_arm_qpos(self.data_ik, arm_key, q5)

    def ee_position(self, arm_key: str) -> np.ndarray:
        return self.data.xpos[self.bid[ARMS[arm_key]["ee_body"]]].copy()

    def ee_position_ik(self, arm_key: str) -> np.ndarray:
        return self.data_ik.xpos[self.bid[ARMS[arm_key]["ee_body"]]].copy()

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

    def read_pos(self, sid):
        """Single-shot position read.  Returns int on success, None on error."""
        with self._lock:
            if not self._pk: return None
            pos, _, comm, _ = self._pk.ReadPosSpeed(sid)
            return None if comm != COMM_SUCCESS else int(pos)

    def read_word(self, sid, addr):
        """Read a 2-byte register (unsigned).  Used for EPROM angle limits."""
        with self._lock:
            if not self._pk: return None
            v, comm, _ = self._pk.read2ByteTxRx(sid, addr)
            return None if comm != COMM_SUCCESS else int(v)


# ── Remote bus (WebSocket → arm_rpi_agent) ───────────────────────────────────
class RemoteBus:
    """Same public API shape as Bus, but talks to the RPi arm agent over
    WebSocket instead of a local serial port.  Runs a private asyncio loop
    in a background thread so Tk-thread callers stay synchronous."""

    def __init__(self):
        self.host = ""
        self.connected = False
        self.latest_tele: dict[int, dict] = {}   # sid → {pos, spd, volt, temp, ...}
        self.status = "idle"
        self._out: "queue.Queue[dict]" = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._ws = None
        # For sync request-reply (read_word etc.): map req_id → Queue that the
        # receiver drops the reply into.
        self._pending: dict[str, "queue.Queue[dict]"] = {}
        self._pending_lock = threading.Lock()
        self._req_seq = 0

    def is_open(self) -> bool: return self.connected

    def open(self, host, _baud=None):
        """host may be ``host`` (implies default port 8100) or ``host:port``."""
        if self._thread and self._thread.is_alive(): self.close()
        h = host.strip()
        if ":" not in h: h = f"{h}:8100"
        self.host = h
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Wait briefly for the WebSocket handshake so the caller gets a
        # meaningful (ok, msg) tuple instead of an eventual-consistency dance.
        for _ in range(30):                 # ~3 seconds
            if self.connected: return True, f"connected {self.host}"
            if self.status.startswith("connect err"): return False, self.status
            time.sleep(0.1)
        return False, "connect timeout"

    def close(self):
        self._stop.set()
        if self._loop:
            try: self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception: pass
        self._thread = None
        self.connected = False
        self.status = "disconnected"

    def write_pos(self, sid, step, speed, acc):
        if not self.connected: return False
        self._out.put({"op": "write", "sid": int(sid), "step": int(step),
                       "speed": int(speed), "acc": int(acc)})
        return True

    def torque(self, sid, on):
        if not self.connected: return False
        self._out.put({"op": "torque", "sid": int(sid), "on": bool(on)})
        return True

    def read_pos(self, sid):
        m = self.latest_tele.get(int(sid))
        if m is None: return None
        pos = m.get("pos")
        return None if pos is None else int(pos)

    def read_quick(self, sid):
        m = self.latest_tele.get(int(sid))
        if m is None: return None
        pos = m.get("pos"); v = m.get("volt", float("nan")); t = m.get("temp", 0)
        return (int(pos) if pos is not None else 0), float(v), int(t)

    def read_word(self, sid, addr, timeout=1.5):
        """Sync read of a 2-byte register.  Returns int or None."""
        if not self.connected: return None
        with self._pending_lock:
            self._req_seq += 1
            req_id = f"rw{self._req_seq}"
            rq: queue.Queue = queue.Queue(maxsize=1)
            self._pending[req_id] = rq
        self._out.put({"op": "read", "sid": int(sid),
                       "addr": int(addr), "size": 2, "req_id": req_id})
        try:
            r = rq.get(timeout=timeout)
        except queue.Empty:
            r = None
        finally:
            with self._pending_lock: self._pending.pop(req_id, None)
        if not r or not r.get("ok"): return None
        v = r.get("value")
        return None if v is None else int(v)

    # ── async worker ────────────────────────────────────────────────────────
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
        import aiohttp
        url = f"ws://{self.host}/ws"
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.ws_connect(url, timeout=5, heartbeat=10) as ws:
                    self._ws = ws
                    self.connected = True
                    self.status = f"connected {self.host}"
                    print(f"[remote] connected {url}", flush=True)
                    send_t = asyncio.create_task(self._sender())
                    recv_t = asyncio.create_task(self._receiver())
                    stop_t = asyncio.create_task(self._wait_stop())
                    done, pending = await asyncio.wait(
                        [send_t, recv_t, stop_t],
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
        import aiohttp
        async for msg in self._ws:
            if msg.type != aiohttp.WSMsgType.TEXT: continue
            try: d = json.loads(msg.data)
            except Exception: continue
            if d.get("t") == "tele":
                self.latest_tele = {m["sid"]: m for m in d.get("motors", [])}
            elif d.get("t") == "reply":
                rid = d.get("req_id")
                if rid:
                    with self._pending_lock:
                        rq = self._pending.get(rid)
                    if rq is not None:
                        try: rq.put_nowait(d)
                        except queue.Full: pass

    async def _wait_stop(self):
        while not self._stop.is_set():
            await asyncio.sleep(0.1)


# ── IK with collision rejection ──────────────────────────────────────────────
def ik_solve(world: World, arm_key: str, q5_init: np.ndarray,
             target: np.ndarray, slider_range: list | None = None):
    """Solve IK for every joint that actually moves the ghost.

    Runs entirely on ``world.data_ik`` (scratch MjData) so the SLSQP cost
    evaluations never touch the rendered ``world.data``; the caller is
    responsible for applying the returned q5 to the display when ready.
    Virtual joints (jname is None) are held fixed.
    """
    spec = ARMS[arm_key]
    # indices to optimise: every joint that maps to a real MJCF joint
    idx = [i for i in range(0, 5) if spec["joints"][i][5] is not None]
    # Prefer runtime slider_range (set by the "讀 EPROM 限位" button) — falls
    # back to the module-default ARMS bounds when no override is passed.
    if slider_range is not None:
        bounds = [slider_range[i] for i in idx]
    else:
        bounds = [(spec["joints"][i][3], spec["joints"][i][4]) for i in idx]

    # sync scratch data from display before solving
    world.data_ik.qpos[:] = world.data.qpos

    def cost(x):
        q5 = q5_init.copy()
        for k, i in enumerate(idx):
            q5[i] = x[k]
        world.set_arm_qpos_ik(arm_key, q5)
        d = world.ee_position_ik(arm_key) - target
        return float(np.dot(d, d))

    x0 = np.array([q5_init[i] for i in idx])
    res = optimize.minimize(cost, x0, method="SLSQP",
                            bounds=bounds,
                            options={"maxiter": 200, "ftol": 1e-9})
    q5_out = q5_init.copy()
    for k, i in enumerate(idx):
        q5_out[i] = res.x[k]
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

        # keep both buses alive so the Local/Remote toggle can swap without reinit
        self._local_bus = Bus()
        self._remote_bus = RemoteBus()
        self.bus = self._local_bus
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
        # per-arm shift so slider q=0 == "operator's chosen home pose"
        self.zero_offset = {"L": np.zeros(5), "R": np.zeros(5)}
        # per-joint slider range (rad).  Starts as the ARMS defaults; the
        # "讀馬達 EPROM 限位" button overwrites this with values derived from
        # each motor's actual EPROM min/max angle.  This is what
        # _refresh_slider_meta and ik_solve read (via _joint_lo_hi).
        self.slider_range = {
            arm: [(float(ARMS[arm]["joints"][i][3]),
                   float(ARMS[arm]["joints"][i][4])) for i in range(5)]
            for arm in ("L", "R")}
        # per-joint enable — only ON joints receive motor step commands, so
        # the user can bring joints up one at a time during IK verification
        self.motor_live = {"L": np.ones(5, dtype=bool),
                           "R": np.ones(5, dtype=bool)}
        self.cal_path = os.path.join(HERE, "qbot_arm_calibration.json")
        self._load_cal()
        # Apply HOME_Q as default zero_offset for arms that weren't restored
        # from the cal file (fresh run): slider zero maps to the mid-range home.
        for arm in ("L", "R"):
            if np.all(self.zero_offset[arm] == 0):
                self.zero_offset[arm] = np.array(HOME_Q[arm], dtype=float)
                self.q_by_arm[arm] = np.array(HOME_Q[arm], dtype=float)
        self.q = self.q_by_arm[self.active]
        # live-send throttle (per joint, 25 Hz)
        self._last_motor_send_t = [0.0] * 5
        self._suppress_slider = False
        # ── pose imitation state ────────────────────────────────────────────
        # Filled by the UDP listener thread; the render tick consumes when
        # the imitation toggle is on.  Timestamps in monotonic ms.
        self._pose_target = {"L": None, "R": None}   # each = list[5] q_rad
        self._pose_conf   = {"L": 0.0,  "R": 0.0}
        self._pose_ts_ms  = 0
        self._pose_sock   = None
        # 2D landmarks (33 points, [x, y, vis] normalised 0..1) — arm GUI's
        # camera panel overlays these so operator can verify pose detection.
        self._pose_lm2d = None
        # Per-arm offset — pose_q comes in some arbitrary frame (depends on
        # MediaPipe's world coord + the operator's stance); we align it with
        # ghost's frame by capturing the pose_q at the moment the operator
        # holds the T-pose (arms side horizontal).  Then:
        #     ghost_q = t_pose_ghost_q + (pose_q_now - pose_q_at_t_pose)
        self._pose_offset = {"L": np.zeros(5), "R": np.zeros(5)}
        # Ghost q that corresponds to the reference pose (arms hanging at
        # sides, elbow straight).  Camera FOV was too narrow for full T-pose;
        # this natural stance calibrates just as well and stays in-frame.
        self._pose_neutral_q = {
            "L": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "R": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        }
        # Calibration state machine — "idle" | "waiting_neutral" | "imitating"
        # In "waiting_neutral" we watch the incoming pose_q for the "arms
        # hanging at sides" signature (lift ≈ 0, elbow ≈ 0 on both arms).
        # Once that pose is stable for ~0.8s we snapshot the offset and jump
        # to "imitating" — no manual countdown needed.
        self._pose_state = "idle"
        self._neutral_samples = {"L": [], "R": []}   # sliding buffer of raw q
        # Detection tolerances (raw retargeted radians):
        self._neutral_lift_thresh  = 0.35    # ~20° max upper-arm→down angle
        self._neutral_elbow_thresh = 0.45    # ~26° max elbow flexion
        self._neutral_hold_frames  = 15      # ~0.6 s at 25Hz UDP rate
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
        # If env asked to start in Remote mode, propagate that through the
        # label/entry/bus swap logic (kept in one place: _on_remote_toggle).
        if self.remote_var.get():
            self._on_remote_toggle()
        self.world.set_arm_qpos(self.active, self._ghost_q())
        self.world.apply_arm_highlight(self.active)  # start with active arm solid
        self._refresh_slider_meta()
        self._refresh_cal_ui()
        self._render_to_canvas()
        # KICK OFF LOOPS (had accidentally been trapped inside _save_cal)
        threading.Thread(target=self._poll_loop, daemon=True).start()
        threading.Thread(target=self._pose_udp_loop, daemon=True).start()
        threading.Thread(target=self._cam_reader_loop, daemon=True).start()
        self.root.after(150, self._drain_tele)
        self.root.after(33, self._render_tick)
        self.root.after(50, self._pose_apply_tick)   # 20 Hz apply loop
        self.root.after(80, self._cam_display_tick)  # ~12 Hz cam refresh

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
                a = d.get(arm)
                if not a: continue
                # accept either legacy [ {tick_lo,...}, ... ] or new
                # { "points": [...], "zero_offset": [...], "motor_live": [...] }
                pts = a["points"] if isinstance(a, dict) and "points" in a else a
                if isinstance(pts, list) and len(pts) == 5 and isinstance(pts[0], dict):
                    self.cal_points[arm] = [
                        {k: float(p.get(k, 0)) for k in
                         ("tick_lo", "q_lo", "tick_hi", "q_hi")}
                        for p in pts
                    ]
                if isinstance(a, dict):
                    zo = a.get("zero_offset")
                    if zo and len(zo) == 5:
                        self.zero_offset[arm][:] = np.array(zo, dtype=float)
                    ml = a.get("motor_live")
                    if ml and len(ml) == 5:
                        self.motor_live[arm][:] = np.array(ml, dtype=bool)
                    sr = a.get("slider_range")
                    if sr and len(sr) == 5:
                        self.slider_range[arm] = [
                            (float(lo), float(hi)) for lo, hi in sr]
            # Auto-derive slider_range from cal_points so IK bounds always
            # match the calibrated joint range — no drift between the two.
            # For each joint, the safe q range is exactly (min(q_lo, q_hi),
            # max(q_lo, q_hi)) from cal_points; virtual joints stay at their
            # ARMS defaults.
            for arm in ("L", "R"):
                for i in range(5):
                    jname = ARMS[arm]["joints"][i][5]
                    if jname is None: continue         # virtual — leave default
                    p = self.cal_points[arm][i]
                    lo, hi = sorted((float(p["q_lo"]), float(p["q_hi"])))
                    self.slider_range[arm][i] = (lo, hi)
        except Exception as e:
            print(f"[cal] load failed: {e}")

    def _save_cal(self):
        d = {arm: {
                "points": [{k: float(v) for k, v in p.items()}
                           for p in self.cal_points[arm]],
                "zero_offset":  self.zero_offset[arm].tolist(),
                "motor_live":   [bool(x) for x in self.motor_live[arm]],
                "slider_range": [[float(lo), float(hi)]
                                 for (lo, hi) in self.slider_range[arm]],
             }
             for arm in ("L", "R")}
        try:
            with open(self.cal_path, "w") as f:
                json.dump(d, f, indent=2)
            self.status_var.set(f"校準存到 {self.cal_path}")
        except Exception as e:
            self.status_var.set(f"save cal failed: {e}")

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

        # ── right column: ghost canvas + inline camera panel ─────────────
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=8)
        # Row split: ghost (left / big) + camera preview (right / narrower)
        ghost_col = tk.Frame(right, bg=C["bg"])
        ghost_col.pack(side="left", fill="both", expand=True)
        cam_col = tk.Frame(right, bg="#0a1220", width=340)
        cam_col.pack(side="right", fill="y", padx=(6, 0))
        cam_col.pack_propagate(False)
        tk.Label(cam_col, text="相機視訊",
                 bg="#0a1220", fg="#7cf",
                 font=("DejaVu Sans", 11, "bold")).pack(pady=(6, 2))
        self.cam_label = tk.Label(cam_col, bg="#000000")
        self.cam_label.pack(padx=6, pady=4)
        self.cam_status = tk.StringVar(value="(等待相機串流)")
        tk.Label(cam_col, textvariable=self.cam_status,
                 bg="#0a1220", fg="#88a",
                 font=("DejaVu Sans Mono", 9)).pack(pady=(2, 6))
        self._cam_tk_img = None
        self._cam_bgr = None
        self._cam_last_t = 0.0
        self._cam_cap = None

        self.canvas = tk.Canvas(ghost_col, width=RENDER_W, height=RENDER_H,
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
        # bus mode toggle: local serial vs remote WebSocket (RPi arm agent).
        # QBOT_ARM_REMOTE_DEFAULT=1 auto-picks Remote mode on launch (used by
        # scripts/arm_remote_start.sh) so the operator doesn't have to toggle.
        self.remote_var = tk.BooleanVar(
            value=bool(int(os.environ.get("QBOT_ARM_REMOTE_DEFAULT", "0") or "0")))
        tk.Checkbutton(bar, text="Remote RPi",
                       variable=self.remote_var,
                       command=self._on_remote_toggle,
                       indicatoron=False, width=10, padx=6, pady=3,
                       relief="raised", borderwidth=2,
                       bg=C["btn"], fg="white",
                       activebackground=C["la"], activeforeground="white",
                       selectcolor=C["la"],
                       font=("DejaVu Sans Mono", 10, "bold")
                       ).pack(side="left", padx=(6, 2))
        self.port_label = tk.Label(bar, text="Port:", bg=C["card"], fg=C["text"])
        self.port_label.pack(side="left", padx=(6, 2))
        self.port_var = tk.StringVar(value=_find_arm_port())
        self.host_var = tk.StringVar(
            value=os.environ.get("QBOT_ARM_REMOTE_HOST", REMOTE_DEFAULT_HOST))
        self.port_entry = tk.Entry(bar, textvariable=self.port_var, width=22,
                                    bg=C["bg"], fg=C["text"], insertbackground=C["text"])
        self.port_entry.pack(side="left")
        self.connect_btn = tk.Button(bar, text="Connect", bg=C["btn"], fg="white",
                                     command=self._toggle_connect, width=10)
        self.connect_btn.pack(side="left", padx=6)
        # Live: when ON, the joint sliders push their motor live (throttled).
        # Rendered as an indicator-off toggle so the full rectangle is
        # clickable (the default check-square hit-box was too small).
        self.live_var = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="Live 拖即送",
                       variable=self.live_var,
                       indicatoron=False, width=10, padx=6, pady=3,
                       relief="raised", borderwidth=2,
                       bg=C["btn"], fg="white",
                       activebackground=C["ok"], activeforeground="white",
                       selectcolor=C["ok"],
                       font=("DejaVu Sans Mono", 10, "bold")
                       ).pack(side="left", padx=4)
        # IK-send: when ON, dragging the ghost end-effector also pushes each
        # moved joint's target to its motor (still gated by the per-joint
        # ON/OFF button below).  Default ON for the sim↔real verification flow.
        self.ik_send_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="IK 送",
                       variable=self.ik_send_var,
                       indicatoron=False, width=6, padx=6, pady=3,
                       relief="raised", borderwidth=2,
                       bg=C["btn"], fg="white",
                       activebackground=C["warn"], activeforeground="white",
                       selectcolor=C["warn"],
                       font=("DejaVu Sans Mono", 10, "bold")
                       ).pack(side="left", padx=4)
        self.conn_var = tk.StringVar(value="● 未連線")
        tk.Label(bar, textvariable=self.conn_var, bg=C["card"], fg=C["err"]
                 ).pack(side="right", padx=8)

        # ── Pose-imitation row (own row, big + bright so it's obvious) ──────
        pose_bar = tk.Frame(p, bg="#0f1a2c")
        pose_bar.pack(fill="x", padx=6, pady=(4, 6))
        tk.Label(pose_bar, text="姿態模仿:",
                 bg="#0f1a2c", fg="#7cf",
                 font=("DejaVu Sans", 11, "bold")
                 ).pack(side="left", padx=(8, 6))
        # Imitation mode toggle — dark OFF, glowing cyan ON.  Store button
        # ref so we can update its text ([模仿 OFF] ↔ [模仿 ON]) on click.
        self.imitate_var = tk.BooleanVar(value=False)
        self.imitate_btn = tk.Checkbutton(
            pose_bar, text="[ 模仿 OFF ]",
            variable=self.imitate_var,
            indicatoron=False, width=18, padx=10, pady=10,
            relief="raised", borderwidth=3, offrelief="raised",
            bg="#1a1a1a", fg="#606060",
            activebackground="#00d4ff", activeforeground="#000000",
            selectcolor="#00d4ff",
            font=("DejaVu Sans", 13, "bold"),
            command=self._on_imitate_btn)
        self.imitate_btn.pack(side="left", padx=6)
        # Motor-arm toggle — dark OFF, glowing red ON.
        self.pose_arm_var = tk.BooleanVar(value=False)
        self.pose_arm_btn = tk.Checkbutton(
            pose_bar, text="[ 送馬達 OFF ]",
            variable=self.pose_arm_var,
            indicatoron=False, width=20, padx=10, pady=10,
            relief="raised", borderwidth=3, offrelief="raised",
            bg="#1a1a1a", fg="#606060",
            activebackground="#ff3050", activeforeground="#ffffff",
            selectcolor="#ff3050",
            font=("DejaVu Sans", 13, "bold"),
            command=self._on_pose_arm_btn)
        self.pose_arm_btn.pack(side="left", padx=6)
        tk.Button(pose_bar, text="[R]重新校準",
                  bg="#7a6818", fg="white",
                  activebackground="#d0b040", activeforeground="black",
                  font=("DejaVu Sans", 10, "bold"),
                  padx=8, pady=6, relief="raised", borderwidth=2,
                  command=self._pose_recalibrate
                  ).pack(side="left", padx=6)
        # BIG status/prompt label — colour + text changes per state.  Sits
        # in its own second row below the buttons so the prompt is wide,
        # readable across the room.
        self.pose_prompt = tk.StringVar(
            value="勾「[模仿]」按鈕啟動 → 站到鏡頭前雙手自然下垂")
        self.pose_prompt_lbl = tk.Label(
            pose_bar, textvariable=self.pose_prompt,
            bg="#334455", fg="#ffd040",
            font=("DejaVu Sans", 13, "bold"),
            padx=12, pady=8, relief="raised", borderwidth=3)
        self.pose_prompt_lbl.pack(side="left", padx=8, fill="x", expand=True)

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
        self.sliders = []; self.id_lbls = []; self.live_btns = []
        self.cmd_vars = [tk.StringVar(value="0.0°") for _ in range(5)]
        self.read_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.step_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.volt_vars = [tk.StringVar(value="—") for _ in range(5)]
        self.temp_vars = [tk.StringVar(value="—") for _ in range(5)]
        for i in range(5):
            row = tk.Frame(jp, bg=C["card"]); row.pack(fill="x", padx=4, pady=2)
            top = tk.Frame(row, bg=C["card"]); top.pack(fill="x")
            # per-joint enable/disable — send-to-motor is gated by this
            lb = tk.Button(top, text="ON", width=3,
                           bg="#2e7d32", fg="white",
                           font=("DejaVu Sans Mono", 9, "bold"))
            lb.config(command=lambda i=i, b=lb: self._toggle_motor_live(i, b))
            lb.pack(side="left", padx=(2, 4))
            self.live_btns.append(lb)
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
            sld = tk.Scale(row, from_=0, to=4095, resolution=1,
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

        # ── zero-offset panel ─────────────────────────────────────────────
        # 精簡版:去掉 2-點 tick 校準 UI(那部份你已在馬達端做完).
        # 只留一個「設為零位」按鈕 — 把當下鬼影姿態記為 q=0,slider 顯示歸零、
        # 鬼影不動、馬達指令也不動(offset 吸收).
        cal = tk.LabelFrame(p, text=" 校準 ",
                            bg=C["panel"], fg=C["text"],
                            font=("DejaVu Sans Mono", 10, "bold"))
        cal.pack(fill="x", padx=6, pady=4)
        # legacy compatibility (some old handlers may touch cal_labels)
        self.cal_labels = [tk.StringVar(value="") for _ in range(5)]
        cb = tk.Frame(cal, bg=C["panel"]); cb.pack(fill="x", padx=4, pady=6)
        tk.Button(cb, text="設為零位(現在鬼影 → q=0)",
                  bg="#2e7d32", fg="white",
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=self._set_zero, width=26
                  ).pack(side="left", padx=2)
        tk.Button(cb, text="清除零位", bg=C["btn"], fg="white",
                  command=self._clear_zero, width=10
                  ).pack(side="left", padx=2)
        tk.Button(cb, text="Save", bg=C["btn"], fg="white",
                  command=self._save_cal, width=8).pack(side="right", padx=2)
        tk.Button(cb, text="Reload", bg=C["btn"], fg="white",
                  command=self._reload_cal, width=8).pack(side="right", padx=2)
        # Second row: pull EPROM angle limits from motors into cal_points.
        cb2 = tk.Frame(cal, bg=C["panel"]); cb2.pack(fill="x", padx=4, pady=(0, 6))
        tk.Button(cb2, text="讀馬達 EPROM 限位 → 拉桿",
                  bg=C["btn_warn"], fg="white",
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=self._pull_motor_limits, width=28
                  ).pack(side="left", padx=2)
        self.limit_info = tk.StringVar(value="")
        tk.Label(cb2, textvariable=self.limit_info,
                 bg=C["panel"], fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(side="left", padx=6)

        # 設新零位(每關節):當下馬達 tick → 中位 q=0,±2048 tick ↔ ±180°。
        # 單圈零位偏差(如肩旋轉 10/20)時,轉到正確中位按對應 J 鍵即對稱。
        cb3 = tk.Frame(cal, bg=C["panel"]); cb3.pack(fill="x", padx=4, pady=(0, 6))
        tk.Label(cb3, text="設新零位(當前tick→中位):", bg=C["panel"], fg=C["text"],
                 font=("DejaVu Sans Mono", 9, "bold")).pack(side="left")
        for j in range(5):
            tk.Button(cb3, text=f"J{j}", bg="#2e7d32", fg="white", width=4,
                      command=lambda i=j: self._set_center_zero(i)
                      ).pack(side="left", padx=1)

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
        # Suppress on-slider callbacks throughout: Tk may fire them when
        # from_/to_ is reduced and the current value gets auto-clamped, which
        # would otherwise clobber self.q with the clamped tick.
        self._suppress_slider = True
        # Sliders always span the full motor tick range (0..4095) so the
        # operator can reach anywhere the physical motor can go — cal_points
        # still defines the tick↔q mapping for ghost + IK.
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            self.sliders[i].config(from_=0, to=4095, resolution=1)
            self.id_lbls[i].config(text=f"{sid}")
        # per-joint Live button colours
        for i in range(5):
            on = bool(self.motor_live[self.active][i])
            self.live_btns[i].config(
                text=("ON" if on else "OFF"),
                bg=("#2e7d32" if on else C["btn_danger"]))
        # Slider tick position reflects the current q via cal_points.
        for i in range(5):
            tick = int(max(0, min(4095, self._q_to_step(i))))
            self.sliders[i].set(tick)
            self.cmd_vars[i].set(f"t={tick}")
        self._suppress_slider = False

    # ── slider / arm logic ───────────────────────────────────────────────────
    def _on_slider(self, i, val_str):
        if self._suppress_slider: return
        try: tick = int(float(val_str))
        except ValueError: return
        # Slider is in raw motor-tick units.  Map back to q (rad) via
        # cal_points so the ghost + IK still work in joint-angle space.
        q_before = self.q[i]
        self.world.set_arm_qpos(self.active, self._ghost_q())
        collide_before = self.world.check_self_collision(self.active)
        self.q[i] = self._step_to_q(i, tick)
        self.q_by_arm[self.active] = self.q
        self.world.set_arm_qpos(self.active, self._ghost_q())
        # Block moves that introduce a new self-collision — revert this joint
        # to its previous value so the operator can't drag through the body.
        if not collide_before and self.world.check_self_collision(self.active):
            self.q[i] = q_before
            self.q_by_arm[self.active] = self.q
            self.world.set_arm_qpos(self.active, self._ghost_q())
            self._suppress_slider = True
            prev_tick = int(max(0, min(4095, self._q_to_step(i))))
            self.sliders[i].set(prev_tick)
            self.cmd_vars[i].set(f"t={prev_tick}")
            self._suppress_slider = False
            self.status_var.set(f"!J{i} 撞到身體 blocked")
            self.err_lbl.config(fg=C["collide"])
            self.err_var.set("blocked: collision")
            return
        self.cmd_vars[i].set(f"t={tick}")
        # if Live: also send the matching motor (throttled per joint)
        self._maybe_send_motor(i)
        self.err_lbl.config(fg=C["ok"])
        self.err_var.set("ok")

    def _maybe_send_motor(self, i: int, source: str = "slider"):
        """Push joint i to its motor live (throttled to 25 Hz per joint).

        Independent toggles per source: ``slider`` -> ``live_var`` (default
        OFF), ``ik`` -> ``ik_send_var`` (default ON).  Both go through the
        per-joint ON/OFF gate so disabled joints never move.
        """
        if not self.bus.is_open(): return
        if source == "slider":
            gate = getattr(self, "live_var", None)
        elif source == "ik":
            gate = getattr(self, "ik_send_var", None)
        else:
            return
        if not (gate and gate.get()): return
        if not self.motor_live[self.active][i]:      # per-joint gate
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
        self._dbg_render_n += 1
        if self._dbg_render_n % 30 == 0:
            print(f"[render] tick {self._dbg_render_n}  "
                  f"q={np.degrees(self.q).round(1).tolist()}  "
                  f"ee={self.world.ee_position(self.active).round(3).tolist()}",
                  flush=True)
        img = self.world.render()
        pil = Image.fromarray(img)
        self._tk_img = ImageTk.PhotoImage(pil)
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._tk_img)
        else:
            self.canvas.itemconfig(self.canvas_img_id, image=self._tk_img)
        # overlay EE marker (project in image-pixel space, not canvas)
        ee = self.world.ee_position(self.active)
        proj = self.world.world_to_screen(ee, RENDER_W, RENDER_H)
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

    _dbg_render_n = 0

    def _render_tick(self):
        self._dbg_render_n += 1
        if self._dbg_render_n <= 3 or self._dbg_render_n % 30 == 0:
            try:
                q_dbg = np.degrees(self.q).round(1).tolist()
                ee_dbg = self.world.ee_position(self.active).round(3).tolist()
                print(f"[tick {self._dbg_render_n}]  q={q_dbg}  ee={ee_dbg}",
                      flush=True)
            except Exception as e:
                print(f"[tick {self._dbg_render_n}]  q-print err: {e}", flush=True)
        try:
            self._render_to_canvas()
        except Exception as e:
            import traceback
            self.status_var.set(f"render err: {e}")
            print(f"[render] EXCEPTION: {e}", flush=True)
            traceback.print_exc()
        self.root.after(33, self._render_tick)

    # ── mouse handlers ───────────────────────────────────────────────────────
    def _ee_pix(self):
        ee = self.world.ee_position(self.active)
        # Use RENDER image size (not canvas size) — the ghost is drawn at
        # (0, 0) of the canvas with a fixed RENDER_W × RENDER_H image, so
        # the click coordinates and marker coordinates must both be in
        # image-pixel space or they'll never match.
        return self.world.world_to_screen(ee, RENDER_W, RENDER_H), ee

    def _on_lmb_press(self, ev):
        # Decide: drag EE (if near) or orbit camera
        proj_ee = self._ee_pix()
        if proj_ee[0] is not None:
            sx, sy, depth = proj_ee[0]
            d = math.hypot(ev.x - sx, ev.y - sy)
            self.status_var.set(
                f"CLICK ({ev.x},{ev.y})  marker ({sx:.0f},{sy:.0f})  d={d:.0f}")
            print(f"[click] ({ev.x},{ev.y}) marker=({sx:.0f},{sy:.0f}) d={d:.0f}", flush=True)
            if d < 60:                # relaxed hit-box (was 24)
                self._drag = "ee"
                self._drag_pix = (ev.x, ev.y)
                self._drag_depth = depth
                self._drag_target0 = proj_ee[1].copy()
                self._drag_shift = bool(ev.state & 0x0001)
                return
        else:
            self.status_var.set(f"CLICK ({ev.x},{ev.y})  marker=off-cam")
            print(f"[click] ({ev.x},{ev.y}) marker=None", flush=True)
        # otherwise orbit
        self._drag = "orbit"
        self._drag_pix = (ev.x, ev.y)
        self._drag_cam0 = (self.world.cam.azimuth, self.world.cam.elevation)

    def _on_lmb_motion(self, ev):
        if self._drag == "ee":
            dx_pix = ev.x - self._drag_pix[0]
            dy_pix = ev.y - self._drag_pix[1]
            cw, ch = RENDER_W, RENDER_H       # match image-pixel space
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
        dv = self.world.screen_delta_to_world(dx_pix, dy_pix,
                                              self.world.cam.distance,
                                              RENDER_W, RENDER_H)
        self.world.cam.lookat[:] = (self._drag_lookat0 - dv)

    def _on_rmb_release(self, ev):
        self._drag = None

    def _on_scroll(self, ev):
        factor = 0.9 if ev.delta > 0 else 1/0.9
        self._zoom(factor, ev)

    def _zoom(self, factor, ev):
        self.world.cam.distance = max(0.3, min(8.0, self.world.cam.distance * factor))

    # ── IK with collision handling; out-of-reach clamps to nearest feasible ─
    def _attempt_target(self, target: np.ndarray):
        q_before = self.q.copy()
        self.world.set_arm_qpos(self.active, q_before)
        collide_before = self.world.check_self_collision(self.active)
        # ik_solve minimises |EE-target|; when target is beyond workspace, the
        # SLSQP result is already the closest reachable pose within joint
        # bounds — so we always apply it (no rejection).
        q5, err = ik_solve(self.world, self.active, q_before.copy(), target,
                           slider_range=self.slider_range[self.active])
        self.world.set_arm_qpos(self.active, q5)
        collide_after = self.world.check_self_collision(self.active)
        if collide_after and not collide_before:
            self.world.set_arm_qpos(self.active, q_before)
            self._drag_collide = True
            self.status_var.set("!collision — reverted")
            self.err_lbl.config(fg=C["collide"])
            self.err_var.set("blocked: collision")
            self.tgt_var.set(f"Target (blocked): ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")
            return
        self._drag_collide = collide_after
        self.q = q5
        self.q_by_arm[self.active] = self.q
        # Sliders show raw motor tick — convert q → tick via cal_points.
        self._suppress_slider = True
        for i in range(5):
            tick = int(max(0, min(4095, self._q_to_step(i))))
            self.sliders[i].set(tick)
            self.cmd_vars[i].set(f"t={tick}")
        self._suppress_slider = False
        # IK-send: push each moved joint's target to its motor (still gated
        # by the per-joint ON/OFF button so disabled joints stay stationary
        # for sim↔real verification).
        for i in range(5):
            if abs(q5[i] - q_before[i]) > 1e-4:
                self._maybe_send_motor(i, source="ik")
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
    def _on_remote_toggle(self):
        """Switch the port entry between serial-device and host:port form.
        Closes the current bus so the next Connect picks the new bus type.
        """
        if self.bus.is_open(): self.bus.close()
        self.connect_btn.config(text="Connect", bg=C["btn"])
        self.conn_var.set("● 未連線")
        remote = self.remote_var.get()
        # swap the entry's underlying var so the field shows the right hint
        if remote:
            self.port_label.config(text="RPi:")
            self.port_entry.config(textvariable=self.host_var)
            # switch the bus object
            self.bus = self._remote_bus
        else:
            self.port_label.config(text="Port:")
            self.port_entry.config(textvariable=self.port_var)
            self.bus = self._local_bus
        self.status_var.set(f"bus → {'remote WebSocket' if remote else 'local serial'}")

    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.connect_btn.config(text="Connect", bg=C["btn"])
            self.conn_var.set("● 未連線")
            return
        if self.remote_var.get():
            host = self.host_var.get().strip()
            ok, msg = self.bus.open(host)
        else:
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
            self.status_var.set(f"connected {msg}")
            # Read each motor's current tick and snap slider + ghost to it so
            # the operator never sees a mismatch between physical arm and
            # display.  Remote bus needs a moment for the first tele frame to
            # populate ``latest_tele`` before read_pos returns anything.
            delay = 800 if self.remote_var.get() else 100
            self.root.after(delay, self._sync_from_motors)
        else:
            self.status_var.set(f"connect failed: {msg}")

    def _sync_from_motors(self):
        """Read live motor ticks and update self.q / sliders / ghost to match.
        Skips any joint the bus can't currently report (keeps old q for that
        one) so a couple of dropped packets don't wreck the whole pose.
        """
        if not self.bus.is_open(): return
        spec = ARMS[self.active]
        synced = []; missed = []
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            tick = self.bus.read_pos(sid)
            if tick is None:
                missed.append(sid); continue
            self.q[i] = self._step_to_q(i, int(tick))
            synced.append(sid)
        self.q_by_arm[self.active] = self.q
        self.world.set_arm_qpos(self.active, self._ghost_q())
        self._refresh_slider_meta()
        parts = [f"從馬達同步 {len(synced)}/5"]
        if missed: parts.append(f"missed {missed}")
        self.status_var.set("  |  ".join(parts))

    def _send_to_robot(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線"); return
        if self.world.check_self_collision(self.active):
            self.status_var.set("!拒送 — 目前姿態 self-collision")
            return
        spec = ARMS[self.active]
        spd = int(self.speed_var.get()); acc = int(self.acc_var.get())
        sent, skipped = [], []
        for i, (sid, label, gear, lo, hi, jname) in enumerate(spec["joints"]):
            if not self.motor_live[self.active][i]:
                skipped.append(sid); continue
            step = max(0, min(4095, self._q_to_step(i)))
            self.bus.write_pos(sid, step, spd, acc)
            sent.append(sid)
        msg = f"sent {self.active}: {sent}"
        if skipped: msg += f"  (skip {skipped})"
        self.status_var.set(msg)

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
        pos = self.bus.read_pos(sid)
        if pos is None:
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

    def _set_zero(self):
        """Capture the current ghost qpos as q=0 reference for this arm.
        Slider display resets to 0°; actual joint q, ghost, motor commands
        do not change (offset absorbs the shift)."""
        self.zero_offset[self.active] = self.q.copy()
        self._refresh_slider_meta()
        self.status_var.set(
            f"零位設定 {self.active}: "
            f"{np.degrees(self.zero_offset[self.active]).round(1).tolist()}°")

    def _clear_zero(self):
        self.zero_offset[self.active][:] = 0.0
        self._refresh_slider_meta()
        self.status_var.set(f"{self.active} 零位已清除")

    def _set_center_zero(self, i):
        """把當下馬達 tick 設為此關節中位 q=0,上下各 ±2048 tick(半圈)對應 ±180°。
        只改校正換算、不送任何馬達指令。用於單圈零位偏差(如肩旋轉 10/20):
        轉到正確中位按一下,上下幅度就對稱、和鬼影/拉條一致。"""
        sid = ARMS[self.active]["joints"][i][0]
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — 無法讀 tick"); return
        pos = self.bus.read_pos(sid)
        if pos is None:
            self.status_var.set(f"J{i} 馬達無回應"); return
        p = self.cal_points[self.active][i]
        p["tick_lo"] = float(pos) - 2048.0; p["q_lo"] = -math.pi
        p["tick_hi"] = float(pos) + 2048.0; p["q_hi"] = +math.pi
        self.zero_offset[self.active][i] = 0.0
        self._refresh_cal_ui()
        self._refresh_slider_meta()
        self.status_var.set(
            f"J{i} ID{sid} 新零位 ← tick={int(pos)}  (±2048↔±180°)")

    def _toggle_motor_live(self, i: int, btn):
        self.motor_live[self.active][i] = not self.motor_live[self.active][i]
        on = bool(self.motor_live[self.active][i])
        btn.config(text=("ON" if on else "OFF"),
                   bg=("#2e7d32" if on else C["btn_danger"]))
        sid = ARMS[self.active]["joints"][i][0]
        self.status_var.set(f"J{i} ID{sid} motor→{'ON' if on else 'OFF'}")

    def _reset_cal(self):
        self.cal_points[self.active] = self._default_cal_points(self.active)
        self._refresh_cal_ui()
        self.status_var.set("校準回到預設(依 gear 自動推算)")

    def _reload_cal(self):
        self._load_cal()
        self._refresh_cal_ui()
        self.status_var.set("校準已從 JSON 重新載入")

    def _pull_motor_limits(self):
        """Read each joint's EPROM min/max angle (addr 9, 11) from the physical
        motor and populate cal_points[i]["tick_lo"/"tick_hi"].  q_lo/q_hi keep
        the ARMS range, so the slider auto-maps its full DOF to the motor's
        physical travel — no manual Set Lo/Set Hi needed.
        """
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — 無法讀 EPROM 限位"); return
        # In Remote mode the read is a WS round-trip; run on a thread so the
        # UI doesn't freeze while we poll 5 joints.
        threading.Thread(target=self._pull_motor_limits_worker,
                         daemon=True).start()
        self.limit_info.set("讀取中...")

    def _pull_motor_limits_worker(self):
        arm = self.active
        spec = ARMS[arm]
        ADDR_MIN_ANGLE = 9
        ADDR_MAX_ANGLE = 11
        results = []
        for i, (sid, label, gear, _lo, _hi, jname) in enumerate(spec["joints"]):
            mn = self.bus.read_word(sid, ADDR_MIN_ANGLE)
            mx = self.bus.read_word(sid, ADDR_MAX_ANGLE)
            if mn is None or mx is None:
                results.append((i, sid, None, None))
                continue
            # If min >= max the motor is in wheel/multi-turn mode (limits
            # meaningless) — skip so we don't clobber sane defaults.
            if mn >= mx:
                results.append((i, sid, mn, mx))
                continue
            # Convert tick range → joint angle range (rad) using the servo
            # constants: q = (tick - CENTER) / STEPS_PER_RAD / gear.
            q_lo = (mn - CENTER) / STEPS_PER_RAD / float(gear)
            q_hi = (mx - CENTER) / STEPS_PER_RAD / float(gear)
            self.slider_range[arm][i] = (float(q_lo), float(q_hi))
            # Keep cal_points consistent so _q_to_step maps slider→tick right
            p = self.cal_points[arm][i]
            p["tick_lo"] = float(mn); p["q_lo"] = float(q_lo)
            p["tick_hi"] = float(mx); p["q_hi"] = float(q_hi)
            results.append((i, sid, mn, mx))
        # marshal UI update back to Tk thread
        self.root.after(0, lambda: self._pull_motor_limits_finish(results))

    def _pull_motor_limits_finish(self, results):
        # Clamp current q to the new (possibly narrower) slider ranges so a
        # stale value doesn't sit outside the visible slider region.
        for i in range(5):
            lo, hi = self.slider_range[self.active][i]
            self.q[i] = float(np.clip(self.q[i], lo, hi))
        self.q_by_arm[self.active] = self.q
        self.world.set_arm_qpos(self.active, self._ghost_q())
        # slider ranges changed → re-apply from_/to and re-render slider values
        self._refresh_slider_meta()
        self._refresh_cal_ui()
        # summarise briefly
        ok = [r for r in results if r[2] is not None and r[3] is not None
              and r[2] < r[3]]
        bad = [r for r in results if r[2] is None or r[3] is None]
        wheel = [r for r in results if r[2] is not None and r[3] is not None
                 and r[2] >= r[3]]
        parts = [f"讀到 {len(ok)}/5"]
        if wheel: parts.append(f"wheel-mode(不動): {[r[1] for r in wheel]}")
        if bad:   parts.append(f"讀失敗: {[r[1] for r in bad]}")
        self.limit_info.set("  |  ".join(parts))
        # Detailed per-joint mapping in the main status bar for confirmation.
        detail = []
        for (i, sid, mn, mx) in results:
            if mn is None or mx is None or mn >= mx:
                detail.append(f"J{i}x"); continue
            lo, hi = self.slider_range[self.active][i]
            detail.append(f"J{i}[{math.degrees(lo):+.0f}°,{math.degrees(hi):+.0f}°]")
        self.status_var.set("拉桿範圍已更新: " + " ".join(detail))

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

    # ── inline camera panel (receives annotated JPEG from pose_imitator) ────
    def _cam_reader_loop(self):
        """Listen on UDP 9998 for annotated JPEG frames sent by
        pose_imitator (drawn with skeleton overlay).  This avoids opening a
        second MJPEG client on the RPi — pose_imitator is the sole camera
        consumer and forwards processed frames to us for display."""
        if cv2 is None:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", POSE_UDP_PORT - 1))    # 9998
            sock.settimeout(0.5)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
            print(f"[cam] listening on UDP 127.0.0.1:{POSE_UDP_PORT - 1}",
                  flush=True)
        except Exception as e:
            print(f"[cam] bind failed: {e}", flush=True); return
        buf_arr = np.zeros((0,), dtype=np.uint8)   # scratch
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except Exception:
                break
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self._cam_bgr = frame
                self._cam_last_t = time.monotonic()

    def _cam_display_tick(self):
        try:
            if cv2 is None or self._cam_bgr is None:
                self.cam_status.set("(等待相機串流)")
            else:
                dt = time.monotonic() - self._cam_last_t
                if dt > 3.0:
                    self.cam_status.set(f"! 相機停頓 {dt:.1f}s")
                else:
                    img = self._cam_bgr
                    h, w = img.shape[:2]
                    new_size = (w, h)
                    # Frame arrives already annotated by pose_imitator; just
                    # convert BGR→RGB for Tk.  No local resize needed since
                    # sender picked 320-wide already.
                    small = img
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    self._cam_tk_img = ImageTk.PhotoImage(pil)
                    self.cam_label.config(image=self._cam_tk_img)
                    self.cam_status.set(f"OK  {new_size[0]}x{new_size[1]}  "
                                        f"dt={dt*1000:.0f}ms")
        finally:
            self.root.after(80, self._cam_display_tick)

    # ── pose imitation: UDP listener + apply tick ────────────────────────────
    def _pose_udp_loop(self):
        """Bind UDP on 127.0.0.1:POSE_UDP_PORT and cache the latest joint
        targets from pose_imitator.  Non-blocking, tiny — the render tick
        consumes when the imitation toggle is on."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("127.0.0.1", POSE_UDP_PORT))
            sock.settimeout(0.5)
            self._pose_sock = sock
            print(f"[pose] listening on UDP 127.0.0.1:{POSE_UDP_PORT}", flush=True)
        except Exception as e:
            print(f"[pose] bind failed: {e}", flush=True)
            return
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(8192)
            except socket.timeout:
                continue
            except Exception:
                break
            try:
                d = json.loads(data.decode())
            except Exception:
                continue
            for arm in ("L", "R"):
                blk = d.get(arm)
                if not blk: continue
                q = blk.get("q")
                conf = float(blk.get("conf", 0.0))
                if isinstance(q, list) and len(q) == 5:
                    self._pose_target[arm] = [float(x) for x in q]
                    self._pose_conf[arm] = conf
            lm2d = d.get("lm2d")
            if isinstance(lm2d, list) and len(lm2d) >= 20:
                self._pose_lm2d = lm2d
            self._pose_ts_ms = int(time.monotonic() * 1000)

    def _on_imitate_btn(self):
        on = self.imitate_var.get()
        self.imitate_btn.config(text=("[* 模仿 ON *]" if on else "[ 模仿 OFF ]"))

    def _on_pose_arm_btn(self):
        on = self.pose_arm_var.get()
        self.pose_arm_btn.config(
            text=("[* 送馬達 ON *]" if on else "[ 送馬達 OFF ]"))

    def _set_pose_prompt(self, text, color="wait"):
        """Update the big status label + change its background so the
        operator can see the state at a glance without reading."""
        palette = {
            "wait":   ("#5c5c1a", "#ffff40"),   # yellow  — waiting
            "detect": ("#5c8c1a", "#e0ff40"),   # yellow-green — detecting
            "ok":     ("#1a7c1a", "#c0ffc0"),   # green  — imitating
            "err":    ("#7c1a1a", "#ffa0a0"),   # red    — error/stale
            "off":    ("#334455", "#a0b0c0"),   # grey   — idle
        }
        bg, fg = palette.get(color, palette["wait"])
        self.pose_prompt.set(text)
        try:
            self.pose_prompt_lbl.config(bg=bg, fg=fg)
        except Exception:
            pass

    def _pose_recalibrate(self):
        """Fast path: as soon as a pose packet arrives, snap the offset and
        start imitating.  No neutral-pose detection, no countdown — the
        first frame we see IS the reference.  The operator can move freely
        and re-press「重新校準」to re-zero at any time."""
        self._pose_state = "arming"
        self._set_pose_prompt("[等待] 等 pose_imitator 送第一幀 …", "wait")

    def _pose_apply_tick(self):
        """Two states of the imitation pipeline:
          * "calibrating" — user has just enabled imitate; system prompts for
            T-pose and collects ~3 s of samples, then computes offset.
          * "imitating"   — apply target = t_pose_ghost_q + (pose_q - offset),
            clip to slider_range, rate-limit, optionally send to motor.
        """
        try:
            imitate = bool(getattr(self, "imitate_var", None)
                           and self.imitate_var.get())
            was_on = getattr(self, "_imitate_prev", False)
            self._imitate_prev = imitate

            # OFF→ON: start calibration
            if imitate and not was_on:
                self._pose_recalibrate()

            # ON→OFF: back to idle
            if was_on and not imitate:
                self._pose_state = "idle"
                self._set_pose_prompt(
                    "勾「模仿」啟動 -> 雙手自然下垂,系統自動偵測後開始",
                    "off")

            stale = (int(time.monotonic() * 1000) - self._pose_ts_ms
                     > POSE_STALE_MS)

            # ── Arming phase — capture first pose then imitate immediately ─
            if self._pose_state == "arming":
                if stale:
                    self._set_pose_prompt(
                        "[等待] 沒收到 pose 訊號 - 檢查 pose_imitator", "err")
                else:
                    qL = self._pose_target.get("L")
                    qR = self._pose_target.get("R")
                    if qL is not None or qR is not None:
                        # Snap offset = incoming pose_q.  Neutral_q = MID of
                        # each joint's calibrated range so aligned has room
                        # to move both ways (avoids clipping when motor's
                        # sync placed the ghost at a range boundary).
                        for arm in ("L", "R"):
                            t = self._pose_target.get(arm)
                            if t is not None:
                                self._pose_offset[arm] = np.array(t, dtype=float)
                            rng = self.slider_range[arm]
                            self._pose_neutral_q[arm] = np.array(
                                [(lo + hi) / 2.0 for (lo, hi) in rng],
                                dtype=float)
                        self._pose_state = "imitating"
                        self._set_pose_prompt(
                            "[OK] 校準完成 - 開始模仿,動手看鬼影跟不跟", "ok")
                    else:
                        self._set_pose_prompt(
                            "[等待] 沒偵測到人 - 站到相機前", "wait")

            # ── Imitation phase (BOTH arms, simultaneous) ─────────────────
            if self._pose_state == "imitating":
                any_moved = False
                arm_motor = bool(getattr(self, "pose_arm_var", None)
                                 and self.pose_arm_var.get())
                # Loop over both arms — pose driven means both arms track
                # the operator; IK / slider testing still switches active
                # arm via the L/R radio buttons for per-arm calibration.
                for arm in ("L", "R"):
                    tgt = self._pose_target.get(arm)
                    if tgt is None or stale: continue
                    off = self._pose_offset[arm]
                    neu = self._pose_neutral_q[arm]
                    psign = POSE_SIGN[arm]
                    aligned = [
                        float(neu[i] + psign[i] * (tgt[i] - off[i]))
                        for i in range(5)
                    ]
                    rng = self.slider_range[arm]
                    old_q = self.q_by_arm[arm].copy()
                    new_q = old_q.copy()
                    for i in range(5):
                        if ARMS[arm]["joints"][i][5] is None and i == 4:
                            continue
                        lo, hi = rng[i]
                        q_clip = float(np.clip(aligned[i], lo, hi))
                        max_step = 0.15
                        dq = q_clip - new_q[i]
                        if abs(dq) > max_step:
                            dq = math.copysign(max_step, dq)
                        new_q[i] = new_q[i] + dq
                    self.q_by_arm[arm] = new_q
                    # Push both arms into the ghost world
                    self.world.set_arm_qpos(
                        arm, new_q + self.cal_trim[arm])
                    # Slider display + motor send only for the *active* arm
                    if arm == self.active:
                        self.q = new_q
                        self._suppress_slider = True
                        for i in range(5):
                            tick = int(max(0, min(4095, self._q_to_step(i))))
                            self.sliders[i].set(tick)
                            self.cmd_vars[i].set(f"t={tick}")
                        self._suppress_slider = False
                    # Motor send — always for BOTH arms when gate is on
                    if arm_motor:
                        # temporarily switch active for _q_to_step / bus send
                        prev_active = self.active
                        self.active = arm
                        self.q = new_q
                        for i in range(5):
                            self._maybe_send_motor(i, source="ik")
                        self.active = prev_active
                        self.q = self.q_by_arm[prev_active]
                    any_moved = True
                if any_moved:
                    tag = "-> MOTOR" if arm_motor else "(ghost only)"
                    cL = self._pose_conf.get("L", 0)
                    cR = self._pose_conf.get("R", 0)
                    self._set_pose_prompt(
                        f"[模仿中] L conf={cL:.2f}  R conf={cR:.2f}  {tag}",
                        "ok")
                elif stale:
                    self._set_pose_prompt(
                        f"[停頓] pose 訊號 stale (>{POSE_STALE_MS}ms) - "
                        "鬼影凍結,檢查 pose_imitator", "err")
        finally:
            self.root.after(50, self._pose_apply_tick)

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
