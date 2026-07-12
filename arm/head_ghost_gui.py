#!/usr/bin/env python3
"""Q-BOT head ghost GUI — drag the head, see it follow in MuJoCo.

A trimmed cousin of qbot_ik_gui.py focused on the three neck joints:
  J0  yaw     (ub_neck_yaw,   MJCF range ±90°,   motor ID 1)
  J1  pitch   (ub_neck_pitch, MJCF range ±30°,   motor ID 2)
  J2  roll    (ub_neck_roll,  MJCF range ±30°,   motor ID 3)

Mouse on the ghost canvas:
  LMB on head      ▸ drag rotates yaw (←→) + pitch (↑↓)
  Shift+LMB on head▸ drag rotates roll
  LMB elsewhere    ▸ orbit camera
  RMB drag         ▸ pan camera (lookat)
  Scroll           ▸ zoom

Stage 1 (this file) : virtual only.  No motors, no IMU.
Stage 2 (TBD)       : phone IMU  → ghost.
Stage 3 (TBD)       : ghost      → real Feetech motors (ID 1/2/3 @ ttyACM1).
"""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

import numpy as np
import mujoco
from PIL import Image, ImageTk

sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

import serial.tools.list_ports as _lp

# Stage-2 deps (phone IMU bridge) — kept optional so the GUI still launches
# even if aiohttp/cryptography aren't installed.
try:
    import asyncio, ipaddress, json, socket, ssl, datetime
    from aiohttp import web
    _HAVE_AIOHTTP = True
except ImportError:
    _HAVE_AIOHTTP = False

# ── paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.normpath(os.path.join(HERE, "..", "models", "QBOT_MJCF", "qbot.xml"))

RENDER_W = 560
RENDER_H = 620
TICKS_PER_DEG = 4096.0 / 360.0
CENTER = 2047

DEFAULT_PORT = "/dev/ttyACM1"
DEFAULT_BAUD = 1_000_000
# CH343 USB-serial serial number printed on the adapter; used to auto-find the
# right /dev/ttyACM* even after a reconnect bumps the device index.
HEAD_USB_SERIAL = "5B14115243"


def _find_head_port(default="/dev/ttyACM1"):
    """Return /dev/ttyACM* whose USB serial number matches HEAD_USB_SERIAL.
    Falls back to ``default`` if no match (e.g. before any motors plugged in)."""
    try:
        for p in _lp.comports():
            sn = (p.serial_number or "").upper()
            if sn == HEAD_USB_SERIAL.upper():
                return p.device
    except Exception:
        pass
    return default
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63
SEND_THROTTLE_S = 0.020  # 50Hz max SyncWrite — plenty smooth, easy on bus
SENDER_TICK_MS = 20      # 50 Hz fixed sender loop (smooths bursty slider input)
SMOOTH_ALPHA = 0.35      # low-pass: q_filt += alpha*(q_target - q_filt) each tick
                         #           0.35 ≈ ~150ms to reach target (clean & responsive)
SEND_EPS_DEG = 0.05      # don't write if motor target moved <0.05° from last send

# Stage-2 (phone IMU)
IMU_PORT = 8443
IMU_HTML = os.path.join(HERE, "head_imu.html")
IMU_CERT = os.path.join(HERE, "head_imu_cert.pem")
IMU_KEY  = os.path.join(HERE, "head_imu_key.pem")

# Head-body for drag-handle projection (computed once via MJCF inspection)
HEAD_BODY = "ub_link_8"

# Bodies that compose the head/neck assembly — geoms here are rendered SOLID
# (highlight colour); everything else stays translucent ghost.  Index list
# was found by walking parents downstream of ub_link_3 → ub_1_2 → ... .
HEAD_BODIES = ("ub_1_2", "ub_sts3215_original_v1_v1", "ub_link_4", "ub_link_8")

# Neck joints — ordered J0..J2 to match drag mapping (yaw, pitch, roll).
# Physical wiring (verified 2026-06 by dragging sliders against real robot):
#   ID 1 = yaw    range narrowed to ±22° (EPROM is 1600..2600 but chassis
#                                         hits the side at tick 1800)
#   ID 2 = roll   (motors 2 and 3 are swapped from the naive ID guess)
#   ID 3 = pitch  direction REVERSED (dir=-1): slider 抬頭 had driven the
#                                              motor 低頭 — flip sign
#
# `dir` ∈ {+1, -1} is applied between the slider value and the motor step;
# the slider's lo_deg/hi_deg always represent the *desired* axis direction
# (+ pitch = head up, + yaw = left turn, + roll = head toward left shoulder).
NECK = [
    # idx label    mjcf_joint        motor_id  dir  lo_deg  hi_deg
    # Physical wiring (verified 2026-07-05):
    #   sid 1 = yaw (turn head left/right)
    #   sid 2 = pitch (nod head up/down)
    #   sid 3 = roll  (tilt head to shoulder)
    # sid 2/3 were previously swapped in this table — corrected here so
    # slider labels match the actual joint that moves.
    (0, "Yaw",   "ub_neck_yaw",   1,   +1,  -22.0,  35.0),
    (1, "Pitch", "ub_neck_pitch", 2,   -1,  -22.0,  22.0),
    (2, "Roll",  "ub_neck_roll",  3,   +1,  -13.0,  13.0),
]

# Drag sensitivity (deg per pixel)
DRAG_YAW_DPP   = 0.25
DRAG_PITCH_DPP = 0.20
DRAG_ROLL_DPP  = 0.25

C = {
    "bg": "#111820", "panel": "#172030", "card": "#1c2a3a",
    "text": "#d0e0f0", "dim": "#6080a0", "bright": "#eef4fc",
    "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
    "accent": "#7ed3ff", "target": "#50c880",
    "btn": "#1e4870", "btn_danger": "#7a2828",
}


# ── World ────────────────────────────────────────────────────────────────────
class World:
    def __init__(self, path: str):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.model.vis.global_.offwidth = RENDER_W
        self.model.vis.global_.offheight = RENDER_H
        self.data = mujoco.MjData(self.model)
        self._ghost()
        self._index()
        self.renderer = mujoco.Renderer(self.model, height=RENDER_H, width=RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # tighter zoom — fill the canvas with head + upper torso (the user
        # still has the rest of the body as context but doesn't waste pixels)
        self.cam.azimuth = 135
        self.cam.elevation = -6
        self.cam.distance = 1.45
        self.cam.lookat[:] = [0.0, 0.0, 1.28]
        self.scene_option = mujoco.MjvOption()
        mujoco.mj_forward(self.model, self.data)
        self.head_bid = self.bid[HEAD_BODY]

    def _ghost(self):
        """Ghost the whole robot, then re-paint the HEAD geoms as solid+bright
        so it's obvious which part is being controlled (and which way is the
        front)."""
        m = self.model
        m.vis.headlight.ambient[:] = [0.55, 0.55, 0.6]
        m.vis.headlight.diffuse[:] = [0.9, 0.9, 0.92]
        m.vis.headlight.specular[:] = [0.3, 0.3, 0.35]
        # 1) wash every material + geom with translucent grey
        GREY = 0.40; ALPHA_GHOST = 0.40
        for i in range(m.nmat):
            m.mat_rgba[i] = [GREY, GREY + 0.04, GREY + 0.10, ALPHA_GHOST]
            m.mat_specular[i] = 0.35
            m.mat_shininess[i] = 0.3
        for i in range(m.ngeom):
            m.geom_rgba[i] = [GREY, GREY + 0.04, GREY + 0.10, ALPHA_GHOST]
        # 2) re-paint head geoms — fully opaque, warm orange so it pops
        head_bids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                     for n in HEAD_BODIES}
        head_bids.discard(-1)
        self._head_geoms = []
        for i in range(m.ngeom):
            if m.geom_bodyid[i] in head_bids:
                m.geom_rgba[i] = [0.95, 0.55, 0.20, 1.0]  # warm orange, opaque
                self._head_geoms.append(i)

    def _index(self):
        m = self.model
        self.jqpos = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i):
                      m.jnt_qposadr[i]
                      for i in range(m.njnt)
                      if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)}
        self.bid = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i
                    for i in range(m.nbody)
                    if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)}

    def set_neck(self, yaw_deg: float, pitch_deg: float, roll_deg: float):
        for idx, label, jname, motor_id, mdir, lo, hi in NECK:
            adr = self.jqpos[jname]
            val = (yaw_deg, pitch_deg, roll_deg)[idx]
            self.data.qpos[adr] = math.radians(max(lo, min(hi, val)))
        mujoco.mj_forward(self.model, self.data)

    def head_pos(self) -> np.ndarray:
        return self.data.xpos[self.head_bid].copy()

    # ── camera projection (same maths as qbot_ik_gui.py) ────────────────────
    def cam_basis(self):
        az = math.radians(self.cam.azimuth)
        el = math.radians(self.cam.elevation)
        fwd = np.array([math.cos(el)*math.cos(az),
                        math.cos(el)*math.sin(az),
                        math.sin(el)])
        up_world = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_world); right /= np.linalg.norm(right) or 1.0
        up = np.cross(right, fwd); up /= np.linalg.norm(up) or 1.0
        return fwd, right, up

    def cam_pos(self):
        fwd, _, _ = self.cam_basis()
        return np.array(self.cam.lookat) - fwd * self.cam.distance

    def fovy_rad(self):
        return math.radians(float(self.model.vis.global_.fovy))

    def world_to_screen(self, p, w, h):
        cpos = self.cam_pos()
        fwd, right, up = self.cam_basis()
        rel = p - cpos
        depth = float(np.dot(rel, fwd))
        if depth <= 1e-6: return None
        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        fov = self.fovy_rad()
        h_world = 2.0 * depth * math.tan(fov/2.0)
        w_world = h_world * (w / h)
        sx = w/2.0 + (x_cam / w_world) * w
        sy = h/2.0 - (y_cam / h_world) * h
        return sx, sy, depth

    def screen_delta_to_world(self, dx, dy, depth, w, h):
        fov = self.fovy_rad()
        h_world = 2.0 * depth * math.tan(fov/2.0)
        scale = h_world / h
        _, right, up = self.cam_basis()
        return right * (dx * scale) - up * (dy * scale)

    def render(self):
        self.renderer.update_scene(self.data, self.cam, self.scene_option)
        return self.renderer.render()


# ── phone IMU HTTPS+WebSocket bridge (Stage 2) ───────────────────────────────
def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_imu_cert(ip):
    if os.path.exists(IMU_CERT) and os.path.exists(IMU_KEY):
        return
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ip)])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address(ip))]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))
    with open(IMU_KEY, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(IMU_CERT, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


class ImuServer:
    """HTTPS + WebSocket bridge for the phone IMU.

    Runs an aiohttp server in a background thread. Phone opens
    ``https://<lan-ip>:8443/`` which serves head_imu.html; the page connects
    back via WebSocket ``/pose`` and streams ``{yaw,pitch,roll}`` ~50 Hz.
    The latest sample is held under a lock and read by the Tk sender_tick.
    """
    def __init__(self, port=IMU_PORT):
        self.port = port
        self.ip = _lan_ip()
        self.lock = threading.Lock()
        self.latest: tuple[float, float, float] | None = None
        self.last_t = 0.0
        self.client_count = 0
        self.message_count = 0     # cumulative WS messages received
        self.connect_count = 0     # cumulative WS upgrades observed
        self.error = ""
        self._thread = None

    def url(self) -> str:
        return f"https://{self.ip}:{self.port}"

    def get_pose(self):
        with self.lock:
            return (self.latest, self.last_t, self.client_count,
                    self.message_count, self.connect_count)

    def start(self):
        if not _HAVE_AIOHTTP:
            self.error = "aiohttp/cryptography not installed"
            return
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ── async server (private) ──────────────────────────────────────────────
    def _run(self):
        try:
            _ensure_imu_cert(self.ip)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._serve())
        except Exception as e:
            self.error = str(e)

    async def _serve(self):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(IMU_CERT, IMU_KEY)
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/pose", self._ws)
        runner = web.AppRunner(app)
        await runner.setup()
        # auto-bump port if already in use (another GUI instance perhaps)
        last_err = None
        for p in (self.port, self.port + 1, self.port + 2, self.port + 3):
            try:
                site = web.TCPSite(runner, "0.0.0.0", p, ssl_context=ctx)
                await site.start()
                self.port = p
                print(f"[imu] HTTPS listening on https://{self.ip}:{p}", flush=True)
                last_err = None
                break
            except OSError as e:
                last_err = e
                print(f"[imu] port {p} busy ({e}); trying next…", flush=True)
        if last_err is not None:
            self.error = f"all ports busy: {last_err}"
            return
        await asyncio.Event().wait()  # serve forever

    async def _index(self, request):
        return web.FileResponse(IMU_HTML)

    async def _ws(self, request):
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        with self.lock:
            self.client_count += 1
            self.connect_count += 1
        print(f"[imu] ws connect from {request.remote}  (total connects={self.connect_count})", flush=True)
        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT: continue
                try:
                    d = json.loads(msg.data)
                    pose = (float(d.get("yaw",   0.0)),
                            float(d.get("pitch", 0.0)),
                            float(d.get("roll",  0.0)))
                    with self.lock:
                        self.latest = pose
                        self.last_t = time.monotonic()
                        self.message_count += 1
                except Exception:
                    continue
        finally:
            with self.lock: self.client_count -= 1
            print(f"[imu] ws disconnect (now {self.client_count} clients)", flush=True)
        return ws


# ── hardware bus ─────────────────────────────────────────────────────────────
class Bus:
    def __init__(self):
        self._ph = None
        self._pk: sms_sts | None = None
        self._lock = threading.Lock()

    def is_open(self):
        return self._pk is not None

    def open(self, port, baud):
        self.close()
        self._ph = PortHandler(port)
        if not self._ph.openPort(): return False, "openPort failed"
        if not self._ph.setBaudRate(baud):
            self._ph.closePort(); return False, "setBaudRate failed"
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

    def write_sync(self, cmds, speed, acc):
        """Single GroupSyncWrite packet for many motors at once.

        cmds: iterable of (sid, step). No ACK, ~30 bytes total for 3 motors,
        so it's both faster and more reliable than a burst of WritePosEx
        which each wait for an ACK round-trip.
        """
        with self._lock:
            if not self._pk: return False
            try:
                self._pk.groupSyncWrite.clearParam()
                spd = int(speed); ac = int(acc)
                for sid, step in cmds:
                    step = max(0, min(4095, int(step)))
                    self._pk.SyncWritePosEx(sid, step, spd, ac)
                r = self._pk.groupSyncWrite.txPacket()
                self._pk.groupSyncWrite.clearParam()
                return r == COMM_SUCCESS
            except Exception:
                # never let a transient serial hiccup blow up the port
                try: self._pk.groupSyncWrite.clearParam()
                except Exception: pass
                return False

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
            v, comm, _ = self._pk.read1ByteTxRx(sid, ADDR_PRESENT_VOLTAGE)
            volt = v * 0.1 if comm == COMM_SUCCESS else float("nan")
            t, comm, _ = self._pk.read1ByteTxRx(sid, ADDR_PRESENT_TEMP)
            temp = t if comm == COMM_SUCCESS else 0
            return int(pos), volt, int(temp)


# ── App ──────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.world = World(MJCF_PATH)
        self.bus = Bus()
        self.imu = ImuServer()
        self.imu.start()                      # background HTTPS server thread
        self.imu_zero = np.zeros(3)           # calibration offsets (yaw,pitch,roll)
        self.imu_dir = np.array([1, 1, 1])    # per-axis ±1 sign flip
        self._imu_last_seen = 0.0

        self.root = tk.Tk()
        self.root.title("Q-BOT 頭部鬼影 / IMU")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1100x700")
        self.root.minsize(960, 660)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # state — q_deg is the slider TARGET (what user is asking for); q_filt
        # is what we actually drive ghost+motor with, low-pass filtered toward
        # q_deg in the 50 Hz sender loop.  This decouples bursty Tk slider
        # events from the smooth stream we want to send the motors.
        self.q_deg = np.zeros(3)
        self.q_filt = np.zeros(3)
        self._q_last_sent = np.full(3, -9999.0)
        self._suppress = False
        self._drag = None              # 'head' | 'orbit' | 'pan'
        self._drag_pix = (0, 0)
        self._drag_q0 = np.zeros(3)
        self._drag_cam0 = None
        self._drag_lookat0 = None
        self._drag_depth = 0.0
        self._last_send_t = 0.0             # one timestamp for the sync packet
        self._tele_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        self._build_ui()
        self.world.set_neck(0, 0, 0)
        self._refresh_sliders()

        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.root.after(150, self._drain_tele)
        self.root.after(33, self._tick)
        self.root.after(SENDER_TICK_MS, self._sender_tick)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#1a237e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Q-BOT  頭部鬼影",
                 font=("DejaVu Sans", 16, "bold"),
                 bg="#1a237e", fg="white").pack(side="left", padx=12, pady=8)
        tk.Label(hdr, text="拖頭=yaw+pitch | Shift+拖=roll | LMB背景=旋轉 | RMB=平移 | 滾輪=縮放",
                 font=("DejaVu Sans Mono", 9),
                 bg="#1a237e", fg="#90caf9").pack(side="left", padx=10)

        body = tk.Frame(self.root, bg=C["bg"]); body.pack(fill="both", expand=True)

        # left
        left = tk.Frame(body, bg=C["panel"], width=500)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        # right
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=8)
        self.canvas = tk.Canvas(right, width=RENDER_W, height=RENDER_H,
                                bg="#000000", highlightthickness=0,
                                cursor="fleur")
        self.canvas.pack(fill="both", expand=True)
        self.canvas_img_id = None
        self._tk_img = None
        self._dot_id = None
        self._ring_id = None
        # mouse
        self.canvas.bind("<ButtonPress-1>",       self._lmb_press)
        self.canvas.bind("<B1-Motion>",           self._lmb_motion)
        self.canvas.bind("<ButtonRelease-1>",     self._lmb_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self._lmb_press)
        self.canvas.bind("<Shift-B1-Motion>",     self._lmb_motion)
        self.canvas.bind("<ButtonPress-3>",       self._rmb_press)
        self.canvas.bind("<B3-Motion>",           self._rmb_motion)
        self.canvas.bind("<ButtonRelease-3>",     self._rmb_release)
        self.canvas.bind("<MouseWheel>",          self._scroll)
        self.canvas.bind("<Button-4>",            lambda e: self._zoom(0.9))
        self.canvas.bind("<Button-5>",            lambda e: self._zoom(1/0.9))

    def _build_controls(self, p):
        title = tk.LabelFrame(p, text=" 3-DOF 頭部 ",
                              bg=C["panel"], fg=C["text"],
                              font=("DejaVu Sans Mono", 10, "bold"))
        title.pack(fill="x", padx=6, pady=(6, 4))

        # connection bar (above the joint sliders) — prominent one-click button
        bar = tk.Frame(p, bg=C["card"]); bar.pack(fill="x", padx=6, pady=(0, 4))
        # big green start button: opens bus + flips Live on
        self.connect_btn = tk.Button(bar, text="▶ 啟動頭部",
                                     bg="#2e7d32", fg="white",
                                     activebackground="#1b5e20",
                                     font=("DejaVu Sans Mono", 11, "bold"),
                                     command=self._toggle_connect, width=12,
                                     pady=2)
        self.connect_btn.pack(side="left", padx=(6, 8), pady=4)
        self.live_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="Live", variable=self.live_var,
                       bg=C["card"], fg=C["ok"], activebackground=C["card"],
                       activeforeground=C["ok"], selectcolor=C["panel"],
                       font=("DejaVu Sans Mono", 9, "bold")
                       ).pack(side="left", padx=2)
        self.conn_var = tk.StringVar(value="● 未連線")
        tk.Label(bar, textvariable=self.conn_var, bg=C["card"], fg=C["err"],
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(side="left", padx=8)
        # Port entry tucked to the right (rarely needed once auto-detect works)
        self.port_var = tk.StringVar(value=_find_head_port())
        tk.Entry(bar, textvariable=self.port_var, width=12,
                 bg=C["bg"], fg=C["dim"], insertbackground=C["text"],
                 font=("DejaVu Sans Mono", 9)
                 ).pack(side="right", padx=(2, 6))
        tk.Label(bar, text="port:", bg=C["card"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="right")

        self.sliders = []
        self.cmd_vars = [tk.StringVar(value="0.0°") for _ in range(3)]
        self.step_vars = [tk.StringVar(value="—") for _ in range(3)]
        self.read_vars = [tk.StringVar(value="—") for _ in range(3)]
        self.volt_vars = [tk.StringVar(value="—") for _ in range(3)]
        self.temp_vars = [tk.StringVar(value="—") for _ in range(3)]

        for idx, label, jname, motor_id, mdir, lo, hi in NECK:
            row = tk.Frame(title, bg=C["card"])
            row.pack(fill="x", padx=4, pady=2)
            top = tk.Frame(row, bg=C["card"]); top.pack(fill="x")
            tk.Label(top, text=f"ID {motor_id}", width=5,
                     bg=C["card"], fg=C["accent"],
                     font=("DejaVu Sans Mono", 10, "bold")).pack(side="left", padx=2)
            tk.Label(top, text=label, width=7,
                     bg=C["card"], fg=C["bright"],
                     font=("DejaVu Sans Mono", 10, "bold")).pack(side="left")
            tk.Label(top, textvariable=self.cmd_vars[idx], width=8,
                     bg=C["card"], fg=C["warn"],
                     font=("DejaVu Sans Mono", 10, "bold")).pack(side="left", padx=2)
            tk.Label(top, text="step:", bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=(8, 1))
            tk.Label(top, textvariable=self.step_vars[idx], width=6,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            tk.Label(top, text=f"[{lo:+.0f},{hi:+.0f}]°",
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 8)).pack(side="right", padx=4)
            # second info row: read / volt / temp
            row2 = tk.Frame(row, bg=C["card"]); row2.pack(fill="x")
            tk.Label(row2, text="     read:", bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            tk.Label(row2, textvariable=self.read_vars[idx], width=9,
                     bg=C["card"], fg=C["ok"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            tk.Label(row2, textvariable=self.volt_vars[idx], width=7,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left", padx=4)
            tk.Label(row2, textvariable=self.temp_vars[idx], width=6,
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)).pack(side="left")
            sld = tk.Scale(row, from_=lo, to=hi, resolution=0.1,
                           orient="horizontal", length=380,
                           bg=C["card"], fg=C["text"],
                           highlightthickness=0, troughcolor=C["panel"],
                           showvalue=False,
                           command=lambda v, i=idx: self._on_slider(i, v))
            sld.pack(fill="x", padx=2, pady=(0, 2))
            self.sliders.append(sld)

        # speed / acc
        sa = tk.Frame(p, bg=C["panel"]); sa.pack(fill="x", padx=6, pady=2)
        tk.Label(sa, text="Speed:", bg=C["panel"], fg=C["text"]
                 ).pack(side="left", padx=2)
        # Speed=0 means "use motor's internal max speed" — best for live drag
        # because the motor doesn't artificially throttle between targets.
        # Sliding to high values (e.g. 1500) acts as a soft cap.
        self.speed_var = tk.IntVar(value=0)
        tk.Scale(sa, from_=0, to=2000, orient="horizontal", length=160,
                 variable=self.speed_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]).pack(side="left")
        tk.Label(sa, text="Acc:", bg=C["panel"], fg=C["text"]
                 ).pack(side="left", padx=(8, 2))
        self.acc_var = tk.IntVar(value=50)
        tk.Scale(sa, from_=1, to=150, orient="horizontal", length=120,
                 variable=self.acc_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]).pack(side="left")

        # buttons
        bp = tk.Frame(p, bg=C["panel"]); bp.pack(fill="x", padx=6, pady=4)
        tk.Button(bp, text="HOME (q=0)", bg=C["btn"], fg="white",
                  command=self._home, width=12).pack(side="left", padx=2)
        tk.Button(bp, text="Demo 點頭", bg=C["btn"], fg="white",
                  command=lambda: self._demo("nod"), width=10).pack(side="left", padx=2)
        tk.Button(bp, text="Demo 轉頭", bg=C["btn"], fg="white",
                  command=lambda: self._demo("shake"), width=10).pack(side="left", padx=2)
        tk.Button(bp, text="E-STOP", bg=C["btn_danger"], fg="white",
                  command=self._estop, width=10).pack(side="right", padx=2)
        tk.Button(bp, text="↻ 重啟", bg=C["btn"], fg="white",
                  command=self._restart, width=8).pack(side="right", padx=2)

        # ── IMU panel (Stage 2: phone IMU → ghost) ──
        ip = tk.LabelFrame(p, text=" 手機 IMU (Stage 2) ",
                           bg=C["panel"], fg=C["text"],
                           font=("DejaVu Sans Mono", 10, "bold"))
        ip.pack(fill="x", padx=6, pady=4)
        # URL line
        url_row = tk.Frame(ip, bg=C["panel"]); url_row.pack(fill="x", padx=6, pady=2)
        tk.Label(url_row, text="URL:", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        self.imu_url_var = tk.StringVar(value=self.imu.url())
        url_lbl = tk.Label(url_row, textvariable=self.imu_url_var,
                           bg=C["panel"], fg=C["accent"],
                           font=("DejaVu Sans Mono", 10, "bold"), cursor="hand2")
        url_lbl.pack(side="left", padx=6)
        url_lbl.bind("<Button-1>", lambda e: self._copy_url())
        tk.Label(url_row, text="(點一下複製)", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left")
        # status line
        st = tk.Frame(ip, bg=C["panel"]); st.pack(fill="x", padx=6, pady=1)
        self.imu_status_var = tk.StringVar(value="● 手機未連")
        tk.Label(st, textvariable=self.imu_status_var, bg=C["panel"],
                 fg=C["err"], font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(side="left")
        self.imu_rate_var = tk.StringVar(value="")
        tk.Label(st, textvariable=self.imu_rate_var, bg=C["panel"],
                 fg=C["dim"], font=("DejaVu Sans Mono", 9)
                 ).pack(side="right", padx=4)
        # raw pose readout
        self.imu_raw_var = tk.StringVar(value="raw: yaw — , pitch — , roll —")
        tk.Label(ip, textvariable=self.imu_raw_var, bg=C["panel"],
                 fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=6, pady=1)
        # graphical bars: phone-frame angles (Yaw -180..180, Pitch -90..90, Roll -90..90)
        self.imu_canvas = tk.Canvas(ip, height=86, bg=C["card"],
                                    highlightthickness=0)
        self.imu_canvas.pack(fill="x", padx=6, pady=3)
        self._imu_bar_ids: dict = {}        # filled by first _draw_imu_bars()
        # controls row 1: From IMU + Calibrate
        ctrl1 = tk.Frame(ip, bg=C["panel"]); ctrl1.pack(fill="x", padx=6, pady=(3, 1))
        self.imu_active_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl1, text="From IMU",
                       variable=self.imu_active_var,
                       bg=C["panel"], fg=C["ok"], activebackground=C["panel"],
                       activeforeground=C["ok"], selectcolor=C["card"],
                       font=("DejaVu Sans Mono", 10, "bold")
                       ).pack(side="left", padx=4)
        tk.Button(ctrl1, text="Calibrate (歸零)",
                  bg=C["btn"], fg="white",
                  command=self._calibrate_imu, width=18
                  ).pack(side="left", padx=4)
        # controls row 2: dir flips (own row so they never get clipped)
        ctrl2 = tk.Frame(ip, bg=C["panel"]); ctrl2.pack(fill="x", padx=6, pady=(0, 3))
        tk.Label(ctrl2, text="方向(IMU↔馬達 反向時點):",
                 bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left", padx=4)
        self.imu_dir_btns = []
        for i, ax in enumerate(("Yaw", "Pitch", "Roll")):
            b = tk.Button(ctrl2, text=f"{ax} +", width=8,
                          bg=C["btn"], fg="white",
                          font=("DejaVu Sans Mono", 9, "bold"))
            b.config(command=lambda i=i, b=b: self._toggle_imu_dir(i, b))
            b.pack(side="left", padx=2)
            self.imu_dir_btns.append(b)
        if not _HAVE_AIOHTTP:
            tk.Label(ip, text="⚠ pip install aiohttp cryptography",
                     bg=C["panel"], fg=C["err"]
                     ).pack(fill="x", padx=6, pady=2)

        self.status_var = tk.StringVar(value="ready")
        tk.Label(p, textvariable=self.status_var, anchor="w",
                 bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=6, pady=(4, 4))

    def _q_to_step(self, i: int) -> int:
        """Convert slider value q_deg[i] into raw motor tick (with dir flip)."""
        mdir = NECK[i][4]
        return CENTER + int(round(self.q_deg[i] * mdir * TICKS_PER_DEG))

    def _refresh_sliders(self):
        self._suppress = True
        for i in range(3):
            self.sliders[i].set(round(self.q_deg[i], 1))
            self.cmd_vars[i].set(f"{self.q_deg[i]:+6.1f}°")
            self.step_vars[i].set(f"{self._q_to_step(i):5d}")
        self._suppress = False

    # ── interaction ──────────────────────────────────────────────────────────
    def _on_slider(self, i, val_str):
        if self._suppress: return
        try: v = float(val_str)
        except ValueError: return
        # only update the TARGET — the 50 Hz sender_tick handles ghost + motor
        # with low-pass smoothing so micro-jitter and stop/start get rounded.
        self.q_deg[i] = v
        self.cmd_vars[i].set(f"{v:+6.1f}°")
        self.step_vars[i].set(f"{self._q_to_step(i):5d}")

    def _home(self):
        # Drop the target — sender_tick will smoothly drive ghost + motors home.
        self.q_deg[:] = 0.0
        self._refresh_sliders()
        if self.bus.is_open() and self.live_var.get():
            self.status_var.set("HOME → 平滑回中心")
        else:
            self.status_var.set("HOME (虛擬)")

    # ── 50 Hz sender loop ────────────────────────────────────────────────────
    def _sender_tick(self):
        """Drive ghost + motors from a low-pass-filtered version of q_deg.

        Runs at 50 Hz via root.after.  Decoupling the *target* (q_deg, set
        from Tk events) from what we actually push to MJCF/motors (q_filt)
        absorbs micro-jitter from slider drag and gives the motor a clean,
        steady stream of small step changes instead of bursty jumps.
        """
        # 0) optionally override q_deg from phone IMU
        if self.imu_active_var.get():
            pose, last_t, *_ = self.imu.get_pose()
            if pose is not None and (time.monotonic() - last_t) < 1.0:
                # wrap-aware diff for yaw (alpha is 0..360°, can roll past zero)
                yz, pz, rz = self.imu_zero
                yd = (((pose[0] - yz + 180.0) % 360.0) - 180.0) * self.imu_dir[0]
                pd = (pose[1] - pz) * self.imu_dir[1]
                rd = (pose[2] - rz) * self.imu_dir[2]
                vals = [yd, pd, rd]
                for i, v in enumerate(vals):
                    lo = NECK[i][5]; hi = NECK[i][6]
                    self.q_deg[i] = max(lo, min(hi, v))
                self._refresh_sliders()
        # 1) advance the filter toward target
        self.q_filt += SMOOTH_ALPHA * (self.q_deg - self.q_filt)
        # snap tiny residuals so we don't dribble bytes forever
        for i in range(3):
            if abs(self.q_deg[i] - self.q_filt[i]) < 0.02:
                self.q_filt[i] = self.q_deg[i]
        # 2) ghost (always, even without bus)
        self.world.set_neck(*self.q_filt)
        # 3) motor write, if hardware connected and "Live" toggled
        if self.bus.is_open() and self.live_var.get():
            now = time.monotonic()
            max_d = float(np.max(np.abs(self.q_filt - self._q_last_sent)))
            if (now - self._last_send_t >= SEND_THROTTLE_S
                    and max_d > SEND_EPS_DEG):
                cmds = []
                for j, row in enumerate(NECK):
                    motor_id = row[3]; mdir = row[4]
                    step = (CENTER
                            + int(round(self.q_filt[j] * mdir * TICKS_PER_DEG)))
                    cmds.append((motor_id, step))
                if self.bus.write_sync(cmds, self.speed_var.get(),
                                       self.acc_var.get()):
                    self._q_last_sent[:] = self.q_filt
                    self._last_send_t = now
        self.root.after(SENDER_TICK_MS, self._sender_tick)

    # ── hardware ─────────────────────────────────────────────────────────────
    def _maybe_send(self, i: int):
        """Throttled SYNC write of all 3 motors. ``i`` is ignored — it exists
        only so callers can pass the changed motor index. We always batch all
        three into one GroupSyncWrite packet (no ACK), which is faster *and*
        more reliable on the shared bus."""
        if not (self.bus.is_open() and self.live_var.get()):
            return
        now = time.monotonic()
        if now - self._last_send_t < SEND_THROTTLE_S:
            return
        self._last_send_t = now
        cmds = [(NECK[j][3], self._q_to_step(j)) for j in range(3)]
        self.bus.write_sync(cmds,
                            self.speed_var.get(), self.acc_var.get())

    def _send_all_now(self):
        """Force-send all 3 motors as one sync packet, bypassing the throttle."""
        if not (self.bus.is_open() and self.live_var.get()):
            return
        cmds = [(NECK[j][3], self._q_to_step(j)) for j in range(3)]
        self.bus.write_sync(cmds,
                            self.speed_var.get(), self.acc_var.get())
        self._last_send_t = time.monotonic()

    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.connect_btn.config(text="▶ 啟動頭部", bg="#2e7d32",
                                    activebackground="#1b5e20")
            self.conn_var.set("● 未連線")
            self.status_var.set("disconnected")
            return
        port = self.port_var.get().strip()
        ok, msg = self.bus.open(port, DEFAULT_BAUD)
        if not ok:
            # auto-retry with serial-number-resolved port (handles ttyACM
            # bumping to a new index after a USB reconnect glitch)
            auto = _find_head_port(default=port)
            if auto != port:
                ok2, msg2 = self.bus.open(auto, DEFAULT_BAUD)
                if ok2:
                    self.port_var.set(auto)
                    ok, msg = True, f"retry@{auto}"
        if ok:
            self.connect_btn.config(text="■ 停止頭部", bg=C["btn_danger"],
                                    activebackground="#5a1f1f")
            self.conn_var.set("● 已連線")
            self.live_var.set(True)        # one-click: also turn on Live
            self.status_var.set(f"connected {self.port_var.get()}")
            # On connect, push current virtual pose to motor (so virtual ↔ real
            # don't disagree on first slider tick). Use slow speed.
            spd_save = self.speed_var.get()
            self.speed_var.set(200)
            self._send_all_now()
            self.speed_var.set(spd_save)
        else:
            self.status_var.set(f"connect failed: {msg}")

    def _estop(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — E-STOP 無作用"); return
        for idx, label, jname, motor_id, mdir, lo, hi in NECK:
            self.bus.torque(motor_id, False)
        self.status_var.set("E-STOP — torque off (扭力已關閉)")

    # ── Stage 2: phone IMU helpers ───────────────────────────────────────────
    def _draw_imu_bars(self, pose):
        """Live yaw/pitch/roll bars on imu_canvas.
        Each row: label, full-range bar with center mark, value bar from
        center to current angle, and the calibrated zero as a tick mark."""
        cv = self.imu_canvas
        w = int(cv.winfo_width()) or 380
        h = 86
        rows = [("Yaw",  0, 180.0),
                ("Pitch", 1, 90.0),
                ("Roll", 2, 90.0)]
        # init shapes once
        if "bg" not in self._imu_bar_ids:
            self._imu_bar_ids["bg"] = []
            self._imu_bar_ids["val"] = []
            self._imu_bar_ids["zero"] = []
            self._imu_bar_ids["lbl"] = []
            for r, (name, _, _) in enumerate(rows):
                y = 8 + r * 26
                self._imu_bar_ids["lbl"].append(
                    cv.create_text(28, y + 7, text=name, fill=C["dim"],
                                   font=("DejaVu Sans Mono", 9, "bold"),
                                   anchor="w"))
                self._imu_bar_ids["bg"].append(
                    cv.create_rectangle(80, y, w-12, y+14,
                                        fill=C["bg"], outline=C["dim"]))
                self._imu_bar_ids["zero"].append(
                    cv.create_line(0, 0, 0, 0, fill=C["target"], width=2))
                self._imu_bar_ids["val"].append(
                    cv.create_rectangle(0, 0, 0, 0, fill=C["accent"],
                                        outline=""))
            # center mark for each bar
            self._imu_bar_ids["center"] = []
            for r in range(3):
                y = 8 + r * 26
                cx = (80 + w-12) // 2
                self._imu_bar_ids["center"].append(
                    cv.create_line(cx, y-3, cx, y+17, fill=C["dim"]))
        # update shapes
        for r, (name, i, mag) in enumerate(rows):
            y = 8 + r * 26
            cv.coords(self._imu_bar_ids["bg"][r], 80, y, w-12, y+14)
            cx = (80 + w-12) / 2.0
            half = (w-12 - 80) / 2.0
            cv.coords(self._imu_bar_ids["center"][r], cx, y-3, cx, y+17)
            # calibrated zero (after calibrate, the zero is at angle = imu_zero[i])
            z = float(np.clip(self.imu_zero[i] / mag, -1, 1))
            zx = cx + z * half
            cv.coords(self._imu_bar_ids["zero"][r], zx, y-2, zx, y+16)
            # current value bar
            if pose is None:
                cv.coords(self._imu_bar_ids["val"][r], 0, 0, 0, 0)
            else:
                v = float(np.clip(pose[i] / mag, -1, 1))
                x1 = cx; x2 = cx + v * half
                if x2 < x1: x1, x2 = x2, x1
                cv.coords(self._imu_bar_ids["val"][r], x1, y+2, x2, y+12)

    def _calibrate_imu(self):
        pose, last_t, *_ = self.imu.get_pose()
        if pose is None or (time.monotonic() - last_t) > 1.0:
            self.status_var.set("校準失敗 — 沒收到手機 pose")
            return
        self.imu_zero[:] = pose
        # also zero q_deg target so the calibration position == ghost home
        self.q_deg[:] = 0.0
        self._refresh_sliders()
        self.status_var.set(
            f"已校準: yaw0={pose[0]:+.1f}  pitch0={pose[1]:+.1f}  roll0={pose[2]:+.1f}")

    def _toggle_imu_dir(self, i: int, btn):
        self.imu_dir[i] = -self.imu_dir[i]
        ax = ("Y", "P", "R")[i]
        sign = "+" if self.imu_dir[i] > 0 else "-"
        btn.config(text=f"{ax}{sign}",
                   bg=(C["btn"] if self.imu_dir[i] > 0 else C["btn_danger"]))
        self.status_var.set(f"IMU {('Yaw','Pitch','Roll')[i]} 方向 → {sign}1")

    def _copy_url(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.imu.url())
            self.status_var.set(f"已複製: {self.imu.url()}")
        except Exception:
            pass

    def _demo(self, kind: str):
        # scripted motion: just update the target — sender_tick smooths and
        # drives ghost+motor.  Demo amplitudes intentionally stay inside the
        # narrowed yaw/roll/pitch limits.
        self.status_var.set(f"demo: {kind}")
        steps = 60
        if kind == "nod":
            seq = [(0, 10 * math.sin(2*math.pi*i/steps), 0) for i in range(steps*2)]
        else:  # shake
            seq = [(28 * math.sin(2*math.pi*i/steps), 0, 0) for i in range(steps*2)]
        seq.append((0, 0, 0))
        for y, p, r in seq:
            self.q_deg[:] = (y, p, r)
            self._refresh_sliders()
            self.root.update_idletasks()
            self.root.update()
            time.sleep(0.02)
        self.status_var.set("demo done")

    # ── mouse ────────────────────────────────────────────────────────────────
    def _head_pix(self):
        cw = int(self.canvas.winfo_width()) or RENDER_W
        ch = int(self.canvas.winfo_height()) or RENDER_H
        p = self.world.head_pos()
        return self.world.world_to_screen(p, cw, ch)

    def _lmb_press(self, ev):
        proj = self._head_pix()
        if proj is not None:
            sx, sy, depth = proj
            if math.hypot(ev.x - sx, ev.y - sy) < 50:
                self._drag = "head"
                self._drag_pix = (ev.x, ev.y)
                self._drag_q0 = self.q_deg.copy()
                self._drag_shift = bool(ev.state & 0x0001)
                self.status_var.set("dragging head")
                return
        self._drag = "orbit"
        self._drag_pix = (ev.x, ev.y)
        self._drag_cam0 = (self.world.cam.azimuth, self.world.cam.elevation)

    def _lmb_motion(self, ev):
        if self._drag == "head":
            dx = ev.x - self._drag_pix[0]
            dy = ev.y - self._drag_pix[1]
            if ev.state & 0x0001:                # Shift → roll
                self.q_deg[2] = self._drag_q0[2] + dx * DRAG_ROLL_DPP
            else:
                self.q_deg[0] = self._drag_q0[0] + dx * DRAG_YAW_DPP    # yaw
                self.q_deg[1] = self._drag_q0[1] - dy * DRAG_PITCH_DPP  # pitch (up=neg dy)
            # clamp to MJCF limits — sender_tick handles ghost + motor.
            for idx, _, _, _, _, lo, hi in NECK:
                self.q_deg[idx] = max(lo, min(hi, self.q_deg[idx]))
            self._refresh_sliders()
            return
        if self._drag == "orbit":
            dx = ev.x - self._drag_pix[0]
            dy = ev.y - self._drag_pix[1]
            az0, el0 = self._drag_cam0
            self.world.cam.azimuth = az0 - dx * 0.4
            self.world.cam.elevation = max(-89.0, min(89.0, el0 - dy * 0.4))
            return

    def _lmb_release(self, ev):
        self._drag = None

    def _rmb_press(self, ev):
        self._drag = "pan"
        self._drag_pix = (ev.x, ev.y)
        self._drag_lookat0 = np.array(self.world.cam.lookat).copy()

    def _rmb_motion(self, ev):
        if self._drag != "pan": return
        dx = ev.x - self._drag_pix[0]
        dy = ev.y - self._drag_pix[1]
        cw = int(self.canvas.winfo_width()) or RENDER_W
        ch = int(self.canvas.winfo_height()) or RENDER_H
        dv = self.world.screen_delta_to_world(dx, dy, self.world.cam.distance, cw, ch)
        self.world.cam.lookat[:] = (self._drag_lookat0 - dv)

    def _rmb_release(self, ev):
        self._drag = None

    def _scroll(self, ev):
        self._zoom(0.9 if ev.delta > 0 else 1/0.9)

    def _zoom(self, factor):
        self.world.cam.distance = max(0.25, min(4.0, self.world.cam.distance * factor))

    # ── render ───────────────────────────────────────────────────────────────
    def _tick(self):
        try:
            img = self.world.render()
            pil = Image.fromarray(img)
            self._tk_img = ImageTk.PhotoImage(pil)
            if self.canvas_img_id is None:
                self.canvas_img_id = self.canvas.create_image(0, 0, anchor="nw",
                                                              image=self._tk_img)
            else:
                self.canvas.itemconfig(self.canvas_img_id, image=self._tk_img)
            # head marker
            proj = self._head_pix()
            if proj is not None:
                sx, sy, _ = proj
                R = 22
                if self._dot_id is None:
                    self._dot_id = self.canvas.create_oval(sx-5, sy-5, sx+5, sy+5,
                                                            fill=C["target"], outline="")
                    self._ring_id = self.canvas.create_oval(sx-R, sy-R, sx+R, sy+R,
                                                            outline=C["target"], width=2)
                else:
                    self.canvas.coords(self._dot_id, sx-5, sy-5, sx+5, sy+5)
                    self.canvas.coords(self._ring_id, sx-R, sy-R, sx+R, sy+R)
                self.canvas.tag_raise(self._ring_id)
                self.canvas.tag_raise(self._dot_id)
        except Exception as e:
            self.status_var.set(f"render err: {e}")
        # IMU status refresh (cheap, every render tick)
        try:
            pose, last_t, count, msg_n, conn_n = self.imu.get_pose()
            now = time.monotonic()
            # refresh URL in case port was bumped
            self.imu_url_var.set(self.imu.url())
            if self.imu.error:
                self.imu_status_var.set(f"● 啟動失敗: {self.imu.error}")
            elif count <= 0:
                if conn_n == 0:
                    self.imu_status_var.set("● 等手機開頁面")
                else:
                    self.imu_status_var.set(f"● 已斷開(過去 {conn_n} 次連線)")
                self.imu_rate_var.set(f"messages: {msg_n}")
            elif pose is None or (now - last_t) > 1.0:
                self.imu_status_var.set("● 連線中…等資料")
                self.imu_rate_var.set(f"messages: {msg_n}")
            else:
                self.imu_status_var.set("● 手機已連接 ✓")
                self.imu_rate_var.set(
                    f"msg: {msg_n}  /  last {(now-last_t)*1000:.0f}ms")
            if pose is None:
                self.imu_raw_var.set("raw: yaw — , pitch — , roll —")
            else:
                self.imu_raw_var.set(
                    f"raw: yaw {pose[0]:+7.1f}  pitch {pose[1]:+7.1f}  roll {pose[2]:+7.1f}")
            # update graphical bars
            self._draw_imu_bars(pose)
        except Exception as e:
            self.status_var.set(f"imu ui err: {e}")
        self.root.after(33, self._tick)

    # ── telemetry ────────────────────────────────────────────────────────────
    def _poll_loop(self):
        while not self._stop.is_set():
            # Slow polling (1 Hz) to leave bus headroom for SyncWrite bursts.
            # Skip entirely if a write happened recently — reads block on ACK
            # and would otherwise serialise behind queued writes.
            if self.bus.is_open():
                if time.monotonic() - self._last_send_t > 0.25:
                    for idx, label, jname, motor_id, mdir, lo, hi in NECK:
                        if self._stop.is_set(): break
                        r = self.bus.read_quick(motor_id)
                        self._tele_q.put((idx, mdir, r))
            time.sleep(1.0)

    def _drain_tele(self):
        try:
            while True:
                i, mdir, r = self._tele_q.get_nowait()
                if r is None:
                    self.read_vars[i].set("—"); self.volt_vars[i].set("—")
                    self.temp_vars[i].set("—")
                else:
                    pos, v, t = r
                    # invert dir so the read° matches the slider's semantic
                    deg = (pos - CENTER) / TICKS_PER_DEG * mdir
                    self.read_vars[i].set(f"{deg:+6.1f}°")
                    self.volt_vars[i].set(f"{v:4.1f}V" if v == v else "—")
                    col = C["err"] if t > 60 else (C["warn"] if t > 50 else C["dim"])
                    self.temp_vars[i].set(f"{t:3d}°C")
        except queue.Empty:
            pass
        self.root.after(150, self._drain_tele)

    def _on_close(self):
        self._stop.set()
        try:
            if self.bus.is_open():
                # leave motors holding their current target (don't kill torque
                # automatically — that would cause head to flop. Use E-STOP if
                # you want torque off.)
                self.bus.close()
        finally:
            self.root.destroy()

    def _restart(self):
        """Tear down + re-exec the same script (frees port 8443 for clean
        rebind, picks up code changes if any)."""
        self._stop.set()
        try:
            if self.bus.is_open():
                self.bus.close()
            self.root.destroy()
        except Exception:
            pass
        os.execvp(sys.executable, [sys.executable, sys.argv[0]] + sys.argv[1:])

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
