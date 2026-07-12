#!/usr/bin/env python3
"""Q-BOT head Raspberry Pi headless agent.

Runs on the RPi that's physically wired to the head hardware:
  * Feetech STS3215 gimbal motors  (USB serial /dev/ttyACM*)
  * USB camera                     (/dev/video*)

Exposes over LAN:
  * ws://<rpi-ip>:8000/ws         — bidirectional JSON motor control + telemetry
  * http://<rpi-ip>:8000/mjpeg    — MJPEG camera stream (multipart)
  * http://<rpi-ip>:8000/snapshot — single JPEG (for debugging)
  * http://<rpi-ip>:8000/status   — JSON health check
  * http://<rpi-ip>:8000/         — tiny HTML dashboard

Design goals:
  * LAN-only (no auth by default; assumes trusted network).
  * Low latency: WebSocket for control (<5ms round-trip on LAN),
                 MJPEG for camera (~50-100ms end-to-end).
  * No display / X server dependency.

Protocol (WebSocket JSON):
  --- Client → Agent ---
    {"op":"sync",     "cmds":[[sid,step], ...], "speed":300, "acc":30}
    {"op":"write",    "sid":1, "step":2048, "speed":300, "acc":30}
    {"op":"torque",   "sid":1, "on":true}
    {"op":"torque_all","on":false}                    # emergency stop
    {"op":"read",     "sid":1, "addr":9, "size":2, "req_id":"x"}
    {"op":"scan",     "range":[1,50], "req_id":"x"}
    {"op":"tele_hz",  "hz":50}                        # subscribe rate
  --- Agent → Client ---
    {"t":"tele","ts":<mono>,"motors":[{"sid":1,"pos":2048,"spd":0,
                                       "load":0.0,"volt":12.3,"temp":35}, ...]}
    {"t":"reply","req_id":"x","ok":true,"value":<...>}
    {"t":"log","level":"warn","msg":"…"}
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import queue
import socket
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field

import serial
import serial.tools.list_ports
from aiohttp import web, WSMsgType
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feetech_lib import SMS_STS  # noqa: E402

# scservo_sdk gives us the big-endian scscl handler for SCS009 (model 1029)
# hand servos which share the daisy chain with SMS/STS motors.
# We wrap the existing serial.Serial into a scservo_sdk PortHandler manually
# so both handlers share the same physical port.
sys.path.insert(0, os.path.expanduser("~/FTServo_Python"))
try:
    from scservo_sdk import PortHandler as _SCS_PortHandler, scscl as _SCScl  # noqa: E402
    _SCS_AVAILABLE = True
except Exception as _e:
    print(f"[warn] scservo_sdk not available: {_e}", flush=True)
    _SCS_AVAILABLE = False

# SCS009 model id — read via _read_bytes(sid, 3, 2); dispatched to scscl.
SCS_MODEL_IDS = {1029}
SCS_TIME_MS = 0                # 0 = follow speed
SCS_SPEED = 1023               # 10-bit max
SCS_TORQUE_REG = 40            # same address as STS


# ── config ───────────────────────────────────────────────────────────────────
DEFAULT_BAUD = 1_000_000
DEFAULT_PORT_HINT = "/dev/ttyACM1"        # if serial-number search fails
# Set this to your CH343 serial number to force auto-detection.  Blank = use
# the first ttyACM* found (Feetech SMS bus adapters look like QinHeng 1a86:55d3).
HEAD_USB_SERIAL = os.environ.get("HEAD_USB_SERIAL", "").strip()

# ── Neck-motor mapping used when driving from phone IMU ─────────────────────
# Each axis (yaw/pitch/roll from phone deviceorientation) → one motor + a
# 2-point linear tick mapping.  Defaults come from the user's earlier
# calibration; can be overridden at runtime via `/api/imu_config` HTTP POST
# or the WebSocket op "set_neck".
NECK_MAPPING = {
    # Physical wiring (verified 2026-07-05):
    #   sid 1 = yaw (turn head left/right)
    #   sid 2 = pitch (nod head up/down)   ← was labelled "roll" in earlier code
    #   sid 3 = roll  (tilt to shoulder)   ← was labelled "pitch" in earlier code
    # Phone control uses yaw + pitch only; sid 3 (roll) is intentionally not
    # driven so the head keeps upright regardless of phone tilt.
    "yaw":   {"sid": 1, "dir": +1, "tick_lo": 1800, "q_lo": -22.0,
              "tick_hi": 2450, "q_hi": +35.0},
    "pitch": {"sid": 2, "dir": -1, "tick_lo": 1800, "q_lo": -22.0,
              "tick_hi": 2500, "q_hi": +22.0},
    "roll":  {"sid": 3, "dir": -1, "tick_lo": 1900, "q_lo": -13.0,
              "tick_hi": 2200, "q_hi": +13.0},
}
NECK_MAPPING_LOCK = threading.Lock()
IMU_ZERO = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

# Fixed conservative half-range applied after wizard.  Regardless of the
# phone's actual tilt magnitude at cal extremes, the resulting mapping puts
# ±MAX_YAW_DEG (yaw) / ±MAX_PITCH_DEG (pitch) as the phone→motor bounds.
# This keeps the mapping symmetric around forward=0 and stops small phone
# twitches from swinging the whole motor range.
MAX_YAW_DEG   = 30.0
MAX_PITCH_DEG = 15.0
# If the physical motor tick_lo/tick_hi wiring is opposite to what the
# phone expects (i.e. you look UP but the head goes DOWN), flip that axis.
# Set once during commissioning; overridable via env var.
AXIS_FLIP = {
    "yaw":   os.environ.get("HEAD_YAW_FLIP",   "0") == "1",
    "pitch": os.environ.get("HEAD_PITCH_FLIP", "1") == "1",
    "roll":  os.environ.get("HEAD_ROLL_FLIP",  "0") == "1",
}
IMU_ACTIVE = False        # gated by phone's ACTIVATE button so we never move
PHONE_CONNECTED = 0       # count of live phone WS clients (0 = no phone)
PHONE_LAST_POSE_MS = 0    # monotonic ms of last IMU packet received
# Latest neck target q (degrees, after IMU_ZERO subtraction + wrap).  The
# PC ghost polls /status and drives ub_neck_yaw/roll/pitch off these so
# operator sees the ghost move whenever the phone moves — regardless of
# whether the physical motors are powered / connected.
HEAD_NECK_Q = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
                          # motors accidentally while the operator is looking

CAMERA_INDEX = int(os.environ.get("CAM_INDEX", "0"))
CAMERA_W     = int(os.environ.get("CAM_W", "640"))
CAMERA_H     = int(os.environ.get("CAM_H", "480"))
CAMERA_ROTATE = int(os.environ.get("CAM_ROTATE", "0"))    # 0/90/180/270
CAMERA_QUALITY = int(os.environ.get("CAM_Q", "70"))

TELE_HZ_DEFAULT = 50
MOTOR_SCAN_RANGE = (1, 50)


# ── Motor hub (single thread owning the serial port) ─────────────────────────
def _find_motor_port() -> str:
    """Look for a Feetech CH343 by serial number, then fall back."""
    try:
        for p in serial.tools.list_ports.comports():
            sn = (p.serial_number or "").upper()
            if HEAD_USB_SERIAL and sn == HEAD_USB_SERIAL.upper():
                return p.device
            if (p.vid, p.pid) == (0x1A86, 0x55D3) and not HEAD_USB_SERIAL:
                return p.device
    except Exception:
        pass
    return DEFAULT_PORT_HINT


@dataclass
class _Cmd:
    op: str
    args: tuple = ()
    reply_q: "queue.Queue|None" = None


class MotorHub:
    """All Feetech serial I/O runs in one thread.  Public methods just queue
    a request; the worker thread executes them in order and, if requested,
    puts a reply on the provided Queue."""

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD):
        self.port = port or _find_motor_port()
        self.baud = baud
        self.ser = None
        self.sts = None
        self.latest: dict[int, dict] = {}      # sid → {pos,spd,load,volt,temp}
        self.tele_hz = TELE_HZ_DEFAULT
        self._cmd_q: queue.Queue = queue.Queue()
        self._known_ids: set[int] = set()
        self._known_models: dict[int, int] = {}
        self._stop = threading.Event()
        self.error = ""
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()

    # ── public ops (all thread-safe) ────────────────────────────────────────
    def enqueue(self, op, args=(), *, want_reply=False):
        rq = queue.Queue(maxsize=1) if want_reply else None
        self._cmd_q.put(_Cmd(op, args, rq))
        return rq

    def sync_reply(self, op, args=(), timeout=0.5):
        rq = self.enqueue(op, args, want_reply=True)
        try:
            return rq.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── worker thread ───────────────────────────────────────────────────────
    def _loop(self):
        # Wait for USB to appear if not yet plugged in — retry every 1 s so
        # a boot-time or unplug-time race doesn't require an agent restart.
        self._ensure_open_with_retry()
        if self.sts is not None:
            self._scan(*MOTOR_SCAN_RANGE)

        last_tele = 0.0
        last_rescan = time.monotonic()
        consec_read_fail = 0
        while not self._stop.is_set():
            # Periodic rescan every 30 s catches sids that drop off + on
            if self.sts is not None and time.monotonic() - last_rescan > 30.0:
                before = set(self._known_ids)
                self._scan(*MOTOR_SCAN_RANGE)
                added = self._known_ids - before
                if added:
                    print(f"[motor] rescan added sids {sorted(added)}",
                          flush=True)
                last_rescan = time.monotonic()
            # USB may have moved: after ~3 s of blank telemetry, reopen.
            if consec_read_fail >= 30:
                print("[motor] many consecutive read failures — re-opening",
                      flush=True)
                self._close(); self.sts = None
                consec_read_fail = 0
                self._ensure_open_with_retry()
                if self.sts is not None:
                    self._scan(*MOTOR_SCAN_RANGE)
            try:
                while True:
                    c = self._cmd_q.get_nowait()
                    self._dispatch(c)
            except queue.Empty:
                pass
            period = 1.0 / max(1, self.tele_hz)
            now = time.monotonic()
            if now - last_tele >= period:
                if self.sts is not None:
                    self._last_read_had_any = False
                    self._read_all()
                    if not self._last_read_had_any and self._known_ids:
                        consec_read_fail += 1
                    else:
                        consec_read_fail = 0
                last_tele = now
            time.sleep(0.002)
        self._close()

    def _ensure_open_with_retry(self):
        """Open the serial port, re-locating it each attempt via USB SN so a
        replug that moves /dev/ttyACM0 → /dev/ttyACM1 self-heals."""
        while not self._stop.is_set():
            try:
                self.port = self.port or _find_motor_port()
                if self.port and not os.path.exists(self.port):
                    self.port = _find_motor_port()
                self._open()
                self.error = ""
                return
            except Exception as e:
                self.error = f"open failed: {e}"
                print(f"[motor] {self.error} - retry in 1 s", flush=True)
                self.port = None
                time.sleep(1.0)

    def _open(self):
        self.ser = serial.Serial(
            self.port, self.baud,
            bytesize=8, parity="N", stopbits=1, timeout=0.02)
        self.sts = SMS_STS(self.ser)
        # Second handler on the SAME serial for SCS009 (big-endian) hand
        # servos.  scservo_sdk's PortHandler opens its own file descriptor by
        # default — we bypass that and inject our already-open serial.Serial.
        self.scs = None
        if _SCS_AVAILABLE:
            ph = _SCS_PortHandler(self.port)
            ph.ser = self.ser              # reuse existing serial
            ph.is_open = True
            ph.baudrate = self.baud
            ph.tx_time_per_byte = 1000.0 / self.baud * 10.0
            self.scs = _SCScl(ph)
        print(f"[motor] open {self.port} @ {self.baud}"
              f"  scscl={'on' if self.scs else 'off'}", flush=True)

    def _close(self):
        try:
            if self.ser and self.ser.is_open: self.ser.close()
        except Exception: pass

    # ── dispatch ────────────────────────────────────────────────────────────
    def _is_scs(self, sid) -> bool:
        """Route sid to scscl handler if scan flagged it as SCS009 (model
        1029) AND scservo_sdk is available."""
        return (self.scs is not None
                and self._known_models.get(int(sid)) in SCS_MODEL_IDS)

    def _dispatch(self, c: _Cmd):
        try:
            if c.op == "write":
                sid, step, spd, acc = c.args
                if self._is_scs(sid):
                    # SCScl uses (pos, time_ms, speed).  speed capped 10-bit.
                    self.scs.WritePos(int(sid), int(step),
                                      SCS_TIME_MS, min(int(spd), 1023))
                else:
                    self.sts.WritePosEx(int(sid), int(step),
                                        int(spd), int(acc))
            elif c.op == "sync":
                cmds, spd, acc = c.args
                # Split by handler — mixed daisy chain needs two sync groups.
                sts_cmds = [(s, st) for s, st in cmds if not self._is_scs(s)]
                scs_cmds = [(s, st) for s, st in cmds if self._is_scs(s)]
                if sts_cmds:
                    ids = [int(s) for s, _ in sts_cmds]
                    pos = [int(st) for _, st in sts_cmds]
                    spds = [int(spd)] * len(sts_cmds)
                    accs = [int(acc)] * len(sts_cmds)
                    self.sts.SyncWritePosEx(ids, len(sts_cmds), pos,
                                            spds, accs)
                if scs_cmds and self.scs is not None:
                    # SCScl broadcast sync: pack all fingers into one packet
                    # so the whole hand moves in the same frame — kills the
                    # per-finger settling jitter we saw with per-sid writes.
                    gsw = self.scs.groupSyncWrite
                    gsw.clearParam()
                    scs_speed = min(int(spd), 1023)
                    for s, st in scs_cmds:
                        pos = int(st)
                        data = [
                            self.scs.scs_lobyte(pos),
                            self.scs.scs_hibyte(pos),
                            self.scs.scs_lobyte(SCS_TIME_MS),
                            self.scs.scs_hibyte(SCS_TIME_MS),
                            self.scs.scs_lobyte(scs_speed),
                            self.scs.scs_hibyte(scs_speed),
                        ]
                        gsw.addParam(int(s), data)
                    gsw.txPacket()
            elif c.op == "torque":
                sid, on = c.args
                if self._is_scs(sid):
                    self.scs.write1ByteTxRx(int(sid), SCS_TORQUE_REG,
                                            1 if on else 0)
                else:
                    self.sts.EnableTorque(int(sid), 1 if on else 0)
            elif c.op == "torque_all":
                (on,) = c.args
                for sid in list(self._known_ids):
                    if self._is_scs(sid):
                        self.scs.write1ByteTxRx(int(sid), SCS_TORQUE_REG,
                                                1 if on else 0)
                    else:
                        self.sts.EnableTorque(int(sid), 1 if on else 0)
            elif c.op == "read":
                sid, addr, size = c.args
                v = self._read_bytes(sid, addr, size)
                if c.reply_q: c.reply_q.put({"ok": v is not None, "value": v})
            elif c.op == "scan":
                lo, hi = c.args
                self._scan(lo, hi)
                if c.reply_q: c.reply_q.put({"ok": True,
                                             "ids": sorted(self._known_ids)})
            elif c.op == "tele_hz":
                (hz,) = c.args
                self.tele_hz = max(1, min(200, int(hz)))
        except Exception as e:
            if c.reply_q: c.reply_q.put({"ok": False, "error": str(e)})
            print(f"[motor] {c.op} err: {e}", flush=True)

    def _scan(self, lo, hi):
        found = set()
        models: dict[int, int] = {}
        for sid in range(lo, hi + 1):
            ok = self.sts.Ping(sid) if hasattr(self.sts, "Ping") \
                    else self._ping(sid)
            if ok is not None:
                found.add(sid)
                # Model number = registers 3-4 (little-endian for STS,
                # big-endian for SCS009 but the raw 2 bytes still tell them
                # apart).  Read as a raw 16-bit and store — the launcher
                # can decode.
                try:
                    m = self._read_bytes(sid, 3, 2)
                    if m is not None:
                        models[sid] = int(m)
                except Exception:
                    pass
        self._known_ids = found
        self._known_models = models
        print(f"[motor] scan → IDs {sorted(found)}  models={models}", flush=True)

    def _ping(self, sid):
        """Fallback ping using feetech_lib register read."""
        try:
            m = self.sts.ReadPos(sid) if hasattr(self.sts, "ReadPos") else None
            return m
        except Exception:
            return None

    def _read_bytes(self, sid, addr, size):
        # feetech_lib exposes _read_packet
        try:
            data = self.sts._read_packet(sid, addr, size)
            if data is None: return None
            v = 0
            for i, b in enumerate(data): v |= (b & 0xFF) << (8 * i)
            # sign extend if 2 bytes and bit-15 set (Feetech signed encoding)
            if size == 2 and (v & 0x8000): v = -(v & 0x7FFF)
            return v
        except Exception:
            return None

    def _read_all(self):
        # Keep previous good values across timeouts so a dropped packet
        # doesn't blank the whole row on the client.  Also track whether
        # ANY motor produced a fresh reading this cycle — the outer loop
        # uses that as the "USB moved, reconnect" trigger.
        latest = dict(self.latest)
        got_any_this_cycle = False
        for sid in list(self._known_ids):
            try:
                pos = self._read_bytes(sid, 56, 2)
                if pos is None: continue
                got_any_this_cycle = True
                spd = self._read_bytes(sid, 58, 2)
                load = self._read_bytes(sid, 60, 2) or 0
                volt = self._read_bytes(sid, 62, 1)
                temp = self._read_bytes(sid, 63, 1)
                if temp is not None and temp > 100:
                    temp = latest.get(sid, {}).get("temp", 0)
                if volt is not None and volt > 200:
                    volt = latest.get(sid, {}).get("volt", 0) * 10
                latest[sid] = {
                    "sid": sid,
                    "pos": pos, "spd": spd,
                    "load": (load & 0x03FF) * 0.1
                            * (-1 if (load & 0x0400) else 1),
                    "volt": (volt or 0) * 0.1,
                    "temp": temp or 0,
                }
            except Exception:
                continue
        self.latest = latest
        self._last_read_had_any = got_any_this_cycle


# ── Camera capture thread ────────────────────────────────────────────────────
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class Camera:
    def __init__(self, index=CAMERA_INDEX, w=CAMERA_W, h=CAMERA_H,
                 rotate=CAMERA_ROTATE, quality=CAMERA_QUALITY):
        self.index = index
        self.w, self.h = w, h
        self.rotate = rotate
        self.quality = quality
        self.frame_jpg: bytes | None = None
        self.frame_count = 0
        self.error = ""
        self._stop = threading.Event()
        self._cap = None
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()

    def stop(self):
        self._stop.set()

    def _open(self):
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            return None
        # tell OpenCV to pull JPEGs straight from the camera (no YUV→RGB in
        # the driver) — with a UVC webcam this is 10-30 ms faster per frame
        try:
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # 1-frame ring buffer → we always get the freshest frame,
        # no accumulated backlog when downstream is slow
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        return cap

    def _loop(self):
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    self.error = f"cam {self.index} open failed"
                    time.sleep(1.0); continue
                self.error = ""
                print(f"[cam] opened index {self.index} "
                      f"{self.w}x{self.h}", flush=True)
            ret, frame = self._cap.read()
            if not ret:
                self._cap.release(); self._cap = None
                self.error = "read failed"
                time.sleep(0.3); continue
            if self.rotate in _ROT:
                frame = cv2.rotate(frame, _ROT[self.rotate])
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                self.frame_jpg = buf.tobytes()
                self.frame_count += 1
        if self._cap: self._cap.release()


# ── HTTP / WebSocket server ──────────────────────────────────────────────────
# ── self-signed HTTPS cert (phone IMU sensors require HTTPS) ────────────────
CERT_PATH = os.path.join(HERE, "agent_cert.pem")
KEY_PATH  = os.path.join(HERE, "agent_key.pem")


def ensure_cert(ip: str):
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    import datetime, ipaddress
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
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()))
    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[cert] wrote self-signed cert for {ip}", flush=True)


def _q_to_step(axis: str, q_deg: float) -> int:
    """Linear 2-point interp with per-axis direction flip and clamp."""
    with NECK_MAPPING_LOCK:
        m = NECK_MAPPING[axis].copy()
    q = q_deg * m["dir"]
    ql, qh = m["q_lo"], m["q_hi"]
    tl, th = m["tick_lo"], m["tick_hi"]
    lo, hi = (ql, qh) if ql <= qh else (qh, ql)
    q = max(lo, min(hi, q))
    denom = qh - ql
    if abs(denom) < 1e-6: return int(round(tl))
    return int(round(tl + (q - ql) * (th - tl) / denom))


def _wrap_deg(v):
    """Normalise an angle to (-180, +180]."""
    return ((v + 180.0) % 360.0) - 180.0


class Agent:
    def __init__(self, motor: MotorHub, cam: Camera, port: int):
        self.motor = motor
        self.cam = cam
        self.port = port

    # ── HTML / status ───────────────────────────────────────────────────────
    async def index(self, request):
        html = """<!doctype html><meta charset=utf-8>
<title>Q-BOT head agent</title>
<style>body{background:#111;color:#dde;font-family:monospace;padding:20px}
img{max-width:640px;border:1px solid #345} a{color:#7cf}</style>
<h2>Q-BOT head Raspberry Pi agent</h2>
<p>Endpoints:</p>
<ul>
 <li><a href="/mjpeg">/mjpeg</a> — live camera stream</li>
 <li><a href="/snapshot">/snapshot</a> — one JPEG</li>
 <li><a href="/status">/status</a> — JSON health check</li>
 <li>ws://.../ws — motor control + telemetry (WebSocket)</li>
</ul>
<img src="/mjpeg">"""
        return web.Response(text=html, content_type="text/html")

    async def status(self, request):
        # Phone status: connected count + freshness of last IMU packet.
        # "phone_pose_age_ms" is how long since we last received a yaw/pitch
        # sample; > 500 ms probably means the phone is idle (browser tab in
        # background) even if the WS is still open.
        pose_age = (int(time.monotonic() * 1000) - PHONE_LAST_POSE_MS
                    if PHONE_LAST_POSE_MS else -1)
        return web.json_response({
            "motor_port": self.motor.port,
            "motor_error": self.motor.error,
            "motor_ids": sorted(self.motor._known_ids),
            "motor_models": {str(k): v for k, v in
                             sorted(self.motor._known_models.items())},
            "motor_tele_hz": self.motor.tele_hz,
            "cam_index": self.cam.index,
            "cam_error": self.cam.error,
            "cam_frames": self.cam.frame_count,
            "phone_connected": int(PHONE_CONNECTED),
            "phone_imu_active": bool(IMU_ACTIVE),
            "phone_pose_age_ms": pose_age,
            "neck_q": dict(HEAD_NECK_Q),
        })

    # ── camera ──────────────────────────────────────────────────────────────
    async def camera_rotate(self, request):
        """POST /camera/rotate?deg=<0|90|180|270>  — hot-swap the capture
        thread's rotation.  Affects EVERY downstream consumer of the MJPEG
        stream (PC client, VR phone page, hand camera_receiver, etc.)."""
        deg_raw = request.query.get("deg", "0")
        try:
            deg = int(deg_raw) % 360
        except ValueError:
            return web.json_response({"ok": False,
                                       "error": "deg must be integer"},
                                      status=400)
        if deg not in (0, 90, 180, 270):
            return web.json_response({"ok": False,
                                       "error": "deg must be 0/90/180/270"},
                                      status=400)
        self.cam.rotate = deg
        print(f"[cam] rotate → {deg}°", flush=True)
        return web.json_response({"ok": True, "rotate": deg})

    async def snapshot(self, request):
        f = self.cam.frame_jpg
        if not f:
            return web.Response(status=503, text="camera not ready")
        return web.Response(body=f, content_type="image/jpeg")

    async def mjpeg(self, request):
        boundary = "frame"
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0",
                "Pragma": "no-cache",
            })
        await resp.prepare(request)
        last_n = -1
        try:
            while True:
                if self.cam.frame_count != last_n and self.cam.frame_jpg:
                    last_n = self.cam.frame_count
                    body = (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(self.cam.frame_jpg)}\r\n\r\n"
                    ).encode() + self.cam.frame_jpg + b"\r\n"
                    await resp.write(body)
                await asyncio.sleep(1.0 / 60)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        return resp

    # ── VR page + IMU (phone) ───────────────────────────────────────────────
    async def vr(self, request):
        return web.FileResponse(os.path.join(HERE, "vr.html"))

    async def pose_ws(self, request):
        """Phone IMU stream — pose messages drive the head motors directly.
        Client sends {"yaw":..., "pitch":..., "roll":...} at ~50 Hz.
        Special ops:  {"cmd":"calibrate"}  (zero the current pose)
                      {"cmd":"active", "on":true/false}
        """
        global IMU_ACTIVE, PHONE_CONNECTED, PHONE_LAST_POSE_MS
        ws = web.WebSocketResponse(heartbeat=8.0)
        await ws.prepare(request)
        PHONE_CONNECTED += 1
        print(f"[pose] phone connect {request.remote} (n={PHONE_CONNECTED})",
              flush=True)
        latest_pose = None
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT: continue
                try: d = json.loads(msg.data)
                except Exception: continue
                if d.get("cmd") == "calibrate" and latest_pose:
                    IMU_ZERO["yaw"]   = latest_pose[0]
                    IMU_ZERO["pitch"] = latest_pose[1]
                    IMU_ZERO["roll"]  = latest_pose[2]
                    print(f"[imu] zero ← {latest_pose}", flush=True)
                    await ws.send_str(json.dumps({"ok": True,
                                                  "imu_zero": IMU_ZERO}))
                    continue
                if d.get("cmd") == "active":
                    IMU_ACTIVE = bool(d.get("on", False))
                    print(f"[imu] active={IMU_ACTIVE}", flush=True)
                    await ws.send_str(json.dumps({"ok": True,
                                                  "imu_active": IMU_ACTIVE}))
                    continue
                if d.get("cmd") == "cal_result":
                    # Wizard sent per-anchor medians:
                    #   data.forward    → IMU_ZERO
                    #   data.yaw_left / yaw_right → NECK_MAPPING["yaw"] range
                    #   data.pitch_up / pitch_down → NECK_MAPPING["pitch"] range
                    #
                    # We DO NOT take min/max of the two extremes — instead we
                    # assign directly:
                    #     q_lo  = phone reading at "head LEFT (or UP)" anchor
                    #     q_hi  = phone reading at "head RIGHT (or DOWN)" anchor
                    # This preserves the phone's sign convention: whether the
                    # phone reports left-turn as +ve or -ve, the mapping still
                    # produces the correct tick because the interp direction
                    # follows the actual q_lo/q_hi ordering.  ``dir`` is
                    # therefore forced to +1 and no longer needed for cal-time.
                    #
                    # A minimum span is enforced so that a shy operator (only
                    # ±5° tilt at the extremes) doesn't end up with a hyper-
                    # sensitive mapping where every tiny head twitch swings
                    # the robot fully.
                    data = d.get("data") or {}
                    fwd = data.get("forward")
                    if fwd:
                        IMU_ZERO["yaw"]   = float(fwd.get("y", 0))
                        IMU_ZERO["pitch"] = float(fwd.get("p", 0))
                        IMU_ZERO["roll"]  = float(fwd.get("r", 0))
                        print(f"[imu] cal zero ← {IMU_ZERO}", flush=True)
                    yl = data.get("yaw_left");  yr = data.get("yaw_right")
                    pu = data.get("pitch_up");  pd_ = data.get("pitch_down")

                    def _fixed_mapping(low_reading, high_reading,
                                       max_deg, flip):
                        """Simple mapping — regardless of how far the operator
                        actually tilted the phone at each cal anchor, the
                        resulting q range is fixed ±max_deg.  Phone sign
                        convention is read from the two samples; ``flip``
                        swaps the two anchors if the physical tick wiring is
                        inverted (empirical override per axis)."""
                        if low_reading <= high_reading:
                            q_lo, q_hi = -max_deg, +max_deg
                        else:
                            q_lo, q_hi = +max_deg, -max_deg
                        if flip:
                            q_lo, q_hi = q_hi, q_lo
                        return q_lo, q_hi

                    with NECK_MAPPING_LOCK:
                        if yl and yr:
                            yl_d = _wrap_deg(yl["y"] - IMU_ZERO["yaw"])
                            yr_d = _wrap_deg(yr["y"] - IMU_ZERO["yaw"])
                            q_lo, q_hi = _fixed_mapping(
                                yl_d, yr_d, MAX_YAW_DEG, AXIS_FLIP["yaw"])
                            NECK_MAPPING["yaw"]["q_lo"] = q_lo
                            NECK_MAPPING["yaw"]["q_hi"] = q_hi
                            NECK_MAPPING["yaw"]["dir"]  = +1
                            print(f"[imu] cal yaw: L raw={yl_d:.1f}°, "
                                  f"R raw={yr_d:.1f}° → q_lo={q_lo:.0f}, "
                                  f"q_hi={q_hi:.0f} "
                                  f"(flip={AXIS_FLIP['yaw']})", flush=True)
                        if pu and pd_:
                            pu_d = pu["p"] - IMU_ZERO["pitch"]
                            pd_d = pd_["p"] - IMU_ZERO["pitch"]
                            q_lo, q_hi = _fixed_mapping(
                                pu_d, pd_d, MAX_PITCH_DEG, AXIS_FLIP["pitch"])
                            NECK_MAPPING["pitch"]["q_lo"] = q_lo
                            NECK_MAPPING["pitch"]["q_hi"] = q_hi
                            NECK_MAPPING["pitch"]["dir"]  = +1
                            print(f"[imu] cal pitch: U raw={pu_d:.1f}°, "
                                  f"D raw={pd_d:.1f}° → q_lo={q_lo:.0f}, "
                                  f"q_hi={q_hi:.0f} "
                                  f"(flip={AXIS_FLIP['pitch']})", flush=True)
                    await ws.send_str(json.dumps({"ok": True,
                                                  "imu_zero": IMU_ZERO,
                                                  "mapping": NECK_MAPPING}))
                    continue
                # normal pose sample
                y = float(d.get("yaw", 0))
                p = float(d.get("pitch", 0))
                r = float(d.get("roll", 0))
                latest_pose = (y, p, r)
                PHONE_LAST_POSE_MS = int(time.monotonic() * 1000)
                # Update ghost neck-target regardless of IMU_ACTIVE so the
                # PC operator can preview head motion before arming motors.
                HEAD_NECK_Q["yaw"]   = _wrap_deg(y - IMU_ZERO["yaw"])
                HEAD_NECK_Q["pitch"] = p - IMU_ZERO["pitch"]
                HEAD_NECK_Q["roll"]  = r - IMU_ZERO["roll"]
                if not IMU_ACTIVE: continue
                # apply zero-offset + wrap
                yd = _wrap_deg(y - IMU_ZERO["yaw"])
                pd = p - IMU_ZERO["pitch"]
                rd = r - IMU_ZERO["roll"]
                # 2-DOF control only: yaw + pitch.  Roll axis intentionally
                # skipped — user found it confusing during calibration and
                # the two remaining DOFs already cover "look direction".  If
                # you later want roll back, add the third cmd row.
                cmds = [
                    (NECK_MAPPING["yaw"]["sid"],   _q_to_step("yaw",   yd)),
                    (NECK_MAPPING["pitch"]["sid"], _q_to_step("pitch", pd)),
                ]
                self.motor.enqueue("sync", (cmds, 0, 30))
        finally:
            PHONE_CONNECTED = max(0, PHONE_CONNECTED - 1)
            print(f"[pose] phone disconnect {request.remote} "
                  f"(n={PHONE_CONNECTED})", flush=True)
        return ws

    # ── WebSocket ───────────────────────────────────────────────────────────
    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        print(f"[ws] connect {request.remote}", flush=True)

        # spawn a task streaming telemetry
        stop = asyncio.Event()

        async def tele_task():
            loop = asyncio.get_running_loop()
            while not stop.is_set():
                data = list(self.motor.latest.values())
                # Always emit a heartbeat, even if motors list is empty, so
                # the client can distinguish "connected but no motors" from
                # "connection dead".
                try:
                    await ws.send_str(json.dumps({
                        "t": "tele",
                        "ts": time.monotonic(),
                        "motors": data,
                    }))
                except (ConnectionResetError, Exception):
                    break
                await asyncio.sleep(1.0 / max(1, self.motor.tele_hz))

        tt = asyncio.create_task(tele_task())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT: continue
                await self._on_cmd(ws, msg.data)
        finally:
            stop.set()
            tt.cancel()
            print(f"[ws] disconnect {request.remote}", flush=True)
        return ws

    async def _on_cmd(self, ws, raw):
        global IMU_ACTIVE
        try: d = json.loads(raw)
        except Exception: return
        op = d.get("op")
        if not op: return
        loop = asyncio.get_running_loop()
        try:
            if op == "write":
                self.motor.enqueue("write",
                                   (d["sid"], d["step"],
                                    d.get("speed", 0), d.get("acc", 30)))
            elif op == "sync":
                self.motor.enqueue("sync",
                                   (d["cmds"], d.get("speed", 0),
                                    d.get("acc", 30)))
            elif op == "torque":
                self.motor.enqueue("torque", (d["sid"], bool(d.get("on", False))))
            elif op == "torque_all":
                self.motor.enqueue("torque_all", (bool(d.get("on", False)),))
            elif op == "read":
                req = d.get("req_id")
                rq = self.motor.enqueue(
                    "read", (d["sid"], d["addr"], d["size"]), want_reply=True)
                r = await loop.run_in_executor(None, lambda: rq.get(timeout=1.0))
                await ws.send_str(json.dumps({"t": "reply", "req_id": req, **r}))
            elif op == "scan":
                lo, hi = d.get("range", MOTOR_SCAN_RANGE)
                req = d.get("req_id")
                rq = self.motor.enqueue("scan", (lo, hi), want_reply=True)
                r = await loop.run_in_executor(None, lambda: rq.get(timeout=8.0))
                await ws.send_str(json.dumps({"t": "reply", "req_id": req, **r}))
            elif op == "tele_hz":
                self.motor.enqueue("tele_hz", (int(d.get("hz", TELE_HZ_DEFAULT)),))
            elif op == "set_neck":
                # Live-edit the IMU→motor mapping (used by PC calibration).
                # {"op":"set_neck","axis":"yaw","tick_lo":.., "q_lo":.., ...}
                axis = d.get("axis")
                if axis in NECK_MAPPING:
                    with NECK_MAPPING_LOCK:
                        for k in ("sid", "dir", "tick_lo", "q_lo",
                                  "tick_hi", "q_hi"):
                            if k in d: NECK_MAPPING[axis][k] = d[k]
            elif op == "get_neck":
                with NECK_MAPPING_LOCK:
                    payload = json.loads(json.dumps(NECK_MAPPING))
                await ws.send_str(json.dumps({"t": "reply",
                                              "req_id": d.get("req_id"),
                                              "ok": True,
                                              "mapping": payload,
                                              "imu_zero": IMU_ZERO,
                                              "imu_active": IMU_ACTIVE}))
            elif op == "imu_active":
                IMU_ACTIVE = bool(d.get("on", False))
                print(f"[imu] active={IMU_ACTIVE}", flush=True)
                if not IMU_ACTIVE:
                    # freeze motors at last position by re-sending torque hold
                    pass
        except queue.Empty:
            await ws.send_str(json.dumps({"t": "reply", "ok": False,
                                          "req_id": d.get("req_id"),
                                          "error": "timeout"}))
        except Exception as e:
            print(f"[ws] cmd err: {e}", flush=True)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000,
                    help="HTTP port (PC clients)")
    ap.add_argument("--https-port", type=int, default=8443,
                    help="HTTPS port (phone VR + IMU — required for iOS)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cam", type=int, default=CAMERA_INDEX)
    ap.add_argument("--motor-port", default=None,
                    help="serial device (default: auto-detect CH343)")
    args = ap.parse_args()

    ip = lan_ip()
    print(f"[agent] LAN IP {ip}", flush=True)
    print(f"        HTTP  → http://{ip}:{args.port}",  flush=True)
    print(f"        HTTPS → https://{ip}:{args.https_port}/vr  (phone VR)",
          flush=True)

    motor = MotorHub(port=args.motor_port)
    motor.start()

    cam = Camera(index=args.cam)
    cam.start()

    agent = Agent(motor, cam, args.port)

    def build_app():
        app = web.Application()
        app.router.add_get("/",           agent.index)
        app.router.add_get("/status",     agent.status)
        app.router.add_post("/camera/rotate", agent.camera_rotate)
        app.router.add_get("/snapshot",   agent.snapshot)
        app.router.add_get("/mjpeg",      agent.mjpeg)
        app.router.add_get("/ws",         agent.ws)
        app.router.add_get("/vr",         agent.vr)
        app.router.add_get("/pose_ws",    agent.pose_ws)
        return app

    async def _serve():
        # Two independent runners: HTTP on args.port, HTTPS on args.https_port.
        # Same handlers, different transports so the phone gets its required
        # HTTPS + the PC gets plain HTTP without needing to trust the cert.
        runners = []
        for cfg, ssl_ctx in [(args.port, None), (args.https_port, "ssl")]:
            app = build_app()
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            runners.append(runner)
            if ssl_ctx == "ssl":
                ensure_cert(ip)
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(CERT_PATH, KEY_PATH)
                site = web.TCPSite(runner, args.host, cfg, ssl_context=ctx)
            else:
                site = web.TCPSite(runner, args.host, cfg)
            await site.start()
        try:
            await asyncio.Event().wait()   # serve forever
        finally:
            motor.stop(); cam.stop()
            for r in runners: await r.cleanup()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
