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


# ── config ───────────────────────────────────────────────────────────────────
DEFAULT_BAUD = 1_000_000
DEFAULT_PORT_HINT = "/dev/ttyACM1"        # if serial-number search fails
# Set this to your CH343 serial number to force auto-detection.  Blank = use
# the first ttyACM* found (Feetech SMS bus adapters look like QinHeng 1a86:55d3).
HEAD_USB_SERIAL = os.environ.get("HEAD_USB_SERIAL", "").strip()

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
        try:
            self._open()
        except Exception as e:
            self.error = f"open failed: {e}"
            print(f"[motor] {self.error}", flush=True)
            return

        # Try to enumerate motors once so telemetry has something to read.
        self._scan(*MOTOR_SCAN_RANGE)

        last_tele = 0.0
        while not self._stop.is_set():
            # 1) drain any queued commands (all handled in-order)
            try:
                while True:
                    c = self._cmd_q.get_nowait()
                    self._dispatch(c)
            except queue.Empty:
                pass
            # 2) periodic telemetry read
            period = 1.0 / max(1, self.tele_hz)
            now = time.monotonic()
            if now - last_tele >= period:
                self._read_all()
                last_tele = now
            # 3) small idle sleep to avoid busy-wait
            time.sleep(0.002)
        self._close()

    def _open(self):
        self.ser = serial.Serial(
            self.port, self.baud,
            bytesize=8, parity="N", stopbits=1, timeout=0.02)
        self.sts = SMS_STS(self.ser)
        print(f"[motor] open {self.port} @ {self.baud}", flush=True)

    def _close(self):
        try:
            if self.ser and self.ser.is_open: self.ser.close()
        except Exception: pass

    # ── dispatch ────────────────────────────────────────────────────────────
    def _dispatch(self, c: _Cmd):
        try:
            if c.op == "write":
                sid, step, spd, acc = c.args
                self.sts.WritePosEx(sid, int(step), int(spd), int(acc))
            elif c.op == "sync":
                cmds, spd, acc = c.args
                # feetech_lib.SMS_STS.SyncWritePosEx takes list ids + arrays
                ids = [int(s) for s, _ in cmds]
                pos = [int(step) for _, step in cmds]
                spds = [int(spd)] * len(cmds)
                accs = [int(acc)] * len(cmds)
                self.sts.SyncWritePosEx(ids, len(cmds), pos, spds, accs)
            elif c.op == "torque":
                sid, on = c.args
                self.sts.EnableTorque(sid, 1 if on else 0)
            elif c.op == "torque_all":
                (on,) = c.args
                for sid in list(self._known_ids):
                    self.sts.EnableTorque(sid, 1 if on else 0)
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
        for sid in range(lo, hi + 1):
            model = self.sts.Ping(sid) if hasattr(self.sts, "Ping") \
                    else self._ping(sid)
            if model is not None:
                found.add(sid)
        self._known_ids = found
        print(f"[motor] scan → IDs {sorted(found)}", flush=True)

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
        latest = {}
        for sid in list(self._known_ids):
            try:
                pos = self._read_bytes(sid, 56, 2)
                spd = self._read_bytes(sid, 58, 2)
                load = self._read_bytes(sid, 60, 2) or 0
                volt = self._read_bytes(sid, 62, 1)
                temp = self._read_bytes(sid, 63, 1)
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
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_FPS, 30)
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
        return web.json_response({
            "motor_port": self.motor.port,
            "motor_error": self.motor.error,
            "motor_ids": sorted(self.motor._known_ids),
            "motor_tele_hz": self.motor.tele_hz,
            "cam_index": self.cam.index,
            "cam_error": self.cam.error,
            "cam_frames": self.cam.frame_count,
        })

    # ── camera ──────────────────────────────────────────────────────────────
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
                if data:
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
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cam", type=int, default=CAMERA_INDEX)
    ap.add_argument("--motor-port", default=None,
                    help="serial device (default: auto-detect CH343)")
    args = ap.parse_args()

    ip = lan_ip()
    print(f"[agent] LAN IP {ip}  →  http://{ip}:{args.port}", flush=True)

    motor = MotorHub(port=args.motor_port)
    motor.start()

    cam = Camera(index=args.cam)
    cam.start()

    agent = Agent(motor, cam, args.port)
    app = web.Application()
    app.router.add_get("/", agent.index)
    app.router.add_get("/status", agent.status)
    app.router.add_get("/snapshot", agent.snapshot)
    app.router.add_get("/mjpeg", agent.mjpeg)
    app.router.add_get("/ws", agent.ws)

    async def on_shutdown(app):
        motor.stop(); cam.stop()
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.port, print=None,
                access_log=None)


if __name__ == "__main__":
    main()
