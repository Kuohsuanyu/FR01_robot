#!/usr/bin/env python3
"""Q-BOT arm Raspberry Pi headless agent.

Runs on the RPi that's physically wired to the arm servo bus:
  * 10 Feetech STS3215 shoulder/elbow motors  (USB serial /dev/ttyACM*)

Exposes over LAN:
  * ws://<rpi-ip>:8100/ws       — bidirectional JSON motor control + telemetry
  * http://<rpi-ip>:8100/status — JSON health check
  * http://<rpi-ip>:8100/       — tiny HTML dashboard

Runs on Python 3.11 (system) — matches head agent.  No camera / no HTTPS.

Protocol (WebSocket JSON) is identical to head_rpi_agent/agent.py:
  --- Client → Agent ---
    {"op":"sync",     "cmds":[[sid,step], ...], "speed":300, "acc":30}
    {"op":"write",    "sid":10, "step":2048, "speed":300, "acc":30}
    {"op":"torque",   "sid":10, "on":true}
    {"op":"torque_all","on":false}                    # emergency stop
    {"op":"read",     "sid":10, "addr":56, "size":2, "req_id":"x"}
    {"op":"scan",     "range":[1,50], "req_id":"x"}
    {"op":"tele_hz",  "hz":50}
  --- Agent → Client ---
    {"t":"tele","ts":<mono>,"motors":[{"sid":10,"pos":2048,"spd":0,
                                       "load":0.0,"volt":12.3,"temp":35}, ...]}
    {"t":"reply","req_id":"x","ok":true,"value":<...>}
    {"t":"log","level":"warn","msg":"…"}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass

import serial
import serial.tools.list_ports
from aiohttp import web, WSMsgType

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feetech_lib import SMS_STS  # noqa: E402


# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_BAUD = int(os.environ.get("ARM_BAUD", "1000000"))
DEFAULT_PORT_HINT = os.environ.get("ARM_PORT", "/dev/ttyACM0")
# If both head + arm are plugged into the same RPi, differentiate by USB serial.
# The default matches the arm's CH343 chip so a USB replug (which shifts
# /dev/ttyACM0 → /dev/ttyACM1) still finds the right device.
ARM_USB_SERIAL = os.environ.get("ARM_USB_SERIAL", "5B14115342")

# Arm servos live at 10-14 (right) and 20-24 (left).  Scan a small window so
# the agent finds them even if the wiring changes.
MOTOR_SCAN_RANGE = (1, 30)
TELE_HZ_DEFAULT = 10   # 10 Hz gives headroom for retries on the flaky sids


# ── Motor hub (single thread owning the serial port) ─────────────────────────
def _find_motor_port() -> str:
    """Prefer a device whose USB serial matches ARM_USB_SERIAL; else fall back
    to any Feetech CH343.  If we're sharing an RPi with the head agent and no
    serial number was pinned, this may pick the head's port — set the env var
    to be safe.
    """
    try:
        for p in serial.tools.list_ports.comports():
            sn = (p.serial_number or "").upper()
            if ARM_USB_SERIAL and sn == ARM_USB_SERIAL.upper():
                return p.device
        if not ARM_USB_SERIAL:
            for p in serial.tools.list_ports.comports():
                if (p.vid, p.pid) == (0x1A86, 0x55D3):
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
    """All Feetech serial I/O runs in one thread.  Public methods queue a
    request; the worker executes them in order and, if requested, puts a
    reply on the provided Queue."""

    def __init__(self, port: str | None = None, baud: int = DEFAULT_BAUD):
        self.port = port or _find_motor_port()
        self.baud = baud
        self.ser = None
        self.sts = None
        self.latest: dict[int, dict] = {}
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

    def _loop(self):
        # Open + scan.  If the initial open fails, keep the worker alive and
        # retry every second — that way a USB unplug at boot doesn't need a
        # manual agent restart when the operator plugs the cable in.
        self._ensure_open_with_retry()
        if self.sts is not None:
            self._scan(*MOTOR_SCAN_RANGE)

        last_tele = 0.0
        last_rescan = time.monotonic()
        consec_read_fail = 0
        while not self._stop.is_set():
            # Periodic full rescan — catches sids that dropped off the bus
            # between the initial scan and now (common on flaky daisy chains).
            if self.sts is not None and time.monotonic() - last_rescan > 30.0:
                before = set(self._known_ids)
                self._scan(*MOTOR_SCAN_RANGE)
                added = self._known_ids - before
                if added:
                    print(f"[motor] rescan added sids {sorted(added)}", flush=True)
                last_rescan = time.monotonic()
            # Detect a USB replug: the port file vanishes / bus errors on write.
            # We use consecutive telemetry read failures as the trigger.
            if consec_read_fail >= 30:            # ~3 s at 10 Hz
                print("[motor] many consecutive read failures — "
                      "re-opening port (USB may have moved)", flush=True)
                self._close()
                self.sts = None
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
                    # Trigger reconnect when NO motor produced a fresh pos read
                    # this cycle — that's the signature of a dead port (USB
                    # unplug/replug that shifted the device path).
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
                # re-scan comports every attempt to catch the new device path
                self.port = self.port or _find_motor_port()
                if self.port and not os.path.exists(self.port):
                    self.port = _find_motor_port()
                self._open()
                self.error = ""
                return
            except Exception as e:
                self.error = f"open failed: {e}"
                print(f"[motor] {self.error} — retry in 1 s", flush=True)
                self.port = None
                time.sleep(1.0)

    def _open(self):
        # 50 ms read timeout — half-duplex bus with 13 motors and downstream
        # daisy-chain latency; 20 ms was chopping tail bytes off reads from
        # the further-out sids so telemetry silently blanked.
        self.ser = serial.Serial(
            self.port, self.baud,
            bytesize=8, parity="N", stopbits=1, timeout=0.05)
        self.sts = SMS_STS(self.ser)
        print(f"[motor] open {self.port} @ {self.baud}", flush=True)

    def _close(self):
        try:
            if self.ser and self.ser.is_open: self.ser.close()
        except Exception: pass

    def _dispatch(self, c: _Cmd):
        try:
            if c.op == "write":
                sid, step, spd, acc = c.args
                self.sts.WritePosEx(sid, int(step), int(spd), int(acc))
            elif c.op == "sync":
                cmds, spd, acc = c.args
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
            elif c.op == "set_mode":
                # mode 3 = 多圈/無限位置;0 = 單圈位置。寫 EPROM(持久)。
                sid, mode = c.args
                if int(mode) == 3:
                    self.sts.SetMultiTurnMode(int(sid))
                else:
                    self.sts.SetPositionMode(int(sid))
                if c.reply_q: c.reply_q.put({"ok": True, "sid": int(sid),
                                             "mode": int(mode)})
            elif c.op == "set_limit":
                # 設 EPROM 角度限位(tick)。min/max 任一為 None 則不改該端。
                sid, mn, mx = c.args
                self.sts.unLockEprom(int(sid))
                if mn is not None: self.sts.SetMinAngleLimit(int(sid), int(mn))
                if mx is not None: self.sts.SetMaxAngleLimit(int(sid), int(mx))
                self.sts.LockEprom(int(sid))
                if c.reply_q: c.reply_q.put({"ok": True, "sid": int(sid),
                                             "min": mn, "max": mx})
            elif c.op == "set_center":
                # 中位校正:當下實體位置 → 2048(寫 40=128,EPROM 持久)
                (sid,) = c.args
                self.sts.CalibrationMiddle(int(sid))
                if c.reply_q: c.reply_q.put({"ok": True, "sid": int(sid)})
        except Exception as e:
            if c.reply_q: c.reply_q.put({"ok": False, "error": str(e)})
            print(f"[motor] {c.op} err: {e}", flush=True)

    def _scan(self, lo, hi):
        # Run the ping sweep three times and union the hits — Feetech's
        # half-duplex bus occasionally drops a response so a single pass can
        # miss a valid motor, especially on 13-servo arms.
        found = set()
        for _ in range(3):
            for sid in range(lo, hi + 1):
                model = self.sts.Ping(sid) if hasattr(self.sts, "Ping") \
                        else self._ping(sid)
                if model is not None:
                    found.add(sid)
        self._known_ids = found
        print(f"[motor] scan → IDs {sorted(found)}", flush=True)

    def _ping(self, sid):
        try:
            m = self.sts.ReadPos(sid) if hasattr(self.sts, "ReadPos") else None
            return m
        except Exception:
            return None

    def _read_bytes(self, sid, addr, size, retries=2):
        # Retry a couple of times — Feetech half-duplex bus drops the
        # occasional response on the further-out chained motors.
        for _ in range(retries + 1):
            try:
                data = self.sts._read_packet(sid, addr, size)
            except Exception:
                data = None
            if data is not None:
                v = 0
                for i, b in enumerate(data): v |= (b & 0xFF) << (8 * i)
                if size == 2 and (v & 0x8000): v = -(v & 0x7FFF)
                return v
        return None

    def _read_all(self):
        # Preserve previous good values across timeouts — a dropped packet
        # shouldn't blank the whole row in the client's telemetry display.
        latest = dict(self.latest)
        got_any_this_cycle = False
        for sid in list(self._known_ids):
            try:
                pos = self._read_bytes(sid, 56, 2)
                if pos is None:            # bus timeout — keep last frame
                    continue
                got_any_this_cycle = True
                spd  = self._read_bytes(sid, 58, 2)
                load = self._read_bytes(sid, 60, 2) or 0
                volt = self._read_bytes(sid, 62, 1)
                temp = self._read_bytes(sid, 63, 1)
                # Feetech temp is 0..100 (byte); anything above ~100 is a
                # corrupted packet — fall back to previous good reading.
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
        # Signal so the outer loop can spot a total-blackout (USB replug).
        self._last_read_had_any = got_any_this_cycle


# ── HTTP + WebSocket agent ───────────────────────────────────────────────────
class Agent:
    def __init__(self, motor: MotorHub, port: int):
        self.motor = motor
        self.port = port

    async def index(self, request):
        html = """<!doctype html><meta charset=utf-8>
<title>Q-BOT arm agent</title>
<style>body{background:#111;color:#dde;font-family:monospace;padding:20px}
a{color:#7cf}</style>
<h2>Q-BOT arm Raspberry Pi agent</h2>
<p>Endpoints:</p>
<ul>
 <li><a href="/status">/status</a> — JSON health check</li>
 <li>ws://.../ws — motor control + telemetry (WebSocket)</li>
</ul>"""
        return web.Response(text=html, content_type="text/html")

    async def status(self, request):
        return web.json_response({
            "motor_port": self.motor.port,
            "motor_error": self.motor.error,
            "motor_ids": sorted(self.motor._known_ids),
            "motor_tele_hz": self.motor.tele_hz,
        })

    async def ws(self, request):
        ws = web.WebSocketResponse(heartbeat=10.0)
        await ws.prepare(request)
        print(f"[ws] connect {request.remote}", flush=True)

        stop = asyncio.Event()

        async def tele_task():
            while not stop.is_set():
                data = list(self.motor.latest.values())
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
            elif op == "set_mode":
                req = d.get("req_id")
                rq = self.motor.enqueue("set_mode", (d["sid"], d["mode"]),
                                        want_reply=True)
                r = await loop.run_in_executor(None, lambda: rq.get(timeout=3.0))
                await ws.send_str(json.dumps({"t": "reply", "req_id": req, **r}))
            elif op == "set_limit":
                req = d.get("req_id")
                rq = self.motor.enqueue("set_limit",
                                        (d["sid"], d.get("min"), d.get("max")),
                                        want_reply=True)
                r = await loop.run_in_executor(None, lambda: rq.get(timeout=3.0))
                await ws.send_str(json.dumps({"t": "reply", "req_id": req, **r}))
            elif op == "set_center":
                req = d.get("req_id")
                rq = self.motor.enqueue("set_center", (d["sid"],), want_reply=True)
                r = await loop.run_in_executor(None, lambda: rq.get(timeout=3.0))
                await ws.send_str(json.dumps({"t": "reply", "req_id": req, **r}))
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
    ap.add_argument("--port", type=int, default=8100,
                    help="HTTP port (PC clients)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--motor-port", default=None,
                    help="serial device (default: auto-detect CH343)")
    args = ap.parse_args()

    ip = lan_ip()
    print(f"[agent] LAN IP {ip}", flush=True)
    print(f"        HTTP  → http://{ip}:{args.port}", flush=True)

    motor = MotorHub(port=args.motor_port)
    motor.start()

    agent = Agent(motor, args.port)

    async def _serve():
        app = web.Application()
        app.router.add_get("/",       agent.index)
        app.router.add_get("/status", agent.status)
        app.router.add_get("/ws",     agent.ws)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, args.host, args.port)
        await site.start()
        try:
            await asyncio.Event().wait()
        finally:
            motor.stop()
            await runner.cleanup()

    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
