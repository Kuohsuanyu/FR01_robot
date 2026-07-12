#!/usr/bin/env python3
"""Exo RPi agent — polls the exoskeleton's Feetech encoders at POLL_HZ and
broadcasts each joint's MJCF-frame q over WebSocket.

HTTP endpoints:
  GET  /status                  → {motor_ids, motor_error, q: {joint: rad}, ...}
  POST /save_zero               → capture current ticks as tick_zero for every
                                  channel and write to ZERO_STORAGE.

WebSocket:
  ws://<host>:8200/pose         → server → client:
                                  {"t":"exo_pose","ts":..., "q":{joint:rad}}

Config in config.py; if `--fake` flag is used the agent skips serial IO and
generates sinusoidal fake positions — useful for validating exo_gui.py
without the real exoskeleton hooked up.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402


# ── Feetech SDK optional import ─────────────────────────────────────────────
_SDK_OK = False
try:
    sys.path.insert(0, os.path.expanduser("~/FTServo_Python"))
    from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402
    _SDK_OK = True
except Exception as e:
    print(f"[exo] scservo_sdk not available: {e} — --fake mode only", flush=True)


class ExoBus:
    """Reads present position (register 56, 2 bytes) from every channel."""
    def __init__(self, port: str, baud: int, fake: bool = False):
        self.fake = fake
        self.error = ""
        self.port = port
        self.q: dict[str, float] = {ch["joint"]: 0.0 for ch in config.CHANNELS}
        self._ticks: dict[int, int] = {ch["sid"]: ch["tick_zero"] for ch in config.CHANNELS}
        self._t0 = time.monotonic()
        if fake or not _SDK_OK:
            self._ph = None
            self._pk = None
            return
        try:
            self._ph = PortHandler(port)
            if not self._ph.openPort():
                raise RuntimeError(f"cannot open {port}")
            if not self._ph.setBaudRate(baud):
                raise RuntimeError(f"setBaudRate({baud}) failed")
            self._pk = sms_sts(self._ph)
            # Force torque OFF on every configured channel — passive mode
            # is the exoskeleton's normal operation (the operator moves
            # the joints, we only READ).  Anything left on from a prior
            # session would resist the operator, so disable defensively.
            n_off = 0
            for ch in config.CHANNELS:
                try:
                    self._pk.write1ByteTxRx(int(ch["sid"]), 40, 0)
                    n_off += 1
                except Exception:
                    pass
            print(f"[exo] torque OFF on {n_off}/{len(config.CHANNELS)} channels",
                  flush=True)
        except Exception as e:
            self.error = str(e)
            self._ph = None; self._pk = None

    def poll(self):
        if self.fake or self._pk is None:
            # Sinusoidal fakes — right and left arm sweep opposite phases so
            # you can eyeball direction in exo_gui.py easily.
            t = time.monotonic() - self._t0
            fake_q = {
                "ub_neck_yaw":              0.30 * math.sin(0.3 * t),
                "ub_neck_pitch":            0.10 * math.sin(0.5 * t),
                "ub_neck_roll":             0.0,
                "ub_right_shoulder":       -0.40 * math.sin(0.4 * t),
                "ub_right_lateral_raise":   0.50 * math.sin(0.4 * t + 0.5),
                "ub_right_arm_twist":       0.20 * math.sin(0.6 * t),
                "ub_right_elbow":  0.80 + 0.60 * math.sin(0.4 * t + 0.8),
                "ub_left_shoulder":         0.40 * math.sin(0.4 * t),
                "ub_left_lateral_raise":   -0.50 * math.sin(0.4 * t + 0.5),
                "ub_left_arm_twist":       -0.20 * math.sin(0.6 * t),
                "ub_left_elbow":   0.80 + 0.60 * math.sin(0.4 * t + 0.8),
            }
            for k in self.q:
                self.q[k] = fake_q.get(k, 0.0)
            return
        # Real: read present position (addr 56) per channel
        for ch in config.CHANNELS:
            try:
                pos, comm, _ = self._pk.read2ByteTxRx(ch["sid"], 56)
                if comm == COMM_SUCCESS:
                    self._ticks[ch["sid"]] = int(pos)
                    self.q[ch["joint"]] = config.tick_to_q(int(pos), ch)
            except Exception as e:
                self.error = f"sid {ch['sid']}: {e}"

    def save_zero(self) -> dict:
        """Freeze current ticks as the new tick_zero for every channel.
        Writes ~/exo_zero.json.  Returns the resulting mapping."""
        mapping = {str(ch["sid"]): int(self._ticks[ch["sid"]]) for ch in config.CHANNELS}
        path = Path(os.path.expanduser(config.ZERO_STORAGE))
        path.write_text(json.dumps(mapping, indent=2))
        # apply to running config
        for ch in config.CHANNELS:
            ch["tick_zero"] = int(self._ticks[ch["sid"]])
        return mapping


# ── HTTP handlers ───────────────────────────────────────────────────────────
async def _status(req):
    bus: ExoBus = req.app["bus"]
    return web.json_response({
        "motor_port": bus.port,
        "motor_error": bus.error,
        "channels": [ch["joint"] for ch in config.CHANNELS],
        "q": dict(bus.q),
    })


async def _save_zero(req):
    bus: ExoBus = req.app["bus"]
    m = bus.save_zero()
    return web.json_response({"ok": True, "zero_ticks": m})


async def _pose_ws(req):
    ws = web.WebSocketResponse(heartbeat=10.0)
    await ws.prepare(req)
    print(f"[exo] pose_ws client {req.remote}", flush=True)
    try:
        bus: ExoBus = req.app["bus"]
        interval = 1.0 / config.POLL_HZ
        while True:
            await ws.send_str(json.dumps({
                "t": "exo_pose",
                "ts": time.time(),
                "q": dict(bus.q),
            }))
            await asyncio.sleep(interval)
    except Exception as e:
        print(f"[exo] pose_ws disconnect ({e})", flush=True)
    return ws


# ── main ────────────────────────────────────────────────────────────────────
async def _bus_poll_loop(app):
    print("[exo] _bus_poll_loop entered", flush=True)
    bus: ExoBus = app["bus"]
    interval = 1.0 / config.POLL_HZ
    n = 0
    t0 = time.monotonic()
    while True:
        try:
            bus.poll()
        except Exception as e:
            print(f"[exo] poll err: {e}", flush=True)
        n += 1
        if n % 200 == 0:
            print(f"[exo] poll #{n}  dt={time.monotonic()-t0:.1f}s", flush=True)
        await asyncio.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--serial-port", default=config.SERVO_PORT)
    ap.add_argument("--baud", type=int, default=config.SERVO_BAUD)
    ap.add_argument("--fake", action="store_true",
                    help="skip real serial; generate sinusoidal fake pose")
    args = ap.parse_args()

    bus = ExoBus(args.serial_port, args.baud, fake=args.fake)
    print(f"[exo] agent starting on :{args.port}  "
          f"(fake={args.fake}, port={args.serial_port}, err={bus.error!r})",
          flush=True)

    app = web.Application()
    app["bus"] = bus
    app.router.add_get("/status", _status)
    app.router.add_post("/save_zero", _save_zero)
    app.router.add_get("/pose", _pose_ws)

    # aiohttp 3.10+ ignores the deprecated ``loop=`` kwarg — attaching the
    # poll loop via ``on_startup`` guarantees it lives on the same event
    # loop the HTTP server uses, otherwise ``_bus_poll_loop`` never runs
    # and every /status returns stale ticks.
    async def _spawn_poll(_app):
        print("[exo] _spawn_poll called", flush=True)
        _app["poll_task"] = asyncio.create_task(_bus_poll_loop(_app))
        print("[exo] poll task scheduled", flush=True)
    async def _cancel_poll(_app):
        _app["poll_task"].cancel()
    app.on_startup.append(_spawn_poll)
    app.on_cleanup.append(_cancel_poll)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
