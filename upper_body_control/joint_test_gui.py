#!/usr/bin/env python3
"""Q-BOT upper-body joint test GUI (Linux).

Slider per joint + live telemetry (position / speed / load / voltage / temp /
current). Hard-coded for the Q-BOT mapping: left arm ID 21-24, right arm
ID 11-14. Talks to Feetech STS3215 via /dev/ttyACM0 @ 1 Mbaud using the
scservo_sdk located at /home/andykuo/FTServo_Python.

SAFETY:
  * Slider range defaults to a single-turn-safe band — gear=4 joints clamp to
    ±30° (joint side ⇒ ±120° motor side), elbow (gear=1) to 0..+90°.
  * Speed/acc defaults are conservative (200 step/s, 20 unit/s²).
  * "EMERGENCY STOP" disables torque on every connected servo.
  * Telemetry runs in a background thread; UI never blocks on serial I/O.
"""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass

# Local SDK (existing checkout, not pip-installed)
sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

# ── Feetech register map (only what we need) ──────────────────────────────────
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_LOAD_L = 60
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63
ADDR_PRESENT_CURRENT_L = 69

STEPS_PER_RAD = 2048.0 / math.pi  # 651.8986…
CENTER = 2048
CURRENT_UNIT_MA = 6.5  # Feetech datasheet: 1 LSB ≈ 6.5 mA

# ── Joint table — (id, label, gear, slider_lo_deg, slider_hi_deg) ────────────
# Slider range is joint-side; motor-side = joint * gear. Defaults stay inside a
# single-turn (±180° motor) envelope so a fresh servo (position mode) is safe.
LEFT_ARM = [
    (20, "L Shoulder Pitch", 4, -30.0, 30.0),   # 前後旋轉 (forward/back swing)
    (21, "L Shoulder Yaw",   4, -30.0, 30.0),
    (22, "L Lateral Raise",  4, -30.0, 30.0),
    (23, "L Arm Twist",      4, -30.0, 30.0),
    (24, "L Elbow",          1,   0.0, 90.0),
]
RIGHT_ARM = [
    (10, "R Shoulder Pitch", 4, -30.0, 30.0),   # 前後旋轉 (forward/back swing)
    (11, "R Shoulder Yaw",   4, -30.0, 30.0),
    (12, "R Lateral Raise",  4, -30.0, 30.0),
    (13, "R Arm Twist",      4, -30.0, 30.0),
    (14, "R Elbow",          1,   0.0, 90.0),
]

# Defaults
DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 1_000_000
SPEED_DEFAULT = 200    # step/s
ACC_DEFAULT = 20       # acc unit
SEND_THROTTLE_S = 0.05  # 20 Hz max
POLL_HZ = 5

# ── Colours ──────────────────────────────────────────────────────────────────
C = {
    "bg": "#1e2433", "panel": "#252d3d", "header": "#1a237e",
    "text": "#eceff1", "dim": "#78909c", "sub": "#546e7a",
    "ok": "#00e676", "err": "#ff1744", "warn": "#ffab00",
    "accent": "#00bcd4", "btn": "#1565c0", "btn_red": "#c62828",
    "btn_green": "#2e7d32",
}


@dataclass
class JointSpec:
    sid: int
    label: str
    gear: int
    lo_deg: float
    hi_deg: float


@dataclass
class Reading:
    ok: bool = False
    pos_step: int = 0
    speed: int = 0          # raw step/s (signed)
    load_pct: float = 0.0   # signed, -100..+100
    volt_v: float = 0.0
    temp_c: int = 0
    current_ma: float = 0.0
    err: str = ""


# ── Serial helper (threadsafe via single lock) ───────────────────────────────
class Bus:
    def __init__(self):
        self._port: PortHandler | None = None
        self._pk: sms_sts | None = None
        self._lock = threading.Lock()
        self.port_path = DEFAULT_PORT
        self.baud = DEFAULT_BAUD

    def open(self, port: str, baud: int) -> tuple[bool, str]:
        self.close()
        self._port = PortHandler(port)
        if not self._port.openPort():
            return False, f"openPort('{port}') failed"
        if not self._port.setBaudRate(baud):
            self._port.closePort()
            return False, f"setBaudRate({baud}) failed"
        self._pk = sms_sts(self._port)
        self.port_path = port
        self.baud = baud
        return True, "open"

    def close(self):
        with self._lock:
            if self._port is not None:
                try: self._port.closePort()
                except Exception: pass
            self._port = None
            self._pk = None

    def is_open(self) -> bool:
        return self._pk is not None

    # ---- writes ------------------------------------------------------------
    def write_pos(self, sid: int, step: int, speed: int, acc: int) -> bool:
        with self._lock:
            if not self._pk: return False
            r, _ = self._pk.WritePosEx(sid, int(step), int(speed), int(acc))
            return r == COMM_SUCCESS

    def torque(self, sid: int, on: bool) -> bool:
        with self._lock:
            if not self._pk: return False
            r, _ = self._pk.write1ByteTxRx(sid, ADDR_TORQUE_ENABLE, 1 if on else 0)
            return r == COMM_SUCCESS

    # ---- reads -------------------------------------------------------------
    def read_all(self, sid: int) -> Reading:
        r = Reading()
        with self._lock:
            if not self._pk: r.err = "closed"; return r
            pk = self._pk
            pos, spd, comm, _ = pk.ReadPosSpeed(sid)
            if comm != COMM_SUCCESS:
                r.err = "no-resp"; return r
            r.pos_step = int(pos)
            r.speed = int(spd)
            load, comm, _ = pk.read2ByteTxRx(sid, ADDR_PRESENT_LOAD_L)
            if comm == COMM_SUCCESS:
                # bit 10 = sign; lower 10 bits = magnitude in 0.1% units
                sign = -1 if (load & 0x0400) else 1
                mag = load & 0x03FF
                r.load_pct = sign * mag * 0.1
            volt, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_VOLTAGE)
            if comm == COMM_SUCCESS:
                r.volt_v = volt * 0.1
            temp, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_TEMP)
            if comm == COMM_SUCCESS:
                r.temp_c = int(temp)
            cur, comm, _ = pk.read2ByteTxRx(sid, ADDR_PRESENT_CURRENT_L)
            if comm == COMM_SUCCESS:
                signed = cur - 65536 if cur > 32767 else cur
                r.current_ma = signed * CURRENT_UNIT_MA
            r.ok = True
            return r


# ── GUI ──────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.bus = Bus()
        self.joints: list[JointSpec] = [JointSpec(*j) for j in (LEFT_ARM + RIGHT_ARM)]
        self.rows: dict[int, "JointRow"] = {}
        self._last_send: dict[int, float] = {}
        self._stop_event = threading.Event()
        self._tele_queue: queue.Queue[tuple[int, Reading]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Q-BOT 手臂關節測試介面")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1180, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build()
        # Telemetry thread + UI pump
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.root.after(100, self._drain_telemetry)

    # ---------------------------------------------------------------- build
    def _build(self):
        # Header
        hdr = tk.Frame(self.root, bg=C["header"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="Q-BOT  手臂關節測試介面",
                 font=("DejaVu Sans", 16, "bold"),
                 bg=C["header"], fg="white").pack(side="left", padx=14, pady=8)
        tk.Label(hdr,
                 text="左臂 ID 21-24 | 右臂 ID 11-14 | gear: 肩×4, 肘×1",
                 font=("DejaVu Sans Mono", 9),
                 bg=C["header"], fg="#90caf9").pack(side="left", padx=8)

        # Top control bar
        bar = tk.Frame(self.root, bg=C["panel"])
        bar.pack(fill="x", padx=8, pady=(8, 0))

        tk.Label(bar, text="Port:", bg=C["panel"], fg=C["text"]).pack(side="left", padx=(8, 2))
        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        tk.Entry(bar, textvariable=self.port_var, width=16,
                 bg=C["bg"], fg=C["text"], insertbackground=C["text"]).pack(side="left")

        tk.Label(bar, text="Baud:", bg=C["panel"], fg=C["text"]).pack(side="left", padx=(10, 2))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD))
        tk.Entry(bar, textvariable=self.baud_var, width=10,
                 bg=C["bg"], fg=C["text"], insertbackground=C["text"]).pack(side="left")

        self.connect_btn = tk.Button(bar, text="Connect",
                                     bg=C["btn_green"], fg="white",
                                     activebackground=C["ok"],
                                     command=self._toggle_connect, width=10)
        self.connect_btn.pack(side="left", padx=8)

        self.estop_btn = tk.Button(bar, text="EMERGENCY STOP (扭力關閉)",
                                   bg=C["btn_red"], fg="white",
                                   activebackground=C["err"],
                                   command=self._emergency_stop, width=24)
        self.estop_btn.pack(side="right", padx=8, pady=6)

        tk.Button(bar, text="HOME(回中心 step 2048)",
                  bg=C["btn"], fg="white", command=self._home_all
                  ).pack(side="right", padx=2)
        tk.Button(bar, text="扭力 全開",
                  bg=C["btn"], fg="white", command=lambda: self._torque_all(True)
                  ).pack(side="right", padx=2)

        # Status line
        self.status_var = tk.StringVar(value="未連線")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=C["bg"], fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=12, pady=(2, 4))

        # Speed / Acc sliders
        sa = tk.Frame(self.root, bg=C["bg"])
        sa.pack(fill="x", padx=8)
        tk.Label(sa, text="Speed (step/s):", bg=C["bg"], fg=C["text"]
                 ).pack(side="left", padx=(4, 4))
        self.speed_var = tk.IntVar(value=SPEED_DEFAULT)
        tk.Scale(sa, from_=50, to=2000, orient="horizontal", length=200,
                 variable=self.speed_var, bg=C["bg"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["panel"]
                 ).pack(side="left")
        tk.Label(sa, text="   Acc:", bg=C["bg"], fg=C["text"]).pack(side="left", padx=(12, 4))
        self.acc_var = tk.IntVar(value=ACC_DEFAULT)
        tk.Scale(sa, from_=1, to=150, orient="horizontal", length=160,
                 variable=self.acc_var, bg=C["bg"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["panel"]
                 ).pack(side="left")

        # Joint rows — column headers then one row per joint
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=8, pady=6)

        headers = ["ID", "Joint", "Gear", "Cmd (deg)",
                   "Slider",
                   "Pos (deg)", "Step", "Speed", "Load %", "Volt V", "Temp °C", "I mA",
                   "Torque"]
        widths  = [4, 16, 5, 9, 26, 9, 6, 6, 7, 6, 7, 7, 7]
        for c, (h, w) in enumerate(zip(headers, widths)):
            tk.Label(body, text=h, width=w, anchor="w",
                     bg=C["panel"], fg=C["accent"],
                     font=("DejaVu Sans Mono", 9, "bold"),
                     padx=4, pady=4
                     ).grid(row=0, column=c, sticky="we", padx=1, pady=1)

        for i, j in enumerate(self.joints, start=1):
            row = JointRow(self, body, i, j, widths)
            self.rows[j.sid] = row

    # ---------------------------------------------------------------- bus ops
    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.connect_btn.config(text="Connect", bg=C["btn_green"])
            self.status_var.set("已斷線")
            return
        port = self.port_var.get().strip()
        try: baud = int(self.baud_var.get())
        except ValueError: baud = DEFAULT_BAUD
        ok, msg = self.bus.open(port, baud)
        if ok:
            self.connect_btn.config(text="Disconnect", bg=C["btn_red"])
            self.status_var.set(f"已連線 {port} @ {baud}")
        else:
            self.status_var.set(f"連線失敗: {msg}")

    def _emergency_stop(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線 — EMERGENCY STOP 無作用")
            return
        for j in self.joints:
            self.bus.torque(j.sid, False)
            row = self.rows[j.sid]
            row.torque_var.set(False)
        self.status_var.set("EMERGENCY STOP — 所有扭力已關閉")

    def _torque_all(self, on: bool):
        if not self.bus.is_open():
            self.status_var.set("尚未連線")
            return
        for j in self.joints:
            self.bus.torque(j.sid, on)
            self.rows[j.sid].torque_var.set(on)
        self.status_var.set(f"扭力 {'開啟' if on else '關閉'}")

    def _home_all(self):
        if not self.bus.is_open():
            self.status_var.set("尚未連線")
            return
        speed = int(self.speed_var.get())
        acc = int(self.acc_var.get())
        for j in self.joints:
            self.bus.write_pos(j.sid, CENTER, speed, acc)
            self.rows[j.sid].cmd_var.set(0.0)
            self.rows[j.sid].slider.set(0.0)
        self.status_var.set("HOME — 所有馬達回 step 2048 (joint 0°)")

    def send_pos(self, j: JointSpec, deg: float):
        if not self.bus.is_open(): return
        now = time.monotonic()
        if now - self._last_send.get(j.sid, 0.0) < SEND_THROTTLE_S:
            return
        self._last_send[j.sid] = now
        rad = math.radians(deg)
        step = int(round(rad * STEPS_PER_RAD * j.gear)) + CENTER
        step = max(0, min(4095, step))
        speed = int(self.speed_var.get())
        acc = int(self.acc_var.get())
        self.bus.write_pos(j.sid, step, speed, acc)

    # ---------------------------------------------------------------- poll
    def _poll_loop(self):
        period = 1.0 / POLL_HZ
        while not self._stop_event.is_set():
            if self.bus.is_open():
                for j in self.joints:
                    if self._stop_event.is_set(): break
                    r = self.bus.read_all(j.sid)
                    self._tele_queue.put((j.sid, r))
            time.sleep(period)

    def _drain_telemetry(self):
        try:
            while True:
                sid, r = self._tele_queue.get_nowait()
                row = self.rows.get(sid)
                if row: row.update_reading(r)
        except queue.Empty:
            pass
        self.root.after(80, self._drain_telemetry)

    # ---------------------------------------------------------------- exit
    def _on_close(self):
        self._stop_event.set()
        try:
            if self.bus.is_open():
                for j in self.joints:
                    self.bus.torque(j.sid, False)
        finally:
            self.bus.close()
            self.root.destroy()

    def run(self):
        self.root.mainloop()


class JointRow:
    def __init__(self, app: App, parent: tk.Widget, row: int,
                 j: JointSpec, widths: list[int]):
        self.app = app
        self.j = j
        self._row = row

        def lab(col, text, fg=C["text"], width=None, anchor="w"):
            w = widths[col] if width is None else width
            l = tk.Label(parent, text=text, width=w, anchor=anchor,
                         bg=C["bg"], fg=fg,
                         font=("DejaVu Sans Mono", 10))
            l.grid(row=row, column=col, sticky="we", padx=1, pady=1)
            return l

        lab(0, str(j.sid), fg=C["accent"])
        lab(1, j.label)
        lab(2, f"×{j.gear}")

        self.cmd_var = tk.DoubleVar(value=0.0)
        tk.Label(parent, textvariable=self.cmd_var, width=widths[3],
                 anchor="e", bg=C["bg"], fg=C["warn"],
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).grid(row=row, column=3, sticky="we", padx=1, pady=1)

        self.slider = tk.Scale(parent, from_=j.lo_deg, to=j.hi_deg,
                               resolution=0.5, orient="horizontal",
                               length=240, bg=C["bg"], fg=C["text"],
                               highlightthickness=0, troughcolor=C["panel"],
                               command=self._on_slide,
                               showvalue=False)
        self.slider.grid(row=row, column=4, sticky="we", padx=1, pady=1)

        self.pos_lab = lab(5, "—", fg=C["dim"], anchor="e")
        self.step_lab = lab(6, "—", fg=C["dim"], anchor="e")
        self.speed_lab = lab(7, "—", fg=C["dim"], anchor="e")
        self.load_lab = lab(8, "—", fg=C["dim"], anchor="e")
        self.volt_lab = lab(9, "—", fg=C["dim"], anchor="e")
        self.temp_lab = lab(10, "—", fg=C["dim"], anchor="e")
        self.cur_lab = lab(11, "—", fg=C["dim"], anchor="e")

        self.torque_var = tk.BooleanVar(value=False)
        cb = tk.Checkbutton(parent, text="ON", variable=self.torque_var,
                            bg=C["bg"], fg=C["ok"], activebackground=C["bg"],
                            activeforeground=C["ok"], selectcolor=C["panel"],
                            command=self._on_torque_toggle,
                            font=("DejaVu Sans Mono", 9, "bold"))
        cb.grid(row=row, column=12, sticky="w", padx=1, pady=1)

    def _on_slide(self, val):
        try: deg = float(val)
        except ValueError: return
        self.cmd_var.set(round(deg, 1))
        self.app.send_pos(self.j, deg)

    def _on_torque_toggle(self):
        on = bool(self.torque_var.get())
        if self.app.bus.is_open():
            self.app.bus.torque(self.j.sid, on)

    def update_reading(self, r: Reading):
        if not r.ok:
            for w in (self.pos_lab, self.step_lab, self.speed_lab,
                      self.load_lab, self.volt_lab, self.temp_lab, self.cur_lab):
                w.config(text="—", fg=C["sub"])
            return
        deg = (r.pos_step - CENTER) / STEPS_PER_RAD * 180.0 / math.pi / self.j.gear
        self.pos_lab.config(text=f"{deg:+7.2f}", fg=C["text"])
        self.step_lab.config(text=f"{r.pos_step:5d}", fg=C["text"])
        self.speed_lab.config(text=f"{r.speed:+5d}", fg=C["text"])
        load_color = C["err"] if abs(r.load_pct) > 70 else (
            C["warn"] if abs(r.load_pct) > 40 else C["text"])
        self.load_lab.config(text=f"{r.load_pct:+5.1f}", fg=load_color)
        volt_color = (C["err"] if r.volt_v < 6.0 or r.volt_v > 8.5
                      else C["text"])
        self.volt_lab.config(text=f"{r.volt_v:5.2f}", fg=volt_color)
        temp_color = (C["err"] if r.temp_c > 65
                      else C["warn"] if r.temp_c > 55 else C["text"])
        self.temp_lab.config(text=f"{r.temp_c:3d}", fg=temp_color)
        self.cur_lab.config(text=f"{r.current_ma:6.0f}", fg=C["text"])


if __name__ == "__main__":
    App().run()
