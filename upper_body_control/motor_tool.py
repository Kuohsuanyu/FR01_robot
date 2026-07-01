#!/usr/bin/env python3
"""Feetech STS3215 motor tool — Python/Tk replacement for FD.exe.

Why: the official FD.exe doesn't run cleanly on Ubuntu 20.04 (Wine 5.0
serial + CJK fonts both broken).  This tool gives the same operations,
written against the scservo_sdk we already use elsewhere in this repo.

Features:
  * port + baud, auto-detect ttyACM* (left/right arm + head ports listed)
  * scan ID 1..50, populate motor list with model number
  * live read of selected motor: pos / speed / load% / volt / temp / current
  * live chart of position over the last ~10 s
  * write goal position with speed + acc sliders (test sweep buttons too)
  * EPROM register editor: ID, baud, mode (0 pos / 1 wheel / 3 multi-turn),
    min/max angle limits, max torque, KP/KI/KD (read + write with EPROM unlock)
"""
from __future__ import annotations

import collections
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

import serial.tools.list_ports as _lp

# ── Feetech register map ─────────────────────────────────────────────────────
# (only what we expose in the EPROM editor)
REGS = [
    # (addr, size_bytes, name_zh, description, writable_in_eprom_only)
    (5,  1, "ID",                "Servo bus ID (1-253)",                    True),
    (6,  1, "Baud Rate",         "0=1M 1=500k 2=250k 3=128k 4=115200 ...",  True),
    (9,  2, "Min Angle Limit",   "Lower bound (tick, 0..4095)",             True),
    (11, 2, "Max Angle Limit",   "Upper bound (tick, 0..4095)",             True),
    (16, 1, "Max Temp Limit",    "Shutdown threshold °C",                   True),
    (21, 1, "KP",                "Position P gain",                         True),
    (22, 1, "KD",                "Position D gain",                         True),
    (23, 1, "KI",                "Position I gain",                         True),
    (33, 1, "Mode",              "0=pos 1=wheel 3=multi-turn",              True),
    (40, 1, "Torque Enable",     "0=off 1=on",                              False),
    (48, 2, "Max Torque",        "0..1000 (0.1%)",                          False),
]
ADDR_LOCK = 55          # 0 = EPROM unlocked (writable), 1 = locked
ADDR_GOAL = 42          # SMS_STS_GOAL_POSITION_L
ADDR_GOAL_SPEED = 46
ADDR_GOAL_ACC = 41
ADDR_TORQUE_EN = 40

ADDR_POS = 56
ADDR_SPEED = 58
ADDR_LOAD = 60
ADDR_VOLT = 62
ADDR_TEMP = 63
ADDR_CURRENT = 69

DEFAULT_BAUD = 1_000_000
SCAN_RANGE = (1, 50)
CHART_SECONDS = 10
POLL_HZ = 25

C = {
    "bg": "#111820", "panel": "#172030", "card": "#1c2a3a",
    "text": "#d0e0f0", "dim": "#6080a0", "bright": "#eef4fc",
    "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
    "accent": "#7ed3ff", "btn": "#1e4870", "btn_danger": "#7a2828",
    "btn_warn": "#7a4818",
    "chart_grid": "#2c4060", "chart_pos": "#7ed3ff", "chart_goal": "#d09840",
}


# ── Bus wrapper ──────────────────────────────────────────────────────────────
class Bus:
    def __init__(self):
        self._ph: PortHandler | None = None
        self._pk: sms_sts | None = None
        self._lock = threading.Lock()
        self.port_path = ""
        self.baud = DEFAULT_BAUD

    def open(self, port, baud):
        self.close()
        ph = PortHandler(port)
        if not ph.openPort():
            return False, f"openPort('{port}') failed"
        if not ph.setBaudRate(baud):
            ph.closePort()
            return False, f"setBaudRate({baud}) failed"
        with self._lock:
            self._ph = ph
            self._pk = sms_sts(ph)
        self.port_path = port; self.baud = baud
        return True, "ok"

    def close(self):
        with self._lock:
            if self._ph is not None:
                try: self._ph.closePort()
                except Exception: pass
            self._ph = None; self._pk = None

    def is_open(self):
        return self._pk is not None

    def ping(self, sid):
        with self._lock:
            if not self._pk: return None
            model, comm, err = self._pk.ping(sid)
            return model if comm == COMM_SUCCESS else None

    def read(self, sid, addr, n):
        """Return int or None on failure."""
        with self._lock:
            if not self._pk: return None
            if n == 1:
                v, c, _ = self._pk.read1ByteTxRx(sid, addr)
            elif n == 2:
                v, c, _ = self._pk.read2ByteTxRx(sid, addr)
            elif n == 4:
                v, c, _ = self._pk.read4ByteTxRx(sid, addr)
            else:
                return None
            return int(v) if c == COMM_SUCCESS else None

    def write(self, sid, addr, n, value):
        with self._lock:
            if not self._pk: return False
            v = int(value)
            if n == 1:
                r, _ = self._pk.write1ByteTxRx(sid, addr, v)
            elif n == 2:
                r, _ = self._pk.write2ByteTxRx(sid, addr, v)
            else:
                return False
            return r == COMM_SUCCESS

    def write_goal(self, sid, pos, speed, acc):
        with self._lock:
            if not self._pk: return False
            r, _ = self._pk.WritePosEx(sid, int(pos), int(speed), int(acc))
            return r == COMM_SUCCESS

    def torque(self, sid, on):
        return self.write(sid, ADDR_TORQUE_EN, 1, 1 if on else 0)

    def unlock_eprom(self, sid):
        return self.write(sid, ADDR_LOCK, 1, 0)

    def lock_eprom(self, sid):
        return self.write(sid, ADDR_LOCK, 1, 1)


# ── helpers ──────────────────────────────────────────────────────────────────
def list_serial_ports():
    out = []
    for p in _lp.comports():
        sn = p.serial_number or ""
        desc = f"{p.device}   ({sn})" if sn else p.device
        out.append((p.device, desc))
    return sorted(out)


def signed_load(v):
    """Feetech load is 11-bit signed magnitude in lower bits."""
    sign = -1 if (v & 0x0400) else 1
    return sign * (v & 0x03FF) * 0.1


def signed_speed(v):
    return v - 65536 if v > 32767 else v


# ── App ──────────────────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.bus = Bus()
        self.root = tk.Tk()
        self.root.title("Q-BOT Feetech motor_tool (Python)")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1180x740")
        self.root.minsize(1040, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.selected_id: int | None = None
        self._chart_data = collections.deque(maxlen=CHART_SECONDS * POLL_HZ)
        self._tele_q: queue.Queue = queue.Queue()
        self._stop = threading.Event()

        self._build_ui()

        threading.Thread(target=self._poll_loop, daemon=True).start()
        self.root.after(100, self._drain_tele)
        self.root.after(80, self._redraw_chart)

    # ── UI build ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#1a237e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="Feetech motor_tool  ·  Python replacement",
                 font=("DejaVu Sans", 14, "bold"),
                 bg="#1a237e", fg="white").pack(side="left", padx=12, pady=8)
        tk.Label(hdr, text="Scan → Select ID → drive Goal Pos / read live / edit EPROM",
                 font=("DejaVu Sans Mono", 9),
                 bg="#1a237e", fg="#90caf9").pack(side="left", padx=8)

        # ── Top bar: port + baud + connect + scan
        bar = tk.Frame(self.root, bg=C["card"]); bar.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(bar, text="Port:", bg=C["card"], fg=C["text"]).pack(side="left", padx=(6, 2))
        self.port_combo = ttk.Combobox(bar, width=42)
        self.port_combo.pack(side="left")
        tk.Button(bar, text="↻", width=2, bg=C["btn"], fg="white",
                  command=self._refresh_ports).pack(side="left", padx=2)
        tk.Label(bar, text="Baud:", bg=C["card"], fg=C["text"]).pack(side="left", padx=(10, 2))
        self.baud_var = tk.IntVar(value=DEFAULT_BAUD)
        ttk.Combobox(bar, width=10, textvariable=self.baud_var,
                     values=[1_000_000, 500_000, 250_000, 128_000, 115_200, 57_600]
                     ).pack(side="left")
        self.connect_btn = tk.Button(bar, text="Connect", bg="#2e7d32",
                                     fg="white",
                                     font=("DejaVu Sans Mono", 10, "bold"),
                                     command=self._toggle_connect, width=10)
        self.connect_btn.pack(side="left", padx=8)
        self.scan_btn = tk.Button(bar, text="Scan IDs", bg=C["btn"], fg="white",
                                  command=self._scan, state="disabled")
        self.scan_btn.pack(side="left", padx=2)
        self.conn_var = tk.StringVar(value="● 未連線")
        tk.Label(bar, textvariable=self.conn_var, bg=C["card"], fg=C["err"],
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(side="right", padx=8)

        # ── Body: 3 columns
        body = tk.Frame(self.root, bg=C["bg"]); body.pack(fill="both", expand=True, padx=6, pady=4)

        # left: motor list
        left = tk.Frame(body, bg=C["panel"], width=200)
        left.pack(side="left", fill="y", padx=(0, 4))
        left.pack_propagate(False)
        tk.Label(left, text="Detected motors",
                 bg=C["panel"], fg=C["bright"],
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(fill="x", padx=4, pady=(6, 2))
        self.motor_list = tk.Listbox(left, bg=C["bg"], fg=C["text"],
                                     selectbackground=C["accent"],
                                     selectforeground=C["bg"],
                                     font=("DejaVu Sans Mono", 11),
                                     highlightthickness=0, bd=0)
        self.motor_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.motor_list.bind("<<ListboxSelect>>", self._on_select_motor)

        # middle: live state + control
        mid = tk.Frame(body, bg=C["bg"])
        mid.pack(side="left", fill="both", expand=True, padx=4)
        self._build_live(mid)
        self._build_control(mid)
        self._build_chart(mid)

        # right: EPROM
        right = tk.Frame(body, bg=C["panel"], width=340)
        right.pack(side="right", fill="y", padx=(4, 0))
        right.pack_propagate(False)
        self._build_eprom(right)

        # status
        self.status_var = tk.StringVar(value="ready")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 bg=C["bg"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=12, pady=(2, 4))

        self._refresh_ports()

    def _build_live(self, p):
        f = tk.LabelFrame(p, text=" Live Telemetry ",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        f.pack(fill="x", padx=2, pady=2)
        self.live_vars: dict[str, tk.StringVar] = {}
        rows = [
            ("Pos",     "pos",     "step"),
            ("Pos°",    "pos_deg", "°"),
            ("Speed",   "speed",   "step/s"),
            ("Load",    "load",    "%"),
            ("Voltage", "volt",    "V"),
            ("Temp",    "temp",    "°C"),
            ("Current", "current", "mA"),
        ]
        row_f = tk.Frame(f, bg=C["panel"]); row_f.pack(fill="x", padx=4, pady=4)
        for i, (label, key, unit) in enumerate(rows):
            cell = tk.Frame(row_f, bg=C["card"])
            cell.grid(row=0, column=i, padx=2, pady=2, sticky="nsew")
            tk.Label(cell, text=label, bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 8)).pack(padx=4, pady=(2, 0))
            v = tk.StringVar(value="—")
            self.live_vars[key] = v
            tk.Label(cell, textvariable=v, bg=C["card"], fg=C["accent"],
                     font=("DejaVu Sans Mono", 13, "bold"), width=6
                     ).pack(padx=4)
            tk.Label(cell, text=unit, bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 8)).pack(padx=4, pady=(0, 2))

    def _build_control(self, p):
        f = tk.LabelFrame(p, text=" Position Control ",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        f.pack(fill="x", padx=2, pady=2)
        r1 = tk.Frame(f, bg=C["panel"]); r1.pack(fill="x", padx=4, pady=3)
        tk.Label(r1, text="Goal pos:", bg=C["panel"], fg=C["text"],
                 width=10, anchor="e").pack(side="left")
        self.goal_var = tk.IntVar(value=2048)
        self.goal_slider = tk.Scale(r1, from_=0, to=4095, orient="horizontal",
                                    length=520, resolution=1,
                                    variable=self.goal_var,
                                    bg=C["panel"], fg=C["text"],
                                    highlightthickness=0,
                                    troughcolor=C["card"],
                                    command=self._on_goal_slider)
        self.goal_slider.pack(side="left", fill="x", expand=True)
        tk.Button(r1, text="GO", bg=C["btn"], fg="white", width=4,
                  command=self._send_goal).pack(side="left", padx=4)

        r2 = tk.Frame(f, bg=C["panel"]); r2.pack(fill="x", padx=4, pady=2)
        tk.Label(r2, text="Speed:", bg=C["panel"], fg=C["text"],
                 width=10, anchor="e").pack(side="left")
        self.speed_var = tk.IntVar(value=0)
        tk.Scale(r2, from_=0, to=3000, orient="horizontal", length=240,
                 variable=self.speed_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]
                 ).pack(side="left")
        tk.Label(r2, text="0=max", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 8)).pack(side="left", padx=4)
        tk.Label(r2, text="Acc:", bg=C["panel"], fg=C["text"],
                 width=6, anchor="e").pack(side="left", padx=(10, 2))
        self.acc_var = tk.IntVar(value=30)
        tk.Scale(r2, from_=1, to=150, orient="horizontal", length=160,
                 variable=self.acc_var, bg=C["panel"], fg=C["text"],
                 highlightthickness=0, troughcolor=C["card"]
                 ).pack(side="left")

        r3 = tk.Frame(f, bg=C["panel"]); r3.pack(fill="x", padx=4, pady=4)
        tk.Button(r3, text="Center (2048)", bg=C["btn"], fg="white",
                  command=lambda: self._set_goal(2048, send=True)
                  ).pack(side="left", padx=2)
        tk.Button(r3, text="-30°", bg=C["btn"], fg="white",
                  command=lambda: self._set_goal(2048 - 341, send=True)
                  ).pack(side="left", padx=2)
        tk.Button(r3, text="+30°", bg=C["btn"], fg="white",
                  command=lambda: self._set_goal(2048 + 341, send=True)
                  ).pack(side="left", padx=2)
        tk.Button(r3, text="Sweep ±30°", bg=C["btn"], fg="white",
                  command=self._sweep).pack(side="left", padx=2)
        self.torque_var = tk.BooleanVar(value=False)
        tk.Checkbutton(r3, text="Torque",
                       variable=self.torque_var,
                       bg=C["panel"], fg=C["ok"],
                       activebackground=C["panel"],
                       activeforeground=C["ok"],
                       selectcolor=C["card"],
                       font=("DejaVu Sans Mono", 10, "bold"),
                       command=self._toggle_torque
                       ).pack(side="left", padx=8)
        tk.Button(r3, text="E-STOP", bg=C["btn_danger"], fg="white",
                  command=self._estop).pack(side="right", padx=2)

    def _build_chart(self, p):
        f = tk.LabelFrame(p, text=f" Position Waveform (last {CHART_SECONDS}s) ",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        f.pack(fill="both", expand=True, padx=2, pady=2)
        self.chart = tk.Canvas(f, bg=C["bg"], highlightthickness=0)
        self.chart.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_eprom(self, p):
        f = tk.LabelFrame(p, text=" EPROM / Registers ",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        f.pack(fill="both", expand=True, padx=2, pady=2)
        # header
        hdr = tk.Frame(f, bg=C["panel"]); hdr.pack(fill="x", padx=2, pady=2)
        for col, (text, w) in enumerate([
                ("Addr", 5), ("Name", 14), ("Read", 7), ("Write", 7), ("", 5)]):
            tk.Label(hdr, text=text, width=w, anchor="w",
                     bg=C["panel"], fg=C["accent"],
                     font=("DejaVu Sans Mono", 9, "bold")
                     ).grid(row=0, column=col, padx=1)
        # rows
        self.reg_read_vars: dict[int, tk.StringVar] = {}
        self.reg_write_vars: dict[int, tk.StringVar] = {}
        for r, (addr, size, name, desc, eprom) in enumerate(REGS, start=1):
            row = tk.Frame(f, bg=C["card"]); row.pack(fill="x", padx=2, pady=1)
            tk.Label(row, text=str(addr), width=5, anchor="w",
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 9)
                     ).grid(row=0, column=0, padx=1)
            tk.Label(row, text=name, width=14, anchor="w",
                     bg=C["card"], fg=C["text"],
                     font=("DejaVu Sans Mono", 9)
                     ).grid(row=0, column=1, padx=1)
            rv = tk.StringVar(value="—"); self.reg_read_vars[addr] = rv
            tk.Label(row, textvariable=rv, width=7, anchor="e",
                     bg=C["card"], fg=C["accent"],
                     font=("DejaVu Sans Mono", 9, "bold")
                     ).grid(row=0, column=2, padx=1)
            wv = tk.StringVar(value=""); self.reg_write_vars[addr] = wv
            tk.Entry(row, textvariable=wv, width=7,
                     bg=C["bg"], fg=C["text"], insertbackground=C["text"],
                     font=("DejaVu Sans Mono", 9)
                     ).grid(row=0, column=3, padx=1)
            tk.Button(row, text="W", width=3, bg=C["btn"], fg="white",
                      font=("DejaVu Sans Mono", 9),
                      command=lambda a=addr, sz=size, name=name: self._write_reg(a, sz, name)
                      ).grid(row=0, column=4, padx=1)
            tk.Label(row, text=desc, anchor="w",
                     bg=C["card"], fg=C["dim"],
                     font=("DejaVu Sans Mono", 8)
                     ).grid(row=1, column=1, columnspan=4, padx=1, sticky="w")
        # actions
        b = tk.Frame(f, bg=C["panel"]); b.pack(fill="x", padx=2, pady=4)
        tk.Button(b, text="Read All", bg=C["btn"], fg="white",
                  command=self._read_all_regs).pack(side="left", padx=2)
        tk.Button(b, text="Unlock EPROM", bg=C["btn_warn"], fg="white",
                  command=lambda: self._lock_eprom(False)
                  ).pack(side="left", padx=2)
        tk.Button(b, text="Lock EPROM", bg=C["btn"], fg="white",
                  command=lambda: self._lock_eprom(True)
                  ).pack(side="left", padx=2)
        tk.Label(f, text="提示:寫 EPROM 前必先 Unlock,改完 Lock 回去",
                 bg=C["panel"], fg=C["warn"],
                 font=("DejaVu Sans Mono", 8)
                 ).pack(fill="x", padx=4, pady=(0, 4))

    # ── Port / connect ───────────────────────────────────────────────────────
    def _refresh_ports(self):
        items = list_serial_ports()
        labels = [d for _, d in items]
        self.port_combo["values"] = labels
        self._port_map = {d: dev for dev, d in items}
        if labels and not self.port_combo.get():
            self.port_combo.set(labels[0])

    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.connect_btn.config(text="Connect", bg="#2e7d32")
            self.conn_var.set("● 未連線")
            self.scan_btn.config(state="disabled")
            self.motor_list.delete(0, "end")
            self.selected_id = None
            self.status_var.set("disconnected")
            return
        label = self.port_combo.get()
        port = self._port_map.get(label, label)
        ok, msg = self.bus.open(port, self.baud_var.get())
        if ok:
            self.connect_btn.config(text="Disconnect", bg=C["btn_danger"])
            self.conn_var.set(f"● 已連線 {self.bus.baud}")
            self.scan_btn.config(state="normal")
            self.status_var.set(f"connected {port}")
        else:
            self.status_var.set(f"connect failed: {msg}")
            messagebox.showerror("連線失敗", msg)

    # ── Scan ──────────────────────────────────────────────────────────────────
    def _scan(self):
        if not self.bus.is_open(): return
        self.motor_list.delete(0, "end")
        self.status_var.set("scanning…")
        # run scan in background so UI doesn't freeze
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self):
        for sid in range(SCAN_RANGE[0], SCAN_RANGE[1] + 1):
            model = self.bus.ping(sid)
            if model is not None:
                text = f"ID {sid:>3}  model {model}"
                self.root.after(0, lambda t=text: self.motor_list.insert("end", t))
        self.root.after(0, lambda: self.status_var.set("scan done"))

    def _on_select_motor(self, _ev):
        sel = self.motor_list.curselection()
        if not sel: return
        text = self.motor_list.get(sel[0])
        # parse "ID  21  model 777"
        try:
            sid = int(text.split()[1])
        except (ValueError, IndexError):
            return
        self.selected_id = sid
        self._chart_data.clear()
        self.status_var.set(f"selected ID {sid}")
        # auto read all EPROM registers
        self._read_all_regs()
        # snap goal slider to current pos
        pos = self.bus.read(sid, ADDR_POS, 2)
        if pos is not None:
            self._set_goal(pos, send=False)
            # also read torque state
            te = self.bus.read(sid, ADDR_TORQUE_EN, 1)
            self.torque_var.set(bool(te))

    # ── Goal / torque ────────────────────────────────────────────────────────
    def _on_goal_slider(self, _v):
        pass  # GO button sends — avoids spamming during drag

    def _set_goal(self, val, send=False):
        val = max(0, min(4095, int(val)))
        self.goal_var.set(val)
        if send: self._send_goal()

    def _send_goal(self):
        if self.selected_id is None or not self.bus.is_open(): return
        ok = self.bus.write_goal(self.selected_id,
                                  self.goal_var.get(),
                                  self.speed_var.get(),
                                  self.acc_var.get())
        if not ok:
            self.status_var.set(f"write_goal failed (ID {self.selected_id})")

    def _sweep(self):
        if self.selected_id is None or not self.bus.is_open(): return
        cur = self.goal_var.get()
        seq = [cur, cur + 341, cur, cur - 341, cur]
        def step(i=0):
            if i >= len(seq): return
            self._set_goal(seq[i], send=True)
            self.root.after(700, lambda: step(i + 1))
        step()

    def _toggle_torque(self):
        if self.selected_id is None: return
        if not self.bus.is_open():
            self.torque_var.set(False); return
        self.bus.torque(self.selected_id, self.torque_var.get())

    def _estop(self):
        if not self.bus.is_open(): return
        # turn off torque on every visible motor
        for i in range(self.motor_list.size()):
            try:
                sid = int(self.motor_list.get(i).split()[1])
                self.bus.torque(sid, False)
            except Exception: pass
        self.torque_var.set(False)
        self.status_var.set("E-STOP — torque off on all listed motors")

    # ── EPROM ────────────────────────────────────────────────────────────────
    def _read_all_regs(self):
        if self.selected_id is None or not self.bus.is_open(): return
        for addr, size, *_ in REGS:
            v = self.bus.read(self.selected_id, addr, size)
            self.reg_read_vars[addr].set(str(v) if v is not None else "—")

    def _write_reg(self, addr, size, name):
        if self.selected_id is None or not self.bus.is_open():
            self.status_var.set("尚未選擇馬達"); return
        s = self.reg_write_vars[addr].get().strip()
        if not s:
            self.status_var.set(f"請填入 {name} 要寫的值"); return
        try: v = int(s, 0)   # support 0x prefix
        except ValueError:
            self.status_var.set(f"無效數字: {s}"); return
        ok = self.bus.write(self.selected_id, addr, size, v)
        if ok:
            self.status_var.set(f"寫 ID{self.selected_id} reg{addr}={v} OK")
            # re-read this reg
            r = self.bus.read(self.selected_id, addr, size)
            self.reg_read_vars[addr].set(str(r) if r is not None else "—")
        else:
            self.status_var.set(f"寫 reg{addr} 失敗 (可能 EPROM 已鎖,Unlock 後再試)")

    def _lock_eprom(self, lock: bool):
        if self.selected_id is None or not self.bus.is_open(): return
        ok = (self.bus.lock_eprom(self.selected_id) if lock
              else self.bus.unlock_eprom(self.selected_id))
        self.status_var.set(f"ID{self.selected_id} EPROM {'已鎖' if lock else '已解鎖'}"
                            if ok else "操作失敗")

    # ── Telemetry / chart ────────────────────────────────────────────────────
    def _poll_loop(self):
        period = 1.0 / POLL_HZ
        while not self._stop.is_set():
            sid = self.selected_id
            if self.bus.is_open() and sid is not None:
                pos, _, comm, _ = (lambda: (None, None, -1, None))()  # placeholder
                # do reads sequentially (single-lock per read inside bus)
                try:
                    pos_v = self.bus.read(sid, ADDR_POS, 2)
                    spd = self.bus.read(sid, ADDR_SPEED, 2)
                    load = self.bus.read(sid, ADDR_LOAD, 2)
                    volt = self.bus.read(sid, ADDR_VOLT, 1)
                    temp = self.bus.read(sid, ADDR_TEMP, 1)
                    cur = self.bus.read(sid, ADDR_CURRENT, 2)
                    self._tele_q.put((sid, pos_v, spd, load, volt, temp, cur,
                                      time.monotonic()))
                except Exception:
                    pass
            time.sleep(period)

    def _drain_tele(self):
        latest = None
        try:
            while True: latest = self._tele_q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            sid, pos, spd, load, volt, temp, cur, t = latest
            if sid != self.selected_id:
                self.root.after(100, self._drain_tele); return
            if pos is not None:
                self.live_vars["pos"].set(f"{pos}")
                deg = (pos - 2048) / (4096/360.0)
                self.live_vars["pos_deg"].set(f"{deg:+.1f}")
                self._chart_data.append((t, pos, self.goal_var.get()))
            if spd is not None:
                self.live_vars["speed"].set(f"{signed_speed(spd):+d}")
            if load is not None:
                self.live_vars["load"].set(f"{signed_load(load):+.1f}")
            if volt is not None:
                self.live_vars["volt"].set(f"{volt*0.1:.1f}")
            if temp is not None:
                col = (C["err"] if temp > 60 else
                       C["warn"] if temp > 50 else C["accent"])
                self.live_vars["temp"].set(f"{temp}")
            if cur is not None:
                ma = signed_speed(cur) * 6.5
                self.live_vars["current"].set(f"{ma:.0f}")
        self.root.after(100, self._drain_tele)

    def _redraw_chart(self):
        cv = self.chart
        cv.delete("all")
        w = int(cv.winfo_width()); h = int(cv.winfo_height())
        if w < 50 or h < 30:
            self.root.after(100, self._redraw_chart); return
        # axes
        for frac, label in [(0.0, "0"), (0.25, "1024"), (0.5, "2048"),
                            (0.75, "3072"), (1.0, "4095")]:
            y = h - 12 - frac * (h - 24)
            cv.create_line(40, y, w - 6, y, fill=C["chart_grid"])
            cv.create_text(34, y, text=label, anchor="e",
                           fill=C["dim"], font=("DejaVu Sans Mono", 8))
        # data
        if len(self._chart_data) > 1:
            now = time.monotonic()
            xs = []
            pos_pts = []
            goal_pts = []
            for (t, p, g) in self._chart_data:
                age = now - t
                if age > CHART_SECONDS: continue
                x = 40 + (1.0 - age / CHART_SECONDS) * (w - 46)
                y_pos = h - 12 - (p / 4095) * (h - 24)
                y_goal = h - 12 - (g / 4095) * (h - 24)
                pos_pts.extend([x, y_pos])
                goal_pts.extend([x, y_goal])
            if len(pos_pts) >= 4:
                cv.create_line(*goal_pts, fill=C["chart_goal"], width=1,
                               dash=(3, 3))
                cv.create_line(*pos_pts, fill=C["chart_pos"], width=2)
            # legend
            cv.create_text(w - 10, 10, anchor="e",
                           text="● position    ─ ─ goal",
                           fill=C["dim"], font=("DejaVu Sans Mono", 8))
        self.root.after(80, self._redraw_chart)

    # ── close ────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._stop.set()
        self.bus.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
