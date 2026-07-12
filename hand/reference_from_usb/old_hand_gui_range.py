#!/usr/bin/env python3
# feetech_gui_range.py — Feetech Servo GUI (SCS009 + STS3032)
# 自動根據每顆馬達的實際範圍設定位置換算

import tkinter as tk
from tkinter import ttk
import serial, time, threading

# ======== 使用者設定區 ========
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46085631-if00"
BAUD = 1_000_000
SERVOS = {
    11: (515, 780),
    12: (518, 285),
    13: (430, 700),
    14: (3400, 2700),
    15: (750, 500),
    16: (2660, 1980),
}
DEFAULT_TIME_MS = 600
DEFAULT_SPEED = 200
SEND_DEBOUNCE_MS = 60
# =================================

def chk(bs): return (~sum(bs)) & 0xFF

def goal_packet(sid, pos, t, spd):
    pos = max(0, min(int(pos), 4095))
    t = max(0, min(int(t), 30000))
    spd = max(0, min(int(spd), 1023))
    data = [pos & 0xFF, (pos >> 8) & 0xFF, t & 0xFF, (t >> 8) & 0xFF, spd & 0xFF, (spd >> 8) & 0xFF]
    core = [sid, 3 + len(data), 0x03, 0x2A] + data
    return bytes([0xFF, 0xFF] + core + [chk(core)])

def torque_packet(sid, on=True):
    core = [sid, 4, 0x03, 0x18, 0x01 if on else 0x00]
    return bytes([0xFF, 0xFF] + core + [chk(core)])

def estop_packet(sid):
    core = [sid, 3 + 6, 0x03, 0x2A, 0, 0, 0, 0, 0, 0]
    return bytes([0xFF, 0xFF] + core + [chk(core)])

class SerialWorker(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.queue = []
        self.lock = threading.Lock()
        self.running = True
        try:
            self.ser = serial.Serial(port, baud, timeout=0.02)
            print(f"✅ Serial connected: {port} @ {baud}")
        except Exception as e:
            print(f"❌ Serial open failed: {e}")
            self.ser = None

    def run(self):
        while self.running and self.ser:
            pkt = None
            with self.lock:
                if self.queue:
                    pkt = self.queue.pop(0)
            if pkt:
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(pkt)
                    time.sleep(0.002)
                except Exception as e:
                    print(f"⚠️ Serial send failed: {e}")
            else:
                time.sleep(0.01)
        if self.ser:
            self.ser.close()

    def send(self, pkt):
        with self.lock:
            self.queue.append(pkt)

    def stop(self):
        self.running = False

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Feetech Range Controller")
        self.geometry("700x480")

        self.worker = SerialWorker(PORT, BAUD)
        self.worker.start()

        self.time_ms = tk.IntVar(value=DEFAULT_TIME_MS)
        self.speed = tk.IntVar(value=DEFAULT_SPEED)
        self.torque = tk.BooleanVar(value=True)
        self.servo_vars = {sid: tk.DoubleVar(value=0.5) for sid in SERVOS}
        self.after_ids = {}

        self._build_ui()
        self.after(300, self._apply_torque_all)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        ttk.Label(self, text=f"Port: {PORT} @ {BAUD}", font=("Arial", 10)).pack(pady=5)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill="x", pady=5)
        ttk.Checkbutton(ctrl, text="Torque Enable", variable=self.torque,
                        command=self._apply_torque_all).pack(side="left", padx=10)
        ttk.Button(ctrl, text="緊急停止(ALL)", command=self._estop_all).pack(side="left")

        ts = ttk.Frame(self)
        ts.pack(fill="x", pady=5)
        ttk.Label(ts, text="Time(ms)").pack(side="left")
        ttk.Scale(ts, from_=0, to=30000, variable=self.time_ms, orient="horizontal").pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(ts, textvariable=self.time_ms, width=5).pack(side="left")
        ttk.Label(ts, text=" Speed").pack(side="left", padx=(10,0))
        ttk.Scale(ts, from_=0, to=1023, variable=self.speed, orient="horizontal").pack(side="left", fill="x", expand=True, padx=5)
        ttk.Label(ts, textvariable=self.speed, width=5).pack(side="left")

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, pady=5)
        for sid, (low, high) in SERVOS.items():
            lf = ttk.Labelframe(frm, text=f"Servo ID {sid} ({low}~{high})")
            lf.pack(fill="x", padx=10, pady=4)
            s = ttk.Scale(lf, from_=0.0, to=1.0, variable=self.servo_vars[sid], orient="horizontal",
                          command=lambda v, sid=sid: self._on_slider(sid, float(v)))
            s.pack(fill="x", expand=True, side="left", padx=5)
            ttk.Label(lf, textvariable=self.servo_vars[sid], width=6).pack(side="left")

    def _on_slider(self, sid, val):
        if sid in self.after_ids:
            self.after_cancel(self.after_ids[sid])
        self.after_ids[sid] = self.after(SEND_DEBOUNCE_MS, lambda: self._send_goal(sid, val))

    def _apply_torque_all(self):
        for sid in SERVOS:
            self.worker.send(torque_packet(sid, self.torque.get()))

    def _send_goal(self, sid, slider_val):
        lo, hi = SERVOS[sid]
        if lo < hi:
            pos = int(lo + (hi - lo) * slider_val)
        else:
            pos = int(lo - (lo - hi) * slider_val)
        pkt = goal_packet(sid, pos, self.time_ms.get(), self.speed.get())
        self.worker.send(pkt)
        print(f"→ ID {sid}: pos={pos}")

    def _estop_all(self):
        for sid in SERVOS:
            self.worker.send(estop_packet(sid))
        print("⛔ Emergency stop sent to all servos")

    def _on_close(self):
        self.worker.stop()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
