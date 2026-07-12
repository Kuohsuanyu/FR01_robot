#!/usr/bin/env python3
"""Hand slider GUI - per-servo slider + finger label picker.

Purpose:
  * Auto-scan the bus, dispatch SMS_STS or SCSCL handler per model.
  * One row per motor: slider = EPROM min..max, live present-position readout.
  * Dropdown per row to assign a finger name - dragged one by one while
    watching which real finger moves.
  * Torque defaults OFF; only when "Torque Enable ALL" is checked will
    position commands actually be written.
  * On save: dumps finger_map_<hand>.json in a shape config.py can consume.

Run:
    python3 hand_slider_gui.py --hand left --min-id 40

Safety:
  - Torque OFF at start; sliders snap to current present.
  - Enable torque explicitly, then commands ride at SPEED=150 (slow).
  - Top-right STOP releases all torque in one click.
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/FTServo_Python"))
from scservo_sdk import PortHandler, sms_sts, scscl, COMM_SUCCESS  # noqa: E402

# ---- Feetech register addresses ----
ADDR_TORQUE_ENABLE = 40
ADDR_MIN_ANGLE_L = 9
ADDR_MAX_ANGLE_L = 11
ADDR_PRESENT_POSITION_L = 56

SCSCL_MODELS = {1029}          # model 1029 = SCS009 (big-endian, 10-bit)
DEFAULT_SPEED = 150            # position servo speed (Feetech units)
DEFAULT_TIME_MS = 500          # SCS009 uses time-based control
DEFAULT_ACC = 30               # STS3032 uses acceleration
POLL_MS = 350                  # present-position polling interval
WRITE_THROTTLE_MS = 40         # min gap between successive writes per sid


def is_scscl(model):
    return model in SCSCL_MODELS


# ---- Bus wrapper ----
class HandBus:
    def __init__(self, port, baud):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            raise RuntimeError(f"cannot open port {port}")
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(f"setBaudRate({baud}) failed")
        self.pk_sts = sms_sts(self.ph)
        self.pk_scs = scscl(self.ph)

    def handler(self, model):
        return self.pk_scs if is_scscl(model) else self.pk_sts

    def scan(self, max_id=50, min_id=1):
        found = []
        for sid in range(min_id, max_id + 1):
            model, comm, _ = self.pk_sts.ping(sid)
            if comm == COMM_SUCCESS:
                found.append((sid, int(model)))
        return found

    def read_limits_and_pos(self, sid, model):
        pk = self.handler(model)
        mn, c1, _ = pk.read2ByteTxRx(sid, ADDR_MIN_ANGLE_L)
        mx, c2, _ = pk.read2ByteTxRx(sid, ADDR_MAX_ANGLE_L)
        ps, c3, _ = pk.read2ByteTxRx(sid, ADDR_PRESENT_POSITION_L)
        if COMM_SUCCESS not in (c1, c2, c3) and any(
                c != COMM_SUCCESS for c in (c1, c2, c3)):
            return None
        return dict(mn=int(mn), mx=int(mx), pos=int(ps))

    def read_pos(self, sid, model):
        pk = self.handler(model)
        ps, c, _ = pk.read2ByteTxRx(sid, ADDR_PRESENT_POSITION_L)
        return int(ps) if c == COMM_SUCCESS else None

    def set_torque(self, sid, model, on):
        pk = self.handler(model)
        return pk.write1ByteTxRx(sid, ADDR_TORQUE_ENABLE, 1 if on else 0)

    def write_pos(self, sid, model, pos):
        pk = self.handler(model)
        if is_scscl(model):
            pk.WritePos(sid, int(pos), DEFAULT_TIME_MS, DEFAULT_SPEED)
        else:
            pk.WritePosEx(sid, int(pos), DEFAULT_SPEED, DEFAULT_ACC)

    def close(self):
        self.ph.closePort()


# Dropdown values (English keys — same as config.py FINGERS dict keys).
FINGER_CHOICES = ["",
                  "pinky",
                  "ring",
                  "middle",
                  "index",
                  "thumb",
                  "thumb_rotate"]


# ---- GUI ----
class HandGUI(tk.Tk):
    def __init__(self, bus, motors, out_path: str, hand_label: str):
        super().__init__()
        self.title(f"Hand slider [{hand_label}] - per-servo test & finger map")
        self.geometry("820x560")
        self.bus = bus
        self.motors = motors      # list of (sid, model)
        self.out_path = out_path
        self.hand_label = hand_label
        self.limits = {}          # sid -> {mn, mx, pos}
        self.rows = {}            # sid -> dict of widgets
        self._last_write_ms: dict[int, float] = {}   # throttle per sid

        self._read_all_limits()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(POLL_MS, self._poll_positions)

    def _read_all_limits(self):
        for sid, model in self.motors:
            r = self.bus.read_limits_and_pos(sid, model)
            if r is None:
                messagebox.showerror(
                    "read failed", f"ID {sid} EPROM read failed")
                self.destroy(); sys.exit(1)
            # SCS009 with min>max means the servo is mounted reversed;
            # swap so slider goes low-to-high visually.
            if r["mn"] > r["mx"]:
                r["mn"], r["mx"] = r["mx"], r["mn"]
                r["swapped"] = True
            else:
                r["swapped"] = False
            self.limits[sid] = r

    def _build_ui(self):
        # Top control bar
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        self.torque_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Torque Enable ALL",
                        variable=self.torque_all,
                        command=self._apply_torque_all).pack(
                            side="left", padx=6)
        ttk.Button(top, text="STOP (release torque)",
                   command=self._stop_all).pack(side="left", padx=6)
        ttk.Button(top,
                   text=f"Save finger map -> {Path(self.out_path).name}",
                   command=self._save_map).pack(side="right", padx=6)

        # Per-servo rows
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)

        header = ttk.Frame(body)
        header.pack(fill="x")
        ttk.Label(header, text="ID", width=5).pack(side="left")
        ttk.Label(header, text="Model", width=10).pack(side="left")
        ttk.Label(header, text="Range", width=14).pack(side="left")
        ttk.Label(header, text="Slider (drag to test)",
                  width=32).pack(side="left")
        ttk.Label(header, text="target", width=8).pack(side="left")
        ttk.Label(header, text="present", width=8).pack(side="left")
        ttk.Label(header, text="Finger?", width=16).pack(side="left")
        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=4)

        for sid, model in self.motors:
            self._build_row(body, sid, model)

    def _build_row(self, parent, sid, model):
        L = self.limits[sid]
        family = "SCS009" if is_scscl(model) else "STS3032"
        row = ttk.Frame(parent, padding=(0, 3))
        row.pack(fill="x")

        ttk.Label(row, text=str(sid), width=5).pack(side="left")
        ttk.Label(row, text=family, width=10).pack(side="left")
        rng = f"{L['mn']}-{L['mx']}"
        if L["swapped"]:
            rng += " (rev)"
        ttk.Label(row, text=rng, width=14).pack(side="left")

        var = tk.IntVar(value=L["pos"])
        scale = ttk.Scale(row, from_=L["mn"], to=L["mx"], variable=var,
                          orient="horizontal", length=280,
                          command=lambda v, s=sid: self._on_slider(s,
                                                                    float(v)))
        scale.pack(side="left", padx=4)

        target_lbl = ttk.Label(row, text=str(L["pos"]), width=8)
        target_lbl.pack(side="left")

        pos_lbl = ttk.Label(row, text=str(L["pos"]), width=8)
        pos_lbl.pack(side="left")

        finger_var = tk.StringVar()
        cbo = ttk.Combobox(row, textvariable=finger_var, width=14,
                           values=FINGER_CHOICES, state="readonly")
        cbo.pack(side="left", padx=4)

        self.rows[sid] = dict(model=model, var=var, scale=scale,
                              target_lbl=target_lbl, pos_lbl=pos_lbl,
                              finger_var=finger_var)

    # ---- Events ----
    def _on_slider(self, sid, val):
        pos = int(round(val))
        self.rows[sid]["target_lbl"].config(text=str(pos))
        if not self.torque_all.get():
            return
        # Throttle to WRITE_THROTTLE_MS between writes for the same sid
        # so a fast drag can't overrun the serial bus.
        now = time.monotonic() * 1000.0
        last = self._last_write_ms.get(sid, 0.0)
        if now - last < WRITE_THROTTLE_MS:
            return
        self._last_write_ms[sid] = now
        model = self.rows[sid]["model"]
        try:
            self.bus.write_pos(sid, model, pos)
        except Exception as e:
            print(f"[write] sid {sid} err: {e}", flush=True)

    def _apply_torque_all(self):
        on = self.torque_all.get()
        # Before enabling, snap each slider to the servo's current present
        # so nothing jerks the moment torque comes up.
        if on:
            for sid, model in self.motors:
                p = self.bus.read_pos(sid, model)
                if p is not None:
                    self.rows[sid]["var"].set(p)
                    self.rows[sid]["target_lbl"].config(text=str(p))
        for sid, model in self.motors:
            self.bus.set_torque(sid, model, on)
        state = "ON" if on else "OFF"
        print(f"[torque] all -> {state}", flush=True)

    def _stop_all(self):
        self.torque_all.set(False)
        for sid, model in self.motors:
            self.bus.set_torque(sid, model, False)
        print("[STOP] all torque released", flush=True)

    def _poll_positions(self):
        for sid, model in self.motors:
            p = self.bus.read_pos(sid, model)
            if p is not None:
                self.rows[sid]["pos_lbl"].config(text=str(p))
        self.after(POLL_MS, self._poll_positions)

    def _save_map(self):
        data = {}
        for sid, model in self.motors:
            L = self.limits[sid]
            finger = self.rows[sid]["finger_var"].get().strip()
            data[str(sid)] = dict(
                model=model,
                family="SCS009" if is_scscl(model) else "STS3032",
                min_tick=L["mn"], max_tick=L["mx"],
                swapped_in_gui=L["swapped"],
                finger=finger or None,
            )
        out = Path(self.out_path)
        if not out.is_absolute():
            out = Path(__file__).with_name(out.name)
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        messagebox.showinfo("saved", f"wrote {out.name}\n{len(data)} motors")
        print(f"[save] {out}", flush=True)

    def _on_close(self):
        try:
            self._stop_all()
        finally:
            self.bus.close()
            self.destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--max-id", type=int, default=50)
    ap.add_argument("--min-id", type=int, default=1,
                    help="skip IDs below this (e.g. 40 to only scan hand)")
    ap.add_argument("--hand", choices=("right", "left"), default="right",
                    help="which hand — picks default output filename")
    ap.add_argument("--out", default=None,
                    help="override output json (default finger_map_<hand>.json)")
    args = ap.parse_args()

    out_path = args.out or f"finger_map_{args.hand}.json"

    bus = HandBus(args.port, args.baud)
    motors = bus.scan(args.max_id, min_id=args.min_id)
    if not motors:
        print("[ERR] no motors found")
        bus.close()
        return 1
    print(f"[scan] hand={args.hand}  found {len(motors)}: {motors}",
          flush=True)

    app = HandGUI(bus, motors, out_path=out_path, hand_label=args.hand)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
