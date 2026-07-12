#!/usr/bin/env python3
"""
RX1 左臂視覺化控制介面
Visual GUI for controlling the RX1 robot left arm on Windows.

Usage:  python gui.py
"""

import tkinter as tk
from tkinter import messagebox
import serial
import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from feetech_lib import SMS_STS

# ── Joint definitions ────────────────────────────────────────────────────────
#
#  ID 21 : 肩膀旋轉    — Shoulder Rotation       (active)
#  ID 22 : 肩膀前後抬升 — Shoulder Pitch/Flexion  (reserved — enable later)
#  ID 23 : 手肘        — Elbow                   (active)
#  ID 24 : 手腕        — Wrist                   (active)
#
JOINTS = [
    dict(label='肩膀旋轉',     name_en='Shoulder Rotation',
         servo_id=21, dir=-1, gear=3, range_deg=50, enabled=True),
    dict(label='肩膀前後抬升', name_en='Shoulder Pitch',
         servo_id=22, dir=-1, gear=3, range_deg=50, enabled=False),
    dict(label='手肘',         name_en='Elbow',
         servo_id=23, dir=1,  gear=3, range_deg=50, enabled=True),
    dict(label='手腕',         name_en='Wrist',
         servo_id=24, dir=-1, gear=3, range_deg=50, enabled=True),
]

# Feetech unit conversion factors
SPEED_K = 652.6051999582332   # (rad/s)  → servo speed unit
ACC_K   = 6.526051999582332   # (rad/s²) → servo acc unit

# ── Colour palette (dark theme) ──────────────────────────────────────────────
C = {
    'bg':       '#1e2433',
    'panel':    '#252d3d',
    'header':   '#1a237e',
    'slider':   '#3d5afe',
    'trough':   '#37474f',
    'trough_d': '#2e3a45',
    'text':     '#eceff1',
    'dim':      '#78909c',
    'green':    '#00e676',
    'green_dk': '#00c853',
    'red':      '#ff1744',
    'red_dk':   '#d50000',
    'orange':   '#ff9100',
    'purple':   '#b39ddb',
    'purple_dk':'#9575cd',
    'yellow':   '#ffd740',
}


# ── Hardware layer ───────────────────────────────────────────────────────────
class ArmController:
    def __init__(self):
        self._ser: serial.Serial | None = None
        self._sts: SMS_STS | None       = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self, port: str, baud: int = 1_000_000) -> str | None:
        """Return None on success, or an error string on failure."""
        try:
            self._ser = serial.Serial(
                port, baud, timeout=0.02,
                bytesize=8, parity='N', stopbits=1,
            )
            self._sts = SMS_STS(self._ser)
            return None
        except serial.SerialException as e:
            self._ser = None
            return str(e)

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def home(self):
        if not self.connected:
            return
        for j in JOINTS:
            self._sts.WritePosEx(j['servo_id'], 2048, 200, 20)
            time.sleep(0.012)

    def move(self, servo_id: int, direction: int, gear: int,
             angle_deg: float, speed: float, acc: float):
        if not self.connected:
            return
        angle_rad = math.radians(angle_deg)
        pos = int(angle_rad / math.pi * 2048 * direction * gear + 2048)
        pos = max(0, min(4095, pos))
        spd = int(speed * gear * SPEED_K)
        ac  = int(acc   * gear * ACC_K)
        self._sts.WritePosEx(servo_id, pos, spd, ac)

    def read_pos_deg(self, servo_id: int, direction: int, gear: int) -> float | None:
        """Read current encoder position and convert to degrees."""
        if not self.connected:
            return None
        try:
            params = [56, 2]          # register 56 = PRESENT_POSITION_L, read 2 bytes
            length = len(params) + 2
            cs = (~(servo_id + length + 0x02 + sum(params))) & 0xFF
            self._ser.reset_input_buffer()
            self._ser.write(bytes(
                [0xFF, 0xFF, servo_id, length, 0x02] + params + [cs]))
            resp = self._ser.read(8)
            if len(resp) >= 8 and resp[0] == 0xFF and resp[1] == 0xFF:
                pos = resp[5] | (resp[6] << 8)
                if pos & 0x8000:
                    pos = -(pos & 0x7FFF)
                # inverse of: pos = angle_rad/π * 2048 * dir * gear + 2048
                angle_rad = (pos - 2048) * math.pi / (2048 * direction * gear)
                return math.degrees(angle_rad)
        except Exception:
            pass
        return None


# ── Main application ─────────────────────────────────────────────────────────
class App:
    def __init__(self):
        self.hw   = ArmController()
        self.root = tk.Tk()
        self.root.title('RX1  左臂控制介面')
        self.root.configure(bg=C['bg'])
        self.root.resizable(False, False)

        # Per-joint widget references (set during _build_joints)
        self._scale_widgets: list[tk.Scale]  = []
        self._angle_vars:    list[tk.StringVar] = []
        self._row_frames:    list[tk.Frame]  = []
        self._status_labels: list[tk.Label] = []

        # Runtime enabled state
        self._enabled = [j['enabled'] for j in JOINTS]

        self._build()
        self._start_refresh()

    # ── UI build ─────────────────────────────────────────────────────────────

    def _build(self):
        root = self.root

        # ─ Header ─────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg=C['header'], pady=14)
        hdr.pack(fill='x')
        tk.Label(hdr, text='RX1  左臂關節控制介面',
                 font=('Microsoft JhengHei UI', 17, 'bold'),
                 bg=C['header'], fg='white').pack()
        tk.Label(hdr, text='Left Arm Joint Controller — Windows',
                 font=('Consolas', 9), bg=C['header'], fg='#90caf9').pack()

        # ─ Connection panel ────────────────────────────────────────────────
        cf = tk.Frame(root, bg=C['panel'], padx=16, pady=10)
        cf.pack(fill='x')

        for col, (lbl, w, var_init) in enumerate([
            ('PORT',  6,  'COM8'),
            ('BAUD',  9,  '1000000'),
        ]):
            tk.Label(cf, text=lbl, font=('Consolas', 9),
                     bg=C['panel'], fg=C['dim']).grid(row=0, column=col*2, padx=(0, 4))
            var = tk.StringVar(value=var_init)
            setattr(self, ('port_var' if col == 0 else 'baud_var'), var)
            tk.Entry(cf, textvariable=var, width=w,
                     font=('Consolas', 12), bg=C['trough'], fg=C['text'],
                     insertbackground='white', relief='flat', bd=4
                     ).grid(row=0, column=col*2+1, padx=(0, 14))

        self._conn_btn = tk.Button(
            cf, text='連  線', width=9,
            font=('Microsoft JhengHei UI', 11, 'bold'),
            bg=C['green'], fg='#102027', activebackground=C['green_dk'],
            relief='flat', bd=0, pady=4, command=self._toggle_connect)
        self._conn_btn.grid(row=0, column=4, padx=(0, 16))

        self._status_var = tk.StringVar(value='⬤  未連線')
        self._status_lbl = tk.Label(
            cf, textvariable=self._status_var,
            font=('Microsoft JhengHei UI', 11, 'bold'),
            bg=C['panel'], fg=C['red'])
        self._status_lbl.grid(row=0, column=5, sticky='w')

        # ─ Joints section ─────────────────────────────────────────────────
        sep = tk.Frame(root, bg=C['bg'], pady=4)
        sep.pack(fill='x')
        tk.Label(sep, text='  關節滑桿控制',
                 font=('Microsoft JhengHei UI', 10),
                 bg=C['bg'], fg=C['dim']).pack(anchor='w')

        self._build_joints(root)

        # ─ Parameters ─────────────────────────────────────────────────────
        pf = tk.Frame(root, bg=C['panel'], padx=16, pady=10)
        pf.pack(fill='x', pady=(2, 0))

        tk.Label(pf, text='速度', font=('Microsoft JhengHei UI', 10),
                 bg=C['panel'], fg=C['text']).grid(row=0, column=0, padx=(0, 6))
        self.speed_var = tk.DoubleVar(value=0.6)
        tk.Scale(pf, variable=self.speed_var, from_=0.1, to=3.0, resolution=0.1,
                 orient='horizontal', length=160, showvalue=True, sliderlength=18,
                 bg=C['panel'], fg=C['text'], troughcolor=C['orange'],
                 highlightbackground=C['panel']
                 ).grid(row=0, column=1, padx=(0, 24))

        tk.Label(pf, text='加速度', font=('Microsoft JhengHei UI', 10),
                 bg=C['panel'], fg=C['text']).grid(row=0, column=2, padx=(0, 6))
        self.acc_var = tk.DoubleVar(value=1.5)
        tk.Scale(pf, variable=self.acc_var, from_=0.5, to=10.0, resolution=0.5,
                 orient='horizontal', length=160, showvalue=True, sliderlength=18,
                 bg=C['panel'], fg=C['text'], troughcolor=C['purple'],
                 highlightbackground=C['panel']
                 ).grid(row=0, column=3)

        # ─ Buttons ────────────────────────────────────────────────────────
        bf = tk.Frame(root, bg=C['bg'], padx=12, pady=10)
        bf.pack(fill='x')

        tk.Button(
            bf, text='⌂   全部歸零  HOME',
            font=('Microsoft JhengHei UI', 12, 'bold'),
            bg=C['red'], fg='white', activebackground=C['red_dk'],
            relief='flat', padx=22, pady=9,
            command=self._home
        ).pack(side='left', padx=(0, 10))

        self._pitch_btn = tk.Button(
            bf, text='▶  啟用  肩膀前後抬升',
            font=('Microsoft JhengHei UI', 11),
            bg=C['purple'], fg='#1a0533', activebackground=C['purple_dk'],
            relief='flat', padx=16, pady=9,
            command=self._enable_pitch)
        self._pitch_btn.pack(side='left')

        # ─ Footer ─────────────────────────────────────────────────────────
        tk.Label(
            root,
            text='Feetech STS  ·  ID 21-24  ·  Left Arm  ·  RX1 Windows',
            font=('Consolas', 8), bg=C['bg'], fg=C['dim']
        ).pack(pady=(0, 8))

    def _build_joints(self, parent):
        """Build one row per joint inside the parent frame."""
        for idx, j in enumerate(JOINTS):
            enabled  = j['enabled']
            row_bg   = C['panel'] if idx % 2 == 0 else C['bg']

            row = tk.Frame(parent, bg=row_bg, pady=7, padx=14)
            row.pack(fill='x', pady=1)
            self._row_frames.append(row)

            # ── Joint name column ──
            name_col = tk.Frame(row, bg=row_bg, width=148)
            name_col.pack_propagate(False)
            name_col.pack(side='left')

            tk.Label(name_col, text=j['label'],
                     font=('Microsoft JhengHei UI', 12,
                            'bold' if enabled else 'normal'),
                     bg=row_bg,
                     fg=C['text'] if enabled else C['dim']
                     ).pack(anchor='w')
            tk.Label(name_col,
                     text=j['name_en'] + ('' if enabled else '  [預留]'),
                     font=('Consolas', 8),
                     bg=row_bg, fg=C['dim']).pack(anchor='w')

            # ── Angle display ──
            angle_var = tk.StringVar(value='  0.0°')
            self._angle_vars.append(angle_var)
            tk.Label(row, textvariable=angle_var,
                     font=('Consolas', 14, 'bold'),
                     bg=row_bg,
                     fg=C['slider'] if enabled else C['dim'],
                     width=8, anchor='e'
                     ).pack(side='left', padx=(0, 10))

            # ── Slider ──
            sv = tk.DoubleVar(value=0.0)
            scale = tk.Scale(
                row, variable=sv,
                from_=-j['range_deg'], to=j['range_deg'],
                resolution=0.5, orient='horizontal',
                length=310, showvalue=False, sliderlength=24,
                bg=row_bg,
                troughcolor=C['slider'] if enabled else C['trough_d'],
                activebackground='#82b1ff',
                highlightbackground=row_bg,
                state='normal' if enabled else 'disabled',
                command=lambda val, i=idx: self._on_slider(i, float(val)))
            scale.pack(side='left', padx=(0, 10))
            self._scale_widgets.append(scale)

            # ── Range hint / status text ──
            r_deg = j['range_deg']
            if enabled:
                status_text = f'±{r_deg}°'
                status_fg   = C['dim']
            else:
                status_text = '— 預留 —\n點「啟用」後解鎖'
                status_fg   = C['purple']

            status_lbl = tk.Label(row, text=status_text,
                                  font=('Microsoft JhengHei UI', 9, 'italic'),
                                  bg=row_bg, fg=status_fg, justify='center')
            status_lbl.pack(side='left', padx=(0, 4))
            self._status_labels.append(status_lbl)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if self.hw.connected:
            self.hw.disconnect()
            self._conn_btn.config(text='連  線', bg=C['green'])
            self._status_var.set('⬤  未連線')
            self._status_lbl.config(fg=C['red'])
        else:
            port = self.port_var.get().strip()
            try:
                baud = int(self.baud_var.get())
            except ValueError:
                baud = 1_000_000
            err = self.hw.connect(port, baud)
            if err is None:
                self._conn_btn.config(text='斷  線', bg=C['red'])
                self._status_var.set(f'⬤  已連線   {port}')
                self._status_lbl.config(fg=C['green'])
                self.hw.home()
            else:
                self._status_var.set('✗  連線失敗')
                self._status_lbl.config(fg=C['orange'])
                messagebox.showerror('連線失敗', f'無法開啟 {port}:\n{err}')

    def _on_slider(self, idx: int, deg: float):
        self._angle_vars[idx].set(f'{deg:+.1f}°')
        if not self._enabled[idx]:
            return
        j = JOINTS[idx]
        self.hw.move(j['servo_id'], j['dir'], j['gear'],
                     deg, self.speed_var.get(), self.acc_var.get())

    def _home(self):
        for i, sv in enumerate(self._scale_widgets):
            self._scale_widgets[i].set(0.0)
            self._angle_vars[i].set('  0.0°')
        self.hw.home()

    def _enable_pitch(self):
        """Enable the reserved shoulder-pitch joint (index 1)."""
        idx = 1
        if self._enabled[idx]:
            return
        self._enabled[idx] = True
        JOINTS[idx]['enabled'] = True

        # Unlock the slider
        row_bg = self._row_frames[idx].cget('bg')
        self._scale_widgets[idx].config(
            state='normal', troughcolor=C['slider'])
        self._angle_vars[idx].set('  0.0°')
        self._status_labels[idx].config(
            text=f'±{JOINTS[idx]["range_deg"]}°', fg=C['dim'])

        # Update the enable button
        self._pitch_btn.config(
            text='✓  肩膀前後抬升  已啟用',
            bg=C['trough'], state='disabled')

    # ── Live position refresh ─────────────────────────────────────────────────

    def _start_refresh(self):
        """Poll servo positions every 500 ms and update angle display."""
        self._refresh()

    def _refresh(self):
        if self.hw.connected:
            for idx, j in enumerate(JOINTS):
                if not self._enabled[idx]:
                    continue
                deg = self.hw.read_pos_deg(j['servo_id'], j['dir'], j['gear'])
                if deg is not None:
                    cur_slider = self._scale_widgets[idx].get()
                    # Only update label from hardware if slider hasn't moved
                    # (avoids fighting the user's input)
                    if abs(cur_slider - deg) > 2.0:
                        pass   # user is dragging; let the slider win
        self.root.after(500, self._refresh)

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()
        self.hw.disconnect()


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = App()
    app.run()
