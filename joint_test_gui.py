#!/usr/bin/env python3
"""
RX1 全關節測試介面
joint_test_gui.py

用途: 逐一測試每個關節馬達是否正常運作
功能:
  - 分頁: 左臂 / 右臂 / 頭部 / 軀幹 / 手掌
  - 每個關節獨立滑桿，即時送出硬體
  - 「掃描」按鈕自動做 ±30° 往返測試
  - 連線後不強制歸零，可分組測試

用法: python joint_test_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from rx1_motor_win import Rx1Motor

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    'bg': '#1e2433', 'panel': '#252d3d', 'header': '#1a237e',
    'text': '#eceff1', 'dim': '#78909c', 'sub': '#546e7a',
    'green': '#00e676', 'red': '#ff1744', 'orange': '#ff9100',
    'purple': '#b39ddb', 'yellow': '#ffd740', 'cyan': '#00bcd4',
    'blue_btn': '#1565c0',
}

# ── Joint specification ───────────────────────────────────────────────────────
#  (servo_id, 名稱, min_deg, max_deg, highlight_tag)
#  highlight_tag → 'active' | 'reserved' | 'normal'

LEFT_ARM = [
    (21, '肩膀旋轉',      -120,  120, 'active'),
    (22, '肩膀前後抬升',   -29,   90, 'reserved'),
    (23, '手肘旋轉',       -90,   90, 'active'),
    (24, '手肘彎曲',      -126,    0, 'active'),
    (25, '前臂旋轉',       -90,   90, 'active'),
    (26, '前臂彎曲',       -45,   45, 'normal'),
    (27, '前臂滾轉',       -45,   45, 'normal'),
]

RIGHT_ARM = [
    (11, '肩膀旋轉',      -120,  120, 'normal'),
    (12, '肩膀前後抬升',   -90,   29, 'normal'),
    (13, '手肘旋轉',       -90,   90, 'normal'),
    (14, '手肘彎曲',         0,  126, 'normal'),
    (15, '前臂旋轉',       -90,   90, 'normal'),
    (16, '前臂彎曲',       -45,   45, 'normal'),
    (17, '前臂滾轉',       -45,   45, 'normal'),
]

HEAD = [
    (4, '頸部上下 (仰俯)',  -60,  60, 'normal'),
    (5, '頸部左右 (偏轉)',  -60,  60, 'normal'),
    (6, '頸部滾轉',         -60,  60, 'normal'),
    (7, '左耳  [SCS]',      -60,  60, 'normal'),
    (8, '右耳  [SCS]',      -60,  60, 'normal'),
]

TORSO = [
    (1, '腰部偏轉 (Yaw)',    -45,  45, 'normal'),
    (2, '腰部俯仰 (Pitch)',  -30,  30, 'normal'),
    (3, '腰部滾轉 (Roll)',   -30,  30, 'normal'),
]

TAG_COLOR = {
    'active':   C['cyan'],
    'reserved': C['yellow'],
    'normal':   C['sub'],
}


# ── Application ───────────────────────────────────────────────────────────────

class JointTestApp:
    def __init__(self):
        self._robot = None

        self.root = tk.Tk()
        self.root.title('RX1 全關節測試介面')
        self.root.configure(bg=C['bg'])
        self.root.resizable(True, True)
        self.root.minsize(860, 560)

        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        r = self.root

        # ── Header
        hdr = tk.Frame(r, bg=C['header'], pady=10)
        hdr.pack(fill='x')
        tk.Label(hdr, text='RX1  全關節測試介面',
                 font=('Microsoft JhengHei UI', 17, 'bold'),
                 bg=C['header'], fg='white').pack()
        tk.Label(hdr,
                 text='Joint Diagnostic — slide to move, 掃描 to sweep, 歸零 to reset',
                 font=('Consolas', 9), bg=C['header'], fg='#90caf9').pack()

        # ── Connection / controls bar
        self._build_top_bar(r)

        # ── Notebook
        nb_wrap = tk.Frame(r, bg=C['bg'])
        nb_wrap.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('T.TNotebook',
                        background=C['bg'], borderwidth=0, relief='flat')
        style.configure('T.TNotebook.Tab',
                        background=C['panel'], foreground=C['text'],
                        padding=(16, 7),
                        font=('Microsoft JhengHei UI', 10, 'bold'))
        style.map('T.TNotebook.Tab',
                  background=[('selected', C['header'])],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(nb_wrap, style='T.TNotebook')
        nb.pack(fill='both', expand=True)

        # Left arm — vars tracked in self._la_vars
        self._la_vars = []
        self._la_sliders = []
        nb.add(self._build_angle_tab(nb, LEFT_ARM,
                                      self._la_vars, self._la_sliders,
                                      self._send_la),
               text='  左臂  ')

        # Right arm
        self._ra_vars = []
        self._ra_sliders = []
        nb.add(self._build_angle_tab(nb, RIGHT_ARM,
                                      self._ra_vars, self._ra_sliders,
                                      self._send_ra),
               text='  右臂  ')

        # Head
        self._hd_vars = []
        self._hd_sliders = []
        nb.add(self._build_angle_tab(nb, HEAD,
                                      self._hd_vars, self._hd_sliders,
                                      self._send_hd),
               text='  頭部  ')

        # Torso
        self._tr_vars = []
        self._tr_sliders = []
        nb.add(self._build_angle_tab(nb, TORSO,
                                      self._tr_vars, self._tr_sliders,
                                      self._send_tr),
               text='  軀幹  ')

        # Grippers
        nb.add(self._build_gripper_tab(nb), text='  手掌  ')

    def _build_top_bar(self, parent):
        bar = tk.Frame(parent, bg=C['panel'], pady=8, padx=14)
        bar.pack(fill='x')

        # Port field + connect
        tk.Label(bar, text='串列埠', font=('Microsoft JhengHei UI', 10),
                 bg=C['panel'], fg=C['text']).pack(side='left', padx=(0, 5))
        self._port = tk.StringVar(value='COM8')
        tk.Entry(bar, textvariable=self._port, width=7,
                 font=('Consolas', 12), bg='#37474f', fg=C['text'],
                 insertbackground='white', relief='flat', bd=3
                 ).pack(side='left', padx=(0, 10))

        self._cbtn = tk.Button(bar, text='連  線', width=9,
                               font=('Microsoft JhengHei UI', 11, 'bold'),
                               bg=C['green'], fg='#102027', relief='flat', pady=3,
                               command=self._toggle_conn)
        self._cbtn.pack(side='left', padx=(0, 10))

        self._cvar = tk.StringVar(value='⬤  未連線')
        self._clbl = tk.Label(bar, textvariable=self._cvar,
                              font=('Microsoft JhengHei UI', 11, 'bold'),
                              bg=C['panel'], fg=C['red'])
        self._clbl.pack(side='left', padx=(0, 20))

        tk.Frame(bar, bg=C['sub'], width=2).pack(side='left', fill='y', padx=(0,16), pady=2)

        # Speed / Acc
        for name, init, lo, hi, res, tc, attr in [
            ('速度',   1.0, 0.1, 3.0, 0.1, C['orange'],  '_spd'),
            ('加速度', 3.0, 0.5,10.0, 0.5, C['purple'],  '_acc'),
        ]:
            tk.Label(bar, text=name, font=('Microsoft JhengHei UI', 10),
                     bg=C['panel'], fg=C['text']).pack(side='left', padx=(0, 4))
            sv = tk.DoubleVar(value=init)
            setattr(self, attr, sv)
            tk.Scale(bar, variable=sv, from_=lo, to=hi, resolution=res,
                     orient='horizontal', length=110, showvalue=True,
                     sliderlength=16, bg=C['panel'], troughcolor=tc,
                     fg=C['text'], highlightbackground=C['panel']
                     ).pack(side='left', padx=(0, 14))

        tk.Frame(bar, bg=C['sub'], width=2).pack(side='left', fill='y', padx=(0,16), pady=2)

        tk.Button(bar, text='⌂  全部歸零',
                  font=('Microsoft JhengHei UI', 11, 'bold'),
                  bg=C['red'], fg='white', relief='flat', padx=12, pady=3,
                  command=self._home_all
                  ).pack(side='left')

    def _build_angle_tab(self, parent, spec, var_list, slider_list, send_fn):
        """
        Build a tab with one slider row per joint.
        spec        : list of (id, label, lo_deg, hi_deg, tag)
        var_list    : empty list — will be populated with DoubleVar per joint
        slider_list : empty list — will be populated with Scale widgets
        send_fn     : callable(deg_list) — sends full joint state to hardware
        """
        frame = tk.Frame(parent, bg=C['bg'])

        # Column header row
        hdr = tk.Frame(frame, bg='#1a2030', pady=5, padx=14)
        hdr.pack(fill='x')
        for txt, w, anchor in [
            ('ID',     5,  'w'),
            ('關節',   16, 'w'),
            ('角度',   40, 'center'),
            ('當前角度', 10, 'center'),
            ('操作',   18, 'center'),
        ]:
            tk.Label(hdr, text=txt, width=w, anchor=anchor,
                     font=('Microsoft JhengHei UI', 9, 'bold'),
                     bg='#1a2030', fg=C['dim']
                     ).pack(side='left')

        # Joint rows
        for i, (sid, label, lo, hi, tag) in enumerate(spec):
            bg  = C['panel'] if i % 2 == 0 else C['bg']
            row = tk.Frame(frame, bg=bg, pady=7, padx=14)
            row.pack(fill='x')

            fg  = TAG_COLOR.get(tag, C['dim'])
            dot = '⬤ ' if tag in ('active', 'reserved') else '  '

            # ID label
            tk.Label(row, text=f'{dot}ID{sid}', width=7, anchor='w',
                     font=('Consolas', 11, 'bold'), bg=bg, fg=fg
                     ).pack(side='left')

            # Joint name
            tk.Label(row, text=label, width=14, anchor='w',
                     font=('Microsoft JhengHei UI', 10), bg=bg, fg=C['text']
                     ).pack(side='left', padx=(0, 6))

            # Range label (min)
            tk.Label(row, text=f'{lo}°', font=('Consolas', 9),
                     bg=bg, fg=C['dim'], width=5, anchor='e'
                     ).pack(side='left')

            # Slider
            sv = tk.DoubleVar(value=0.0)
            var_list.append(sv)
            sl = tk.Scale(row, variable=sv, from_=lo, to=hi,
                          resolution=0.5, orient='horizontal', length=280,
                          showvalue=False, sliderlength=20,
                          bg=bg, troughcolor='#37474f',
                          highlightbackground=bg,
                          command=lambda v, vl=var_list, sf=send_fn:
                              sf([x.get() for x in vl]))
            sl.pack(side='left')
            slider_list.append(sl)

            # Range label (max)
            tk.Label(row, text=f'{hi}°', font=('Consolas', 9),
                     bg=bg, fg=C['dim'], width=5, anchor='w'
                     ).pack(side='left', padx=(0, 8))

            # Angle display
            angle_var = tk.StringVar(value='  0.0°')
            sv.trace_add('write', lambda *_a, av=angle_var, s=sv:
                         av.set(f'{s.get():+.1f}°'))
            tk.Label(row, textvariable=angle_var, width=8, anchor='e',
                     font=('Consolas', 13, 'bold'), bg=bg, fg=fg
                     ).pack(side='left', padx=(0, 10))

            # Sweep button
            tk.Button(row, text='◉ 掃描',
                      font=('Microsoft JhengHei UI', 9, 'bold'),
                      bg=C['blue_btn'], fg='white', relief='flat',
                      padx=8, pady=2,
                      command=lambda s=sl, vl=var_list, sf=send_fn:
                          self._start_sweep(s, vl, sf)
                      ).pack(side='left', padx=(0, 6))

            # Zero button
            tk.Button(row, text='⌂ 歸零',
                      font=('Microsoft JhengHei UI', 9),
                      bg=C['sub'], fg=C['text'], relief='flat',
                      padx=8, pady=2,
                      command=lambda s=sl, vl=var_list, sf=send_fn:
                          self._zero_joint(s, vl, sf)
                      ).pack(side='left')

        return frame

    def _build_gripper_tab(self, parent):
        frame = tk.Frame(parent, bg=C['bg'])

        for title, grip_attr, send_fn in [
            ('左手掌  Left Gripper',  '_lg_var', self._send_lg),
            ('右手掌  Right Gripper', '_rg_var', self._send_rg),
        ]:
            gf = tk.LabelFrame(frame, text=f'  {title}  ',
                               bg=C['panel'], fg=C['dim'],
                               font=('Microsoft JhengHei UI', 11, 'bold'),
                               padx=20, pady=16)
            gf.pack(fill='x', padx=20, pady=14)

            tk.Label(gf, text='0% = 完全張開    100% = 完全閉合',
                     font=('Microsoft JhengHei UI', 10),
                     bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(0, 10))

            sv = tk.DoubleVar(value=0.0)
            setattr(self, grip_attr, sv)

            pct_var = tk.StringVar(value='0%   (張開)')
            def _upd(v, pv=pct_var):
                p = float(v)
                s = '張開' if p < 10 else '閉合' if p > 90 else f'{p:.0f}%'
                pv.set(f'{p:.0f}%   ({s})')

            row = tk.Frame(gf, bg=C['panel'])
            row.pack(fill='x')

            tk.Scale(row, variable=sv, from_=0, to=100, resolution=1,
                     orient='horizontal', length=420, showvalue=False,
                     sliderlength=24, bg=C['panel'], troughcolor='#37474f',
                     highlightbackground=C['panel'],
                     command=lambda v, sf=send_fn, upd=_upd:
                         (upd(v), sf(float(v) / 100))
                     ).pack(side='left', padx=(0, 14))

            tk.Label(row, textvariable=pct_var,
                     font=('Consolas', 14, 'bold'),
                     bg=C['panel'], fg=C['cyan'], width=16
                     ).pack(side='left')

            btn_row = tk.Frame(gf, bg=C['panel'])
            btn_row.pack(anchor='w', pady=(8, 0))
            for txt, val in [('張開 (0%)', 0), ('50%', 50), ('閉合 (100%)', 100)]:
                tk.Button(btn_row, text=txt,
                          font=('Microsoft JhengHei UI', 10),
                          bg=C['sub'], fg=C['text'], relief='flat',
                          padx=10, pady=4,
                          command=lambda sv_=sv, v=val, sf=send_fn:
                              (sv_.set(v), sf(v / 100))
                          ).pack(side='left', padx=(0, 8))

        return frame

    # ── Event handlers ────────────────────────────────────────────────────────

    def _toggle_conn(self):
        if self._robot is not None:
            self._robot.close()
            self._robot = None
            self._cbtn.config(text='連  線', bg=C['green'])
            self._cvar.set('⬤  未連線')
            self._clbl.config(fg=C['red'])
            return

        port = self._port.get().strip()
        try:
            robot = Rx1Motor(port)
            robot.connect()
            self._robot = robot
            self._cbtn.config(text='斷  線', bg=C['red'])
            self._cvar.set(f'⬤  已連線   {port}')
            self._clbl.config(fg=C['green'])
        except RuntimeError as e:
            self._cvar.set('✗  連線失敗')
            self._clbl.config(fg=C['orange'])
            messagebox.showerror('連線失敗', str(e))

    def _zero_joint(self, slider, var_list, send_fn):
        slider.set(0.0)
        send_fn([v.get() for v in var_list])

    def _home_all(self):
        for sliders in (self._la_sliders, self._ra_sliders,
                        self._hd_sliders, self._tr_sliders):
            for sl in sliders:
                sl.set(0.0)
        if self._robot:
            self._send_la([0.0]*7)
            self._send_ra([0.0]*7)
            self._send_hd([0.0]*5)
            self._send_tr([0.0]*3)
            self._send_lg(0.0)
            self._send_rg(0.0)
        try:
            self._lg_var.set(0); self._rg_var.set(0)
        except Exception:
            pass

    # ── Sweep (runs on main thread via root.after) ────────────────────────────

    def _start_sweep(self, slider, var_list, send_fn):
        lo  = float(slider.cget('from'))
        hi  = float(slider.cget('to'))
        amp = min(30.0, (hi - lo) / 4)

        # 0 → +amp → -amp → 0
        steps = (
            [amp * p / 100 for p in range(0, 101, 5)]
            + [amp * p / 100 for p in range(100, -101, -5)]
            + [amp * p / 100 for p in range(-100, 1, 5)]
            + [0.0]
        )

        def step(n):
            if n >= len(steps):
                return
            slider.set(steps[n])               # fires slider command → send_fn
            self.root.after(70, step, n + 1)

        step(0)

    # ── Hardware senders ──────────────────────────────────────────────────────

    def _r(self, degs):
        return [math.radians(d) for d in degs]

    def _sa(self):
        return self._spd.get(), self._acc.get()

    def _send_la(self, degs):
        if not self._robot: return
        s, a = self._sa()
        self._robot.left_arm(self._r(degs), speeds=[s]*7, accs=[a]*7)

    def _send_ra(self, degs):
        if not self._robot: return
        s, a = self._sa()
        self._robot.right_arm(self._r(degs), speeds=[s]*7, accs=[a]*7)

    def _send_hd(self, degs):
        if not self._robot: return
        s, a = self._sa()
        self._robot.head(self._r(degs), speeds=[s]*5, accs=[a]*5)

    def _send_tr(self, degs):
        if not self._robot: return
        s, a = self._sa()
        self._robot.torso(self._r(degs), speeds=[s]*3, accs=[a]*3)

    def _send_lg(self, ratio):
        if not self._robot: return
        self._robot.left_gripper(max(0.0, min(1.0, ratio)))

    def _send_rg(self, ratio):
        if not self._robot: return
        self._robot.right_gripper(max(0.0, min(1.0, ratio)))

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()
        if self._robot:
            self._robot.close()


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app = JointTestApp()
    app.run()
