#!/usr/bin/env python3
"""
RX1 Dual Arm IK Controller  —  arm_ik_gui.py

Workflow:
  01 CONNECT  →  02 JOINT CONFIG  →  03 MOTOR MODE  →  04 MOTOR TEST  →  05 IK CONTROL

Mouse (3D viewport):
  LMB drag  = move end-effector (IK)
  RMB drag  = rotate view
  Scroll    = zoom

Requirements: numpy  scipy  matplotlib  (Anaconda)
              feetech_lib.py  in same folder
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np
from scipy import optimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import math, time, serial, threading, sys, os, json

sys.path.insert(0, os.path.dirname(__file__))
try:
    from feetech_lib import SMS_STS
    _HW_OK = True
except ImportError:
    _HW_OK = False

# ── Colour palette  (professional / mid-brightness) ───────────────────────────
C = {
    'bg':      '#111820',    # main background
    'panel':   '#172030',    # side-panel background
    'card':    '#1c2a3a',    # card / row background
    'border':  '#2c4060',    # dividers
    'text':    '#d0e0f0',    # body text
    'dim':     '#6080a0',    # secondary / muted labels
    'bright':  '#eef4fc',    # highlighted text
    'ok':      '#50c880',    # success green
    'warn':    '#d09840',    # warning amber
    'err':     '#e05858',    # error red
    'la_col':  '#5898e8',    # left arm  – bright blue
    'ra_col':  '#e08848',    # right arm – bright orange
    'active':  '#78c8e8',    # active value readout
    'tg_col':  '#50c880',    # IK target marker
    # Phase accent colours (one per workflow step)
    'ph1': '#60a0d8',        # 01 CONNECT     – sky blue
    'ph2': '#58c090',        # 02 JOINT CFG   – mint teal
    'ph3': '#d0a850',        # 03 MOTOR MODE  – gold
    'ph4': '#9888d0',        # 04 MOTOR TEST  – lavender
    'ph5': '#6888d8',        # 05 IK CTRL     – periwinkle
    # Button fills
    'btn_connect':  '#1e4870',
    'btn_apply':    '#38206a',
    'btn_home':     '#401818',
    'btn_read':     '#183828',
    'btn_sbtn':     '#223040',
    # Slider / scale
    'sl_trough':   '#203448',
    'sl_test':     '#203828',
    # Mode buttons
    'mode_pos':    '#1e3e78',
    'mode_multi':  '#3e1e68',
    'dir_p':       '#1e5830',
    'dir_n':       '#581e1e',
}

FONT_MONO = ('Consolas', 10)
FONT_LG   = ('Consolas', 13, 'bold')
FONT_SM   = ('Consolas', 9)
FONT_XS   = ('Consolas', 8)
FONT_HDR  = ('Consolas', 8, 'bold')

SPEED_K = 652.6051999582332
ACC_K   = 6.526051999582332

# ── Arm definitions ───────────────────────────────────────────────────────────
# [sid, dir, gear, label, lo_rad, hi_rad, mode, ticks_lo, ticks_hi, mt_gear]
_ARM_L_DEFAULTS = [
    # sid  dir gear  label          lo_rad   hi_rad  mode  tlo   thi  mt_gear
    (20, -1, 3, 'Shoul Rot',   -2.5132,  2.5132, 0,    0, 4095, 1),
    (21, -1, 3, 'Shoul Pitch', -0.5026,  1.5708, 0,    0, 4095, 1),
    (22,  1, 3, 'Elbow Rot',   -1.5708,  1.5708, 0,    0, 4095, 1),
    (23, -1, 1, 'Elbow Flex',  -1.5708,  1.5708, 0,  550, 3600, 1),
    (24,  1, 1, 'Wrist Rot',   -1.5708,  1.5708, 0,    0, 4095, 1),
]
_ARM_R_DEFAULTS = [
    (10, -1, 3, 'Shoul Rot',   -2.5132,  2.5132, 0,    0, 4095, 1),
    (11, -1, 3, 'Shoul Pitch', -1.5708,  0.5026, 0,    0, 4095, 1),
    (12,  1, 3, 'Elbow Rot',   -1.5708,  1.5708, 0,    0, 4095, 1),
    (13,  1, 3, 'Elbow Flex',  -2.1991,  0.0000, 0,    0, 4095, 1),
    (14,  1, 1, 'Wrist Rot',   -1.5708,  1.5708, 0,    0, 4095, 1),
]

LEFT_CHAIN = [
    ('F', [ 0.00,  0.12, -0.050], [-1.04706, 0, 0], None,    None),
    ('R', [ 0.00,  0.00,  0.075], [ 0,       0, 0], [0,0,1],    0),
    ('R', [ 0.00,  0.00,  0.080], [ 1.8279,  0, 0], [1,0,0],    1),  # +45° offset baked (1.04706+0.7854)
    ('F', [ 0.00,  0.00, -0.0625], [ 0,       0, 0], None,    None),
    ('R', [ 0.00,  0.00, -0.0625], [ 0,       0, 0], [0,0,1],    2),
    ('R', [ 0.00,  0.00, -0.220], [ 0,       0, 0], [0,1,0],    3),
    ('F', [ 0.00,  0.00, -0.045], [ 0,       0, 0], None,    None),
    ('R', [ 0.00,  0.00, -0.045], [ 0,       0, 0], [0,0,1],    4),
    ('F', [ 0.00,  0.00, -0.180], [ 0,       0, 0], None,    None),
    ('F', [ 0.00,  0.00, -0.100], [ 0,       0, 0], None,    None),
]
RIGHT_CHAIN = [
    ('F', [ 0.00, -0.12, -0.050], [ 1.04706, 0, 0], None,    None),
    ('R', [ 0.00,  0.00,  0.075], [ 0,       0, 0], [0,0,1],    0),
    ('R', [ 0.00,  0.00,  0.080], [-1.8279,  0, 0], [1,0,0],    1),  # -45° offset baked (-1.04706-0.7854)
    ('F', [ 0.00,  0.00, -0.0625], [ 0,       0, 0], None,    None),
    ('R', [ 0.00,  0.00, -0.0625], [ 0,       0, 0], [0,0,1],    2),
    ('R', [ 0.00,  0.00, -0.220], [ 0,       0, 0], [0,1,0],    3),
    ('F', [ 0.00,  0.00, -0.045], [ 0,       0, 0], None,    None),
    ('R', [ 0.00,  0.00, -0.045], [ 0,       0, 0], [0,0,1],    4),
    ('F', [ 0.00,  0.00, -0.180], [ 0,       0, 0], None,    None),
    ('F', [ 0.00,  0.00, -0.100], [ 0,       0, 0], None,    None),
]

# ── Math / FK / IK ────────────────────────────────────────────────────────────

def _Txyz(v):
    T = np.eye(4); T[:3,3] = v; return T

def _Trpy(rpy):
    r,p,y = rpy
    cr,sr = math.cos(r),math.sin(r)
    cp,sp = math.cos(p),math.sin(p)
    cy,sy = math.cos(y),math.sin(y)
    T = np.eye(4)
    T[:3,:3] = (np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
               @np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
               @np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]]))
    return T

def _Rax(ax, a):
    c,s = math.cos(a),math.sin(a); x,y,z = ax
    T = np.eye(4)
    T[:3,:3] = np.array([
        [c+x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)  ]])
    return T

def fk_chain(chain, arm_def, q):
    T = np.eye(4); pts = [(T[:3,3].copy(),'base',None)]
    for jtype,xyz,rpy,axis,qi in chain:
        T = T @ _Txyz(xyz) @ _Trpy(rpy)
        if jtype == 'R':
            pts.append((T[:3,3].copy(), arm_def[qi][3], arm_def[qi][0]))
            T = T @ _Rax(axis, float(q[qi]) if qi < len(q) else 0.0)
        else:
            pts.append((T[:3,3].copy(),'link',None))
    pts.append((T[:3,3].copy(),'EE',None)); return pts

def ee_pos(chain, arm_def, q):
    return fk_chain(chain, arm_def, q)[-1][0]

def ik_solve(chain, arm_def, target, q0=None):
    if q0 is None: q0 = np.zeros(5)
    bounds = [(arm_def[i][4], arm_def[i][5]) for i in range(5)]
    def cost(q): return float(np.sum((ee_pos(chain,arm_def,q)-target)**2))
    res = optimize.minimize(cost, q0, method='SLSQP', bounds=bounds,
                            options={'maxiter':300,'ftol':1e-9})
    return res.x, res.fun

# ── Hardware ──────────────────────────────────────────────────────────────────

class Hardware:
    def __init__(self):
        self._ser = self._sts = None
        self._lock = threading.Lock()

    @property
    def ok(self): return self._ser is not None and self._ser.is_open

    def connect(self, port, baud=1_000_000):
        try:
            self._ser = serial.Serial(port, baud, timeout=0.05,
                                      bytesize=8, parity='N', stopbits=1)
            self._sts = SMS_STS(self._ser)
        except Exception as e:
            self._ser = None; return str(e)

    def disconnect(self):
        with self._lock:
            if self._ser and self._ser.is_open: self._ser.close()
            self._ser = None

    def home_arm(self, arm_def, skip_ids=None):
        if not self.ok: return
        with self._lock:
            for row in arm_def:
                sid = row[0]
                if skip_ids and sid in skip_ids: continue
                self._sts.WritePosEx(sid, 2048, 200, 20)
                time.sleep(0.012)

    def send_arm(self, arm_def, q, speed, acc, skip_ids=None):
        if not self.ok: return
        with self._lock:
            for i,row in enumerate(arm_def):
                sid,d,g = row[0],row[1],row[2]
                if skip_ids and sid in skip_ids: continue
                lo,hi   = row[7],row[8]
                pos = int(float(q[i])/math.pi*2048*d*g + 2048)
                pos = max(lo, min(hi, pos))
                self._sts.WritePosEx(sid, pos,
                                     int(speed*g*SPEED_K), int(acc*g*ACC_K))

    def send_ticks(self, sid, ticks, speed=200, acc=20):
        if not self.ok: return
        with self._lock:
            ticks = max(-32767, min(32767, int(ticks)))
            self._sts.WritePosEx(sid, ticks, speed, acc)

    def read_pos(self, sid):
        if not self.ok: return None
        with self._lock: return self._sts.ReadPos(sid)

    def read_all_arm(self, arm_l, arm_r):
        if not self.ok: return []
        results = []
        for arm_def in (arm_l, arm_r):
            for row in arm_def:
                sid = row[0]
                with self._lock:
                    pos  = self._sts.ReadPos(sid)
                    mode = self._sts.ReadMode(sid)
                results.append({'sid':sid,'label':row[3],
                                'pos':pos,'mode':mode,'ok':pos is not None})
        return results

    def set_position_mode(self, sid):
        if not self.ok: return
        with self._lock: self._sts.SetPositionMode(sid)

    def set_multi_turn_mode(self, sid):
        if not self.ok: return
        with self._lock: self._sts.SetMultiTurnMode(sid)

# ── UI helpers ────────────────────────────────────────────────────────────────

def _lbl(parent, text, fg=None, **kw):
    return tk.Label(parent, text=text, font=FONT_XS,
                    bg=C['panel'], fg=fg or C['dim'], **kw)

def _ph_strip(parent, ph_col):
    """2-px coloured top strip to mark a phase section."""
    tk.Frame(parent, bg=ph_col, height=2).pack(fill='x')

def _ph_header(parent, number, title, ph_col, extra=None):
    """Numbered phase header row with coloured accent."""
    _ph_strip(parent, ph_col)
    hdr = tk.Frame(parent, bg=C['panel']); hdr.pack(fill='x', padx=8, pady=(5,3))
    tk.Label(hdr, text=f'{number:02d}', font=FONT_HDR,
             bg=C['panel'], fg=ph_col, width=2).pack(side='left')
    tk.Label(hdr, text=f'  {title}', font=FONT_HDR,
             bg=C['panel'], fg=ph_col).pack(side='left')
    if extra:
        tk.Label(hdr, text=extra, font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(side='right')
    return hdr   # caller may add widgets to right side

def _divider(parent):
    tk.Frame(parent, bg=C['border'], height=1).pack(fill='x', pady=3)

def _arm_tabs(parent, var, cmd, bg=None):
    bg = bg or C['panel']
    for val,txt,col in [('L','LEFT',C['la_col']),('R','RIGHT',C['ra_col'])]:
        tk.Radiobutton(parent, text=txt, variable=var, value=val,
                       font=FONT_XS, bg=bg, fg=col,
                       selectcolor=C['border'], activebackground=bg,
                       activeforeground=col, command=cmd,
                       indicatoron=0, width=5, relief='flat',
                       padx=2, pady=2,
                       ).pack(side='left', padx=(0,2))

# ── Application ───────────────────────────────────────────────────────────────

class App:
    _DEFAULT_WS = dict(x_min=-0.45, x_max=0.45,
                       y_min=-0.25, y_max=0.65,
                       z_min=-0.65, z_max=0.35)
    _CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'arm_ik_config.json')

    # ── Config persistence ────────────────────────────────────────────────────

    def _arm_to_dict(self, arm_def):
        out = []
        for r in arm_def:
            out.append({'sid':r[0],'dir':r[1],'gear':r[2],'label':r[3],
                        'lo_deg':round(math.degrees(r[4]),4),
                        'hi_deg':round(math.degrees(r[5]),4),
                        'mode':r[6],'ticks_lo':r[7],'ticks_hi':r[8],
                        'mt_gear':r[9] if len(r)>9 else 1})
        return out

    def _dict_to_arm(self, data, defaults):
        arm = [list(r) for r in defaults]
        for i, d in enumerate(data):
            if i >= len(arm): break
            arm[i][1] = d.get('dir',      arm[i][1])
            arm[i][4] = math.radians(d.get('lo_deg', math.degrees(arm[i][4])))
            arm[i][5] = math.radians(d.get('hi_deg', math.degrees(arm[i][5])))
            arm[i][6] = d.get('mode',     arm[i][6])
            arm[i][7] = d.get('ticks_lo', arm[i][7])
            arm[i][8] = d.get('ticks_hi', arm[i][8])
            while len(arm[i]) <= 9: arm[i].append(1)
            arm[i][9] = d.get('mt_gear',  1)
        return arm

    def _load_config(self):
        try:
            with open(self._CFG_FILE, encoding='utf-8') as f:
                cfg = json.load(f)
            if 'arm_l' in cfg:
                self._arm_l = self._dict_to_arm(cfg['arm_l'], _ARM_L_DEFAULTS)
            if 'arm_r' in cfg:
                self._arm_r = self._dict_to_arm(cfg['arm_r'], _ARM_R_DEFAULTS)
            if 'port' in cfg:
                self._saved_port = cfg['port']
            if 'ws' in cfg:
                self._ws.update(cfg['ws'])
        except FileNotFoundError:
            pass   # first run — use defaults
        except Exception as e:
            print(f'[config] load error: {e}')

    def _save_config(self):
        try:
            cfg = {
                'port':  getattr(self, '_port', None) and self._port.get() or 'COM8',
                'arm_l': self._arm_to_dict(self._arm_l),
                'arm_r': self._arm_to_dict(self._arm_r),
                'ws':    {**self._ws},
            }
            with open(self._CFG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            if hasattr(self, '_save_lbl'):
                self._save_lbl.config(fg=C['ok'])
                self._save_lbl.after(1500, lambda: self._save_lbl.config(fg=C['dim']))
        except Exception as e:
            print(f'[config] save error: {e}')

    def _schedule_save(self):
        """Debounced save — write 400 ms after the last change."""
        if hasattr(self, '_save_job') and self._save_job:
            self._root.after_cancel(self._save_job)
        self._save_job = self._root.after(400, self._save_config)

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self.hw = Hardware()
        self._arm_l = [list(r) for r in _ARM_L_DEFAULTS]
        self._arm_r = [list(r) for r in _ARM_R_DEFAULTS]
        self._ws    = {k:v for k,v in self._DEFAULT_WS.items()}
        self._saved_port = 'COM8'
        self._save_job   = None

        # Load saved config before building UI
        self._load_config()

        self._q_l  = np.zeros(len(self._arm_l))
        self._q_r  = np.zeros(len(self._arm_r))
        self._tgt_l = ee_pos(LEFT_CHAIN,  self._arm_l, self._q_l).copy()
        self._tgt_r = ee_pos(RIGHT_CHAIN, self._arm_r, self._q_r).copy()
        self._err_l = self._err_r = 0.0
        self._disabled = {}   # {(an, i): tk.BooleanVar}
        self._arm_sel = 'L'

        self._elev = 18.0; self._azim = -52.0; self._zoom = 1.0
        self._lbtn = self._rbtn = None
        self._fb_job = None
        self._last_test_send = {}

        self._root = tk.Tk()
        self._root.title('RX1  ARM  IK  CONTROLLER')
        self._root.configure(bg=C['bg'])
        self._root.resizable(True, True)
        self._root.minsize(1180, 660)
        self._root.protocol('WM_DELETE_WINDOW', self._on_close)
        self._build()
        self._redraw()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # Title bar
        bar = tk.Frame(self._root, bg='#0e1828', pady=6)
        bar.pack(fill='x')
        tk.Label(bar, text='RX1  ARM  IK  CONTROLLER',
                 font=('Consolas',14,'bold'),
                 bg='#0e1828', fg=C['ph5']).pack(side='left', padx=14)
        tk.Label(bar,
                 text='LMB = move EE    RMB = rotate    Scroll = zoom',
                 font=FONT_XS, bg='#0e1828', fg=C['dim']).pack(side='left', padx=16)
        # Workflow guide strip
        wf = tk.Frame(self._root, bg='#141e30', pady=3)
        wf.pack(fill='x')
        guide = [('01 CONNECT',C['ph1']),('  →  02 JOINT CFG',C['ph2']),
                 ('  →  03 MOTOR MODE',C['ph3']),('  →  04 MOTOR TEST',C['ph4']),
                 ('  →  05 IK CONTROL',C['ph5'])]
        for txt,col in guide:
            tk.Label(wf, text=txt, font=FONT_XS, bg='#141e30', fg=col
                     ).pack(side='left', padx=(10,0))

        body = tk.Frame(self._root, bg=C['bg']); body.pack(fill='both', expand=True)

        # Scrollable left panel
        lp_outer = tk.Frame(body, bg=C['panel'], width=350,
                            highlightbackground=C['border'], highlightthickness=1)
        lp_outer.pack(side='left', fill='y'); lp_outer.pack_propagate(False)
        cvs = tk.Canvas(lp_outer, bg=C['panel'], highlightthickness=0, bd=0)
        sb  = tk.Scrollbar(lp_outer, orient='vertical', command=cvs.yview,
                           bg=C['border'], troughcolor=C['bg'])
        cvs.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y'); cvs.pack(side='left', fill='both', expand=True)
        lp = tk.Frame(cvs, bg=C['panel'])
        wid = cvs.create_window((0,0), window=lp, anchor='nw')
        def _conf(e):
            cvs.configure(scrollregion=cvs.bbox('all'))
            cvs.itemconfig(wid, width=cvs.winfo_width())
        lp.bind('<Configure>', _conf)
        cvs.bind('<Configure>', lambda e: cvs.itemconfig(wid, width=e.width))
        lp.bind_all('<MouseWheel>',
                    lambda e: cvs.yview_scroll(int(-1*(e.delta/120)),'units'))
        self._build_panel(lp)

        # 3-D viewport
        rp = tk.Frame(body, bg=C['bg']); rp.pack(side='right', fill='both', expand=True)
        self._build_canvas(rp)

    def _build_panel(self, p):
        pk = dict(padx=10)
        self._build_connect(p, pk)       # 01
        self._build_joint_cfg(p, pk)     # 02
        self._build_motor_mode(p, pk)    # 03
        self._build_motor_test(p, pk)    # 04
        self._build_ik_ctrl(p, pk)       # 05
        tk.Frame(p, bg=C['panel'], height=10).pack()  # bottom padding

    # ═══════════════════════════════════════════════════════════════════════════
    # 01  CONNECT
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_connect(self, p, pk):
        _ph_header(p, 1, 'CONNECT', C['ph1'],
                   extra='1 Mbaud  8N1')
        f = tk.Frame(p, bg=C['panel']); f.pack(fill='x', pady=(2,6), **pk)
        row = tk.Frame(f, bg=C['panel']); row.pack(fill='x')
        tk.Label(row, text='PORT', font=FONT_XS,
                 bg=C['panel'], fg=C['dim'], width=5).pack(side='left')
        self._port = tk.StringVar(value=self._saved_port)
        tk.Entry(row, textvariable=self._port, width=7, font=FONT_MONO,
                 bg=C['border'], fg=C['bright'],
                 insertbackground=C['ph1'], relief='flat', bd=3
                 ).pack(side='left', padx=(4,8))
        self._cbtn = tk.Button(row, text='CONNECT', width=10,
                               font=FONT_XS, bg=C['btn_connect'],
                               fg=C['ph1'], relief='flat', pady=4,
                               cursor='hand2', command=self._toggle_conn)
        self._cbtn.pack(side='left', padx=(0,8))
        self._cvar = tk.StringVar(value='OFFLINE')
        self._clbl = tk.Label(row, textvariable=self._cvar,
                              font=FONT_HDR, bg=C['panel'], fg=C['err'])
        self._clbl.pack(side='left')
        self._save_lbl = tk.Label(row, text='● CFG', font=FONT_XS,
                                  bg=C['panel'], fg=C['dim'])
        self._save_lbl.pack(side='right', padx=(0,4))

    # ═══════════════════════════════════════════════════════════════════════════
    # 02  JOINT CONFIG  —  IK direction & angle limits
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_joint_cfg(self, p, pk):
        hdr = _ph_header(p, 2, 'JOINT CONFIG', C['ph2'],
                         extra='IK direction & limits')
        # Arm sub-tab
        f = tk.Frame(p, bg=C['panel']); f.pack(fill='x', pady=(2,6), **pk)
        tab_row = tk.Frame(f, bg=C['panel']); tab_row.pack(fill='x', pady=(0,4))
        self._jcfg_var = tk.StringVar(value='L')
        _arm_tabs(tab_row, self._jcfg_var, self._refresh_jcfg)
        tk.Label(tab_row, text='ID    DIR    LO°      HI°',
                 font=FONT_XS, bg=C['panel'], fg=C['dim']
                 ).pack(side='left', padx=(12,0))

        self._jcfg_frames = {}
        if not hasattr(self,'_dir_btn_vars'): self._dir_btn_vars={}
        if not hasattr(self,'_lim_vars'):     self._lim_vars={}
        for an,ad,col in [('L',self._arm_l,C['la_col']),
                           ('R',self._arm_r,C['ra_col'])]:
            frm = tk.Frame(f, bg=C['panel'])
            self._jcfg_frames[an] = frm
            self._build_jcfg_rows(frm, an, ad, col)
        self._refresh_jcfg()

    def _build_jcfg_rows(self, parent, an, arm_def, col):
        self._dir_btn_vars[an]=[]
        self._lim_vars[an]=[]
        for i,row in enumerate(arm_def):
            sid,d,lo,hi = row[0],row[1],row[4],row[5]
            r = tk.Frame(parent, bg=C['card']); r.pack(fill='x', pady=1)
            tk.Label(r, text=f'ID{sid}', font=FONT_XS,
                     bg=C['card'], fg=col, width=5, anchor='w'
                     ).pack(side='left', padx=(4,2))
            dv = tk.StringVar(value=f'{d:+d}')
            self._dir_btn_vars[an].append(dv)
            btn = tk.Button(r, textvariable=dv, font=FONT_XS,
                            bg=C['dir_p'] if d>0 else C['dir_n'],
                            fg=C['bright'], width=3, relief='flat',
                            pady=1, cursor='hand2')
            btn.config(command=lambda b=btn,dv=dv,an=an,i=i:
                       self._flip_dir(an,i,dv,b))
            btn.pack(side='left', padx=(0,6))
            lo_sv = tk.StringVar(value=f'{math.degrees(lo):.1f}')
            hi_sv = tk.StringVar(value=f'{math.degrees(hi):.1f}')
            self._lim_vars[an].append((lo_sv,hi_sv))
            for sv,wh in [(lo_sv,'lo'),(hi_sv,'hi')]:
                e = tk.Entry(r, textvariable=sv, width=7, font=FONT_XS,
                             bg=C['border'], fg=C['active'],
                             insertbackground=C['ph2'], relief='flat', bd=2)
                e.pack(side='left', padx=(0,3))
                e.bind('<Return>',   lambda ev,an=an,i=i,s=sv,w=wh:
                       self._apply_lim(an,i,w,s))
                e.bind('<FocusOut>', lambda ev,an=an,i=i,s=sv,w=wh:
                       self._apply_lim(an,i,w,s))

    def _refresh_jcfg(self):
        sel = self._jcfg_var.get()
        for n,frm in self._jcfg_frames.items():
            if n==sel: frm.pack(fill='x')
            else:      frm.pack_forget()

    def _flip_dir(self, an, i, dv, btn):
        arm = self._arm_l if an=='L' else self._arm_r
        arm[i][1] *= -1; d = arm[i][1]
        dv.set(f'{d:+d}')
        btn.config(bg=C['dir_p'] if d>0 else C['dir_n'])
        self._schedule_save()

    def _apply_lim(self, an, i, wh, sv):
        arm = self._arm_l if an=='L' else self._arm_r
        col = 4 if wh=='lo' else 5
        try:    arm[i][col] = math.radians(float(sv.get()))
        except: sv.set(f'{math.degrees(arm[i][col]):.1f}')
        self._schedule_save()

    # ═══════════════════════════════════════════════════════════════════════════
    # 03  MOTOR MODE  —  POS / MULTI-TURN  +  tick limits  +  EPROM write
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_motor_mode(self, p, pk):
        hdr = _ph_header(p, 3, 'MOTOR MODE', C['ph3'],
                         extra='EPROM write — persistent')
        f = tk.Frame(p, bg=C['panel']); f.pack(fill='x', pady=(2,4), **pk)

        tab_row = tk.Frame(f, bg=C['panel']); tab_row.pack(fill='x', pady=(0,4))
        self._mmode_var = tk.StringVar(value='L')
        _arm_tabs(tab_row, self._mmode_var, self._refresh_mmode)

        self._mmode_frames = {}
        if not hasattr(self,'_mode_btn_vars'): self._mode_btn_vars={}
        if not hasattr(self,'_tick_vars'):     self._tick_vars={}
        if not hasattr(self,'_mt_gear_vars'):  self._mt_gear_vars={}
        for an,ad,col in [('L',self._arm_l,C['la_col']),
                           ('R',self._arm_r,C['ra_col'])]:
            frm = tk.Frame(f, bg=C['panel'])
            self._mmode_frames[an] = frm
            self._build_mmode_rows(frm, an, ad, col)

        # Apply button
        abf = tk.Frame(f, bg=C['panel']); abf.pack(fill='x', pady=(6,0))
        tk.Button(abf, text='APPLY MODES TO HW',
                  font=FONT_XS, bg=C['btn_apply'], fg=C['ph3'],
                  relief='flat', padx=10, pady=4, cursor='hand2',
                  command=self._apply_all_modes
                  ).pack(side='left', padx=(0,8))
        tk.Label(abf, text='Overwrites EPROM  (power-cycle safe)',
                 font=FONT_XS, bg=C['panel'], fg=C['dim']).pack(side='left')

        self._refresh_mmode()

    def _build_mmode_rows(self, parent, an, arm_def, col):
        self._mode_btn_vars[an]=[]
        self._tick_vars[an]=[]
        self._mt_gear_vars[an]=[]
        # Column labels
        ch = tk.Frame(parent, bg=C['panel']); ch.pack(fill='x', pady=(0,2))
        for txt,w in [('ID',5),('MODE',7),('LO ticks',9),('HI ticks',9),('Gear×',5)]:
            tk.Label(ch, text=txt, font=FONT_XS,
                     bg=C['panel'], fg=C['dim'], width=w, anchor='w'
                     ).pack(side='left', padx=(0,2))
        for i,row in enumerate(arm_def):
            sid=row[0]; mode=row[6]; tlo=row[7]; thi=row[8]
            mt_g = row[9] if len(row)>9 else 1
            r = tk.Frame(parent, bg=C['card']); r.pack(fill='x', pady=1)
            tk.Label(r, text=f'ID{sid}', font=FONT_XS,
                     bg=C['card'], fg=col, width=5, anchor='w'
                     ).pack(side='left', padx=(4,2))
            mv = tk.StringVar(value='POS' if mode==0 else 'MULTI')
            self._mode_btn_vars[an].append(mv)
            mbtn = tk.Button(r, textvariable=mv, font=FONT_XS,
                             bg=C['mode_pos'] if mode==0 else C['mode_multi'],
                             fg=C['bright'], width=6, relief='flat',
                             pady=1, cursor='hand2')
            mbtn.config(command=lambda b=mbtn,mv=mv,an=an,i=i:
                        self._toggle_mode(an,i,mv,b))
            mbtn.pack(side='left', padx=(0,4))
            tlo_sv = tk.StringVar(value=str(tlo))
            thi_sv = tk.StringVar(value=str(thi))
            self._tick_vars[an].append((tlo_sv,thi_sv))
            for sv,wh in [(tlo_sv,'tlo'),(thi_sv,'thi')]:
                e = tk.Entry(r, textvariable=sv, width=9, font=FONT_XS,
                             bg=C['border'], fg=C['warn'],
                             insertbackground=C['ph3'], relief='flat', bd=2)
                e.pack(side='left', padx=(0,2))
                e.bind('<Return>',   lambda ev,an=an,i=i,s=sv,w=wh:
                       self._apply_ticks(an,i,w,s))
                e.bind('<FocusOut>', lambda ev,an=an,i=i,s=sv,w=wh:
                       self._apply_ticks(an,i,w,s))
            gv = tk.IntVar(value=mt_g)
            self._mt_gear_vars[an].append(gv)
            sp = tk.Spinbox(r, textvariable=gv, from_=1, to=100, increment=1,
                            width=4, font=FONT_XS,
                            bg='#0e1a10', fg='#70a870',
                            buttonbackground=C['btn_sbtn'], relief='flat')
            sp.pack(side='left', padx=(2,4))
            sp.bind('<Return>',   lambda ev,an=an,i=i: self._apply_mt_gear(an,i))
            sp.bind('<FocusOut>', lambda ev,an=an,i=i: self._apply_mt_gear(an,i))
            gv.trace_add('write', lambda *_,an=an,i=i: self._apply_mt_gear(an,i))

    def _refresh_mmode(self):
        sel = self._mmode_var.get()
        for n,frm in self._mmode_frames.items():
            if n==sel: frm.pack(fill='x')
            else:      frm.pack_forget()

    def _get_mt_gear(self, an, i):
        try:
            if hasattr(self,'_mt_gear_vars') and an in self._mt_gear_vars:
                return max(1, int(self._mt_gear_vars[an][i].get()))
        except Exception: pass
        arm = self._arm_l if an=='L' else self._arm_r
        return arm[i][9] if len(arm[i])>9 else 1

    def _apply_mt_gear(self, an, i):
        arm = self._arm_l if an=='L' else self._arm_r
        g = self._get_mt_gear(an, i)
        while len(arm[i]) <= 9: arm[i].append(1)
        arm[i][9] = g
        if arm[i][6] == 3:   # already MULTI — recalc immediately
            lo, hi = -(4096*g), (4096*g)
            arm[i][7]=lo; arm[i][8]=hi
            if an in self._tick_vars:
                self._tick_vars[an][i][0].set(str(lo))
                self._tick_vars[an][i][1].set(str(hi))
            self._update_test_slider_range(an, i)
        self._schedule_save()

    def _toggle_mode(self, an, i, mv, btn):
        arm = self._arm_l if an=='L' else self._arm_r
        new_mode = 3 if arm[i][6]==0 else 0
        arm[i][6] = new_mode
        if new_mode == 3:
            mv.set('MULTI'); btn.config(bg=C['mode_multi'])
            g = self._get_mt_gear(an, i)
            lo, hi = -(4096*g), (4096*g)
            arm[i][7]=lo; arm[i][8]=hi
            if an in self._tick_vars:
                self._tick_vars[an][i][0].set(str(lo))
                self._tick_vars[an][i][1].set(str(hi))
        else:
            mv.set('POS'); btn.config(bg=C['mode_pos'])
            arm[i][7]=0; arm[i][8]=4095
            if an in self._tick_vars:
                self._tick_vars[an][i][0].set('0')
                self._tick_vars[an][i][1].set('4095')
        self._update_test_slider_range(an, i)
        self._schedule_save()

    def _apply_ticks(self, an, i, wh, sv):
        arm = self._arm_l if an=='L' else self._arm_r
        col = 7 if wh=='tlo' else 8
        try:    arm[i][col] = int(float(sv.get()))
        except: sv.set(str(arm[i][col]))
        self._update_test_slider_range(an, i)
        self._schedule_save()

    def _apply_all_modes(self):
        if not self.hw.ok:
            messagebox.showwarning('Not Connected','Connect to hardware first.')
            return
        lines = []
        for an,ad in [('L',self._arm_l),('R',self._arm_r)]:
            for row in ad:
                tag = 'MULTI-TURN (MODE 3)' if row[6]==3 else 'POSITION  (MODE 0)'
                lines.append(f'  ID{row[0]:2d}  →  {tag}')
        if not messagebox.askyesno('Confirm EPROM Write',
            'Write MODE register to each servo EPROM.\n'
            'Setting is PERMANENT until re-applied.\n\n'
            + '\n'.join(lines) + '\n\nProceed?'): return
        errs = []
        for ad in (self._arm_l, self._arm_r):
            for row in ad:
                try:
                    if row[6]==3: self.hw.set_multi_turn_mode(row[0])
                    else:         self.hw.set_position_mode(row[0])
                    time.sleep(0.05)
                except Exception as e:
                    errs.append(f'ID{row[0]}: {e}')
        if errs: messagebox.showerror('EPROM Errors', '\n'.join(errs))
        else:    messagebox.showinfo('Done', 'All modes applied successfully.')

    # ═══════════════════════════════════════════════════════════════════════════
    # 04  MOTOR TEST  —  per-joint sliders  +  live position feedback
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_motor_test(self, p, pk):
        hdr = _ph_header(p, 4, 'MOTOR TEST', C['ph4'],
                         extra='slide to move  |  SYNC reads hw pos')
        f = tk.Frame(p, bg=C['panel']); f.pack(fill='x', pady=(2,4), **pk)

        # Controls row
        cf = tk.Frame(f, bg=C['panel']); cf.pack(fill='x', pady=(0,4))
        tk.Button(cf, text='READ ALL', font=FONT_XS,
                  bg=C['btn_read'], fg=C['ok'],
                  relief='flat', padx=8, pady=3, cursor='hand2',
                  command=self._read_all_now).pack(side='left', padx=(0,6))
        self._auto_var = tk.BooleanVar(value=False)
        tk.Checkbutton(cf, text='AUTO', variable=self._auto_var,
                       font=FONT_XS, bg=C['panel'], fg=C['dim'],
                       selectcolor=C['border'], activebackground=C['panel'],
                       command=self._toggle_auto).pack(side='left', padx=(0,10))
        tk.Label(cf, text='SPD', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(side='left')
        self._test_spd = tk.IntVar(value=200)
        tk.Spinbox(cf, textvariable=self._test_spd,
                   from_=50, to=2000, increment=50, width=5,
                   font=FONT_XS, bg=C['border'], fg=C['active'],
                   buttonbackground=C['btn_sbtn'], relief='flat'
                   ).pack(side='left', padx=(3,8))
        tk.Label(cf, text='ACC', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(side='left')
        self._test_acc = tk.IntVar(value=20)
        tk.Spinbox(cf, textvariable=self._test_acc,
                   from_=5, to=200, increment=5, width=4,
                   font=FONT_XS, bg=C['border'], fg=C['active'],
                   buttonbackground=C['btn_sbtn'], relief='flat'
                   ).pack(side='left', padx=(3,0))

        # Arm sub-tab
        tab_row = tk.Frame(f, bg=C['panel']); tab_row.pack(fill='x', pady=(0,4))
        self._fb_arm_var = tk.StringVar(value='L')
        _arm_tabs(tab_row, self._fb_arm_var, self._refresh_fb_tab)
        tk.Label(tab_row, text='  ID    MODE   POSITION   ST  [SYNC]',
                 font=FONT_XS, bg=C['panel'], fg=C['dim']
                 ).pack(side='left', padx=(6,0))

        self._fb_frames    = {}
        self._fb_rows      = {}   # sid → (mode_v, pos_v, stat_v)
        self._test_scales  = {}   # an  → [Scale, ...]
        self._test_sl_vars = {}   # an  → [DoubleVar, ...]
        self._test_cur_vars = {}  # an  → [StringVar, ...]  current tick display
        self._rng_label_vars = {}

        for an,ad,col in [('L',self._arm_l,C['la_col']),
                           ('R',self._arm_r,C['ra_col'])]:
            frm = tk.Frame(f, bg=C['panel'])
            self._fb_frames[an]     = frm
            self._test_scales[an]   = []
            self._test_sl_vars[an]  = []
            self._test_cur_vars[an] = []
            self._build_test_rows(frm, an, ad, col)

        self._refresh_fb_tab()

    def _build_test_rows(self, parent, an, arm_def, col):
        for i,row in enumerate(arm_def):
            sid=row[0]; mode=row[6]; tlo=row[7]; thi=row[8]

            # Info row
            info = tk.Frame(parent, bg=C['card']); info.pack(fill='x', pady=(3,0))
            tk.Label(info, text=f'ID{sid}', font=FONT_XS,
                     bg=C['card'], fg=col, width=5, anchor='w'
                     ).pack(side='left', padx=(4,2))
            mode_v = tk.StringVar(value='POS' if mode==0 else 'MULTI')
            tk.Label(info, textvariable=mode_v, font=FONT_XS,
                     bg=C['card'], fg=C['ph3'], width=6, anchor='w'
                     ).pack(side='left', padx=(0,2))
            pos_v  = tk.StringVar(value='——')
            tk.Label(info, textvariable=pos_v, font=FONT_MONO,
                     bg=C['card'], fg=C['active'], width=8, anchor='e'
                     ).pack(side='left', padx=(0,4))
            stat_v = tk.StringVar(value='——')
            tk.Label(info, textvariable=stat_v, font=FONT_XS,
                     bg=C['card'], fg=C['dim'], width=6, anchor='w'
                     ).pack(side='left', padx=(0,4))
            tk.Button(info, text='SYNC', font=FONT_XS,
                      bg=C['btn_sbtn'], fg=C['ok'],
                      relief='flat', padx=4, pady=1, cursor='hand2',
                      command=lambda an=an,i=i,pv=pos_v:
                      self._sync_slider(an,i,pv)).pack(side='left', padx=(0,4))
            self._fb_rows[sid] = (mode_v, pos_v, stat_v)

            # Slider row — DISABLE button is the first element, large and coloured
            sl_fr = tk.Frame(parent, bg=C['card']); sl_fr.pack(fill='x', pady=(0,1))

            dis_v = tk.BooleanVar(value=False)
            self._disabled[(an, i)] = dis_v
            dis_btn = tk.Button(sl_fr, text='ACT', font=('Consolas',10,'bold'),
                                bg='#1a5c1a', fg='#70ff70', width=5,
                                relief='raised', padx=4, pady=3, cursor='hand2',
                                borderwidth=2, activebackground='#2a8c2a',
                                activeforeground='#ffffff')
            dis_btn.pack(side='left', padx=(4,6))
            def _toggle_dis(an=an, i=i, v=dis_v, btn=dis_btn):
                v.set(not v.get())
                arm = self._arm_l if an=='L' else self._arm_r
                sid_ = arm[i][0]
                if v.get():
                    btn.config(text='LOCK', bg='#7a1a1a', fg='#ff7070',
                               activebackground='#aa2a2a')
                    if an in self._test_scales and i < len(self._test_scales[an]):
                        self._test_scales[an][i].config(state='disabled',
                            troughcolor='#252525')
                else:
                    btn.config(text='ACT', bg='#1a5c1a', fg='#70ff70',
                               activebackground='#2a8c2a')
                    if an in self._test_scales and i < len(self._test_scales[an]):
                        self._test_scales[an][i].config(state='normal',
                            troughcolor=C['sl_test'])
            dis_btn.config(command=_toggle_dis)

            sl_var = tk.DoubleVar(value=2048)
            self._test_sl_vars[an].append(sl_var)
            cur_v = tk.StringVar(value='2048')
            self._test_cur_vars[an].append(cur_v)
            tk.Label(sl_fr, textvariable=cur_v, font=FONT_MONO,
                     bg=C['card'], fg=C['warn'], width=7, anchor='e'
                     ).pack(side='left', padx=(0,2))
            sc = tk.Scale(sl_fr, variable=sl_var,
                          from_=tlo, to=thi, orient='horizontal',
                          length=145, showvalue=False, resolution=1,
                          bg=C['card'], troughcolor=C['sl_test'],
                          fg=C['ok'], activebackground=C['ok'],
                          highlightbackground=C['card'], sliderlength=12,
                          command=lambda v,an=an,i=i,cv=cur_v: (
                              cv.set(str(int(float(v)))),
                              self._test_slider_moved(an,i,v)))
            sc.pack(side='left', padx=(0,4))
            self._test_scales[an].append(sc)
            rng_v = tk.StringVar(value=f'{tlo}~{thi}')
            self._rng_label_vars[(an,i)] = rng_v
            tk.Label(sl_fr, textvariable=rng_v, font=FONT_XS,
                     bg=C['card'], fg=C['dim']).pack(side='left')
            tk.Frame(parent, bg=C['border'], height=1).pack(fill='x')

    def _refresh_fb_tab(self):
        sel = self._fb_arm_var.get()
        for n,frm in self._fb_frames.items():
            if n==sel: frm.pack(fill='x')
            else:      frm.pack_forget()

    def _update_test_slider_range(self, an, i):
        arm = self._arm_l if an=='L' else self._arm_r
        tlo, thi = arm[i][7], arm[i][8]
        if an in self._test_scales and i<len(self._test_scales[an]):
            self._test_scales[an][i].config(from_=tlo, to=thi)
            cur = self._test_sl_vars[an][i].get()
            self._test_sl_vars[an][i].set(max(tlo, min(thi, cur)))
        key = (an, i)
        if key in self._rng_label_vars:
            self._rng_label_vars[key].set(f'{tlo}~{thi}')

    def _test_slider_moved(self, an, i, val):
        if self._disabled.get((an, i), tk.BooleanVar()).get(): return
        arm = self._arm_l if an=='L' else self._arm_r
        sid = arm[i][0]
        now = time.monotonic()
        if now - self._last_test_send.get(sid, 0) < 0.05: return
        self._last_test_send[sid] = now
        self.hw.send_ticks(sid, int(float(val)),
                           speed=self._test_spd.get(),
                           acc=self._test_acc.get())

    def _sync_slider(self, an, i, pos_var):
        arm = self._arm_l if an=='L' else self._arm_r
        sid = arm[i][0]
        def _do():
            pos = self.hw.read_pos(sid)
            def _upd():
                if pos is not None:
                    pos_var.set(f'{pos:+6d}')
                    tlo, thi = arm[i][7], arm[i][8]
                    clamped = max(tlo, min(thi, pos))
                    if an in self._test_sl_vars and i<len(self._test_sl_vars[an]):
                        self._test_sl_vars[an][i].set(clamped)
                    if an in self._test_cur_vars and i<len(self._test_cur_vars[an]):
                        self._test_cur_vars[an][i].set(str(clamped))
                else:
                    pos_var.set('NO RSP')
            self._root.after(0, _upd)
        threading.Thread(target=_do, daemon=True).start()

    def _read_all_now(self):
        if not self.hw.ok:
            for mv,pv,sv in self._fb_rows.values():
                mv.set('—'); pv.set('NO HW'); sv.set('OFFLN')
            return
        def _do():
            res = self.hw.read_all_arm(self._arm_l, self._arm_r)
            self._root.after(0, lambda: self._update_fb_display(res))
        threading.Thread(target=_do, daemon=True).start()

    def _update_fb_display(self, results):
        MN = {0:'POS',1:'WHEEL',3:'MULTI',None:'?'}
        for entry in results:
            sid = entry['sid']
            if sid not in self._fb_rows: continue
            mv,pv,sv = self._fb_rows[sid]
            if entry['ok']:
                mv.set(MN.get(entry['mode'],'?'))
                pv.set(f'{entry["pos"]:+6d}')
                sv.set('OK')
            else:
                mv.set('?'); pv.set('——'); sv.set('NO RSP')

    def _toggle_auto(self):
        if self._auto_var.get(): self._schedule_auto_read()
        elif self._fb_job:
            self._root.after_cancel(self._fb_job); self._fb_job=None

    def _schedule_auto_read(self):
        if not self._auto_var.get(): return
        self._read_all_now()
        self._fb_job = self._root.after(100, self._schedule_auto_read)

    # ═══════════════════════════════════════════════════════════════════════════
    # 05  IK CONTROL  —  arm select, EE, joint angles, motion, workspace
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ik_ctrl(self, p, pk):
        _ph_header(p, 5, 'IK CONTROL', C['ph5'],
                   extra='LMB = drag target')
        f = tk.Frame(p, bg=C['panel']); f.pack(fill='x', pady=(2,6), **pk)

        # Arm selector
        sel_row = tk.Frame(f, bg=C['panel']); sel_row.pack(fill='x', pady=(0,4))
        tk.Label(sel_row, text='ACTIVE ARM', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(side='left', padx=(0,8))
        self._arm_var = tk.StringVar(value='L')
        _arm_tabs(sel_row, self._arm_var, self._on_arm_change)
        tk.Label(sel_row, text='  (both always visible)',
                 font=FONT_XS, bg=C['panel'], fg=C['dim']).pack(side='left')

        _divider(f)

        # EE positions
        tk.Label(f, text='END-EFFECTOR  (m)', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(0,3))
        eg = tk.Frame(f, bg=C['panel']); eg.pack(fill='x')
        # headers
        for ci,txt in [(1,'LEFT 21-25'),(2,'RIGHT 11-15')]:
            tk.Label(eg, text=txt, font=FONT_XS,
                     bg=C['panel'],
                     fg=C['la_col'] if ci==1 else C['ra_col'],
                     width=10, anchor='e').grid(row=0, column=ci, padx=(0,6))
        self._xyz_l, self._xyz_r = [], []
        for ri,ax in enumerate(['X','Y','Z']):
            tk.Label(eg, text=ax, font=FONT_XS, bg=C['panel'],
                     fg=C['dim'], width=2).grid(row=ri+1, column=0,
                                                sticky='e', padx=(0,4))
            sv_l = tk.StringVar(value=f'{self._tgt_l[ri]:+.3f}')
            self._xyz_l.append(sv_l)
            tk.Label(eg, textvariable=sv_l, font=FONT_LG,
                     bg=C['panel'], fg=C['la_col'],
                     width=10, anchor='e').grid(row=ri+1, column=1, padx=(0,6))
            sv_r = tk.StringVar(value=f'{self._tgt_r[ri]:+.3f}')
            self._xyz_r.append(sv_r)
            tk.Label(eg, textvariable=sv_r, font=FONT_LG,
                     bg=C['panel'], fg=C['ra_col'],
                     width=10, anchor='e').grid(row=ri+1, column=2)
        self._ik_l_var = tk.StringVar(value='IK L: —')
        self._ik_r_var = tk.StringVar(value='IK R: —')
        tk.Label(f, textvariable=self._ik_l_var, font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(4,0))
        tk.Label(f, textvariable=self._ik_r_var, font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w')

        _divider(f)

        # Joint angles
        tk.Label(f, text='JOINT ANGLES  (°)', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(0,3))
        jg = tk.Frame(f, bg=C['panel']); jg.pack(fill='x')
        self._q_lbls_l, self._q_lbls_r = [], []
        for i in range(5):
            id_l,id_r = self._arm_l[i][0],self._arm_r[i][0]
            tk.Label(jg, text=f'ID{id_l}', font=FONT_XS,
                     bg=C['panel'], fg=C['la_col'], width=5, anchor='w'
                     ).grid(row=i, column=0)
            lv_l = tk.StringVar(value='  0.0'); self._q_lbls_l.append(lv_l)
            tk.Label(jg, textvariable=lv_l, font=FONT_MONO,
                     bg=C['panel'], fg=C['la_col'], width=7, anchor='e'
                     ).grid(row=i, column=1, padx=(0,10))
            tk.Label(jg, text=f'ID{id_r}', font=FONT_XS,
                     bg=C['panel'], fg=C['ra_col'], width=5, anchor='w'
                     ).grid(row=i, column=2)
            lv_r = tk.StringVar(value='  0.0'); self._q_lbls_r.append(lv_r)
            tk.Label(jg, textvariable=lv_r, font=FONT_MONO,
                     bg=C['panel'], fg=C['ra_col'], width=7, anchor='e'
                     ).grid(row=i, column=3)

        _divider(f)

        # Motion params
        tk.Label(f, text='MOTION PARAMS', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(0,3))
        mg = tk.Frame(f, bg=C['panel']); mg.pack(fill='x')
        for ri,(name,attr,init,lo,hi,res) in enumerate([
            ('SPEED','_spd',0.5,0.1,3.0,0.1),
            ('ACC',  '_acc',1.5,0.5,8.0,0.5)]):
            tk.Label(mg, text=name, font=FONT_XS,
                     bg=C['panel'], fg=C['dim'], width=6, anchor='w'
                     ).grid(row=ri, column=0, padx=(0,4))
            sv = tk.DoubleVar(value=init); setattr(self,attr,sv)
            vv = tk.StringVar(value=f'{init:.1f}')
            sv.trace_add('write', lambda *_,v=sv,vv=vv: vv.set(f'{v.get():.1f}'))
            tk.Scale(mg, variable=sv, from_=lo, to=hi, resolution=res,
                     orient='horizontal', length=150, showvalue=False,
                     bg=C['panel'], troughcolor=C['sl_trough'],
                     fg=C['dim'], highlightbackground=C['panel'],
                     sliderlength=14).grid(row=ri, column=1)
            tk.Label(mg, textvariable=vv, font=FONT_XS,
                     bg=C['panel'], fg=C['text'], width=4
                     ).grid(row=ri, column=2, padx=(4,0))

        bf = tk.Frame(f, bg=C['panel']); bf.pack(fill='x', pady=(8,0))
        tk.Button(bf, text='HOME ALL', font=FONT_XS,
                  bg=C['btn_home'], fg=C['err'],
                  relief='flat', padx=12, pady=4, cursor='hand2',
                  command=self._home).pack(side='left', padx=(0,10))
        self._hw_var = tk.BooleanVar(value=True)
        tk.Checkbutton(bf, text='LIVE SEND', variable=self._hw_var,
                       font=FONT_XS, bg=C['panel'], fg=C['dim'],
                       selectcolor=C['border'], activebackground=C['panel'],
                       activeforeground=C['text']).pack(side='left')

        _divider(f)

        # Workspace bounds
        tk.Label(f, text='WORKSPACE BOUNDS  (m)', font=FONT_XS,
                 bg=C['panel'], fg=C['dim']).pack(anchor='w', pady=(0,3))
        wg = tk.Frame(f, bg=C['panel']); wg.pack(fill='x')
        self._ws_vars = {}
        for ri,(ax,lk,hk) in enumerate([('X','x_min','x_max'),
                                         ('Y','y_min','y_max'),
                                         ('Z','z_min','z_max')]):
            tk.Label(wg, text=ax, font=FONT_XS, bg=C['panel'],
                     fg=C['dim'], width=2).grid(row=ri, column=0,
                                                sticky='e', padx=(0,4))
            for ci,key in [(1,lk),(2,hk)]:
                sv = tk.StringVar(value=f'{self._ws[key]:.2f}')
                self._ws_vars[key] = sv
                e = tk.Entry(wg, textvariable=sv, width=7, font=FONT_XS,
                             bg=C['border'], fg=C['ph5'],
                             insertbackground=C['ph5'], relief='flat', bd=2)
                e.grid(row=ri, column=ci, padx=(0,6), pady=1)
                e.bind('<Return>',   lambda ev,k=key,s=sv: self._apply_ws(k,s))
                e.bind('<FocusOut>', lambda ev,k=key,s=sv: self._apply_ws(k,s))
        wr = tk.Frame(f, bg=C['panel']); wr.pack(fill='x', pady=(4,0))
        tk.Button(wr, text='RESET WS', font=FONT_XS,
                  bg=C['btn_sbtn'], fg=C['dim'],
                  relief='flat', padx=8, cursor='hand2',
                  command=self._reset_ws).pack(side='left')
        tk.Label(wr, text='  clamps IK drag target',
                 font=FONT_XS, bg=C['panel'], fg=C['dim']).pack(side='left')

    # ── 3D canvas ─────────────────────────────────────────────────────────────

    def _build_canvas(self, parent):
        # ── big RESET button at top of 3D panel ──
        tk.Button(parent, text='⟳  RESET TO HOME',
                  font=('Consolas', 13, 'bold'),
                  bg='#9a1010', fg='#ffffff',
                  activebackground='#cc1818', activeforeground='#ffffff',
                  relief='flat', padx=20, pady=7, cursor='hand2',
                  borderwidth=0,
                  command=self._home).pack(fill='x', padx=6, pady=(6,2))

        self._fig = Figure(figsize=(7.0,6.5), facecolor='#111820')
        self._ax  = self._fig.add_subplot(111, projection='3d')
        self._ax.set_facecolor('#151f2e')
        canvas = FigureCanvasTkAgg(self._fig, master=parent)
        w = canvas.get_tk_widget(); w.pack(fill='both', expand=True)
        w.bind('<Button-1>',        self._lbtn_press)
        w.bind('<B1-Motion>',       self._lbtn_drag)
        w.bind('<ButtonRelease-1>', self._lbtn_release)
        w.bind('<Button-3>',        self._rbtn_press)
        w.bind('<B3-Motion>',       self._rbtn_drag)
        w.bind('<ButtonRelease-3>', self._rbtn_release)
        w.bind('<MouseWheel>',      self._scroll)
        self._canvas = canvas

    def _lbtn_press(self,e): self._lbtn=(e.x,e.y)
    def _lbtn_release(self,e): self._lbtn=None
    def _rbtn_press(self,e): self._rbtn=(e.x,e.y)
    def _rbtn_release(self,e): self._rbtn=None

    def _lbtn_drag(self, e):
        if self._lbtn is None: return
        lx,ly=self._lbtn; dx,dy=e.x-lx,e.y-ly; self._lbtn=(e.x,e.y)
        θ = math.radians(self._azim)
        φ = math.radians(self._elev)
        s = 0.003 * self._zoom
        # Viewport right vector (horizontal drag direction in world space)
        right = np.array([-math.sin(θ), math.cos(θ), 0.0])
        # Viewport up vector (vertical drag direction, accounting for tilt)
        # = right × view_direction, normalised
        up = np.array([-math.cos(θ)*math.sin(φ),
                       -math.sin(θ)*math.sin(φ),
                        math.cos(φ)])
        # dx → right, dy → down in screen = -up in world
        delta = s * (dx * right - dy * up)
        if self._arm_sel=='L':
            self._tgt_l=np.clip(self._tgt_l+delta,self._ws_lo(),self._ws_hi())
        else:
            self._tgt_r=np.clip(self._tgt_r+delta,self._ws_lo(),self._ws_hi())
        self._solve_and_update()

    def _rbtn_drag(self, e):
        if self._rbtn is None: return
        lx,ly=self._rbtn; dx,dy=e.x-lx,e.y-ly; self._rbtn=(e.x,e.y)
        self._azim -= dx*0.5
        self._elev  = max(-85.0, min(85.0, self._elev+dy*0.3))
        self._ax.view_init(elev=self._elev, azim=self._azim)
        self._canvas.draw_idle()

    def _scroll(self, e):
        self._zoom *= 0.92 if e.delta>0 else 1.09
        self._zoom  = max(0.25, min(5.0, self._zoom))
        self._update_limits(); self._canvas.draw_idle()

    def _ws_lo(self):
        return np.array([self._ws['x_min'],self._ws['y_min'],self._ws['z_min']])
    def _ws_hi(self):
        return np.array([self._ws['x_max'],self._ws['y_max'],self._ws['z_max']])
    def _apply_ws(self,k,sv):
        try: self._ws[k]=float(sv.get())
        except: sv.set(f'{self._ws[k]:.2f}')
        self._schedule_save()
    def _reset_ws(self):
        self._ws={k:v for k,v in self._DEFAULT_WS.items()}
        for k,sv in self._ws_vars.items(): sv.set(f'{self._ws[k]:.2f}')
        self._schedule_save()

    def _on_arm_change(self):
        self._arm_sel=self._arm_var.get(); self._redraw()

    def _toggle_conn(self):
        if self.hw.ok:
            self.hw.disconnect()
            self._cbtn.config(text='CONNECT', bg=C['btn_connect'])
            self._cvar.set('OFFLINE'); self._clbl.config(fg=C['err'])
        else:
            port=self._port.get().strip(); err=self.hw.connect(port)
            if err is None:
                self._cbtn.config(text='DISCONNECT', bg=C['btn_sbtn'])
                self._cvar.set(f'ONLINE  {port}')
                self._clbl.config(fg=C['ok'])
                self.hw.home_arm(self._arm_l)
                self.hw.home_arm(self._arm_r)
            else:
                self._cvar.set('FAILED'); self._clbl.config(fg=C['warn'])
                messagebox.showerror('Connection Error', f'{port}:\n{err}')

    def _get_skip_ids(self, an):
        arm = self._arm_l if an == 'L' else self._arm_r
        return {arm[i][0] for (a,i),v in self._disabled.items()
                if a == an and v.get()}

    def _home(self):
        skip_l = self._get_skip_ids('L')
        skip_r = self._get_skip_ids('R')
        self._q_l = np.zeros(len(self._arm_l))
        self._q_r = np.zeros(len(self._arm_r))
        self._tgt_l = ee_pos(LEFT_CHAIN, self._arm_l, self._q_l).copy()
        self._tgt_r = ee_pos(RIGHT_CHAIN, self._arm_r, self._q_r).copy()
        self._err_l = self._err_r = 0.0
        self._refresh_panel(); self._redraw()
        self.hw.home_arm(self._arm_l, skip_ids=skip_l)
        self.hw.home_arm(self._arm_r, skip_ids=skip_r)

    def _solve_and_update(self):
        sel=self._arm_sel
        if sel=='L':
            q,err=ik_solve(LEFT_CHAIN, self._arm_l,self._tgt_l,self._q_l.copy())
            self._q_l,self._err_l=q,err
            if self.hw.ok and self._hw_var.get():
                self.hw.send_arm(self._arm_l,q,self._spd.get(),self._acc.get(),
                                 skip_ids=self._get_skip_ids('L'))
        else:
            q,err=ik_solve(RIGHT_CHAIN,self._arm_r,self._tgt_r,self._q_r.copy())
            self._q_r,self._err_r=q,err
            if self.hw.ok and self._hw_var.get():
                self.hw.send_arm(self._arm_r,q,self._spd.get(),self._acc.get(),
                                 skip_ids=self._get_skip_ids('R'))
        self._refresh_panel(); self._redraw()

    def _refresh_panel(self):
        for i,sv in enumerate(self._xyz_l): sv.set(f'{float(self._tgt_l[i]):+.3f}')
        for i,sv in enumerate(self._xyz_r): sv.set(f'{float(self._tgt_r[i]):+.3f}')
        cm_l=float(math.sqrt(self._err_l)*100)
        cm_r=float(math.sqrt(self._err_r)*100)
        def tag(v): return 'OK' if v<0.5 else 'WARN' if v<2 else 'FAIL'
        self._ik_l_var.set(f'IK L: {cm_l:.2f} cm  [{tag(cm_l)}]')
        self._ik_r_var.set(f'IK R: {cm_r:.2f} cm  [{tag(cm_r)}]')
        for i,lv in enumerate(self._q_lbls_l):
            lv.set(f'{math.degrees(self._q_l[i]):+.1f}')
        for i,lv in enumerate(self._q_lbls_r):
            lv.set(f'{math.degrees(self._q_r[i]):+.1f}')

    # ── 3D drawing ────────────────────────────────────────────────────────────

    def _update_limits(self):
        z=self._zoom
        self._ax.set_xlim(-0.45*z,0.45*z)
        self._ax.set_ylim(-0.25*z,0.65*z)
        self._ax.set_zlim(-0.65*z,0.35*z)

    def _draw_arm(self, pts, arm_def, arm_col, ee_col, target, is_active):
        ax=self._ax; pos=[p[0] for p in pts]
        aids={row[0] for row in arm_def}
        alpha = 1.0 if is_active else 0.28
        lw    = 3.2 if is_active else 1.8
        for i in range(len(pos)-1):
            p0,p1=pos[i],pos[i+1]
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],
                    color='#05090f', linewidth=8, solid_capstyle='round')
            ax.plot([p0[0],p1[0]],[p0[1],p1[1]],[p0[2],p1[2]],
                    color=arm_col, linewidth=lw, solid_capstyle='round', alpha=alpha)
        for p,label,sid in pts:
            if sid in aids:       c,sz=C['active'],120
            elif sid is not None: c,sz=C['dim'],50
            else:                 c,sz='#12202e',16
            ax.scatter(*p, s=sz, c=c, depthshade=False, zorder=5, alpha=alpha)
        if is_active:
            for p,label,sid in pts:
                if sid in aids:
                    ax.text(p[0]+0.012,p[1]+0.012,p[2]+0.012,
                            str(sid), color='#405060', fontsize=7,
                            fontfamily='monospace')
        ee=pos[-1]
        ax.scatter(*ee, s=220, c=ee_col, marker='*',
                   depthshade=False, zorder=10, alpha=alpha)
        if is_active:
            ax.scatter(*target, s=180, c=C['tg_col'], marker='+',
                       linewidths=2.2, depthshade=False, zorder=10)
            ax.plot([ee[0],target[0]],[ee[1],target[1]],[ee[2],target[2]],
                    color=C['tg_col'], linewidth=1.0,
                    linestyle='--', alpha=0.35)

    def _draw_ws_box(self):
        lo,hi=self._ws_lo(),self._ws_hi()
        edges=[([lo[0],hi[0]],[lo[1],lo[1]],[lo[2],lo[2]]),
               ([lo[0],hi[0]],[hi[1],hi[1]],[lo[2],lo[2]]),
               ([lo[0],hi[0]],[lo[1],lo[1]],[hi[2],hi[2]]),
               ([lo[0],hi[0]],[hi[1],hi[1]],[hi[2],hi[2]]),
               ([lo[0],lo[0]],[lo[1],hi[1]],[lo[2],lo[2]]),
               ([hi[0],hi[0]],[lo[1],hi[1]],[lo[2],lo[2]]),
               ([lo[0],lo[0]],[lo[1],hi[1]],[hi[2],hi[2]]),
               ([hi[0],hi[0]],[lo[1],hi[1]],[hi[2],hi[2]]),
               ([lo[0],lo[0]],[lo[1],lo[1]],[lo[2],hi[2]]),
               ([hi[0],hi[0]],[lo[1],lo[1]],[lo[2],hi[2]]),
               ([lo[0],lo[0]],[hi[1],hi[1]],[lo[2],hi[2]]),
               ([hi[0],hi[0]],[hi[1],hi[1]],[lo[2],hi[2]])]
        for xs,ys,zs in edges:
            self._ax.plot(xs,ys,zs, color='#14283a',
                          linewidth=0.6, linestyle='--', alpha=0.5)

    def _redraw(self):
        ax=self._ax; ax.cla(); ax.set_facecolor('#151f2e')
        sel=self._arm_sel
        pts_l=fk_chain(LEFT_CHAIN, self._arm_l,self._q_l)
        pts_r=fk_chain(RIGHT_CHAIN,self._arm_r,self._q_r)
        self._draw_arm(pts_l,self._arm_l,C['la_col'],'#7090d0',
                       self._tgt_l, sel=='L')
        self._draw_arm(pts_r,self._arm_r,C['ra_col'],'#d08050',
                       self._tgt_r, sel=='R')
        self._draw_ws_box()
        sc=0.06*self._zoom
        for vec,c in [([sc,0,0],'#904040'),([0,sc,0],'#407040'),([0,0,sc],'#405080')]:
            ax.quiver(0,0,0,*vec,color=c,linewidth=1.2,arrow_length_ratio=0.35)
        sel_txt='LEFT' if sel=='L' else 'RIGHT'
        cm_l=float(math.sqrt(self._err_l)*100)
        cm_r=float(math.sqrt(self._err_r)*100)
        ax.set_title(
            f'[ {sel_txt} ACTIVE ]    IK L {cm_l:.2f} cm    R {cm_r:.2f} cm',
            color=C['dim'], fontsize=9, pad=8, fontfamily='monospace')
        ax.set_xlabel('X',color=C['dim'],labelpad=1,fontsize=8)
        ax.set_ylabel('Y',color=C['dim'],labelpad=1,fontsize=8)
        ax.set_zlabel('Z',color=C['dim'],labelpad=1,fontsize=8)
        ax.tick_params(colors=C['dim'],labelsize=6)
        ax.xaxis.pane.fill=ax.yaxis.pane.fill=ax.zaxis.pane.fill=False
        for pane in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            pane.set_edgecolor('#243450')
        ax.grid(True,color='#1e3050',linewidth=0.6)
        self._update_limits()
        ax.view_init(elev=self._elev,azim=self._azim)
        self._canvas.draw_idle()

    # ── Run ───────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._save_config()   # final save on exit
        if self._fb_job: self._root.after_cancel(self._fb_job)
        self.hw.disconnect()
        self._root.destroy()

    def run(self):
        self._root.mainloop()


if __name__ == '__main__':
    app = App()
    app.run()
