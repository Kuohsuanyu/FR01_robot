#!/usr/bin/env python3
"""Q-BOT motion replay / imitator.

Reads a recorded motion (npz or csv with joint angles in radians) and
drives the same full-body ghost used by ``qbot_ik_gui.py``.  When "送馬達"
is on, the 8 arm joints (sid 10-13, 20-23) are converted to tick using
the existing ``qbot_arm_calibration.json`` and streamed to the RPi arm
agent via ``RemoteBus``.  Wrist joints (sid 14, 24) are not in the
recording — left at 0.

Nothing here modifies ``qbot_ik_gui.py``; classes are imported from it
so calibration + MJCF stay in sync.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog

import mujoco
import numpy as np
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))         # upper_body_control/
from qbot_ik_gui import (
    World, RemoteBus, ARMS, MJ_SIGN,
    MJCF_PATH, REMOTE_DEFAULT_HOST, RENDER_W, RENDER_H, C,
)

CAL_PATH = os.path.join(os.path.dirname(HERE), "qbot_arm_calibration.json")

DEFAULT_MOTION = os.path.expanduser("~/下載/health_full_qbot_motion.npz")

# Recording column names → (arm key, joint index 0-4 in ARMS[arm]).
# Only 4 per arm — sid 14/24 wrist is not in the recording.
JOINT_MAP = {
    "ub_right_shoulder":       ("R", 0),
    "ub_right_lateral_raise":  ("R", 1),
    "ub_right_arm_twist":      ("R", 2),
    "ub_right_elbow":          ("R", 3),
    "ub_left_shoulder":        ("L", 0),
    "ub_left_lateral_raise":   ("L", 1),
    "ub_left_arm_twist":       ("L", 2),
    "ub_left_elbow":           ("L", 3),
}

# Per-joint direction override for the motor path.  +1 = motion_min maps
# to cal tick_lo, motion_max maps to cal tick_hi; -1 = swap.  Flip if the
# real motor turns the opposite way from what the ghost shows for a joint.
# (Only affects the motor path.  Ghost always follows the motion directly.)
MOTOR_DIR = {
    "R": [+1, +1, +1, +1, +1],
    "L": [+1, +1, +1, +1, +1],
}


def load_motion(path: str) -> dict:
    """Return {'names': [str], 'q': (N,J) float32, 'fps': float,
                'base': (N,7) or None}.

    Accepts npz (preferred, keeps float32 + base_pose) or csv with
    ``time_s`` header column."""
    if path.lower().endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        return {
            "names": [str(n) for n in d["joint_names"]],
            "q": np.asarray(d["joint_angles"], dtype=np.float32),
            "fps": float(d["fps"]),
            "base": (np.asarray(d["base_pose"], dtype=np.float32)
                     if "base_pose" in d.files else None),
        }
    # csv fallback
    import csv
    with open(path) as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [r for r in rd]
    if header[0] != "time_s":
        raise ValueError(f"csv header[0] must be 'time_s', got {header[0]!r}")
    names = header[1:]
    times = np.array([float(r[0]) for r in rows], dtype=np.float64)
    q = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    fps = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    return {"names": names, "q": q, "fps": fps, "base": None}


class MotionApp:
    def __init__(self):
        self.world = World(MJCF_PATH)
        self.bus = RemoteBus()
        self.cal = self._load_cal()

        self.root = tk.Tk()
        self.root.title("Q-BOT motion imitator")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1180x780")

        self.motion: dict | None = None
        self.frame_idx = 0
        self.playing = False
        # Motor catch-up bookkeeping
        self._last_ticks: dict[int, int] = {}      # sid → last commanded tick
        self._settle_until = 0.0                    # monotonic deadline
        self._missed_frames: list[int] = []         # frames motor couldn't reach in time
        # Constants
        self._CATCHUP_TOL = 40                      # ticks (~3.5° with STS3215)
        self._CATCHUP_MAX_S = 2.0                   # max wait per frame before giving up
        self.speed = 1.0
        self.loop = True
        self.apply_base = False
        self.send_motors = False
        self._last_tick_t = time.monotonic()

        # Cache MJCF joint qpos addresses for every recorded joint we know
        self._name_to_qpos: dict[str, int] = {}

        self._build_ui()
        # Try default motion file if it exists
        if os.path.exists(DEFAULT_MOTION):
            self._load_file(DEFAULT_MOTION)

        self.root.after(33, self._render_tick)
        self.root.after(15, self._playback_tick)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # left = ghost, right = controls
        wrap = tk.Frame(self.root, bg=C["bg"]); wrap.pack(fill="both", expand=True)
        gl = tk.Frame(wrap, bg=C["bg"]); gl.pack(side="left", fill="both", expand=True)
        rc = tk.Frame(wrap, bg=C["bg"], width=380)
        rc.pack(side="right", fill="y", padx=(4, 8), pady=8)
        rc.pack_propagate(False)

        self.canvas = tk.Canvas(gl, width=RENDER_W, height=RENDER_H,
                                bg="#000", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas_img_id = None
        self._tk_img = None

        # File row
        fr = tk.LabelFrame(rc, text=" Motion file ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        fr.pack(fill="x", pady=(0, 4))
        self.file_var = tk.StringVar(value="(none)")
        tk.Label(fr, textvariable=self.file_var, bg=C["panel"], fg=C["text"],
                 anchor="w", font=("DejaVu Sans Mono", 9), wraplength=340,
                 justify="left").pack(fill="x", padx=6, pady=(4, 0))
        tk.Button(fr, text="Open .npz / .csv",
                  bg=C["btn"], fg="white",
                  command=self._pick_file).pack(fill="x", padx=6, pady=6)
        self.meta_var = tk.StringVar(value="")
        tk.Label(fr, textvariable=self.meta_var, bg=C["panel"],
                 fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=6, pady=(0, 6))

        # Transport
        tp = tk.LabelFrame(rc, text=" Playback ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        tp.pack(fill="x", pady=4)
        row = tk.Frame(tp, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        self.play_btn = tk.Button(row, text="▶ Play",
                                  bg="#2e7d32", fg="white", width=10,
                                  command=self._toggle_play)
        self.play_btn.pack(side="left", padx=2)
        tk.Button(row, text="■ Stop", bg=C["btn_danger"], fg="white", width=8,
                  command=self._stop).pack(side="left", padx=2)
        tk.Button(row, text="|◀ 0", bg=C["btn"], fg="white", width=6,
                  command=lambda: self._seek(0)).pack(side="left", padx=2)

        # Frame slider
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = tk.Scale(tp, from_=0, to=0, orient="horizontal",
                                    variable=self.frame_var, resolution=1,
                                    bg=C["panel"], fg=C["text"],
                                    highlightthickness=0, showvalue=1,
                                    troughcolor=C["card"],
                                    command=self._on_scrub)
        self.frame_scale.pack(fill="x", padx=6, pady=(0, 4))
        self.frame_lbl = tk.StringVar(value="frame 0 / 0   t=0.00 s")
        tk.Label(tp, textvariable=self.frame_lbl, bg=C["panel"],
                 fg=C["bright"], anchor="w",
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=6)

        # Speed + loop + base-pose toggle
        row = tk.Frame(tp, bg=C["panel"]); row.pack(fill="x", padx=6, pady=4)
        tk.Label(row, text="speed", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(row, from_=0.1, to=2.0, resolution=0.1, orient="horizontal",
                 variable=self.speed_var, showvalue=1, length=140,
                 bg=C["panel"], fg=C["text"],
                 troughcolor=C["card"], highlightthickness=0,
                 command=lambda v: setattr(self, "speed", float(v))
                 ).pack(side="left", padx=6)
        self.loop_var = tk.BooleanVar(value=True)
        tk.Checkbutton(row, text="loop",
                       variable=self.loop_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["card"], activebackground=C["panel"],
                       command=lambda: setattr(self, "loop",
                                               self.loop_var.get())
                       ).pack(side="left", padx=6)
        self.base_var = tk.BooleanVar(value=False)
        tk.Checkbutton(tp, text="apply base pose (floating base moves in world)",
                       variable=self.base_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["card"], activebackground=C["panel"],
                       command=lambda: setattr(self, "apply_base",
                                               self.base_var.get())
                       ).pack(anchor="w", padx=6, pady=(0, 6))

        # Motor sending
        mm = tk.LabelFrame(rc, text=" 送真實馬達 (arms only) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        mm.pack(fill="x", pady=4)
        row = tk.Frame(mm, bg=C["panel"]); row.pack(fill="x", padx=6, pady=4)
        tk.Label(row, text="arm agent:", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        self.host_var = tk.StringVar(value=REMOTE_DEFAULT_HOST)
        tk.Entry(row, textvariable=self.host_var, width=18,
                 bg=C["card"], fg=C["text"]).pack(side="left", padx=4)
        self.conn_btn = tk.Button(row, text="Connect",
                                   bg=C["btn"], fg="white", width=9,
                                   command=self._toggle_connect)
        self.conn_btn.pack(side="left", padx=2)
        self.bus_status_var = tk.StringVar(value="disconnected")
        tk.Label(mm, textvariable=self.bus_status_var, bg=C["panel"],
                 fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)).pack(fill="x", padx=6)

        self.send_var = tk.BooleanVar(value=False)
        tk.Checkbutton(mm, text="送馬達  (未連 / 無校準表 → 自動忽略)",
                       variable=self.send_var, bg=C["panel"], fg=C["text"],
                       selectcolor=C["card"], activebackground=C["panel"],
                       command=lambda: setattr(self, "send_motors",
                                               self.send_var.get())
                       ).pack(anchor="w", padx=6, pady=4)
        # Torque + acc + speed for arm motors
        row = tk.Frame(mm, bg=C["panel"]); row.pack(fill="x", padx=6, pady=4)
        tk.Label(row, text="speed", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        # Conservative defaults for first-run: half of what feels normal.
        # STS3215 typical: speed 3000-4000, acc 50-100 for arm motion.
        # We start at 750 / 25 so the first playback is visibly slow and
        # gentle — user can bump up once they trust the motion.
        self.motor_speed_var = tk.IntVar(value=750)
        tk.Spinbox(row, from_=100, to=3000, increment=100, width=6,
                   textvariable=self.motor_speed_var,
                   bg=C["card"], fg=C["text"]).pack(side="left", padx=4)
        tk.Label(row, text="acc", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        self.motor_acc_var = tk.IntVar(value=25)
        tk.Spinbox(row, from_=1, to=254, increment=5, width=5,
                   textvariable=self.motor_acc_var,
                   bg=C["card"], fg=C["text"]).pack(side="left", padx=4)
        tk.Button(row, text="Torque ON",
                  bg="#2e7d32", fg="white", width=10,
                  command=lambda: self._torque_all(True)
                  ).pack(side="right", padx=2)
        tk.Button(row, text="Torque OFF",
                  bg=C["btn_danger"], fg="white", width=10,
                  command=lambda: self._torque_all(False)
                  ).pack(side="right", padx=2)
        # ── Safety toggles: which arm + wait for motor catch-up ────────
        toggles = tk.Frame(mm, bg=C["panel"]); toggles.pack(fill="x", padx=6, pady=(4, 0))
        tk.Label(toggles, text="arms:", bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9)).pack(side="left")
        self.arm_filter_var = tk.StringVar(value="R")
        for label, val in (("both", "both"), ("R only", "R"), ("L only", "L")):
            tk.Radiobutton(toggles, text=label,
                           variable=self.arm_filter_var, value=val,
                           bg=C["panel"], fg=C["text"],
                           selectcolor=C["card"],
                           activebackground=C["panel"],
                           font=("DejaVu Sans Mono", 9)
                           ).pack(side="left", padx=2)
        self.wait_motor_var = tk.BooleanVar(value=True)
        tk.Checkbutton(toggles, text="等馬達跟上",
                       variable=self.wait_motor_var,
                       bg=C["panel"], fg=C["ok"],
                       selectcolor=C["card"],
                       activebackground=C["panel"],
                       font=("DejaVu Sans Mono", 9, "bold")
                       ).pack(side="left", padx=8)

        # ── Pre-flight tools: sample-frame test + collision scan ────────
        tools = tk.Frame(mm, bg=C["panel"]); tools.pack(fill="x", padx=6, pady=(6, 4))
        tk.Button(tools, text="▶ Sample test (5 frames)",
                  bg="#c78500", fg="white", width=22,
                  font=("DejaVu Sans Mono", 9, "bold"),
                  command=self._sample_test
                  ).pack(side="left", padx=2)
        tk.Button(tools, text="⚠ 檢查碰撞",
                  bg=C["btn"], fg="white", width=14,
                  font=("DejaVu Sans Mono", 9, "bold"),
                  command=self._check_collisions
                  ).pack(side="left", padx=2)
        self.tools_var = tk.StringVar(value="")
        tk.Label(mm, textvariable=self.tools_var, bg=C["panel"], fg=C["text"],
                 anchor="w", justify="left", wraplength=340,
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=6, pady=(0, 4))

        # Diag / joint values readout
        di = tk.LabelFrame(rc, text=" 當前幀關節 (rad) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        di.pack(fill="both", expand=True, pady=4)
        self.joint_txt = tk.Text(di, height=14, width=44, bg=C["card"],
                                 fg=C["text"], font=("DejaVu Sans Mono", 9),
                                 relief="flat", borderwidth=0)
        self.joint_txt.pack(fill="both", expand=True, padx=6, pady=6)

    # ── file handling ───────────────────────────────────────────────────────
    def _pick_file(self):
        p = filedialog.askopenfilename(
            initialdir=os.path.expanduser("~/下載"),
            title="Choose motion file",
            filetypes=[("motion", "*.npz *.csv"), ("all", "*.*")])
        if p:
            self._load_file(p)

    def _load_file(self, path: str):
        try:
            m = load_motion(path)
        except Exception as e:
            self.file_var.set(f"load err: {e}")
            return
        self.motion = m
        self._name_to_qpos = {n: self.world.jqpos[n]
                              for n in m["names"] if n in self.world.jqpos}
        n_missing = len(m["names"]) - len(self._name_to_qpos)
        n_frames = int(m["q"].shape[0])
        # Retarget: fit each joint's motion range inside the MJCF joint
        # limits with a per-joint affine (scale, shift).  Rewrites
        # `motion["q"]` in-place so both ghost and motor path see the
        # feasible values.
        retarget_info = self._retarget_to_mjcf()
        # Also build per-joint motor mapping: motion range → cal tick range.
        # This decouples the motor path from the cal's absolute q values
        # (which can be offset / partial per operator's slider-based
        # calibration) — we just use cal tick_lo/tick_hi as endpoints.
        self._motor_map = self._build_motor_maps()
        self.file_var.set(os.path.basename(path))
        clip_note = f"  [{retarget_info}]" if retarget_info else ""
        self.meta_var.set(
            f"{n_frames} frames  @ {m['fps']:.1f} fps   "
            f"({n_frames / m['fps']:.1f} s)   "
            f"joints matched: {len(self._name_to_qpos)}/{len(m['names'])}"
            + (f"  {n_missing} missing" if n_missing else "")
            + clip_note)
        self.frame_scale.config(to=max(0, n_frames - 1))
        self._seek(0)

    def _build_motor_maps(self) -> dict:
        """For each joint listed in JOINT_MAP, precompute (tick_lo, tick_hi)
        as absolute endpoints derived from the cal file, and remember the
        motion's own range for that joint.  At send time we do a single
        affine  tick = tick_lo + (q - m_lo) * (tick_hi - tick_lo) / m_span.

        This bypasses the cal's absolute q_lo/q_hi values (which may be
        offset or partial in the operator's slider-frame captures) and
        simply says "motion_min → one cal endpoint tick, motion_max → the
        other".  Direction can be flipped per joint via MOTOR_DIR."""
        maps: dict[tuple[str, int], tuple[int, int, float, float]] = {}
        if not self.cal or not self.motion:
            return maps
        m = self.motion
        for name, (arm, ji) in JOINT_MAP.items():
            if name not in m["names"]: continue
            col = m["names"].index(name)
            vals = m["q"][:, col]
            m_lo, m_hi = float(vals.min()), float(vals.max())
            if m_hi - m_lo < 1e-6:                    # constant column
                continue
            c = (self.cal.get(arm) or [])
            if ji >= len(c): continue
            p = c[ji]
            # Preserve the cal's implicit direction: cal encodes q_lo/q_hi
            # as pairs with their tick endpoints — if q_lo > q_hi the cal's
            # relationship is inverted (higher q ↔ lower tick).  We want
            # motion's max (high MJCF value) to end up on the same tick as
            # cal's high-q side, so swap tick endpoints when cal is inverted.
            if float(p["q_lo"]) > float(p["q_hi"]):
                t_lo = float(p["tick_hi"])
                t_hi = float(p["tick_lo"])
            else:
                t_lo = float(p["tick_lo"])
                t_hi = float(p["tick_hi"])
            if MOTOR_DIR[arm][ji] < 0:
                t_lo, t_hi = t_hi, t_lo               # user manual override
            maps[(arm, ji)] = (t_lo, t_hi, m_lo, m_hi)
        return maps

    def _retarget_to_mjcf(self) -> str:
        """Per-joint affine fit — for every joint that maps to an MJCF
        joint with a defined range, ensure the recorded motion stays inside
        that range.  Uses ``scale = min(1, mjcf_span / motion_span)`` and
        shifts the motion midpoint to the MJCF midpoint only when the
        motion doesn't already fit (leaves natural clips alone).  Modifies
        ``self.motion['q']`` in place; returns a human-readable summary."""
        import mujoco
        m = self.motion
        q = m["q"]                # (N, J) float32
        model = self.world.model
        jname_to_id = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
                       i for i in range(model.njnt)}
        scaled = []
        for col, name in enumerate(m["names"]):
            jid = jname_to_id.get(name)
            if jid is None: continue
            if not bool(model.jnt_limited[jid]):
                continue              # no MJCF limit → skip
            lo, hi = float(model.jnt_range[jid][0]), \
                     float(model.jnt_range[jid][1])
            col_vals = q[:, col]
            m_lo, m_hi = float(col_vals.min()), float(col_vals.max())
            fits = (m_lo >= lo) and (m_hi <= hi)
            if fits:
                continue              # motion already inside limits
            m_span = max(m_hi - m_lo, 1e-6)
            mjcf_span = hi - lo
            scale = min(1.0, mjcf_span / m_span)
            m_mid = 0.5 * (m_lo + m_hi)
            mjcf_mid = 0.5 * (lo + hi)
            shift = mjcf_mid - scale * m_mid
            new_col = col_vals * scale + shift
            # Safety clamp — kills any residual over-range from float noise
            np.clip(new_col, lo, hi, out=new_col)
            q[:, col] = new_col.astype(q.dtype)
            scaled.append(f"{name}(×{scale:.2f})")
        return ("retargeted: " + ", ".join(scaled)) if scaled else ""

    # ── transport ───────────────────────────────────────────────────────────
    def _toggle_play(self):
        if not self.motion: return
        self.playing = not self.playing
        self.play_btn.config(
            text="■ Pause" if self.playing else "▶ Play",
            bg=(C["btn_warn"] if self.playing else "#2e7d32"))
        self._last_tick_t = time.monotonic()

    def _stop(self):
        self.playing = False
        self.play_btn.config(text="▶ Play", bg="#2e7d32")
        self._seek(0)

    def _seek(self, i: int):
        if not self.motion: return
        n = int(self.motion["q"].shape[0])
        self.frame_idx = max(0, min(n - 1, int(i)))
        self.frame_var.set(self.frame_idx)
        self._apply_frame(self.frame_idx)

    def _on_scrub(self, v):
        if not self.motion: return
        i = int(float(v))
        if i != self.frame_idx:
            self.frame_idx = i
            self._apply_frame(i)

    # ── frame → ghost + motors ──────────────────────────────────────────────
    def _apply_frame(self, i: int):
        m = self.motion
        row = m["q"][i]
        # 1) Recorded angles are already MJCF-native (joint names match the
        #    MJCF exactly).  Write them straight into qpos — no MJ_SIGN.
        #    MJ_SIGN belongs on the slider→MJCF path in qbot_ik_gui, not here.
        for col, name in enumerate(m["names"]):
            adr = self._name_to_qpos.get(name)
            if adr is None: continue
            self.world.data.qpos[adr] = float(row[col])
        # 2) optional: apply base pose (floating_base free joint)
        if self.apply_base and m["base"] is not None:
            bp = m["base"][i]
            adr = 0  # free joint occupies qpos[0:7] as (x,y,z,qw,qx,qy,qz)
            self.world.data.qpos[adr:adr + 7] = bp
        else:
            # Freeze base at origin so pose is viewed in body frame
            self.world.data.qpos[0:3] = [0.0, 0.0, 1.0]
            self.world.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.world.model, self.world.data)
        # 3) update diag text with the 8 arm angles
        self._update_joint_text(i)
        # 4) stream to real motors if enabled
        if self.send_motors and self.bus.is_open() and self.cal:
            self._send_arms(row, m["names"])

    def _update_joint_text(self, i: int):
        m = self.motion
        lines = [f"frame {i:4d} / {m['q'].shape[0] - 1:4d}",
                 f"t = {i / m['fps']:6.2f} s",
                 ""]
        for name in m["names"]:
            if name.startswith("ub_"):
                col = m["names"].index(name)
                v = float(m["q"][i][col])
                lines.append(f"  {name:26s}  {v:+.3f}")
        self.joint_txt.delete("1.0", "end")
        self.joint_txt.insert("1.0", "\n".join(lines))
        self.frame_lbl.set(f"frame {i} / {m['q'].shape[0] - 1}   "
                           f"t={i / m['fps']:.2f} s")

    def _playback_tick(self):
        if self.playing and self.motion is not None:
            now = time.monotonic()
            # Motor catch-up gate: when "等馬達跟上" is on and we're actually
            # streaming motor commands, hold the frame advance until either
            # (a) all commanded sids are within CATCHUP_TOL of their target,
            # or (b) the CATCHUP_MAX_S deadline has passed.
            # In case (b) we log the frame index into _missed_frames — you
            # can review these later ("追不上" 的節點).
            waiting = (self.wait_motor_var.get()
                       and self.send_motors
                       and self.bus.is_open()
                       and bool(self._last_ticks))
            if waiting:
                if not self._motors_caught_up():
                    if now < self._settle_until:
                        # keep waiting; DO NOT advance the frame
                        self._last_tick_t = now
                        self.root.after(30, self._playback_tick)
                        return
                    else:
                        # timed out — record and continue
                        self._missed_frames.append(self.frame_idx)
                        if hasattr(self, "tools_var"):
                            self.tools_var.set(
                                f"⚠ frame {self.frame_idx} 追不上"
                                f" (累積 missed={len(self._missed_frames)})")
                        # clear so we don't spam-log again for the same frame
                        self._last_ticks.clear()
            dt = now - self._last_tick_t
            step = int(dt * self.motion["fps"] * self.speed)
            if step >= 1:
                self._last_tick_t = now
                new = self.frame_idx + step
                n = int(self.motion["q"].shape[0])
                if new >= n:
                    if self.loop:
                        new = new % n
                    else:
                        new = n - 1
                        self.playing = False
                        self.play_btn.config(text="▶ Play",
                                             bg="#2e7d32")
                self.frame_idx = new
                self.frame_var.set(new)
                self._apply_frame(new)
        self.root.after(15, self._playback_tick)

    # ── motor sender ────────────────────────────────────────────────────────
    def _load_cal(self):
        if not os.path.exists(CAL_PATH):
            return None
        try:
            d = json.load(open(CAL_PATH))
            out = {}
            for arm in ("L", "R"):
                a = d.get(arm) or {}
                pts = a["points"] if isinstance(a, dict) and "points" in a \
                    else a
                if isinstance(pts, list) and len(pts) == 5:
                    out[arm] = [
                        {k: float(p.get(k, 0)) for k in
                         ("tick_lo", "q_lo", "tick_hi", "q_hi")}
                        for p in pts
                    ]
            return out or None
        except Exception as e:
            print(f"[cal] load error: {e}", flush=True)
            return None

    def _q_to_step(self, arm: str, ji: int, q: float) -> int | None:
        c = (self.cal or {}).get(arm)
        if c is None: return None
        p = c[ji]
        # Safety clamp: q must stay within the arm-cal's q_lo/q_hi range so
        # the tick can't shoot past tick_lo/tick_hi.  cal_points can be
        # inverted (q_lo > q_hi), so clamp to sorted bounds.
        q_min = min(p["q_lo"], p["q_hi"])
        q_max = max(p["q_lo"], p["q_hi"])
        q = max(q_min, min(q_max, q))
        denom = p["q_hi"] - p["q_lo"]
        if abs(denom) < 1e-9: return int(round(p["tick_lo"]))
        tick = p["tick_lo"] + (q - p["q_lo"]) * (p["tick_hi"] - p["tick_lo"]) / denom
        # Absolute physical guardrail — Feetech tick range is 0..4095.
        return int(round(max(0, min(4095, tick))))

    def _send_arms(self, row: np.ndarray, names: list[str]):
        spd = int(self.motor_speed_var.get())
        acc = int(self.motor_acc_var.get())
        arm_filter = self.arm_filter_var.get()   # "both" | "R" | "L"
        self._last_ticks.clear()
        for name, (arm, ji) in JOINT_MAP.items():
            if arm_filter != "both" and arm != arm_filter:
                continue                          # skip the disabled side
            if name not in names: continue
            key = (arm, ji)
            if key not in self._motor_map:
                continue                          # constant column / no cal
            col = names.index(name)
            q_mjcf = float(row[col])
            t_lo, t_hi, m_lo, m_hi = self._motor_map[key]
            # Direct affine: motion range → cal tick range.
            # Independent of the cal's q_lo/q_hi absolute values — this
            # sidesteps the offset problem we hit with R J2 where the
            # calibrated q range didn't include 0.
            frac = (q_mjcf - m_lo) / (m_hi - m_lo)
            frac = max(0.0, min(1.0, frac))       # clip to motion range
            tick = t_lo + frac * (t_hi - t_lo)
            step = int(round(max(0, min(4095, tick))))
            sid = ARMS[arm]["joints"][ji][0]
            self.bus.write_pos(sid, step, spd, acc)
            self._last_ticks[sid] = step
        # Start motor-settle window for the catch-up wait.
        if self._last_ticks:
            self._settle_until = time.monotonic() + self._CATCHUP_MAX_S

    def _motors_caught_up(self) -> bool:
        """Poll bus.latest_tele (updated at motor_tele_hz by the agent).
        Returns True when EVERY commanded sid is within CATCHUP_TOL of its
        last target tick.  Missing telemetry → assume not caught up yet."""
        for sid, target in self._last_ticks.items():
            pos = self.bus.read_pos(sid)
            if pos is None:
                return False
            if abs(pos - target) > self._CATCHUP_TOL:
                return False
        return True

    def _toggle_connect(self):
        if self.bus.is_open():
            self.bus.close()
            self.bus_status_var.set("disconnected")
            self.conn_btn.config(text="Connect", bg=C["btn"])
            return
        host = self.host_var.get().strip()
        ok, msg = self.bus.open(host)
        self.bus_status_var.set(msg)
        if ok:
            self.conn_btn.config(text="Disconnect", bg=C["btn_danger"])

    def _sample_test(self):
        """Send N evenly-spaced sample frames to real motors with dwell —
        lets the operator eyeball whether ghost & real robot land at the
        same pose at each anchor point BEFORE committing to full playback.

        Ghost + sliders track the anchor being tested; motor write is done
        via _send_arms (same code path as playback) so the values match
        exactly."""
        from tkinter import messagebox
        if not self.motion:
            messagebox.showerror("sample test", "先載入 motion 檔"); return
        if not self.bus.is_open():
            messagebox.showerror("sample test",
                "先按 Connect 連上 arm agent"); return
        if not self.cal:
            messagebox.showerror("sample test",
                f"找不到校準檔:\n{CAL_PATH}"); return

        N_SAMPLES = 5
        DWELL_MS  = 2000
        n = int(self.motion["q"].shape[0])
        frames = np.linspace(0, n - 1, N_SAMPLES, dtype=int).tolist()
        # Ensure motion isn't playing while we run the test
        was_playing = self.playing
        if was_playing:
            self._toggle_play()
        # Force-enable motor send just for this test
        prev_send = self.send_motors
        self.send_motors = True

        self.tools_var.set(
            f"Sample test: {len(frames)} frames "
            f"[{', '.join(map(str, frames))}]  ({DWELL_MS/1000:.1f}s each)")

        def _step(k):
            if k >= len(frames):
                # Restore state
                self.send_motors = prev_send
                self.send_var.set(prev_send)
                self.tools_var.set(
                    "Sample test 完成 — 確認幽靈與實體姿勢是否一致")
                if was_playing:
                    self._toggle_play()
                return
            i = frames[k]
            self._seek(i)   # updates ghost + sends motors (send_motors on)
            self.tools_var.set(
                f"Sample test {k+1}/{len(frames)}  → frame {i} "
                f"({i / self.motion['fps']:.1f}s)")
            self.root.after(DWELL_MS, lambda: _step(k + 1))
        _step(0)

    def _check_collisions(self):
        """Sweep every frame with mj_forward and count contacts beyond the
        baseline (feet-on-floor).  Reports the first ~10 violating frames
        so the operator can seek to them and inspect visually."""
        if not self.motion:
            self.tools_var.set("collision check: 先載入 motion")
            return
        baseline = int(getattr(self.world, "_baseline_ncon", 0))
        n = int(self.motion["q"].shape[0])
        viols: list[tuple[int, int]] = []      # (frame, ncon)
        # Cache current frame so we can restore after the scan
        saved_qpos = self.world.data.qpos.copy()
        try:
            for i in range(n):
                self._apply_frame(i)
                # Ensure contacts are populated — mj_forward inside
                # _apply_frame already ran mj_collision, so data.ncon is
                # authoritative.
                ncon = int(self.world.data.ncon)
                if ncon > baseline:
                    viols.append((i, ncon - baseline))
        finally:
            # Restore whichever frame the operator was looking at
            self.world.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.world.model, self.world.data)
        if not viols:
            self.tools_var.set(f"✓ 檢查 {n} frames 無碰撞 (baseline={baseline})")
            return
        first10 = viols[:10]
        summary = ", ".join(f"f{i}({c})" for i, c in first10)
        more = f" +{len(viols)-10} more" if len(viols) > 10 else ""
        self.tools_var.set(
            f"⚠ 發現 {len(viols)}/{n} frames 有碰撞:{summary}{more}")

    def _torque_all(self, on: bool):
        if not self.bus.is_open(): return
        for arm in ("L", "R"):
            for sid, *_ in ARMS[arm]["joints"]:
                self.bus.torque(sid, on)

    # ── rendering ───────────────────────────────────────────────────────────
    def _render_tick(self):
        try:
            img = self.world.render()
            pil = Image.fromarray(img)
            self._tk_img = ImageTk.PhotoImage(pil)
            if self.canvas_img_id is None:
                self.canvas_img_id = self.canvas.create_image(
                    0, 0, image=self._tk_img, anchor="nw")
            else:
                self.canvas.itemconfig(self.canvas_img_id,
                                       image=self._tk_img)
        except Exception as e:
            print(f"[render] {e}", flush=True)
        self.root.after(33, self._render_tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    MotionApp().run()
