#!/usr/bin/env python3
"""Q-BOT master launcher — one-click buttons for every tool.

Tk window with a big button per component.  Each button:
  * shows whether the underlying tool / process is currently running
  * clicks either start it or stop it
  * shows relevant status (RPi ping, motor bus, camera, endpoints)

Run:  python3 scripts/launcher.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

import socket

def _mdns_ok(name: str) -> bool:
    """True if a .local name resolves (mDNS/avahi)."""
    if not name:
        return False
    try:
        socket.gethostbyname(name)
        return True
    except OSError:
        return False

def _load_host(env_key: str, cache_name: str, default: str = "",
               mdns: str = "") -> str:
    """解析順序:env 覆寫 → mDNS 名稱(主)→ 快取 IP → 預設。
    名稱能解析就用名稱,IP 變動免管;解不到才退回快取,再靠『尋找 IP』掃描。"""
    env = os.environ.get(env_key, "").strip()
    if env:
        return env
    if _mdns_ok(mdns):
        return mdns
    cache = os.path.join(REPO, cache_name)
    if os.path.isfile(cache):
        try:
            v = open(cache).read().strip()
            if v:
                return v
        except Exception:
            pass
    return default

RPI_HOST     = _load_host("QBOT_RPI_HOST",     ".rpi_host",     "fr01-head.local", "fr01-head.local")
RPI_LEG_HOST = _load_host("QBOT_RPI_LEG_HOST", ".rpi_host_leg", "",                "fr01-leg.local")
RPI_EXO_HOST = _load_host("QBOT_RPI_EXO_HOST", ".rpi_host_exo", "",                "fr01-exo.local")
RPI_USER     = "robot"
RPI_LEG_USER = "fr01"
RPI_EXO_USER = "robot"        # default — override in .rpi_exo_user if different
RPI_KEY      = os.path.expanduser("~/.ssh/qbot_rpi")

C = {
    "bg": "#111820", "panel": "#172030", "card": "#1c2a3a",
    "text": "#d0e0f0", "dim": "#6080a0", "bright": "#eef4fc",
    "ok": "#50c880", "warn": "#d09840", "err": "#e05858",
    "btn": "#1e4870", "btn_green": "#2e7d32", "btn_red": "#7a2828",
    "accent": "#7ed3ff",
}


def run_bg(script: str, extra_args=None):
    """Launch a script asynchronously in a new terminal-less process."""
    path = os.path.join(HERE, script)
    cmd = ["bash", path] + list(extra_args or [])
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def rpi_ssh(cmd, timeout=4):
    """Run a command on the RPi via key auth. Returns (rc, stdout)."""
    try:
        r = subprocess.run(
            ["ssh", "-i", RPI_KEY,
             "-o", "StrictHostKeyChecking=no",
             "-o", "BatchMode=yes",
             "-o", f"ConnectTimeout={timeout}",
             f"{RPI_USER}@{RPI_HOST}", cmd],
            capture_output=True, text=True, timeout=timeout + 2)
        return r.returncode, r.stdout.strip()
    except Exception:
        return 255, ""


def http_get(url, timeout=2):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()[:8192]
    except Exception as e:
        return 0, str(e)


# Feetech model-number → friendly name.  Populated from real scans of the
# Q-BOT daisy chain; extend if new servos appear.
MODEL_NAME = {
    777:  "STS3215",   # head + both arms
    1029: "SCS009",    # hand middle 4 servos
    521:  "SCServo",   # hand two-end servos (family variant of SCS)
}

# Expected motor layout — used for auto-scan status painting.
EXPECTED_MOTORS = [
    ("HEAD",  (1, 2, 3),              "STS3215"),
    ("R arm", (10, 11, 12, 13, 14),   "STS3215"),
    ("L arm", (20, 21, 22, 23, 24),   "STS3215"),
    ("HAND",  (41, 42, 43, 44, 45, 46), None),  # mixed model chain
]
EXPECTED_TOTAL = sum(len(g[1]) for g in EXPECTED_MOTORS)


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Q-BOT launcher")
        self.root.configure(bg=C["bg"])
        self.root.geometry("860x820")
        self.root.minsize(800, 760)

        self._build()
        self._refresh()             # first probe
        self.root.after(3000, self._auto_refresh)

    def _build(self):
        hdr = tk.Frame(self.root, bg="#1a237e"); hdr.pack(fill="x")
        tk.Label(hdr, text="Q-BOT  一鍵啟動 launcher",
                 font=("DejaVu Sans", 16, "bold"),
                 bg="#1a237e", fg="white").pack(side="left", padx=12, pady=8)

        # Status card at top
        st = tk.LabelFrame(self.root, text=" 系統狀態 ",
                           bg=C["panel"], fg=C["text"],
                           font=("DejaVu Sans Mono", 10, "bold"))
        st.pack(fill="x", padx=8, pady=(8, 4))
        self.status_var = tk.StringVar(value="probing…")
        tk.Label(st, textvariable=self.status_var, bg=C["panel"], fg=C["dim"],
                 font=("DejaVu Sans Mono", 9), anchor="w", justify="left"
                 ).pack(fill="x", padx=8, pady=6)

        # Motor status card — auto-scans on start via _probe, shows each
        # daisy-chain group's presence + model + missing IDs.
        mm = tk.LabelFrame(self.root, text=" 馬達狀況 (自動掃描) ",
                           bg=C["panel"], fg=C["text"],
                           font=("DejaVu Sans Mono", 10, "bold"))
        mm.pack(fill="x", padx=8, pady=(0, 4))
        self.motor_head_var = tk.StringVar(value="probing…")
        self.motor_body_var = tk.StringVar(value="")
        tk.Label(mm, textvariable=self.motor_head_var, bg=C["panel"],
                 fg=C["accent"], anchor="w",
                 font=("DejaVu Sans Mono", 10, "bold")
                 ).pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(mm, textvariable=self.motor_body_var, bg=C["panel"],
                 fg=C["text"], anchor="w", justify="left",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Head section
        h = tk.LabelFrame(self.root, text=" 頭部(RPi agent + PC 上位機)",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        h.pack(fill="x", padx=8, pady=4)
        # Row 1: main head start/stop + head lock
        row1 = tk.Frame(h, bg=C["panel"]); row1.pack(fill="x", padx=6, pady=(6, 2))
        tk.Button(row1, text="▶ 啟動 head",
                  bg=C["btn_green"], fg="white", width=16,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("head_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row1, text="■ 停 RPi agent",
                  bg=C["btn_red"], fg="white", width=14,
                  command=lambda: self._start("head_stop.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row1, text="[鎖] 頭部固定",
                  bg=C["warn"], fg="white", width=14,
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=lambda: self._start("head_lock.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row1, text="掃描馬達 IDs",
                  bg=C["btn"], fg="white", width=14,
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=self._scan_motors_popup
                  ).pack(side="left", padx=2)
        # Row 2: phone VR URL + live phone connection status
        row2 = tk.Frame(h, bg=C["panel"]); row2.pack(fill="x", padx=6, pady=(0, 2))
        tk.Button(row2, text="複製手機 VR URL",
                  bg=C["btn"], fg="white", width=16,
                  command=self._open_vr_url
                  ).pack(side="left", padx=2)
        self.phone_status = tk.StringVar(value="手機: (未連)")
        self.phone_lbl = tk.Label(row2, textvariable=self.phone_status,
                                   bg=C["panel"], fg=C["err"],
                                   font=("DejaVu Sans Mono", 11, "bold"))
        self.phone_lbl.pack(side="left", padx=10)
        self.head_info = tk.StringVar(value="")
        tk.Label(h, textvariable=self.head_info,
                 bg=C["panel"], fg=C["accent"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Arm section
        a = tk.LabelFrame(self.root, text=" 手臂 IK GUI ",
                         bg=C["panel"], fg=C["text"],
                         font=("DejaVu Sans Mono", 10, "bold"))
        a.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(a, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        tk.Button(row, text="▶ 本機 arm IK",
                  bg=C["btn_green"], fg="white", width=16,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("arm_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row, text="▶ RPi 遠端 arm IK",
                  bg=C["btn"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("arm_remote_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row, text="■ 停 RPi arm",
                  bg=C["btn_red"], fg="white", width=12,
                  command=lambda: self._start("arm_remote_stop.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row, text="motor_tool",
                  bg=C["btn"], fg="white", width=12,
                  command=lambda: self._start("motor_tool_start.sh")
                  ).pack(side="left", padx=2)
        self.arm_info = tk.StringVar(value="")
        tk.Label(a, textvariable=self.arm_info,
                 bg=C["panel"], fg=C["accent"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Pose imitation section
        pi = tk.LabelFrame(self.root, text=" 姿態模仿 (相機 → arm GUI UDP) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        pi.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(pi, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        tk.Button(row, text="▶ RPi 相機模仿",
                  bg=C["btn_green"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("pose_imitate_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row, text="▶ 筆電相機模仿",
                  bg=C["btn"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start_with_args(
                      "pose_imitate_start.sh", ["--local"])
                  ).pack(side="left", padx=2)
        tk.Label(pi, text="  arm GUI 勾「模仿」看鬼影,對了勾「送馬達」",
                 bg=C["panel"], fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Motion replay section (offline npz/csv → full-body ghost + arm motors)
        mr = tk.LabelFrame(self.root, text=" 動作重現 (npz/csv → ghost + arm) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        mr.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(mr, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        tk.Button(row, text="▶ Motion imitator",
                  bg="#2e7d32", fg="white", width=20,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=self._start_motion_imitator
                  ).pack(side="left", padx=2)
        tk.Label(mr, text="  讀 ~/下載/*.npz → 幽靈跟著動,勾「送馬達」→ 手臂馬達跟動",
                 bg=C["panel"], fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Hand section
        hd = tk.LabelFrame(self.root, text=" 靈巧手 (hand) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        hd.pack(fill="x", padx=8, pady=4)
        # Right-hand row
        row_r = tk.Frame(hd, bg=C["panel"]); row_r.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(row_r, text="右手:", bg=C["panel"], fg=C["dim"], width=6,
                 font=("DejaVu Sans Mono", 10, "bold")).pack(side="left")
        tk.Button(row_r, text="▶ 相機影像模式",
                  bg=C["btn_green"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("hand_camera_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row_r, text="▶ 手套 UDP 模式",
                  bg=C["btn"], fg="white", width=16,
                  command=lambda: self._start("hand_glove_start.sh")
                  ).pack(side="left", padx=2)
        # Left-hand row
        row_l = tk.Frame(hd, bg=C["panel"]); row_l.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(row_l, text="左手:", bg=C["panel"], fg=C["dim"], width=6,
                 font=("DejaVu Sans Mono", 10, "bold")).pack(side="left")
        tk.Button(row_l, text="▶ 相機影像模式",
                  bg=C["btn_green"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=lambda: self._start("hand_camera_left_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row_l, text="▶ 手套 UDP 模式",
                  bg=C["btn"], fg="white", width=16,
                  command=lambda: self._start("hand_glove_left_start.sh")
                  ).pack(side="left", padx=2)
        # Calibration (left) — launches slider over SSH X11 forwarding.
        # Universal STOP button here — kills local + RPi hand processes.
        row_cal = tk.Frame(hd, bg=C["panel"]); row_cal.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(row_cal, text="校準:", bg=C["panel"], fg=C["dim"], width=6,
                 font=("DejaVu Sans Mono", 10, "bold")).pack(side="left")
        tk.Button(row_cal, text="▶ 右手滑桿校準 (RPi X11)",
                  bg=C["warn"], fg="white", width=24,
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=lambda: self._start("hand_slider_right_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row_cal, text="▶ 左手滑桿校準 (RPi X11)",
                  bg=C["warn"], fg="white", width=24,
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=lambda: self._start("hand_slider_left_start.sh")
                  ).pack(side="left", padx=2)
        tk.Button(row_cal, text="■ 停止靈巧手",
                  bg=C["btn_red"], fg="white", width=16,
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=lambda: self._start("hand_stop.sh")
                  ).pack(side="left", padx=2)
        self.hand_info = tk.StringVar(
            value="  camera_receiver.py --show  或  glove_receiver.py")
        tk.Label(hd, textvariable=self.hand_info,
                 bg=C["panel"], fg=C["accent"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # ── Exo (exoskeleton) — 3rd RPi + dual-ghost GUI ───────────────────
        ex = tk.LabelFrame(self.root, text=" 外骨骼 exo (dual ghost) ",
                          bg=C["panel"], fg=C["text"],
                          font=("DejaVu Sans Mono", 10, "bold"))
        ex.pack(fill="x", padx=8, pady=4)
        row = tk.Frame(ex, bg=C["panel"]); row.pack(fill="x", padx=6, pady=6)
        tk.Button(row, text="▶ 啟動 exo (真機)",
                  bg=C["btn_green"], fg="white", width=18,
                  font=("DejaVu Sans Mono", 11, "bold"),
                  command=self._start_exo
                  ).pack(side="left", padx=2)
        tk.Button(row, text="▶ 啟動 exo (fake)",
                  bg=C["btn"], fg="white", width=18,
                  command=self._start_exo_fake
                  ).pack(side="left", padx=2)
        tk.Button(row, text="■ 停 exo",
                  bg=C["btn_red"], fg="white", width=10,
                  command=self._stop_exo
                  ).pack(side="left", padx=2)
        tk.Button(row, text="[?] 尋找 Exo IP",
                  bg=C["btn"], fg="white", width=14,
                  command=self._find_rpi_exo
                  ).pack(side="left", padx=2)
        tk.Label(ex, text=("  fake 模式:本機跑 sinusoidal 假外骨骼,適合先確認 GUI。"
                          "  真機:需先設好 .rpi_exo_mac 讓「尋找 Exo IP」找得到。"),
                 bg=C["panel"], fg=C["dim"], anchor="w",
                 font=("DejaVu Sans Mono", 9)
                 ).pack(fill="x", padx=8, pady=(0, 6))

        # Refresh + shutdown row
        b = tk.Frame(self.root, bg=C["bg"]); b.pack(fill="x", padx=8, pady=6)
        tk.Button(b, text="⏻ 遠端關機 RPi",
                  bg=C["btn_red"], fg="white",
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=self._confirm_rpi_shutdown, width=18
                  ).pack(side="left")
        # New: auto-discover RPi IP when we're on a strange network.
        tk.Button(b, text="[?] 尋找 RPi IP",
                  bg=C["btn"], fg="white",
                  font=("DejaVu Sans Mono", 10, "bold"),
                  command=self._find_rpi, width=14
                  ).pack(side="left", padx=8)
        tk.Button(b, text="↻ Refresh status",
                  bg=C["btn"], fg="white",
                  command=self._refresh, width=18).pack(side="right")

    def _phone_lbl_colour(self, mode):
        colours = {"ok": C["ok"], "warn": C["warn"], "err": C["err"]}
        try: self.phone_lbl.config(fg=colours.get(mode, C["dim"]))
        except Exception: pass

    def _update_motor_status(self, st: dict):
        """Paint the auto-scan card from a /status response dict."""
        ids = set(st.get("motor_ids") or [])
        models = {int(k): int(v) for k, v in
                  (st.get("motor_models") or {}).items()}
        n_found = len(ids)
        port = st.get("motor_port", "?")
        # Header: total + port
        self.motor_head_var.set(
            f"{'[OK] ' if n_found == EXPECTED_TOTAL else '[!] '}"
            f"detected {n_found}/{EXPECTED_TOTAL} on {port}")
        # Body: one line per group with SID list, model summary, missing
        lines = []
        for name, expect, expect_model in EXPECTED_MOTORS:
            have = [i for i in expect if i in ids]
            miss = [i for i in expect if i not in ids]
            model_counts: dict[str, int] = {}
            for i in have:
                m = models.get(i)
                nm = MODEL_NAME.get(m, f"m{m}") if m else "?"
                model_counts[nm] = model_counts.get(nm, 0) + 1
            mdl = ", ".join(f"{k}×{v}" for k, v in model_counts.items()) or "-"
            tag = "OK" if not miss else "!!"
            line = f"  [{tag}] {name:5s} {have}  →  {mdl}"
            if miss:
                line += f"   MISSING: {miss}"
            lines.append(line)
        err = st.get("motor_error", "") or ""
        if err:
            lines.append(f"  err: {err}")
        self.motor_body_var.set("\n".join(lines))

    def _scan_motors_popup(self):
        """Query head_agent /status and show currently-detected motor IDs
        in a modal popup — one-glance verification of the daisy chain."""
        from tkinter import messagebox
        code, body = http_get(f"http://{RPI_HOST}:8000/status", timeout=3)
        if code != 200 or not body:
            messagebox.showerror("掃描馬達 IDs",
                f"head_agent 沒回應 (code={code})\n"
                "確認 RPi 有跑 head agent")
            return
        import json as _j
        try:
            st = _j.loads(body)
        except Exception:
            messagebox.showerror("掃描馬達 IDs", "回傳不是 JSON"); return
        ids = set(st.get("motor_ids") or [])
        models = {int(k): int(v) for k, v in
                  (st.get("motor_models") or {}).items()}
        port = st.get("motor_port", "?")
        err  = st.get("motor_error", "") or ""
        lines = [
            f"Serial port: {port}",
            f"Detected {len(ids)}/{EXPECTED_TOTAL} motors",
            "",
        ]
        for name, expect, _ in EXPECTED_MOTORS:
            have = [i for i in expect if i in ids]
            miss = [i for i in expect if i not in ids]
            per = ", ".join(
                f"{i}:{MODEL_NAME.get(models.get(i), f'm{models.get(i)}')}"
                for i in have)
            lines.append(f"{name:5s} ({expect[0]}-{expect[-1]}):")
            lines.append(f"    {per if per else '(none)'}")
            if miss:
                lines.append(f"    MISSING: {miss}")
        if err:
            lines.append("")
            lines.append(f"ERROR: {err}")
        # Also invite a fresh scan on the RPi if some missing
        missing_any = any(i not in ids
                          for _, exp, _ in EXPECTED_MOTORS for i in exp)
        if missing_any:
            lines.append("")
            lines.append("(若剛插上,等 30 秒 agent 會自動 rescan)")
        messagebox.showinfo("掃描馬達 IDs", "\n".join(lines))

    def _find_any(self, mode: str):
        """Run find_rpi.sh --<mode>, take stdout IP, cache + apply."""
        global RPI_HOST, RPI_LEG_HOST, RPI_EXO_HOST
        from tkinter import messagebox
        script = os.path.join(HERE, "find_rpi.sh")
        label = {"head": "上半身", "leg": "腿部", "exo": "外骨骼"}.get(mode, mode)
        try:
            r = subprocess.run(["bash", script, f"--{mode}"],
                               capture_output=True, text=True, timeout=90)
        except Exception as e:
            messagebox.showerror(f"尋找 RPi ({label})",
                                 f"執行 find_rpi.sh 失敗:\n{e}")
            return
        ip = (r.stdout.strip().splitlines() or [""])[-1]
        if r.returncode != 0 or not ip:
            messagebox.showerror(f"尋找 RPi ({label})",
                                 f"找不到 {label} RPi\n\n{r.stderr[-400:]}")
            return
        if mode == "head":
            RPI_HOST = ip
        elif mode == "leg":
            RPI_LEG_HOST = ip
        else:
            RPI_EXO_HOST = ip
        self.status_var.set(f"{label} RPi IP set to {ip}")
        messagebox.showinfo(f"尋找 RPi ({label})",
            f"找到 {label} RPi 在 {ip}")
        self.root.after(400, self._refresh)

    def _find_rpi(self):     self._find_any("head")
    def _find_rpi_leg(self): self._find_any("leg")
    def _find_rpi_exo(self): self._find_any("exo")

    def _start_exo(self):
        self._start_rel_script("exo/scripts/exo_start.sh")

    def _start_exo_fake(self):
        # In fake mode we can skip RPi entirely — fake agent runs locally.
        self._start_rel_script("exo/scripts/exo_start.sh", "--fake")

    def _stop_exo(self):
        self._start_rel_script("exo/scripts/exo_stop.sh")

    def _start_rel_script(self, rel: str, *extra):
        path = os.path.join(REPO, rel)
        subprocess.Popen(["bash", path, *extra],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL,
                         start_new_session=True)
        self.status_var.set(f"launched: {rel} {' '.join(extra)}")

    def _confirm_rpi_shutdown(self):
        from tkinter import messagebox
        if messagebox.askyesno("遠端關機",
                               f"要把 RPi ({RPI_HOST}) 關機嗎?\n"
                               "所有 agent 會停,SSH 會斷,大約 15 秒後 RPi 斷電。"):
            self._start("rpi_shutdown.sh")

    def _start(self, script):
        run_bg(script)
        self.status_var.set(f"launched: {script}")
        self.root.after(1500, self._refresh)

    def _start_with_args(self, script, args):
        run_bg(script, args)
        self.status_var.set(f"launched: {script} {' '.join(args)}")
        self.root.after(1500, self._refresh)

    def _start_motion_imitator(self):
        """Launch the offline motion-replay GUI directly (no shell script)."""
        script = os.path.join(REPO, "arm",
                              "motion_imitator", "motion_replay_gui.py")
        subprocess.Popen(
            [sys.executable, script],
            cwd=os.path.dirname(script),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.status_var.set("launched: motion_replay_gui.py")

    def _open_vr_url(self):
        url = f"https://{RPI_HOST}:8443/vr"
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(url)
            self.status_var.set(f"copied {url}  (paste to phone browser)")
        except Exception:
            self.status_var.set(url)

    # ── async probe (don't block UI) ────────────────────────────────────────
    def _refresh(self):
        threading.Thread(target=self._probe, daemon=True).start()

    def _auto_refresh(self):
        self._refresh()
        self.root.after(5000, self._auto_refresh)

    def _probe(self):
        # RPi + head agent + system stats (temp/load/mem in one SSH call)
        rc, _ = rpi_ssh("true", timeout=3)
        rpi_up = (rc == 0)
        agent_pid = None
        rpi_sys = ""
        if rpi_up:
            rc2, out = rpi_ssh("pgrep -f agent.py | head -1", timeout=3)
            agent_pid = out.strip() if out.strip() else None
            rc3, sys_out = rpi_ssh(
                "printf '%s\\t%s\\t%s\\n' "
                "\"$(cat /sys/class/thermal/thermal_zone0/temp)\" "
                "\"$(cut -d' ' -f1-3 /proc/loadavg)\" "
                "\"$(free -m | awk '/^Mem:/ {print $3\"/\"$2\"MB\"}')\"",
                timeout=3)
            if sys_out:
                try:
                    temp_raw, load, mem = sys_out.strip().split("\t")
                    temp_c = int(temp_raw) / 1000.0
                    rpi_sys = f"  RPi {temp_c:.1f}°C  load {load}  mem {mem}"
                except Exception:
                    rpi_sys = "  RPi sys ??"
        # status endpoint
        status_code, status_body = http_get(
            f"http://{RPI_HOST}:8000/status", timeout=2) if agent_pid else (0, "")
        # arm serial
        arm_ports = [p for p in os.listdir("/dev") if p.startswith("ttyACM")]

        # summarise
        def line(ok, txt): return ("[OK] " if ok else "[XX] ") + txt
        s = []
        s.append(line(rpi_up, f"RPi {RPI_HOST}: "
                              f"{'reachable' if rpi_up else 'unreachable'}"))
        if rpi_sys:
            s.append(rpi_sys)
        s.append(line(bool(agent_pid),
                      f"head agent: "
                      f"{'running PID '+agent_pid if agent_pid else 'stopped'}"))
        if status_body and status_code == 200:
            import json as _j
            try: st = _j.loads(status_body)
            except Exception: st = {}
            self.head_info.set(
                f"  motor_ids {st.get('motor_ids')}"
                f"   cam_frames {st.get('cam_frames')}"
                f"   port {st.get('motor_port')}")
            self._update_motor_status(st)
            # Phone status badge (green ON, orange PARTIAL, red OFF)
            pc = int(st.get("phone_connected", 0) or 0)
            active = bool(st.get("phone_imu_active", False))
            age = int(st.get("phone_pose_age_ms", -1) or -1)
            if pc == 0:
                self.phone_status.set("手機: (未連)")
                self._phone_lbl_colour("err")
            elif age < 0 or age > 2000:
                self.phone_status.set(f"手機: 連線但沒 pose (n={pc})")
                self._phone_lbl_colour("warn")
            else:
                mode = "ACTIVE (控制中)" if active else "連線中"
                self.phone_status.set(
                    f"手機: {mode}  age {age}ms  n={pc}")
                self._phone_lbl_colour("ok")
        else:
            self.head_info.set("")
            self.phone_status.set("手機: (未連)")
            self._phone_lbl_colour("err")
            self.motor_head_var.set("agent 未回應 — 馬達掃描無法執行")
            self.motor_body_var.set("啟動 head agent 後會自動列出所有馬達")
        s.append(line(bool(arm_ports),
                      f"local /dev/ttyACM*: "
                      f"{arm_ports if arm_ports else 'none'}"))
        self.arm_info.set("  端點: " + ",".join(arm_ports) if arm_ports else "  端點: —")

        self.status_var.set("\n".join(s))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Make bash scripts executable if not already (once)
    for f in os.listdir(HERE):
        if f.endswith(".sh"):
            p = os.path.join(HERE, f)
            try: os.chmod(p, 0o755)
            except Exception: pass
    App().run()
