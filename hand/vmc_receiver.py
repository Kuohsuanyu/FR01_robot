#!/usr/bin/env python3
"""VMC OSC 手套接收 → FR01 右手巧手驅動。

架構:
  - OSC 伺服器綁 127.0.0.1:39539(VMC 官方 port),收 /VMC/Ext/Bone/Pos
  - 每個手指骨骼 twist 角度 → 0..1 → config.tick_from_norm() → 送 Feetech 匯流排
  - 依 config.FINGERS[*]["model"] 派 scscl / sms_sts handler(SCS009 用 BE)
  - config.FIXED (ID 35 拇指轉動) 在啟動時送到 hold_tick 並保持,不動態控制

校準:
  - `o` = 張開手保持 3 秒 → 抓每根 quaternion baseline
  - `g` = 握緊手保持 3 秒 → 抓每根 twist 角度上限
  - baseline 齊了才會進 RUN,才會實際送位置給馬達

安全:
  - 校準未完成前,馬達不會動(target 保持 0)
  - Ctrl-C 或 q 退出時,馬達解除扭矩
"""
import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import socket

import numpy as np
from pythonosc.osc_packet import OscPacket
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/andykuo/FTServo_Python")
sys.path.insert(0, str(Path(__file__).parent))
from scservo_sdk import PortHandler, sms_sts, scscl, COMM_SUCCESS  # noqa: E402
import config as cfg  # noqa: E402

# ---- OSC ----
# 預設綁 0.0.0.0,接受本機 loopback + LAN 進來的封包。
# 若上位機在另一台電腦(常見情境),必定要 0.0.0.0 才收得到。
OSC_BIND_IP = "0.0.0.0"
OSC_PORT = 39539  # VMC 官方 port
OSC_ADDR = "/VMC/Ext/Bone/Pos"

# ---- VMC bone → 我方手指名 (config.FINGERS 的 key) ----
# 實測上位機主要送 *Intermediate,少數 *Proximal / *Distal。
# 三段都接,tick 時每根手指取「值最大的那段」當彎曲量。
# 這樣 sender 不論送哪段都能用,單段掉包也能被別段補上,更穩定。
_RIGHT_FINGERS = ["pinky", "ring", "middle", "index"]
_FINGER_TO_VMC_STEM = {
    "pinky":  "RightLittle",
    "ring":   "RightRing",
    "middle": "RightMiddle",
    "index":  "RightIndex",
    "thumb":  "RightThumb",
}
_SEGMENTS = ("Proximal", "Intermediate", "Distal")

RIGHT_BONE_TO_FINGER: Dict[str, str] = {}
for finger, stem in _FINGER_TO_VMC_STEM.items():
    for seg in _SEGMENTS:
        RIGHT_BONE_TO_FINGER[f"{stem}{seg}"] = finger

# 每根骨骼從 open baseline 抽 twist 的本地軸。
# 實測手指 quaternion 都是純 Z 軸旋轉 (0,0,z,w) → local Z 沒錯。
DEFAULT_AXIS = (0.0, 0.0, 1.0)
AXIS_LOCAL: Dict[str, Tuple[float, float, float]] = {
    b: DEFAULT_AXIS for b in RIGHT_BONE_TO_FINGER
}

# ---- Socket ----
RCVBUF_BYTES = 8 * 1024 * 1024   # 8MB 收 buffer,防止爆量掉包
CONTROL_HZ = 100                 # 主 loop 從 latest dict 讀值的頻率

# ---- 控制參數 ----
CALIB_SECONDS = 3.0
EMA_ALPHA = 1.0             # 1.0 = 無 EMA;<1 越平滑但延遲越大
SLEW_PER_SEC = 10.0          # 每秒 0..1 最大變化量
MARGIN_DEG = 8.0            # open/close 兩端加 margin,減少飽和
MIN_CLOSE_ABS_DEG = 8.0     # close_deg 過小 → 校準或軸不對,警告

MOTOR_HZ = 100              # 馬達 tick 送出頻率
LERP_ALPHA = 0.8            # target 到 current 的插值(每 tick)


# =========================================================================
# quaternion 工具(從 glove_testv4 抽離,穩定版)
# =========================================================================
def quat_bad(q: np.ndarray) -> bool:
    return q is None or np.any(np.isnan(q)) or np.any(np.isinf(q)) or np.linalg.norm(q) < 1e-6


def normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q if n < 1e-12 else q / n


def avg_quats(quats: List[np.ndarray]) -> Optional[np.ndarray]:
    if not quats:
        return None
    ref = quats[0]
    acc = np.zeros(4, dtype=np.float64)
    for q in quats:
        q = q.astype(np.float64)
        if np.dot(q, ref) < 0:
            q = -q
        acc += q
    n = np.linalg.norm(acc)
    return None if n < 1e-12 else acc / n


def twist_deg(q_rel: np.ndarray, axis_local: Tuple[float, float, float]) -> float:
    """quaternion 的 twist-swing 分解,回傳沿 axis_local 的 twist 角度 (度,含正負號)。"""
    ax = np.array(axis_local, dtype=np.float64)
    ax /= max(np.linalg.norm(ax), 1e-9)
    q = np.array(q_rel, dtype=np.float64)
    if q[3] < 0:
        q = -q
    v = q[:3]
    proj = ax * np.dot(v, ax)
    twist = np.array([proj[0], proj[1], proj[2], q[3]], dtype=np.float64)
    tn = np.linalg.norm(twist)
    if tn < 1e-12:
        return 0.0
    twist /= tn
    angle = 2.0 * math.atan2(np.linalg.norm(twist[:3]), max(1e-12, twist[3]))
    sign = 1.0 if np.dot(twist[:3], ax) >= 0 else -1.0
    return math.degrees(angle) * sign


def map01_with_margin(a_deg: float, open_deg: float, close_deg: float, margin: float) -> float:
    if close_deg > open_deg:
        a0, a1 = open_deg - margin, close_deg + margin
        t = (a_deg - a0) / (a1 - a0)
    else:
        a0, a1 = open_deg + margin, close_deg - margin
        t = (a0 - a_deg) / (a0 - a1)
    return float(np.clip(t, 0.0, 1.0))


def slew(prev: float, target: float, max_delta: float) -> float:
    d = target - prev
    if d > max_delta:
        return prev + max_delta
    if d < -max_delta:
        return prev - max_delta
    return target


# =========================================================================
# 馬達層 — 用 SDK,依 model 派 handler
# =========================================================================
class HandBus:
    def __init__(self, port: str, baud: int):
        self.ph = PortHandler(port)
        if not self.ph.openPort():
            raise RuntimeError(f"open port {port} failed")
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(f"setBaudRate {baud} failed")
        self.pk_sts = sms_sts(self.ph)
        self.pk_scs = scscl(self.ph)

    def _pk(self, model):
        return self.pk_scs if cfg.is_scscl(model) else self.pk_sts

    def torque(self, sid: int, model: int, on: bool):
        self._pk(model).write1ByteTxRx(sid, 40, 1 if on else 0)

    def write_pos(self, sid: int, model: int, tick: int):
        pk = self._pk(model)
        if cfg.is_scscl(model):
            pk.WritePos(sid, int(tick), 500, 200)         # time_ms, speed
        else:
            pk.WritePosEx(sid, int(tick), 200, 30)        # speed, acc

    def close(self):
        self.ph.closePort()


class HandDriver(threading.Thread):
    """背景 100Hz 迴圈:讀 targets_norm (0..1) → tick → 送。"""
    def __init__(self, bus: HandBus):
        super().__init__(daemon=True)
        self.bus = bus
        self.running = True
        # 每根手指:target_norm 是輸入;current_norm 是實際插值後送出的
        self.targets_norm: Dict[str, float] = {n: 0.0 for n in cfg.FINGERS}
        self.currents_norm: Dict[str, float] = {n: 0.0 for n in cfg.FINGERS}

    def set_target(self, finger: str, norm: float):
        self.targets_norm[finger] = max(0.0, min(1.0, norm))

    def enable_all(self):
        for name, c in cfg.FINGERS.items():
            self.bus.torque(c["id"], c["model"], True)
        for name, c in cfg.FIXED.items():
            self.bus.torque(c["id"], c["model"], True)
            self.bus.write_pos(c["id"], c["model"], c["hold_tick"])
            print(f"[fixed] {name} (id={c['id']}) → hold_tick={c['hold_tick']}")

    def disable_all(self):
        for c in list(cfg.FINGERS.values()) + list(cfg.FIXED.values()):
            self.bus.torque(c["id"], c["model"], False)

    def run(self):
        period = 1.0 / MOTOR_HZ
        while self.running:
            for name, c in cfg.FINGERS.items():
                tgt = self.targets_norm[name]
                cur = self.currents_norm[name]
                if abs(tgt - cur) < 0.002:
                    continue
                cur += (tgt - cur) * LERP_ALPHA
                self.currents_norm[name] = cur
                tick = cfg.tick_from_norm(name, cur)
                self.bus.write_pos(c["id"], c["model"], tick)
            time.sleep(period)


# =========================================================================
# 骨骼校準狀態
# =========================================================================
@dataclass
class BoneState:
    q_now: Optional[np.ndarray] = None
    q_open: Optional[np.ndarray] = None
    open_deg: float = 0.0
    close_deg: float = 60.0
    ema01: float = 0.0
    last01: float = 0.0
    open_samples: List[np.ndarray] = field(default_factory=list)
    close_samples: List[float] = field(default_factory=list)


# =========================================================================
# VMC 系統
# =========================================================================
class VMCHand:
    def __init__(self, driver: HandDriver):
        self.driver = driver
        self.lock = threading.Lock()
        self.running = True
        self.bones: Dict[str, BoneState] = {b: BoneState() for b in RIGHT_BONE_TO_FINGER}
        self.mode = "IDLE"           # IDLE / CALIB_OPEN / CALIB_CLOSE / RUN
        self.deadline = 0.0
        self.debug = False
        self.last_run_t = time.time()
        threading.Thread(target=self._keyboard_worker, daemon=True).start()

    # ---- 使用者按鍵 ----
    def _keyboard_worker(self):
        print("\n[Keys]  s=status  o=cal OPEN(3s)  g=cal CLOSE(3s)  p=print RUN  d=debug  q=quit\n")
        while self.running:
            ch = sys.stdin.read(1).strip().lower()
            if not ch:
                continue
            if ch == "s": self._status()
            elif ch == "o": self._start_calib("OPEN")
            elif ch == "g": self._start_calib("CLOSE")
            elif ch == "p": self._print_run()
            elif ch == "d": self.debug = not self.debug; print(f"[debug]={self.debug}")
            elif ch == "q": self.running = False; break

    def _status(self):
        with self.lock:
            got = [b for b, st in self.bones.items() if st.q_now is not None]
            cal = [b for b, st in self.bones.items() if st.q_open is not None]
        print(f"\n[STATUS] received {len(got)}/{len(self.bones)} bones,open-calibrated {len(cal)}")
        if got: print("  got:", got)
        if cal: print("  cal:", cal)

    def _start_calib(self, which: str):
        with self.lock:
            for st in self.bones.values():
                if which == "OPEN": st.open_samples.clear()
                else:               st.close_samples.clear()
            self.mode = f"CALIB_{which}"
            self.deadline = time.time() + CALIB_SECONDS
        print(f"[Calib {which}] {CALIB_SECONDS:.1f}s — 保持{'張開' if which=='OPEN' else '握緊'}姿勢")

    def _print_run(self):
        with self.lock:
            print(f"\n[RUN] mode={self.mode}")
            for b, st in self.bones.items():
                if st.q_now is None:
                    print(f"  {b}: no data"); continue
                if st.q_open is None:
                    print(f"  {b}: no OPEN baseline (press o)"); continue
                r_rel = R.from_quat(st.q_open).inv() * R.from_quat(st.q_now)
                ang = twist_deg(r_rel.as_quat(), AXIS_LOCAL[b])
                v01 = map01_with_margin(ang, st.open_deg, st.close_deg, MARGIN_DEG)
                print(f"  {b}: close_deg={st.close_deg:+.1f}  ang={ang:+.1f}  v01={v01:.2f}  ema={st.ema01:.2f}")
            print()

    # ---- 從 UDP 執行緒接進最新 quaternion(每根骨骼只保留最新)----
    def ingest_bone(self, bone: str, q: np.ndarray):
        if bone not in self.bones or quat_bad(q):
            return
        with self.lock:
            self.bones[bone].q_now = normalize(q)

    # ---- 定頻 tick(由 control thread 呼叫,不再每個封包跑一次)----
    def _tick(self):
        now = time.time()
        with self.lock:
            if self.mode == "CALIB_OPEN":
                for st in self.bones.values():
                    if st.q_now is not None:
                        st.open_samples.append(st.q_now.copy())
                if now >= self.deadline:
                    for b, st in self.bones.items():
                        avg = avg_quats(st.open_samples)
                        st.open_samples.clear()
                        st.q_open = normalize(avg) if avg is not None else None
                        st.open_deg = 0.0; st.ema01 = 0.0; st.last01 = 0.0
                    self.mode = "IDLE"
                    ok = [b for b, st in self.bones.items() if st.q_open is not None]
                    print(f"[Calib OPEN] done, ok={len(ok)}/{len(self.bones)}. 下一步:握緊按 g")

            elif self.mode == "CALIB_CLOSE":
                for b, st in self.bones.items():
                    if st.q_now is None or st.q_open is None:
                        continue
                    r_rel = R.from_quat(st.q_open).inv() * R.from_quat(st.q_now)
                    st.close_samples.append(twist_deg(r_rel.as_quat(), AXIS_LOCAL[b]))
                if now >= self.deadline:
                    for b, st in self.bones.items():
                        if not st.close_samples:
                            print(f"  WARN {b}: no close samples"); continue
                        med = float(np.median(st.close_samples))
                        std = float(np.std(st.close_samples))
                        st.close_deg = med
                        st.close_samples.clear()
                        warn = "  <— close_deg 太小,檢查軸或姿勢" if abs(med) < MIN_CLOSE_ABS_DEG else ""
                        print(f"  {b}: close_deg={med:+.1f} std={std:.1f}{warn}")
                    self.mode = "RUN"
                    print("\n[Calib CLOSE] done → RUN。想看即時值按 p。\n")

            elif self.mode == "RUN":
                dt = max(1e-3, now - self.last_run_t)
                self.last_run_t = now
                max_delta = SLEW_PER_SEC * dt

                # 先依 finger 分組,每根手指從其 3 段候選骨骼取「最大 v01」
                per_finger: Dict[str, Tuple[float, str]] = {}  # finger -> (v01, source_bone)
                for bone, st in self.bones.items():
                    if st.q_now is None or st.q_open is None:
                        continue
                    r_rel = R.from_quat(st.q_open).inv() * R.from_quat(st.q_now)
                    ang = twist_deg(r_rel.as_quat(), AXIS_LOCAL[bone])
                    v01 = map01_with_margin(ang, st.open_deg, st.close_deg, MARGIN_DEG)
                    finger = RIGHT_BONE_TO_FINGER[bone]
                    prev = per_finger.get(finger)
                    if prev is None or v01 > prev[0]:
                        per_finger[finger] = (v01, bone)

                # 每根手指:EMA + slew,寫進 driver.
                # 用 per-finger ema/last state(存在 self._finger_state)
                if not hasattr(self, "_finger_state"):
                    self._finger_state = {f: dict(ema=0.0, last=0.0) for f in cfg.FINGERS}
                for finger, (v01, src) in per_finger.items():
                    st_f = self._finger_state[finger]
                    st_f["ema"] = (1.0 - EMA_ALPHA) * st_f["ema"] + EMA_ALPHA * v01
                    st_f["ema"] = slew(st_f["last"], st_f["ema"], max_delta)
                    st_f["last"] = st_f["ema"]
                    self.driver.set_target(finger, st_f["ema"])
                    if self.debug:
                        print(f"[RUN] {finger:>6}  v01={v01:.2f}  ema={st_f['ema']:.2f}  src={src}")


# =========================================================================
# 高頻 UDP 執行緒:raw socket + OSC parser,每根骨骼只保留最新
# =========================================================================
def udp_receiver_thread(ip: str, port: int, system: "VMCHand", stop_flag):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
    s.settimeout(0.1)
    s.bind((ip, port))
    print(f"[udp] listening on {ip}:{port}  rcvbuf={RCVBUF_BYTES//1024}KB")
    pkt_count = 0
    msg_count = 0
    hit_count = 0
    last_report = time.time()
    while not stop_flag():
        try:
            data, _ = s.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break
        pkt_count += 1
        try:
            pkt = OscPacket(data)
        except Exception:
            continue
        for m in pkt.messages:
            msg_count += 1
            msg = m.message
            if msg.address != OSC_ADDR:
                continue
            params = list(msg)
            if len(params) != 8:
                continue
            bone = str(params[0])
            if bone not in RIGHT_BONE_TO_FINGER:
                continue
            q = np.array([float(params[4]), float(params[5]),
                          float(params[6]), float(params[7])])
            system.ingest_bone(bone, q)
            hit_count += 1
        # 每 5 秒印一次 throughput
        now = time.time()
        if now - last_report >= 5.0:
            dt = now - last_report
            print(f"[udp] {pkt_count/dt:.0f} pkt/s  {msg_count/dt:.0f} msg/s  "
                  f"命中手指骨骼 {hit_count/dt:.0f}/s")
            pkt_count = msg_count = hit_count = 0
            last_report = now
    s.close()


# =========================================================================
# 控制執行緒:定頻讓 VMCHand._tick() 跑校準/算 target
# =========================================================================
def control_thread(system: "VMCHand", stop_flag):
    period = 1.0 / CONTROL_HZ
    while not stop_flag():
        system._tick()
        time.sleep(period)


# =========================================================================
# main
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=OSC_BIND_IP)
    ap.add_argument("--port", type=int, default=OSC_PORT)
    ap.add_argument("--no-motor", action="store_true", help="OSC only,不真的送馬達 (dry-run)")
    args = ap.parse_args()

    assert cfg.HAND_SIDE == "right", "目前 receiver 只支援右手(config.HAND_SIDE 應為 'right')"

    # ---- 初始化馬達 ----
    if not args.no_motor:
        bus = HandBus(cfg.SERVO_PORT, cfg.SERVO_BAUD)
        driver = HandDriver(bus)
        driver.enable_all()
        driver.start()
    else:
        bus = None; driver = _DryRunDriver()

    system = VMCHand(driver)

    print(f"🖐️ VMC receiver up: {args.ip}:{args.port}  (map: {OSC_ADDR})")
    print(f"    config: right hand, {len(cfg.FINGERS)} active fingers,fixed {list(cfg.FIXED)}")

    # 兩個背景執行緒:UDP 高頻進 + 控制定頻算
    stop = lambda: not system.running
    t_udp = threading.Thread(target=udp_receiver_thread,
                             args=(args.ip, args.port, system, stop), daemon=True)
    t_ctl = threading.Thread(target=control_thread,
                             args=(system, stop), daemon=True)
    t_udp.start()
    t_ctl.start()

    try:
        while system.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Ctrl-C]")
    finally:
        system.running = False
        time.sleep(0.2)
        if not args.no_motor:
            driver.running = False
            time.sleep(0.05)
            driver.disable_all()
            bus.close()
        print("bye.")


class _DryRunDriver:
    """--no-motor 模式用:只吸目標值,不觸串口。"""
    def set_target(self, finger, norm):
        pass
    def enable_all(self): pass
    def disable_all(self): pass
    running = True


if __name__ == "__main__":
    sys.exit(main())
