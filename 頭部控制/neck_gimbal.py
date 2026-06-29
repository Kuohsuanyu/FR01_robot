"""
三自由度頭頸雲台控制 — 絕對角度跟隨 (給 VR 頭追用)

把手機 IMU 的 yaw/pitch/roll(度) → 三顆 Feetech STS3215 的目標位置(tick)，
每 tick 做：度→tick 換算 → 軟限位 → 低通濾波 → 限速 → 下發 WritePosEx。
驅動沿用 RX1 的 feetech_lib.py（STS 協定、baud 1,000,000）。

實機設定（2026-06，使用者提供）:
  ID1 左右轉頭(yaw)  : 1000~3000，越低(往1000)=往左轉
  ID2 上下轉頭(pitch): 1500~3000，1500=低頭到最下
  ID3 側面轉頭(roll) : 1500~2500，越低(往1500)=向右傾斜
  零位(中心)皆 2047

用法:
  python neck_gimbal.py --check   # 只讀取三軸目前位置，不會移動（先確認連線/ID）
  python neck_gimbal.py --test    # 連線→回中心→各軸在限位內輕輕掃動（確認方向/限位）
  python neck_gimbal.py --demo    # 展示模式：前後→左右→自然擺盪，循環播放
"""
import math
import sys
import threading
import time

import serial

from feetech_lib import SMS_STS

# 讓中文/符號在 Windows cp950 主控台也不會印爆
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 單位換算 ──────────────────────────────────────────────────────────────
SPEED_K = 652.6051999582332      # rad/s → 伺服 speed 單位（沿用 RX1）
ACC_K = 6.526051999582332
TICKS_PER_DEG = 4096.0 / 360.0   # STS3215: 4096 ticks = 360°  → 11.378 tick/度


# ============================================================================
#  雲台設定（已依實機填好）
# ============================================================================
PORT = "COM8"
BAUD = 1_000_000

# 每軸:
#   sid    : 伺服 ID
#   dir    : 方向 +1 / -1（裝起來上下或左右相反就改這個）
#   scale  : 手機角度 → 雲台角度 倍率（1.0 = 1:1 跟隨）
#   center : 零位 tick
#   lo/hi  : 軟限位 tick（保護機構，超過會被夾住）
AXES = {
    #          sid  dir   scale center  lo    hi
    "yaw":   dict(sid=1, dir=+1, scale=1.0, center=2047, lo=1000, hi=3000),  # 左右轉頭(1000=往左轉到底)
    "pitch": dict(sid=2, dir=+1, scale=1.0, center=2047, lo=1500, hi=3000),  # 上下(1500=低頭最下)
    "roll":  dict(sid=3, dir=+1, scale=1.0, center=2047, lo=1500, hi=2500),  # 側傾(1500=向右傾)
}

# 動態參數
LOOP_HZ = 50           # 控制迴圈頻率
FILTER_ALPHA = 0.35    # 低通濾波 0~1，越小越穩但越延遲
MAX_SPEED_DPS = 180    # 每軸最大角速度(deg/s)，限速防過衝/防撞限位
SERVO_SPEED_U = 0      # 伺服內部目標速度(0=最大，實際速度由上面限速器控制)
SERVO_ACC_U = 30       # 伺服加速度單位
# ============================================================================


def _find_ch343():
    """自動找 CH343 轉接器的 COM 埠（VID 1A86 / PID 55D3，或描述含 CH343）。"""
    try:
        import serial.tools.list_ports as lp
        for p in lp.comports():
            if (p.vid == 0x1A86 and p.pid == 0x55D3) or "CH343" in (p.description or ""):
                return p.device
    except Exception:
        pass
    return None


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class Axis:
    def __init__(self, name, sid, dir=+1, scale=1.0, center=2047, lo=0, hi=4095):
        self.name = name
        self.sid = sid
        self.dir = dir
        self.scale = scale
        self.center = center
        self.lo = lo
        self.hi = hi
        self.head = float(center)    # 來自 IMU 的目標(tick，已夾限位)
        self.filt = float(center)    # 濾波後
        self.target = float(center)  # 實際下發(tick)

    def set_deg(self, deg):
        tick = self.center + self.dir * self.scale * deg * TICKS_PER_DEG
        self.head = clamp(tick, self.lo, self.hi)

    def update(self, sts, max_step):
        self.filt += FILTER_ALPHA * (self.head - self.filt)
        goal = clamp(self.filt, self.lo, self.hi)
        d = clamp(goal - self.target, -max_step, max_step)
        self.target += d
        sts.WritePosEx(self.sid, int(self.target), SERVO_SPEED_U, SERVO_ACC_U)


class NeckGimbal:
    def __init__(self, port=PORT, baud=BAUD):
        self.port = port
        self.baud = baud
        self.ser = None
        self.sts = None
        self.axes = {name: Axis(name, **cfg) for name, cfg in AXES.items()}
        self._max_step = MAX_SPEED_DPS * TICKS_PER_DEG / LOOP_HZ   # tick/迴圈
        self._run = False
        self._thread = None

    def connect(self):
        port = _find_ch343() or self.port      # 自動找 CH343，不管它是 COM5/6/8…
        self.ser = serial.Serial(port, self.baud, bytesize=8,
                                 parity='N', stopbits=1, timeout=0.02)
        self.port = port
        self.sts = SMS_STS(self.ser)
        for ax in self.axes.values():
            self.sts.EnableTorque(ax.sid, 1)

    def set_head_pose(self, yaw_deg, pitch_deg, roll_deg):
        """手機 IMU 進來：yaw→左右(id1), pitch→上下(id2), roll→側傾(id3)。"""
        self.axes["yaw"].set_deg(yaw_deg)
        self.axes["pitch"].set_deg(pitch_deg)
        self.axes["roll"].set_deg(roll_deg)

    def center(self, settle=1.2):
        """緩慢回到零位(2047)，避免上電瞬間暴衝。"""
        for ax in self.axes.values():
            ax.head = ax.filt = ax.target = float(ax.center)
            self.sts.WritePosEx(ax.sid, ax.center, 300, 20)
        time.sleep(settle)

    def _loop(self):
        dt = 1.0 / LOOP_HZ
        while self._run:
            t0 = time.perf_counter()
            for ax in self.axes.values():
                ax.update(self.sts, self._max_step)
            time.sleep(max(0, dt - (time.perf_counter() - t0)))

    def start(self):
        if self.sts is None:
            self.connect()
        self.center()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            self.ser.close()


# ── 命令列工具 ──────────────────────────────────────────────────────────────
def _check():
    """只讀目前位置，不移動。確認 COM 埠 / ID 連得到。"""
    port = _find_ch343() or PORT
    ser = serial.Serial(port, BAUD, bytesize=8, parity='N', stopbits=1, timeout=0.1)
    sts = SMS_STS(ser)
    print(f"讀取 {port} 上的伺服位置（不會移動）:")
    for name, cfg in AXES.items():
        pos = sts.ReadPos(cfg["sid"])
        ok = "OK" if pos is not None else "讀不到(檢查ID/接線/電源)"
        print(f"  {name:5s} id{cfg['sid']}  位置={pos}  {ok}  限位[{cfg['lo']},{cfg['hi']}]")
    ser.close()


def _test():
    """連線→回中心→各軸在限位內輕輕掃動。"""
    print("⚠️ 3 秒後連線並輕輕掃動三軸（Ctrl+C 取消）")
    for i in range(3, 0, -1):
        print(f"  {i}…", end="\r"); time.sleep(1)
    neck = NeckGimbal()
    neck.start()
    try:
        # 在各軸限位內，用「度」掃動（會被 set_deg 換算並夾限位）
        print("\nyaw 左右")
        for d in list(range(0, 30)) + list(range(30, -30, -1)) + list(range(-30, 1)):
            neck.set_head_pose(d, 0, 0); time.sleep(0.02)
        print("pitch 上下")
        for d in list(range(0, 25)) + list(range(25, -25, -1)) + list(range(-25, 1)):
            neck.set_head_pose(0, d, 0); time.sleep(0.02)
        print("roll 側傾")
        for d in list(range(0, 20)) + list(range(20, -20, -1)) + list(range(-20, 1)):
            neck.set_head_pose(0, 0, d); time.sleep(0.02)
        neck.set_head_pose(0, 0, 0); time.sleep(0.5)
        print("完成。")
    finally:
        neck.stop()


def _demo():
    """展示模式：上下點頭(前後) → 左右轉頭 → 自然擺盪，循環播放。"""
    print("展示模式：前後 → 左右 → 自然擺盪（循環，Ctrl+C 結束）")
    print("  3 秒後開始…")
    for i in range(3, 0, -1):
        print(f"  {i}…", end="\r"); time.sleep(1)
    neck = NeckGimbal()
    neck.start()

    def play(duration, fn, label):
        print(f"\n  >> {label}")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            e = time.perf_counter() - t0
            y, p, r = fn(e)
            neck.set_head_pose(y, p, r)
            time.sleep(0.02)

    try:
        while True:
            # 1) 前後：上下點頭（pitch 正弦）
            play(6, lambda e: (0, 16 * math.sin(2 * math.pi * e / 3), 0), "上下點頭 (前後)")
            # 2) 左右：轉頭（yaw 正弦）
            play(7, lambda e: (35 * math.sin(2 * math.pi * e / 3.5), 0, 0), "左右轉頭")
            # 3) 自然擺盪：三軸不同頻率/相位疊加，像人自然張望
            play(14, lambda e: (
                25 * math.sin(2 * math.pi * e / 4.0),
                7 * math.sin(2 * math.pi * e / 3.0 + 1.0),
                9 * math.sin(2 * math.pi * e / 5.0 + 2.0),
            ), "自然擺盪")
            # 回中心稍停
            neck.set_head_pose(0, 0, 0)
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\n結束展示，回中心。")
        neck.set_head_pose(0, 0, 0)
        time.sleep(0.8)
    finally:
        neck.stop()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--check"
    if mode == "--check":
        _check()
    elif mode == "--test":
        _test()
    elif mode == "--demo":
        _demo()
    else:
        print("用法: python neck_gimbal.py [--check | --test | --demo]")
