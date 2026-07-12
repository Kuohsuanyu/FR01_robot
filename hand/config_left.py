# -*- coding: utf-8 -*-
"""FR01 巧手伺服設定 — LEFT hand (5 隻手指 / 6 顆馬達).

**這是左手手掌 (left hand).**

出自 2026-07-05 現場逐顆搖桿測試 (見 finger_map_left.json)。
硬體:Feetech BUS servo,同一條 TTL 匯流排混掛兩系列
      * SCS009 (model 1029) → scservo_sdk.scscl handler,big-endian,10-bit tick
      * STS3032 (model 521)  → scservo_sdk.sms_sts handler,little-endian,12-bit tick
匯流排:CH340 → /dev/ttyACM0 @ 1 Mbps

方向規則 (現場實測):
      * 41 食指 (STS3032)  數值大是縮起來 → open=low, close=high
      * 42 小拇指 (SCS009) 數值小是縮起來 → open=high, close=low  (反向)
      * 43 無名指 (SCS009) 數值大是縮起來 → open=low, close=high
      * 44 中指 (SCS009)   數值小是縮起來 → open=high, close=low  (反向)
      * 45 拇指轉動 (SCS009) 固定 180 (不入動態控制)
      * 46 拇指開合 (STS3032) 數值大是縮起來 → open=low, close=high
"""

HAND_SIDE = "left"

SERVO_PORT = "/dev/ttyACM0"
SERVO_BAUD = 1_000_000

# EPROM min/max tick 由 read2ByteTxRx(addr=9/11) 讀出,方向依現場測試填入
FINGERS = {
    "pinky":  dict(id=42, model=1029, open_tick=550,  close_tick=110),   # 小拇指 (反向)
    "ring":   dict(id=43, model=1029, open_tick=410,  close_tick=840),   # 無名指
    "middle": dict(id=44, model=1029, open_tick=550,  close_tick=200),   # 中指 (反向)
    "index":  dict(id=41, model=521,  open_tick=1800, close_tick=2700),  # 食指
    "thumb":  dict(id=46, model=521,  open_tick=1300, close_tick=2500),  # 拇指開合
}

# 上電時保持在指定 tick,不列入動態控制
# ID 45 = 拇指轉動軸,現場測試後固定在 180 (面向掌心)
FIXED = {
    "thumb_rotate": dict(id=45, model=1029, hold_tick=180),
}

ACTIVE_IDS = [c["id"] for c in FINGERS.values()]
FIXED_IDS = [c["id"] for c in FIXED.values()]
ALL_IDS = ACTIVE_IDS + FIXED_IDS


def is_scscl(model: int) -> bool:
    """model 1029 = SCS009 (big-endian);其他 (521 等 STS 系) 走 sms_sts."""
    return model == 1029


def finger_range(name: str):
    """回傳 (lo_tick, hi_tick):不管 open/close 誰大,固定為排序後 (低, 高)。"""
    c = FINGERS[name]
    lo, hi = c["open_tick"], c["close_tick"]
    return (lo, hi) if lo <= hi else (hi, lo)


def tick_from_norm(name: str, norm: float) -> int:
    """把 0.0 (張開) ~ 1.0 (握起) 的正規化值轉成 tick,自動處理反向。"""
    c = FINGERS[name]
    o, k = c["open_tick"], c["close_tick"]
    norm = max(0.0, min(1.0, norm))
    return int(round(o + (k - o) * norm))
