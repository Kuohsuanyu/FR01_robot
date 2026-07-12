# -*- coding: utf-8 -*-
"""FR01 巧手伺服設定 — 5 隻手指 / 6 顆馬達。

**這是右手手掌 (right hand)。**

出自 2026-07-02 現場逐顆搖桿測試 (見 finger_map.json)。
硬體:Feetech BUS servo,同一條 TTL 匯流排混掛兩系列
      * SCS009 (model 1029) → scservo_sdk.scscl handler,big-endian,10-bit tick
      * STS3032 (model 521)  → scservo_sdk.sms_sts handler,little-endian,12-bit tick
匯流排:CH340 (USB 序號 5B14115243) → /dev/ttyACM0 @ 1 Mbps

方向規則 (現場實測):
      * 31 小拇指  變高是縮 → open=low, close=high
      * 32 無名指  變低是縮 → open=high, close=low  (反向)
      * 33 中指    變高是縮 → open=low, close=high
      * 34 食指    變低是縮 → open=high, close=low  (反向)
      * 35 拇指轉動 固定 650 (不入動態控制)
      * 36 拇指開合 變低是縮 → open=high, close=low  (反向)
"""

HAND_SIDE = "right"

SERVO_PORT = "/dev/ttyACM0"
SERVO_BAUD = 1_000_000

# --- Active fingers ---
# id      : Feetech bus id
# model   : 由 ping 回來的型號 (決定 handler / byte order)
# open_tick, close_tick : 手指張開 / 握起兩端 tick
#   * 若 open > close 表示裝配方向反相 (數字越大手指越張開)
# 目前先把 EPROM min/max 當 open/close 用;下一輪再逐顆精修行程。
FINGERS = {
    "pinky":  dict(id=31, model=1029, open_tick=515,  close_tick=780),   # 小拇指
    "ring":   dict(id=32, model=1029, open_tick=518,  close_tick=240),   # 無名指 (反向:數字大=張開)
    "middle": dict(id=33, model=1029, open_tick=430,  close_tick=700),   # 中指
    "index":  dict(id=34, model=521,  open_tick=3400, close_tick=2650),  # 食指 (反向)
    "thumb":  dict(id=36, model=521,  open_tick=2660, close_tick=2200),  # 拇指開合 (反向)
}

# --- Fixed / trim servos ---
# 上電時要保持在指定 tick,不列入動態控制。
# ID 35 = 拇指轉動軸,現場測試後固定在 650 (面向掌心)。
FIXED = {
    "thumb_rotate": dict(id=35, model=1029, hold_tick=650),
}

# 便於外部使用
ACTIVE_IDS = [c["id"] for c in FINGERS.values()]
FIXED_IDS = [c["id"] for c in FIXED.values()]
ALL_IDS = ACTIVE_IDS + FIXED_IDS


def is_scscl(model: int) -> bool:
    """model 1029 = SCS009 (big-endian);其他 (521 等 STS 系) 走 sms_sts。"""
    return model == 1029


def finger_range(name: str):
    """回傳 (lo_tick, hi_tick):不管 open/close 誰大,固定為排序後 (低, 高)。
    給滑桿邊界檢查用。"""
    c = FINGERS[name]
    lo, hi = c["open_tick"], c["close_tick"]
    return (lo, hi) if lo <= hi else (hi, lo)


def tick_from_norm(name: str, norm: float) -> int:
    """把 0.0 (張開) ~ 1.0 (握起) 的正規化值轉成 tick。自動處理反向。"""
    c = FINGERS[name]
    o, k = c["open_tick"], c["close_tick"]
    norm = max(0.0, min(1.0, norm))
    return int(round(o + (k - o) * norm))
