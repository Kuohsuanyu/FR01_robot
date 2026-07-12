# -*- coding: utf-8 -*-
"""手臂關節↔馬達 校正的『唯讀真相來源』。

重點:這裡的 tick 校正(兩點線性 q↔step)與 qbot_ik_gui 的 `_q_to_step` **完全相同**,
且讀的是同一份 `qbot_arm_calibration.json`(你已手動調好的)。任何連動/橋接工具都
import 這裡,保證零位/方向/縮放和 IK 一致 —— 不重新推算任何零位。

僅提供純函式,不含 GUI,可安全被 bridge / 測試 import。
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "qbot_arm_calibration.json")

# 手臂關節規格(與 qbot_ik_gui.ARMS 一致):index → (sid, 名稱, lo, hi, MJCF關節名)
# lo/hi 是『真實安全限位』(rad),連動送手臂前用它夾;wrist 無 MJCF 關節(jname=None)。
ARM_SPEC = {
    "R": [
        (10, "Shoul Rot",  -1.5708, 1.5708, "ub_right_shoulder"),
        (11, "Shoul Lift", -1.5708, 1.5708, "ub_right_lateral_raise"),
        (12, "Arm Twist",  -1.5708, 1.5708, "ub_right_arm_twist"),
        (13, "Elbow",       0.0,    2.35619, "ub_right_elbow"),
        (14, "Wrist",      -3.1416, 3.1416, None),
    ],
    "L": [
        (20, "Shoul Rot",  -1.5708, 1.5708, "ub_left_shoulder"),
        (21, "Shoul Lift", -1.5708, 1.5708, "ub_left_lateral_raise"),
        (22, "Arm Twist",  -1.5708, 1.5708, "ub_left_arm_twist"),
        (23, "Elbow",       0.0,    2.35619, "ub_left_elbow"),
        (24, "Wrist",      -3.1416, 3.1416, None),
    ],
}

# exo 關節名 → (arm, index)。exo 的 wrist/grip 不屬手臂(巧手),不在此。
EXO_JOINT_TO_ARM = {
    "ub_right_shoulder":      ("R", 0), "ub_right_lateral_raise": ("R", 1),
    "ub_right_arm_twist":     ("R", 2), "ub_right_elbow":         ("R", 3),
    "ub_left_shoulder":       ("L", 0), "ub_left_lateral_raise":  ("L", 1),
    "ub_left_arm_twist":      ("L", 2), "ub_left_elbow":          ("L", 3),
}


def _default_points(arm):
    # 沒有校正檔時的保守預設(q=0↔tick2048,滿量程 ±π);正式應存在校正檔。
    import math
    return [{"tick_lo": 0.0, "q_lo": -math.pi, "tick_hi": 4095.0, "q_hi": math.pi}
            for _ in range(5)]


def load_cal(path: str = CAL_PATH) -> dict:
    """回傳 {'L': [5 points], 'R': [5 points]}(與 IK 同格式)。"""
    if not os.path.exists(path):
        return {"L": _default_points("L"), "R": _default_points("R")}
    d = json.load(open(path, encoding="utf-8"))
    return {arm: d.get(arm, {}).get("points", _default_points(arm)) for arm in ("L", "R")}


def q_to_step(cal: dict, arm: str, i: int, q: float) -> int:
    """q(rad) → 馬達 tick。與 qbot_ik_gui._q_to_step 逐字相同。"""
    p = cal[arm][i]
    denom = p["q_hi"] - p["q_lo"]
    if abs(denom) < 1e-9:
        return int(round(p["tick_lo"]))
    return int(round(p["tick_lo"] + (q - p["q_lo"]) * (p["tick_hi"] - p["tick_lo"]) / denom))


def clamp_to_limits(arm: str, i: int, q: float) -> tuple[float, bool]:
    """把 q 夾進手臂真實限位。回傳 (夾後q, 是否被夾)。"""
    lo, hi = ARM_SPEC[arm][i][2], ARM_SPEC[arm][i][3]
    c = max(lo, min(hi, q))
    return c, (c != q)
