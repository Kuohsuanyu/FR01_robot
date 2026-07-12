"""Exo channel → robot MJCF joint mapping.

Assumes the exoskeleton uses Feetech STS3215 servos as absolute-position
encoders (same protocol as the FR01 arms), one per DOF, on a single
daisy-chain at /dev/ttyACM0.  If the real hardware uses a different bus,
just change SERVO_PORT / SERVO_BAUD and keep the CHANNELS layout.

Each channel:
    sid          - Feetech ID on the daisy-chain
    joint        - MJCF joint name it drives
    tick_zero    - tick reading when the operator stands neutral (arms down,
                   head straight).  Set once at commissioning with the
                   `save_zero` op below.
    ticks_per_rad- signed calibration constant.  Sign encodes direction:
                   +1 means "tick increasing = MJCF q increasing";
                   -1 means "tick increasing = MJCF q decreasing".
    q_min / q_max- clipping range (rad, MJCF frame) to keep ghost/motor
                   inside safe joint limits.
"""
from __future__ import annotations
import math

SERVO_PORT = "/dev/ttyACM0"
SERVO_BAUD = 1_000_000
POLL_HZ    = 50               # broadcast rate on /pose WS
SCAN_RANGE = (1, 30)          # daisy-chain scan range

# STS3215 has 4096 ticks / 360°  → 2π rad
STS_TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)

# All values below are placeholders — recalibrate against the real exo
# before enabling LIVE mode.  Use exo_rpi_agent/agent.py's `save_zero` op
# once the operator is standing neutral.
CHANNELS = [
    # sid, joint,                    tick_zero, ticks_per_rad,       q_min,   q_max
    dict(sid=1,  joint="ub_neck_yaw",             tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=2,  joint="ub_neck_pitch",           tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/6, q_max=+math.pi/6),
    dict(sid=3,  joint="ub_neck_roll",            tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/6, q_max=+math.pi/6),
    dict(sid=10, joint="ub_right_shoulder",       tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=11, joint="ub_right_lateral_raise",  tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=12, joint="ub_right_arm_twist",      tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=13, joint="ub_right_elbow",          tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min= 0.0,       q_max=+2.35),
    dict(sid=20, joint="ub_left_shoulder",        tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=21, joint="ub_left_lateral_raise",   tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=22, joint="ub_left_arm_twist",       tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min=-math.pi/2, q_max=+math.pi/2),
    dict(sid=23, joint="ub_left_elbow",           tick_zero=2048, ticks_per_rad= +STS_TICKS_PER_RAD, q_min= 0.0,       q_max=+2.35),
]

ZERO_STORAGE = "~/exo_zero.json"     # tick_zero override cached here after `save_zero`


def tick_to_q(tick: int, ch: dict) -> float:
    """Convert raw tick to MJCF-frame q (rad), clipped to channel range."""
    q = (tick - ch["tick_zero"]) / ch["ticks_per_rad"]
    return max(ch["q_min"], min(ch["q_max"], q))
