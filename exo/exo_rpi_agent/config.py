"""Exo channel → robot MJCF joint mapping.  Mid-calibration state."""
from __future__ import annotations
import math

SERVO_PORT = '/dev/ttyACM0'
SERVO_BAUD = 1_000_000
POLL_HZ    = 50
SCAN_RANGE = (1, 100)

STS_TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)

# q_min / q_max intentionally very wide (±3.2 rad) — clipping disabled until
# the exo→ghost mapping is validated per joint.  q_offset shifts the
# reading so exo neutral (arms at sides) maps to the ghost's chosen zero.
CHANNELS = [
    # ── right arm ──
    dict(sid=1,  joint='ub_right_shoulder',       tick_zero=233, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=-0.491639,             q_min=-3.2, q_max=+3.2),
    dict(sid=56, joint='ub_right_arm_twist',      tick_zero=453, ticks_per_rad=-STS_TICKS_PER_RAD, q_offset=-1.813200,             q_min=-3.2, q_max=+3.2),
    dict(sid=55, joint='ub_right_elbow',          tick_zero=3599, ticks_per_rad=-STS_TICKS_PER_RAD, q_offset=-0.122758,             q_min=-3.2, q_max=+3.2),
    dict(sid=66, joint='ub_right_lateral_raise',  tick_zero=161, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=1.138476,      q_min=-3.2, q_max=+3.2),
    dict(sid=64, joint='exo_right_wrist',         tick_zero=237,  ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
    dict(sid=65, joint='exo_right_grip',          tick_zero=2428, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
    # ── left arm (symmetry-guessed) ──
    dict(sid=2,  joint='ub_left_shoulder',        tick_zero=3327, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
    dict(sid=63, joint='ub_left_arm_twist',      tick_zero=3693, ticks_per_rad=-STS_TICKS_PER_RAD, q_offset=1.476817,      q_min=-3.2, q_max=+3.2),
    dict(sid=54, joint='ub_left_lateral_raise',  tick_zero=564,  ticks_per_rad=-STS_TICKS_PER_RAD, q_offset=-math.pi/2,      q_min=-3.2, q_max=+3.2),
    dict(sid=57, joint='ub_left_elbow',           tick_zero=2222, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
    dict(sid=53, joint='exo_left_grip' ,          tick_zero=2540, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
    dict(sid=67, joint='exo_left_wrist',           tick_zero=3877, ticks_per_rad=+STS_TICKS_PER_RAD, q_offset=0.0,             q_min=-3.2, q_max=+3.2),
]

ZERO_STORAGE = '~/exo_zero.json'


def tick_to_q(tick: int, ch: dict) -> float:
    # Wraparound: STS3215 encoder is 0-4095 single-turn.  A joint near
    # tick 0 or 4095 that moves across the boundary reports a delta of
    # ~+/-4096 relative to tick_zero — snap it back into the -2048..+2048
    # window so a small physical move never produces a huge q spike.
    delta = int(tick) - int(ch['tick_zero'])
    if delta > 2048:  delta -= 4096
    elif delta < -2048: delta += 4096
    q = delta / ch['ticks_per_rad'] + ch.get('q_offset', 0.0)
    return max(ch['q_min'], min(ch['q_max'], q))
