# -*- coding: utf-8 -*-
"""Q-BOT upper-body (arm) servo configuration.

Standalone module for the upper-body Feetech servo control — kept separate from
the leg/policy deployment on purpose. Develop + verify here first; wire into the
full deployment only once it's proven.

Hardware: 8 arm joints, Feetech STS3215 (SMS_STS protocol).
Gearbox:  4:1 planetary on the 3 SHOULDER DOF (shoulder / lateral_raise /
          arm_twist); elbow is direct-drive (1:1). Matches the training model
          (QBOT_TRAIN/qbot.xml) and the ksim ZEROS arm order.

Joint <-> servo mapping (same scheme as Red-Rabbit rx1_motor):
    servo_steps = round(angle_rad * STEPS_PER_RAD * gear * dir) + CENTER + home
    servo_speed = round(joint_speed_rad_s * gear * SPEED_CONST)
    servo_acc   = round(joint_acc * gear * ACC_CONST)
Because gear*angle can exceed one turn (e.g. 90 deg * 4 = 360 deg of servo
travel), the STS3215 must run in MULTI-TURN / step position mode.

>>> HARDWARE-SPECIFIC values below (id / dir / home) are PLACEHOLDERS <<<
Fill them in from the real robot (see README "Bring-up" section). gear/limits
are known and correct.
"""
import math

# --- Feetech STS3215 constants ---
STS_RESOLUTION = 4096                       # steps per revolution (12-bit)
STEPS_PER_RAD = STS_RESOLUTION / (2 * math.pi)   # 651.9 steps/rad  (== 2048/pi)
CENTER = 2048                               # step at servo zero (mid of one turn)
# rad/s -> Feetech speed units, rad/s^2 -> acc units (Feetech manual / rx1_motor)
SPEED_CONST = 652.6051999582332
ACC_CONST = 6.526051999582332

# Per-joint config. Order matches the policy's arm outputs (ksim ZEROS arms).
#   id   : Feetech servo bus id            (PLACEHOLDER - set from robot)
#   gear : reduction ratio                 (KNOWN: shoulder x4, elbow x1)
#   dir  : +1 / -1 to match mounting       (PLACEHOLDER - set during bring-up)
#   home : extra step offset at angle 0    (PLACEHOLDER - calibrate)
#   lo/hi: joint software limits [rad]     (KNOWN, from qbot.xml)
JOINTS = {
    # right arm
    "ub_right_shoulder":      dict(id=11, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_right_lateral_raise": dict(id=12, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_right_arm_twist":     dict(id=13, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_right_elbow":         dict(id=14, gear=1, dir=+1, home=0, lo=0.0,     hi=2.35619),
    # left arm
    "ub_left_shoulder":       dict(id=21, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_left_lateral_raise":  dict(id=22, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_left_arm_twist":      dict(id=23, gear=4, dir=+1, home=0, lo=-1.5708, hi=1.5708),
    "ub_left_elbow":          dict(id=24, gear=1, dir=+1, home=0, lo=0.0,     hi=2.35619),
}

# Joint order the walking policy emits its 8 arm targets in (must match training).
ARM_ORDER = list(JOINTS.keys())

# Training start/hold pose (rad) — from QBOT_TRAIN/train_qbot.py ZEROS arms.
HOME_POSE_RAD = {
    "ub_right_shoulder": 0.0,
    "ub_right_lateral_raise": math.radians(80.0),
    "ub_right_arm_twist": math.radians(80.0),
    "ub_right_elbow": math.radians(67.5),
    "ub_left_shoulder": 0.0,
    "ub_left_lateral_raise": math.radians(-80.0),
    "ub_left_arm_twist": math.radians(80.0),
    "ub_left_elbow": math.radians(67.5),
}

# Serial port of the Feetech controller board (PLACEHOLDER).
SERVO_PORT = "/dev/ttyUSB0"
SERVO_BAUD = 1000000
