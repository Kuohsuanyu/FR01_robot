# -*- coding: utf-8 -*-
"""Verify the joint<->servo mapping WITHOUT hardware.

Checks: round-trip angle->steps->angle, the 4x gear amplification on the
shoulder DOF, multi-turn range needed, and prints the servo commands for the
training home pose.
"""
import math
from config import JOINTS, ARM_ORDER, HOME_POSE_RAD, CENTER, STEPS_PER_RAD
from feetech_arm import angle_to_steps, steps_to_angle, speed_to_units, FeetechArm

print(f"STEPS_PER_RAD = {STEPS_PER_RAD:.2f}  (2048/pi = {2048/math.pi:.2f})\n")

print("=== round-trip + gear check ===")
ok = True
for name in ARM_ORDER:
    j = JOINTS[name]
    a = 0.5 * (j["lo"] + j["hi"]) or 0.7   # a mid-ish test angle
    steps = angle_to_steps(name, a)
    back = steps_to_angle(name, steps)
    off = steps - CENTER - j["home"]
    err = abs(back - max(j["lo"], min(j["hi"], a)))
    ok &= err < 1e-3
    print(f"{name:24} gear={j['gear']}  a={a:+.3f}rad -> {steps:6d} steps "
          f"(offset {off:+5d}) -> {back:+.3f}rad  err={err:.1e}")
print("round-trip:", "OK" if ok else "FAIL", "\n")

print("=== multi-turn range (servo travel at joint limits) ===")
for name in ARM_ORDER:
    j = JOINTS[name]
    s_lo = angle_to_steps(name, j["lo"]) - CENTER - j["home"]
    s_hi = angle_to_steps(name, j["hi"]) - CENTER - j["home"]
    turns = (max(abs(s_lo), abs(s_hi)) / 4096.0)
    note = "  <-- needs MULTI-TURN mode" if turns > 0.5 else ""
    print(f"{name:24} servo offset range [{s_lo:+6d},{s_hi:+6d}]  "
          f"= {turns:.2f} turns{note}")
print()

print("=== home pose -> servo commands (speed 0.5 rad/s) ===")
arm = FeetechArm()           # mock bus (no hardware) -> prints commands
arm.connect()
arm.go_home(speed_rad_s=0.5)
print("\nNote: shoulder joints' servo 'speed' units are 4x the elbow's for the "
      "same joint speed (gear). Servo max speed therefore caps the JOINT speed "
      "at servo_max/4 ~ 1.18 rad/s (see README).")
