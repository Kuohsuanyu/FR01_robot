# -*- coding: utf-8 -*-
"""Gear-aware Feetech arm controller for the Q-BOT upper body.

Converts joint angles (rad, the policy's native units) into Feetech STS3215
servo commands, accounting for the per-joint gear reduction (4x on the shoulder
DOF). Talks to the bus through the Feetech `scservo_sdk` if available; otherwise
falls back to a MockBus that just records the commands, so the mapping can be
verified WITHOUT any hardware.

Usage:
    arm = FeetechArm()                 # auto: real bus if SDK+port, else mock
    arm.connect()
    arm.set_pose_rad({"ub_right_elbow": 1.0, ...}, speed_rad_s=0.8)
    arm.go_home()                      # move to the training start pose
"""
import math
from config import (JOINTS, ARM_ORDER, HOME_POSE_RAD, STEPS_PER_RAD, CENTER,
                    SPEED_CONST, ACC_CONST, SERVO_PORT, SERVO_BAUD)


# ----------------------------- angle <-> steps --------------------------------
def angle_to_steps(name: str, angle_rad: float) -> int:
    """Joint angle [rad] -> absolute Feetech step target (multi-turn)."""
    j = JOINTS[name]
    angle_rad = max(j["lo"], min(j["hi"], angle_rad))           # clamp to limits
    return int(round(angle_rad * STEPS_PER_RAD * j["gear"] * j["dir"])) + CENTER + j["home"]


def steps_to_angle(name: str, steps: int) -> float:
    """Inverse of angle_to_steps (for reading feedback)."""
    j = JOINTS[name]
    return (steps - CENTER - j["home"]) / (STEPS_PER_RAD * j["gear"] * j["dir"])


def speed_to_units(name: str, speed_rad_s: float) -> int:
    return int(round(abs(speed_rad_s) * JOINTS[name]["gear"] * SPEED_CONST))


def acc_to_units(name: str, acc_rad_s2: float) -> int:
    return int(round(abs(acc_rad_s2) * JOINTS[name]["gear"] * ACC_CONST))


# --------------------------------- buses --------------------------------------
class MockBus:
    """Stand-in for the Feetech SDK: records the last SyncWrite so the mapping
    can be checked with no hardware/SDK installed."""

    def __init__(self):
        self.last = {}

    def connect(self):
        print("[MockBus] no Feetech SDK / port — running in DRY-RUN (no hardware).")

    def sync_write(self, cmds):
        self.last = {c["id"]: c for c in cmds}
        for c in cmds:
            print(f"  [dry] id={c['id']:<3} pos={c['pos']:<7} "
                  f"speed={c['speed']:<6} acc={c['acc']}")

    def close(self):
        pass


class FeetechBus:
    """Real bus via scservo_sdk (Feetech STS/SMS). SyncWritePosEx per command."""

    def __init__(self, port=SERVO_PORT, baud=SERVO_BAUD):
        self.port, self.baud = port, baud
        self._h = None

    def connect(self):
        import scservo_sdk as scs  # feetech-servo-sdk
        self._scs = scs
        self._ph = scs.PortHandler(self.port)
        self._pk = scs.sms_sts(self._ph)
        if not self._ph.openPort():
            raise IOError(f"failed to open {self.port}")
        if not self._ph.setBaudRate(self.baud):
            raise IOError(f"failed to set baud {self.baud}")
        print(f"[FeetechBus] connected {self.port} @ {self.baud}")

    def sync_write(self, cmds):
        gsw = self._scs.GroupSyncWrite(
            self._pk, self._scs.SMS_STS_ACC, 7)  # acc(1)+pos(2)+time(2)+speed(2)
        for c in cmds:
            # WritePosEx packs target/speed/acc for one servo
            self._pk.SyncWritePosEx(c["id"], c["pos"], c["speed"], c["acc"])
        self._pk.groupSyncWrite.txPacket() if hasattr(self._pk, "groupSyncWrite") else None

    def close(self):
        if self._h:
            self._ph.closePort()


def _make_bus():
    try:
        import scservo_sdk  # noqa: F401
        import os
        if os.path.exists(SERVO_PORT):
            return FeetechBus()
    except Exception:
        pass
    return MockBus()


# ------------------------------- controller -----------------------------------
class FeetechArm:
    def __init__(self, bus=None):
        self.bus = bus or _make_bus()

    def connect(self):
        self.bus.connect()

    def _commands(self, pose_rad, speed_rad_s, acc_rad_s2):
        cmds = []
        for name, ang in pose_rad.items():
            if name not in JOINTS:
                raise KeyError(f"unknown joint {name}")
            cmds.append(dict(
                id=JOINTS[name]["id"],
                pos=angle_to_steps(name, ang),
                speed=speed_to_units(name, speed_rad_s),
                acc=acc_to_units(name, acc_rad_s2),
            ))
        return cmds

    def set_pose_rad(self, pose_rad: dict, speed_rad_s: float = 0.8, acc_rad_s2: float = 5.0):
        """Command a set of joint angles (rad). speed/acc are at the JOINT side;
        they get multiplied by the gear ratio per joint for the servo."""
        self.bus.sync_write(self._commands(pose_rad, speed_rad_s, acc_rad_s2))

    def set_arm_vector(self, angles8, speed_rad_s: float = 0.8, acc_rad_s2: float = 5.0):
        """Command the 8 arm joints from a flat vector in ARM_ORDER (i.e. the
        policy's arm outputs)."""
        assert len(angles8) == len(ARM_ORDER), f"expected {len(ARM_ORDER)} angles"
        self.set_pose_rad(dict(zip(ARM_ORDER, angles8)), speed_rad_s, acc_rad_s2)

    def go_home(self, speed_rad_s: float = 0.5):
        """Move to the training start/hold pose (slow)."""
        self.set_pose_rad(dict(HOME_POSE_RAD), speed_rad_s=speed_rad_s)

    def close(self):
        self.bus.close()
