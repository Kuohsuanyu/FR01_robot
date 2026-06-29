"""
RX1 humanoid robot motor controller — Windows standalone (no ROS).

Translated from rx1_motor_lib (C++) and rx1_motor (ROS node).
Uses feetech_lib.py for serial communication with Feetech STS/SCS servos.

Quick start:
    from rx1_motor_win import Rx1Motor
    robot = Rx1Motor("COM8")
    robot.initialize()
    robot.right_arm([0]*7)
"""

import math
import time
import serial

from feetech_lib import SMS_STS, SCSCL


class Rx1Motor:
    # ---- servo IDs, direction signs, gear ratios --------------------------------
    _RIGHT_ARM_IDS   = [11, 12, 13, 14, 15, 16, 17]
    _RIGHT_ARM_DIRS  = [-1, -1,  1,  1,  1,  1, -1]
    _RIGHT_ARM_GEARS = [ 3,  3,  3,  3,  1,  1,  1]

    _LEFT_ARM_IDS    = [21, 22, 23, 24, 25, 26, 27]
    _LEFT_ARM_DIRS   = [-1, -1,  1, -1,  1, -1, -1]
    _LEFT_ARM_GEARS  = [ 3,  3,  3,  3,  1,  1,  1]

    _HEAD_IDS        = [4, 5, 6, 7, 8]
    _HEAD_DIRS       = [-1, -1, -1,  1, -1]
    _HEAD_GEARS      = [ 1,  1,  1,  1,  1]

    _TORSO_IDS       = [1, 2, 3]
    _TORSO_DIRS      = [-1,  1, -1]
    _TORSO_GEARS     = [ 3,  3,  3]

    _RIGHT_HAND_IDS     = [31, 32, 33, 34, 35, 36]
    _RIGHT_HAND_DEFAULT = [200, 2400, 2300, 420, 600, 440]
    _RIGHT_HAND_RANGE   = [  0, -300, -400, 200,-200, 200]

    _LEFT_HAND_IDS      = [41, 42, 43, 44, 45, 46]
    _LEFT_HAND_DEFAULT  = [512, 1700, 1700, 650, 420, 630]
    _LEFT_HAND_RANGE    = [  0,  300,  500,-210, 210,-240]

    # ---- torso parallel-linkage constants ---------------------------------------
    _TORSO_D  = 0.0865
    _TORSO_L1 = 0.05
    _TORSO_H1 = 0.11
    _TORSO_H2 = 0.11

    # ---- default motion parameters ----------------------------------------------
    _TORSO_SPEED = 1.0;  _TORSO_ACC = 1.5
    _ARM_SPEED   = 1.0;  _ARM_ACC   = 3.0
    _HEAD_SPEED  = 1.0;  _HEAD_ACC  = 7.0
    _HAND_SPEED  = 0.0;  _HAND_ACC  = 15.0

    # ---- Feetech unit conversions -----------------------------------------------
    # 50 step/s = 0.732 RPM  →  1 step/s = 0.00153232 rad/s  →  1/0.00153232 ≈ 652.6
    _SPEED_K = 652.6051999582332
    # 1 acc unit = 100 step/s²  →  factor ≈ 6.526
    _ACC_K   = 6.526051999582332

    # -------------------------------------------------------------------------
    def __init__(self, port: str, baudrate: int = 1_000_000):
        self._port     = port
        self._baudrate = baudrate
        self._ser      = None
        self._sts      = None
        self._scs      = None

    def connect(self):
        """Open the serial port only — does NOT move any joint."""
        try:
            self._ser = serial.Serial(
                self._port, self._baudrate,
                bytesize=8, parity='N', stopbits=1,
                timeout=0.02
            )
        except serial.SerialException as e:
            raise RuntimeError(f"Cannot open {self._port}: {e}")
        self._sts = SMS_STS(self._ser)
        self._scs = SCSCL(self._ser)

    def initialize(self):
        """Open the serial port and drive every joint to its home position."""
        self.connect()

        # Arms and torso → center position (2048)
        for sid in (self._RIGHT_ARM_IDS + self._LEFT_ARM_IDS
                    + self._TORSO_IDS):
            self._sts.WritePosEx(sid, 2048, 200, 20)
            time.sleep(0.012)

        # Head servos
        for i, sid in enumerate(self._HEAD_IDS):
            if i == 0 or i == 2:
                self._sts.WritePosEx(sid, 2048, 200, 20)
            elif i == 1:
                self._sts.WritePosEx(sid, 1600, 200, 20)
            else:                           # ears (SCS)
                self._scs.WritePos(sid, 512, 0, 100)
            time.sleep(0.012)

    def close(self):
        """Close the serial connection."""
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ------------------------------------------------------------------ arms

    def right_arm(self, angles_rad: list, speeds=None, accs=None):
        """Command right arm.  angles_rad: 7 joint angles in radians."""
        n   = len(self._RIGHT_ARM_IDS)
        spd = speeds or [self._ARM_SPEED] * n
        acc = accs   or [self._ARM_ACC]   * n
        self._sync_cmd(self._RIGHT_ARM_IDS, self._RIGHT_ARM_DIRS,
                       self._RIGHT_ARM_GEARS, angles_rad, spd, acc)

    def left_arm(self, angles_rad: list, speeds=None, accs=None):
        """Command left arm.  angles_rad: 7 joint angles in radians."""
        n   = len(self._LEFT_ARM_IDS)
        spd = speeds or [self._ARM_SPEED] * n
        acc = accs   or [self._ARM_ACC]   * n
        self._sync_cmd(self._LEFT_ARM_IDS, self._LEFT_ARM_DIRS,
                       self._LEFT_ARM_GEARS, angles_rad, spd, acc)

    # ------------------------------------------------------------------ torso

    def torso(self, angles_rad: list, speeds=None, accs=None):
        """
        Command torso (3 joints).
        angles_rad: [yaw, pitch, roll] in radians.
        Parallel-linkage IK is applied internally.
        """
        n   = len(self._TORSO_IDS)
        spd = speeds or [self._TORSO_SPEED] * n
        acc = accs   or [self._TORSO_ACC]   * n

        joints = list(angles_rad)
        ik = self._torso_ik(self._TORSO_D, self._TORSO_L1,
                            self._TORSO_H1, self._TORSO_H2,
                            angles_rad[2], angles_rad[1])
        joints[1] = ik[0]
        joints[2] = ik[1]
        self._sync_cmd(self._TORSO_IDS, self._TORSO_DIRS,
                       self._TORSO_GEARS, joints, spd, acc)

    # ------------------------------------------------------------------ head

    def head(self, angles_rad: list, speeds=None, accs=None):
        """
        Command head (5 joints).
        joints 0-2: neck (STS), joints 3-4: ears (SCS).
        """
        n   = len(self._HEAD_IDS)
        spd = speeds or [self._HEAD_SPEED] * n
        acc = accs   or [self._HEAD_ACC]   * n

        for i, sid in enumerate(self._HEAD_IDS):
            speed_u = int(spd[i] * self._SPEED_K)
            acc_u   = int(acc[i] * self._ACC_K)
            if i in (0, 1, 2):  # neck — STS
                pos = int(angles_rad[i] / math.pi * 2048
                          * self._HEAD_DIRS[i] * self._HEAD_GEARS[i] + 2048)
                self._sts.WritePosEx(sid, pos, speed_u, acc_u)
            else:               # ears — SCS
                pos = int(angles_rad[i] / math.pi * 512
                          * self._HEAD_DIRS[i] * self._HEAD_GEARS[i] + 512)
                self._scs.WritePos(sid, pos, 0, 0)

    # ---------------------------------------------------------------- grippers

    def right_gripper(self, ratio: float):
        """Right gripper: 0.0 = fully open, 1.0 = fully closed."""
        speed_u = int(self._HAND_SPEED * self._SPEED_K)
        acc_u   = int(self._HAND_ACC   * self._ACC_K)
        for i, sid in enumerate(self._RIGHT_HAND_IDS):
            pos = int(self._RIGHT_HAND_DEFAULT[i]
                      + ratio * self._RIGHT_HAND_RANGE[i])
            if i in (1, 2):                     # thumb/index → STS
                self._sts.WritePosEx(sid, pos, speed_u, acc_u)
            elif i == 3:
                self._scs.WritePos(sid, pos, 0, 400)
            elif i == 4:
                self._scs.WritePos(sid, pos, 0, 300)
            elif i == 5:
                self._scs.WritePos(sid, pos, 0, 200)
        # thumb yaw
        self._scs.WritePos(self._RIGHT_HAND_IDS[0], 200, 0, 400)

    def left_gripper(self, ratio: float):
        """Left gripper: 0.0 = fully open, 1.0 = fully closed."""
        speed_u = int(self._HAND_SPEED * self._SPEED_K)
        acc_u   = int(self._HAND_ACC   * self._ACC_K)
        for i, sid in enumerate(self._LEFT_HAND_IDS):
            pos = int(self._LEFT_HAND_DEFAULT[i]
                      + ratio * self._LEFT_HAND_RANGE[i])
            if i in (1, 2):
                self._sts.WritePosEx(sid, pos, speed_u, acc_u)
            elif i == 3:
                self._scs.WritePos(sid, pos, 0, 400)
            elif i == 4:
                self._scs.WritePos(sid, pos, 0, 300)
            elif i == 5:
                self._scs.WritePos(sid, pos, 0, 200)
        # thumb yaw
        self._scs.WritePos(self._LEFT_HAND_IDS[0], 512, 0, 400)

    # ----------------------------------------------------------------- internal

    def _sync_cmd(self, ids, dirs, gears, angles, speeds, accs):
        """Build position/speed/acc arrays and issue a sync write."""
        n          = len(ids)
        pos_arr    = []
        speed_arr  = []
        acc_arr    = []
        for i in range(n):
            pos   = int(angles[i] / math.pi * 2048
                        * dirs[i] * gears[i] + 2048)
            spd   = int(speeds[i] * gears[i] * self._SPEED_K)
            ac    = int(accs[i]   * gears[i] * self._ACC_K)
            # Forearm joints (i >= 3): free speed, 10× acc
            if i >= 3:
                spd = 0
                ac *= 10
            pos_arr.append(pos)
            speed_arr.append(spd)
            acc_arr.append(ac)
        self._sts.SyncWritePosEx(ids, n, pos_arr, speed_arr, acc_arr)

    @staticmethod
    def _torso_ik(d, L1, h1, h2, roll, pitch):
        """Parallel-linkage IK for the torso (from the original C++ code)."""
        cx = math.cos(pitch); sx = math.sin(pitch)
        cy = math.cos(roll);  sy = math.sin(roll)

        AL = -L1*L1*cy + L1*d*sx*sy
        BL = -L1*L1*sy + L1*h1 - L1*d*sx*cy
        CL = -(L1*L1 + d*d - d*d*cx - L1*h1*sy - d*h1*sx*cy)
        LenL = math.sqrt(AL*AL + BL*BL)

        AR = -L1*L1*cy - L1*d*sx*sy
        BR = -L1*L1*sy + L1*h2 + L1*d*sx*cy
        CR = -(L1*L1 + d*d - d*d*cx - L1*h2*sy + d*h2*sx*cy)
        LenR = math.sqrt(AR*AR + BR*BR)

        if LenL <= abs(CL) or LenR <= abs(CR):
            return (0.0, 0.0)

        tL = math.asin(CL / LenL) - math.asin(AL / LenL)
        tR = math.asin(CR / LenR) - math.asin(AR / LenR)
        return (tL, tR)
