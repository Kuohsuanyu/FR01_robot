"""
Feetech SCServo protocol — Python/Windows implementation.

Packet format (write):
    0xFF 0xFF  ID  LEN  INST  ADDR  data...  ~CS
    LEN = len(INST + ADDR + data) + 1  (for CS byte)
    CS  = ~(ID + LEN + INST + ADDR + sum(data)) & 0xFF

Read request:
    0xFF 0xFF  ID  0x04  0x02  ADDR  COUNT  ~CS
Read response:
    0xFF 0xFF  ID  (COUNT+2)  ERROR  data[0..COUNT-1]  ~CS

Sync-write (SMS_STS only, no ACK):
    0xFF 0xFF  0xFE  ((dataLen+1)*IDN+4)  0x83  ADDR  dataLen
        [ID0 d0...]  [ID1 d1...]  ~SUM
"""

import serial
import time

# Instructions
INST_READ       = 0x02
INST_WRITE      = 0x03
INST_SYNC_WRITE = 0x83
BROADCAST_ID    = 0xFE

# ── STS register map ──────────────────────────────────────────────────────────
SMS_STS_ID                 = 5
SMS_STS_MIN_ANGLE_LIMIT_L  = 9
SMS_STS_MIN_ANGLE_LIMIT_H  = 10
SMS_STS_MAX_ANGLE_LIMIT_L  = 11
SMS_STS_MAX_ANGLE_LIMIT_H  = 12
SMS_STS_MODE               = 33     # EPROM: 0=position, 1=wheel, 3=multi-turn
SMS_STS_TORQUE_ENABLE      = 40
SMS_STS_ACC                = 41
SMS_STS_GOAL_POSITION_L    = 42
SMS_STS_GOAL_POSITION_H    = 43
SMS_STS_GOAL_TIME_L        = 44
SMS_STS_GOAL_TIME_H        = 45
SMS_STS_GOAL_SPEED_L       = 46
SMS_STS_GOAL_SPEED_H       = 47
SMS_STS_LOCK               = 55     # 0=unlock EPROM, 1=lock EPROM
SMS_STS_PRESENT_POSITION_L = 56
SMS_STS_PRESENT_POSITION_H = 57
SMS_STS_PRESENT_SPEED_L    = 58
SMS_STS_PRESENT_SPEED_H    = 59
SMS_STS_MOVING             = 66
SMS_STS_PRESENT_CURRENT_L  = 69
SMS_STS_PRESENT_CURRENT_H  = 70

# ── SCS register map ──────────────────────────────────────────────────────────
SCSCL_GOAL_POSITION_L      = 42


def _cs(seq: list) -> int:
    return (~sum(seq)) & 0xFF


class SMS_STS:
    """
    Feetech STS-series servo driver.
    Data byte order: little-endian (LSB first).
    Negative positions encode direction in bit-15 of the MSB.

    Control modes (MODE register, addr 33, stored in EPROM):
      0 – Position servo   (0-4095 ticks, angle limits apply)
      1 – Wheel / speed    (continuous rotation, speed command)
      3 – Multi-turn servo (unlimited position, angle limit = 0)
    """

    def __init__(self, ser: serial.Serial):
        self._ser = ser

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _pack_le(self, val: int) -> tuple:
        return val & 0xFF, (val >> 8) & 0xFF

    def _write_packet(self, sid: int, addr: int, data: list):
        """WRITE instruction (with ACK, which is discarded)."""
        n      = len(data)
        length = n + 3           # INST + ADDR + data(n) + CS
        params = [addr] + data
        cs     = _cs([sid, length, INST_WRITE] + params)
        self._ser.reset_input_buffer()
        self._ser.write(bytes([0xFF, 0xFF, sid, length, INST_WRITE]
                              + params + [cs]))
        self._ser.read(6)        # discard ACK

    def _read_packet(self, sid: int, addr: int, n: int):
        """
        READ instruction — returns list of n data bytes, or None on timeout /
        protocol error.
        """
        body   = [INST_READ, addr, n]
        length = len(body) + 1  # body + CS
        cs     = _cs([sid, length] + body)
        self._ser.reset_input_buffer()
        self._ser.write(bytes([0xFF, 0xFF, sid, length] + body + [cs]))
        # Response: 0xFF 0xFF ID (n+2) ERR data[n] CS
        resp   = self._ser.read(n + 6)
        if (len(resp) != n + 6
                or resp[0] != 0xFF or resp[1] != 0xFF
                or resp[2] != sid):
            return None
        return list(resp[5:5 + n])

    def _write_byte(self, sid: int, addr: int, val: int):
        self._write_packet(sid, addr, [val & 0xFF])

    def _write_word_le(self, sid: int, addr: int, val: int):
        lo, hi = self._pack_le(val)
        self._write_packet(sid, addr, [lo, hi])

    # ── Position commands ─────────────────────────────────────────────────────

    def WritePosEx(self, sid: int, pos: int, speed: int, acc: int):
        """
        Write goal position (with speed and acc) to one STS servo.
        pos  : encoder ticks; negative encodes via bit-15
        speed: steps/s (0 = max)
        acc  : acceleration units
        """
        if pos < 0:
            pos = (-pos) | 0x8000
        pos_l, pos_h = self._pack_le(pos)
        spd_l, spd_h = self._pack_le(speed)
        buf = [acc & 0xFF, pos_l, pos_h, 0, 0, spd_l, spd_h]
        self._write_packet(sid, SMS_STS_ACC, buf)

    def SyncWritePosEx(self, ids: list, n: int,
                       positions: list, speeds: list, accs: list):
        """Broadcast goal positions to N STS servos simultaneously (no ACK)."""
        data_len = 7
        mes_len  = (data_len + 1) * n + 4
        servo_data = []
        for i in range(n):
            pos = positions[i]
            if pos < 0:
                pos = (-pos) | 0x8000
            pos_l, pos_h = self._pack_le(pos)
            spd_l, spd_h = self._pack_le(speeds[i])
            servo_data.extend([ids[i], accs[i] & 0xFF,
                                pos_l, pos_h, 0, 0, spd_l, spd_h])
        params = [SMS_STS_ACC, data_len] + servo_data
        cs     = _cs([BROADCAST_ID, mes_len, INST_SYNC_WRITE] + params)
        packet = bytes([0xFF, 0xFF, BROADCAST_ID, mes_len, INST_SYNC_WRITE]
                       + params + [cs])
        self._ser.reset_input_buffer()
        self._ser.write(packet)

    # ── Read feedback ─────────────────────────────────────────────────────────

    def ReadPos(self, sid: int):
        """
        Read current position.
        Returns signed int (negative if direction bit set), or None on error.
        """
        data = self._read_packet(sid, SMS_STS_PRESENT_POSITION_L, 2)
        if data is None:
            return None
        pos = data[0] | (data[1] << 8)
        if pos & 0x8000:
            pos = -(pos & 0x7FFF)
        return pos

    def ReadMode(self, sid: int):
        """Read current MODE register byte (0/1/3), or None on error."""
        data = self._read_packet(sid, SMS_STS_MODE, 1)
        return data[0] if data is not None else None

    def ReadMoving(self, sid: int):
        """Read MOVING status (1=moving, 0=stopped), or None on error."""
        data = self._read_packet(sid, SMS_STS_MOVING, 1)
        return data[0] if data is not None else None

    def ReadAll(self, sid: int):
        """
        Read position, speed, and moving status in one burst.
        Returns dict {'pos': int, 'speed': int, 'moving': int} or None.
        """
        # Read 11 bytes from PRESENT_POSITION_L (56) through MOVING (66)
        data = self._read_packet(sid, SMS_STS_PRESENT_POSITION_L, 11)
        if data is None:
            return None
        pos   = data[0] | (data[1] << 8)
        if pos & 0x8000: pos = -(pos & 0x7FFF)
        spd   = data[2] | (data[3] << 8)
        if spd & 0x8000: spd = -(spd & 0x7FFF)
        moving = data[10]   # offset 66-56 = 10
        return {'pos': pos, 'speed': spd, 'moving': moving}

    # ── EPROM lock / unlock ───────────────────────────────────────────────────

    def unLockEprom(self, sid: int):
        """Unlock EPROM for write (must call before writing persistent regs)."""
        self._write_byte(sid, SMS_STS_LOCK, 0)
        time.sleep(0.010)

    def LockEprom(self, sid: int):
        """Re-lock EPROM after persistent writes."""
        self._write_byte(sid, SMS_STS_LOCK, 1)
        time.sleep(0.010)

    # ── Mode management ───────────────────────────────────────────────────────

    def SetPositionMode(self, sid: int):
        """
        Switch servo to standard position mode (MODE 0).
        Restores MAX_ANGLE_LIMIT to 4095.
        Writes EPROM — persistent across power cycles.
        """
        self.unLockEprom(sid)
        self._write_word_le(sid, SMS_STS_MAX_ANGLE_LIMIT_L, 4095)
        time.sleep(0.005)
        self._write_byte(sid, SMS_STS_MODE, 0)
        time.sleep(0.005)
        self.LockEprom(sid)

    def SetMultiTurnMode(self, sid: int):
        """
        Switch servo to multi-turn / infinite position mode (MODE 3).
        Sets MAX_ANGLE_LIMIT = 0 to remove angle restriction.
        Writes EPROM — persistent across power cycles.
        """
        self.unLockEprom(sid)
        self._write_byte(sid, SMS_STS_MODE, 3)
        time.sleep(0.005)
        self._write_word_le(sid, SMS_STS_MAX_ANGLE_LIMIT_L, 0)
        time.sleep(0.005)
        self.LockEprom(sid)

    def SetMinAngleLimit(self, sid: int, limit: int):
        """Write MIN_ANGLE_LIMIT to EPROM (call unLockEprom first)."""
        self._write_word_le(sid, SMS_STS_MIN_ANGLE_LIMIT_L, limit)

    def SetMaxAngleLimit(self, sid: int, limit: int):
        """Write MAX_ANGLE_LIMIT to EPROM (call unLockEprom first)."""
        self._write_word_le(sid, SMS_STS_MAX_ANGLE_LIMIT_L, limit)

    # ── Torque ────────────────────────────────────────────────────────────────

    def EnableTorque(self, sid: int, enable: int):
        """1 = enable torque, 0 = disable (limp)."""
        self._write_byte(sid, SMS_STS_TORQUE_ENABLE, enable)

    def CalibrationMiddle(self, sid: int):
        """中位校正:寫 torque 暫存器(40)=128,把馬達當下實體位置設為 2048
        中心(寫入 EPROM offset,持久)。單圈零位偏差時用來歸中。"""
        self._write_byte(sid, SMS_STS_TORQUE_ENABLE, 128)

    # ── Wheel (speed) mode ────────────────────────────────────────────────────

    def WheelMode(self, sid: int):
        """Switch to wheel/speed mode (MODE 1). Writes EPROM."""
        self._write_byte(sid, SMS_STS_MODE, 1)

    def WriteSpe(self, sid: int, speed: int, acc: int = 0):
        """Speed command for wheel mode. Negative = reverse."""
        if speed < 0:
            speed = (-speed) | 0x8000
        spd_l, spd_h = self._pack_le(speed)
        self._write_byte(sid, SMS_STS_ACC, acc & 0xFF)
        self._write_packet(sid, SMS_STS_GOAL_SPEED_L, [spd_l, spd_h])


class SCSCL:
    """
    Feetech SCS-series servo driver.
    Data byte order: big-endian (MSB first).
    """

    def __init__(self, ser: serial.Serial):
        self._ser = ser

    def _pack_be(self, val: int) -> tuple:
        return (val >> 8) & 0xFF, val & 0xFF

    def _write_packet(self, sid: int, addr: int, data: list):
        n      = len(data)
        length = n + 3
        params = [addr] + data
        cs     = _cs([sid, length, INST_WRITE] + params)
        self._ser.reset_input_buffer()
        self._ser.write(bytes([0xFF, 0xFF, sid, length, INST_WRITE]
                              + params + [cs]))
        self._ser.read(6)

    def WritePos(self, sid: int, pos: int, time_ms: int, speed: int):
        """
        Write goal position to one SCS servo.
        pos     : encoder ticks (0-1023)
        time_ms : move duration ms (0 = use speed)
        speed   : steps/s (0 = max)
        """
        pos_h,  pos_l  = self._pack_be(pos)
        time_h, time_l = self._pack_be(time_ms)
        spd_h,  spd_l  = self._pack_be(speed)
        self._write_packet(sid, SCSCL_GOAL_POSITION_L,
                           [pos_h, pos_l, time_h, time_l, spd_h, spd_l])
