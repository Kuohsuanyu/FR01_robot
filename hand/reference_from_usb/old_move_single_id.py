#!/usr/bin/env python3
# feetech_move_id11.py — Move ID=11 to position 650 (SCS009 / STS3032)
# 對 Feetech BUS（單線半雙工）發原始封包：WRITE @ 0x2A (Pos, Time, Speed)

import serial, time

# === 請依實際調整 ===
PORT     = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46085631-if00"  # 或 /dev/ttyACM0
BAUD     = 1_000_000
SERVO_ID = 2
POS      = 650      # 目標位置（常見 0~1023 或 0~4095，依型號/設定）
TIME_MS  = 600      # 動作時間 (0~30000 ms)
SPEED    = 200      # 速度 (0~1023；依型號)
# ====================

def chk(bs):  # Feetech checksum：~sum(data) & 0xFF
    return (~sum(bs)) & 0xFF

def write_reg(ser, sid, addr, payload_bytes):
    core = [sid, 3 + len(payload_bytes), 0x03, addr] + list(payload_bytes)
    pkt  = bytes([0xFF, 0xFF] + core + [chk(core)])
    ser.reset_input_buffer()
    ser.write(pkt)
    # 某些型號會回狀態封包，某些不會；等一下下看是否有回
    time.sleep(0.02)
    resp = ser.read_all()
    if resp:
        print(f"  ↳ status: {resp.hex(' ')}")
    return resp

def torque_enable(ser, sid, on=True):
    # 常見 Torque Enable 位址 0x18（不同型號可能不同，若無效可略過）
    try:
        return write_reg(ser, sid, 0x18, [0x01 if on else 0x00])
    except Exception as e:
        print(f"  (torque 設定略過) {e}")

def goal_position(ser, sid, pos, time_ms=600, speed=200):
    # 常見 STS/SCS 目標位置區塊從 0x2A 起：Pos(2), Time(2), Speed(2)
    pos  = max(0, min(int(pos), 4095))
    t    = max(0, min(int(time_ms), 30000))
    spd  = max(0, min(int(speed), 1023))
    data = [pos & 0xFF, (pos >> 8) & 0xFF, t & 0xFF, (t >> 8) & 0xFF, spd & 0xFF, (spd >> 8) & 0xFF]
    return write_reg(ser, sid, 0x2A, data)

def main():
    print(f"Open {PORT} @ {BAUD} …")
    ser = serial.Serial(PORT, BAUD, timeout=0.05)

    print(f"\n[ID {SERVO_ID}] Enable torque → move to {POS}")
    torque_enable(ser, SERVO_ID, True)
    goal_position(ser, SERVO_ID, POS, TIME_MS, SPEED)

    # 可選：回起點或再發一個位置
    # time.sleep(1.0)
    # goal_position(ser, SERVO_ID, 0, 800, 200)

    ser.close()
    print("\nDone. 若不動：1) 確認鮑率=1Mbps 2) ID 是否 11 3) 是否共地+外部供電 4) 該型號是否在 wheel 模式")

if __name__ == "__main__":
    main()
