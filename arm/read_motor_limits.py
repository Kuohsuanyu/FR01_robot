#!/usr/bin/env python3
"""Read Feetech STS3215 EPROM limits + current state. Read-only (no movement).

Scans both USB serial ports and dumps for every responding motor:
  * mode           (reg 33)   0=position  1=wheel  3=multi-turn
  * min/max angle  (reg 9-12) software limits stored in EPROM (ticks)
  * current pos    (reg 56-57)
  * voltage / temp (reg 62 / 63)
"""
import sys
sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402

PORTS = {
    "/dev/ttyACM0": dict(label="左臂",   ids=[21, 22, 23, 24]),
    "/dev/ttyACM1": dict(label="頭部",   ids=[1, 2, 3]),
}

# Register addresses (sms_sts.py)
ADDR_MIN_ANGLE_L = 9
ADDR_MAX_ANGLE_L = 11
ADDR_MODE = 33
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63

MODE_NAMES = {0: "position", 1: "wheel/speed", 3: "multi-turn"}


def read_motor(pk, sid):
    """Return dict or None if unreachable."""
    pos, _, comm, _ = pk.ReadPosSpeed(sid)
    if comm != COMM_SUCCESS:
        return None
    out = dict(pos=int(pos))

    mn, comm, _ = pk.read2ByteTxRx(sid, ADDR_MIN_ANGLE_L)
    out["min_tick"] = int(mn) if comm == COMM_SUCCESS else None
    mx, comm, _ = pk.read2ByteTxRx(sid, ADDR_MAX_ANGLE_L)
    out["max_tick"] = int(mx) if comm == COMM_SUCCESS else None

    mode, comm, _ = pk.read1ByteTxRx(sid, ADDR_MODE)
    out["mode"] = int(mode) if comm == COMM_SUCCESS else None

    v, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_VOLTAGE)
    out["volt"] = v * 0.1 if comm == COMM_SUCCESS else None
    t, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_TEMP)
    out["temp"] = int(t) if comm == COMM_SUCCESS else None
    return out


def fmt_tick(t):
    if t is None: return "—"
    deg = (t - 2047) / (4096/360)
    return f"{t:>5d}({deg:+6.1f}°)"


def main():
    print(f"{'port':<18}{'id':>4}  {'mode':<11}  {'min':>14}  {'max':>14}  "
          f"{'pos':>14}  {'volt':>6}  {'temp':>5}")
    print("-" * 110)
    for port, cfg in PORTS.items():
        ph = PortHandler(port)
        if not ph.openPort():
            print(f"{port:<18}  open failed")
            continue
        if not ph.setBaudRate(1_000_000):
            print(f"{port:<18}  setBaudRate failed"); ph.closePort(); continue
        pk = sms_sts(ph)
        for sid in cfg["ids"]:
            r = read_motor(pk, sid)
            if r is None:
                print(f"{port:<18}{sid:>4}  no response"); continue
            mode_s = MODE_NAMES.get(r['mode'], f"unknown({r['mode']})")
            print(f"{port:<18}{sid:>4}  {mode_s:<11}  "
                  f"{fmt_tick(r['min_tick']):>14}  "
                  f"{fmt_tick(r['max_tick']):>14}  "
                  f"{fmt_tick(r['pos']):>14}  "
                  f"{r['volt']:>5.1f}V  {r['temp']:>3d}°C")
        ph.closePort()
    print("\n注:")
    print("  - position mode: tick 0..4095 = ±180°,limits 都不能超出單圈")
    print("  - multi-turn mode: tick 可超出 0-4095(多圈累加),limits 通常都 0")
    print("  - wheel/speed mode: 連續旋轉,不吃位置 limit")


if __name__ == "__main__":
    main()
