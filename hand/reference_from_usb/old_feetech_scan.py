#!/usr/bin/env python3
# feetech_scan.py — 掃描 Feetech 智慧總線伺服的 ID 與鮑率
# 適用 STS/SMS/SCS 系列
# Author: Ted's BirdBot adaptation (2025)

import serial
import time

# === 基本設定 ===
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46085631-if00"  # 或 /dev/ttyUSB0
BAUD_RATES = [115200, 57600, 1000000, 250000, 460800, 38400]
ID_RANGE = range(1, 21)
# =================

def checksum(data):
    """計算校驗位元"""
    return (~sum(data)) & 0xFF

def ping_servo(ser, sid):
    """傳送 ping 封包並讀取回應"""
    packet = [0xFF, 0xFF, sid, 0x02, 0x01]
    packet.append(checksum(packet[2:]))
    ser.reset_input_buffer()
    ser.write(bytes(packet))
    time.sleep(0.03)
    resp = ser.read_all()
    return resp

def scan_ids(baud):
    """掃描所有 ID"""
    try:
        ser = serial.Serial(PORT, baud, timeout=0.3)
    except serial.SerialException as e:
        print(f"❌ 無法開啟埠 {PORT}: {e}")
        return []

    print(f"\n🔎 正在掃描鮑率 {baud}...")
    found_ids = []
    for sid in ID_RANGE:
        resp = ping_servo(ser, sid)
        if resp:
            print(f"  ✅ ID {sid:02d} → 回應: {resp.hex(' ')}")
            found_ids.append(sid)
        else:
            print(f"  -- ID {sid:02d} 無回應")
    ser.close()
    return found_ids

def main():
    print(f"=== 掃描 Feetech 智慧伺服 on {PORT} ===")
    all_found = {}
    for baud in BAUD_RATES:
        ids = scan_ids(baud)
        all_found[baud] = ids
        if ids:
            print(f"✅ 鮑率 {baud} 偵測到 ID: {ids}")
        else:
            print(f"⚠️ 鮑率 {baud} 無任何伺服回應")

    print("\n=== 掃描結果總結 ===")
    for baud, ids in all_found.items():
        print(f"{baud} → {ids if ids else '無回應'}")

if __name__ == "__main__":
    main()
