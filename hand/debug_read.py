#!/usr/bin/env python3
"""每顆馬達 raw byte 讀取 + 多次取樣,診斷讀值是否穩定/污染。"""
import sys
import time
sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, COMM_SUCCESS  # noqa: E402
from scservo_sdk.protocol_packet_handler import protocol_packet_handler  # noqa: E402

PORT = "/dev/ttyACM0"
BAUD = 1_000_000
IDS = [31, 32, 33, 34, 35, 36]


def main():
    ph = PortHandler(PORT)
    ph.openPort()
    ph.setBaudRate(BAUD)
    pk = sms_sts(ph)

    for sid in IDS:
        print(f"\n===== ID {sid} =====")
        # --- 各方式讀 min/max ---
        for i in range(3):
            # method 1: read2ByteTxRx (組字後 uint16)
            mn_raw, comm1, err1 = pk.read2ByteTxRx(sid, 9)
            mx_raw, comm2, err2 = pk.read2ByteTxRx(sid, 11)
            # method 2: readTxRx 4 bytes 一次拿 min_L, min_H, max_L, max_H
            data, comm3, err3 = pk.readTxRx(sid, 9, 4)
            # method 3: ReadPos (SDK 內部已 scs_tohost)
            pos_sdk, comm4, err4 = pk.ReadPos(sid)
            # method 4: 直接讀 present position 兩 byte
            pos_raw, comm5, err5 = pk.read2ByteTxRx(sid, 56)

            if all(c == COMM_SUCCESS for c in [comm1, comm2, comm3, comm4, comm5]):
                print(f" round{i}: "
                      f"min_word=0x{mn_raw:04X}({mn_raw})  "
                      f"max_word=0x{mx_raw:04X}({mx_raw})  "
                      f"raw4=[{data[0]:02X} {data[1]:02X} {data[2]:02X} {data[3]:02X}]  "
                      f"pos_sdk={pos_sdk}  pos_word=0x{pos_raw:04X}({pos_raw})  "
                      f"err=[{err1},{err2},{err3},{err4},{err5}]")
            else:
                print(f" round{i}: comm fail: {comm1},{comm2},{comm3},{comm4},{comm5}")
            time.sleep(0.05)

    ph.closePort()


if __name__ == "__main__":
    main()
