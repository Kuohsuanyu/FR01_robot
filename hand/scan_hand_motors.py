#!/usr/bin/env python3
"""巧手馬達掃描 + 限位讀取(唯讀,不會轉動馬達)。

掃描指定串口上的所有 Feetech STS 馬達 (ID 1..MAX_ID),對每個回應的馬達印出:
  * 模型編號 (ping)
  * 工作模式 (reg 33)   0=position  1=wheel/speed  3=multi-turn
  * 最小/最大角度限位 (reg 9-10 / 11-12,EPROM 內存值,單位 tick)
  * 目前位置 (reg 56-57)
  * 電壓 / 溫度 (reg 62 / 63)

用法:
    python3 scan_hand_motors.py                 # 預設 /dev/ttyACM0, 自動試多個 baud, 掃 1..30
    python3 scan_hand_motors.py --baud 1000000  # 鎖定 baud
    python3 scan_hand_motors.py --port /dev/ttyACM1 --max-id 50
"""
import argparse
import sys

sys.path.insert(0, "/home/andykuo/FTServo_Python")
from scservo_sdk import PortHandler, sms_sts, scscl, COMM_SUCCESS  # noqa: E402


# --- Feetech 內部 byte order 依系列而異 ---
#   SMS/STS/HLS 系列 → little-endian (scs_end=0);位置 12-bit,0..4095/圈
#   SCSCL / SCS009 系列 → big-endian  (scs_end=1);位置 10-bit,0..1023/圈
# 同一條匯流排可以混掛;必須依 ping 回來的 model 號決定用哪個 handler。
SCSCL_MODELS = {1029}  # 已知 model 1029 = SCS009 (big-endian, 10-bit)


def is_scscl(model):
    return model in SCSCL_MODELS

# STS3215 EPROM/RAM register 位址 (參考 scservo_sdk/sms_sts.py)
ADDR_MIN_ANGLE_L = 9
ADDR_MAX_ANGLE_L = 11
ADDR_MODE = 33
ADDR_PRESENT_POSITION_L = 56
ADDR_PRESENT_VOLTAGE = 62
ADDR_PRESENT_TEMP = 63

MODE_NAMES = {0: "position", 1: "wheel/speed", 3: "multi-turn"}


def scs_tohost(a, b=15):
    """Feetech STS/SMS 慣用 sign-magnitude 解碼(bit b 是符號)。HLS 系列不一定用此編碼。"""
    if a & (1 << b):
        return -(a & ~(1 << b))
    return a


def to_s16(v):
    """two's complement int16 解讀。"""
    return v - 65536 if v >= 32768 else v


def fmt_raw(raw):
    """把 uint16 用三種可能解讀並列印出來,方便對照 Feetech 官方工具。"""
    if raw is None:
        return "         —              "
    u = raw
    s16 = to_s16(raw)
    sm = scs_tohost(raw, 15)
    return f"0x{raw:04X}  u={u:>5d}  s16={s16:>+6d}  sm={sm:>+6d}"


def read_motor(pk, sid):
    """回傳 dict;若某欄位讀失敗則為 None。"""
    out = {}
    # raw uint16 直接拿 EPROM 內容,不解碼(避免猜錯編碼方式)
    mn, comm, _ = pk.read2ByteTxRx(sid, ADDR_MIN_ANGLE_L)
    out["min_raw"] = int(mn) if comm == COMM_SUCCESS else None
    mx, comm, _ = pk.read2ByteTxRx(sid, ADDR_MAX_ANGLE_L)
    out["max_raw"] = int(mx) if comm == COMM_SUCCESS else None
    ps, comm, _ = pk.read2ByteTxRx(sid, ADDR_PRESENT_POSITION_L)
    out["pos_raw"] = int(ps) if comm == COMM_SUCCESS else None

    mode, comm, _ = pk.read1ByteTxRx(sid, ADDR_MODE)
    out["mode"] = int(mode) if comm == COMM_SUCCESS else None

    v, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_VOLTAGE)
    out["volt"] = v * 0.1 if comm == COMM_SUCCESS else None
    t, comm, _ = pk.read1ByteTxRx(sid, ADDR_PRESENT_TEMP)
    out["temp"] = int(t) if comm == COMM_SUCCESS else None
    return out


BAUD_CANDIDATES = [1_000_000, 500_000, 115_200, 250_000, 128_000, 76_800, 57_600, 38_400]


def scan_ids(pk, max_id):
    found = []
    for sid in range(1, max_id + 1):
        model, comm, _ = pk.ping(sid)
        if comm == COMM_SUCCESS:
            found.append((sid, int(model)))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=None,
                    help="鎖定 baud;若不指定則自動輪詢常見值")
    ap.add_argument("--max-id", type=int, default=50,
                    help="掃描 ID 範圍 1..MAX_ID (預設 50;本機巧手實測 ID 落在 31-36)")
    args = ap.parse_args()

    ph = PortHandler(args.port)
    if not ph.openPort():
        print(f"[ERR] 無法開啟串口 {args.port}")
        return 1
    # 兩個 handler 共用同一個 port,依 model 決定用哪一個(byte order 不同)
    pk_sts = sms_sts(ph)   # little-endian, SMS/STS/HLS
    pk_scs = scscl(ph)     # big-endian,   SCSCL / SCS009

    bauds = [args.baud] if args.baud else BAUD_CANDIDATES
    found = []
    used_baud = None
    for baud in bauds:
        if not ph.setBaudRate(baud):
            print(f"[WARN] 設定 baud={baud} 失敗,略過")
            continue
        print(f"[INFO] 嘗試 {args.port} @ {baud:>7} bps 掃描 ID 1..{args.max_id} ...", end=" ", flush=True)
        found = scan_ids(pk_sts, args.max_id)  # ping 只用 1-byte 回應,任一 handler 皆可
        if found:
            print(f"命中 {len(found)} 顆!")
            used_baud = baud
            break
        print("無回應")

    if not found:
        print("[WARN] 所有 baud 都沒有回應。檢查電源 / 排線 / 通訊協定。")
        ph.closePort()
        return 0

    print(f"\n[INFO] 使用 baud = {used_baud}")
    for sid, model in found:
        family = "SCS009 (BE, 10-bit)" if is_scscl(model) else "STS/SMS (LE, 12-bit)"
        print(f"  [ID:{sid:03d}] model={model:>4}  → {family}")

    # --- Phase 2: 依 model 使用對應 handler 讀限位 + 當前位置 ---
    print()
    print(f"共發現 {len(found)} 顆馬達  ({len([m for _,m in found if is_scscl(m)])} 顆 SCS009,{len([m for _,m in found if not is_scscl(m)])} 顆 STS/SMS)")
    print("=" * 90)
    print(f"{'ID':>3}  {'model':>5}  {'family':<22}  {'min':>6}  {'max':>6}  {'pos':>6}  {'V':>5}  {'°C':>4}")
    print("-" * 90)
    for sid, model in found:
        pk = pk_scs if is_scscl(model) else pk_sts
        family = "SCS009 (BE, 0..1023)" if is_scscl(model) else "STS/SMS (LE, 0..4095)"
        r = read_motor(pk, sid)
        volt_s = f"{r['volt']:>4.1f}V" if r["volt"] is not None else "  — "
        temp_s = f"{r['temp']:>3d}°C" if r["temp"] is not None else "  — "
        mn = r["min_raw"] if r["min_raw"] is not None else "—"
        mx = r["max_raw"] if r["max_raw"] is not None else "—"
        ps = r["pos_raw"] if r["pos_raw"] is not None else "—"
        print(f"{sid:>3}  {model:>5}  {family:<22}  {str(mn):>6}  {str(mx):>6}  {str(ps):>6}  {volt_s}  {temp_s}")
    print("=" * 90)

    ph.closePort()
    return 0


if __name__ == "__main__":
    sys.exit(main())
