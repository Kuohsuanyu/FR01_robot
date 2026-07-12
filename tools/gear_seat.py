"""
慢速持續旋轉 — 幫齒輪咬合就位
ID=11, /dev/ttyACM0, 每隔一段時間切換正反轉
Ctrl+C 停止
"""
import sys
import time

sys.path.append("/home/andykuo/FTServo_Python")
from scservo_sdk import *

PORT       = "/dev/ttyACM0"
BAUD       = 1000000
SERVO_ID   = 11

SPEED      = 2000    # 加大：約 29 rpm，克服齒輪摩擦
ACC        = 30
SWITCH_SEC = 2.5

SMS_STS_MODE          = 33
SMS_STS_TORQUE_ENABLE = 40
SMS_STS_PRESENT_SPEED_L = 58
SMS_STS_PRESENT_VOLTAGE = 62
SMS_STS_PRESENT_TEMPERATURE = 63


def check(comm, err, tag):
    ph = packetHandler
    if comm != COMM_SUCCESS:
        print(f"[{tag}] FAIL {ph.getTxRxResult(comm)}", flush=True); return False
    if err != 0:
        print(f"[{tag}] ERR  {ph.getRxPacketError(err)}", flush=True); return False
    print(f"[{tag}] ok", flush=True); return True


portHandler   = PortHandler(PORT)
packetHandler = sms_sts(portHandler)

if not portHandler.openPort():
    print(f"開啟 {PORT} 失敗"); sys.exit(1)
if not portHandler.setBaudRate(BAUD):
    print("設定 baudrate 失敗"); sys.exit(1)
print(f"連線成功  ID={SERVO_ID}  speed=±{SPEED}  每 {SWITCH_SEC}s 換向", flush=True)

model, comm, err = packetHandler.ping(SERVO_ID)
if comm != COMM_SUCCESS:
    print(f"ping ID={SERVO_ID} 失敗：{packetHandler.getTxRxResult(comm)}", flush=True); sys.exit(1)
print(f"ping ok model={model}", flush=True)

volt, _, _ = packetHandler.read1ByteTxRx(SERVO_ID, SMS_STS_PRESENT_VOLTAGE)
temp, _, _ = packetHandler.read1ByteTxRx(SERVO_ID, SMS_STS_PRESENT_TEMPERATURE)
print(f"電壓 {volt/10:.1f} V   溫度 {temp} °C", flush=True)
if volt < 60:
    print("⚠️  電壓 < 6V，可能沒接電或供電不足", flush=True)

# 進 wheel mode + 開扭力（順序：先關扭力→切模式→開扭力）
comm, err = packetHandler.write1ByteTxRx(SERVO_ID, SMS_STS_TORQUE_ENABLE, 0)
check(comm, err, "torque off")
comm, err = packetHandler.write1ByteTxRx(SERVO_ID, SMS_STS_MODE, 1)
check(comm, err, "wheel mode")
comm, err = packetHandler.write1ByteTxRx(SERVO_ID, SMS_STS_TORQUE_ENABLE, 1)
check(comm, err, "torque on")

direction = 1
try:
    while True:
        comm, err = packetHandler.WriteSpec(SERVO_ID, SPEED * direction, ACC)
        check(comm, err, f"spin {'+' if direction>0 else '-'}{SPEED}")
        time.sleep(0.5)
        spd, _, _ = packetHandler.read2ByteTxRx(SERVO_ID, SMS_STS_PRESENT_SPEED_L)
        # 15-bit 有號
        spd_signed = spd if spd < 0x8000 else -(spd & 0x7FFF)
        print(f"   實測轉速 raw={spd_signed}", flush=True)
        time.sleep(SWITCH_SEC - 0.5)
        direction = -direction
except KeyboardInterrupt:
    print("\n停止中...", flush=True)
    packetHandler.WriteSpec(SERVO_ID, 0, ACC)
    time.sleep(0.3)
    portHandler.closePort()
    print("bye.")
