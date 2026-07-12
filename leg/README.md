# Leg control (kbot lower-body) — 架構分析

**來源**:leg RPi (`fr01@raspberrypi`, MAC `b8:27:eb:02:db:cf`) 的 `~/robot_data/` + `~/fr01_requirements.txt`

**已排除(可再產生 / 太大)**:
- `**/target/` — Rust build 產物 (原 ~1 GB)
- `**/.git/` — 4 個內部 upstream repos
- `~/fr01/` — Python venv (原 ~255 MB)
- `~/kbot_deployment` — 28 MB executable(用 firmware 重編即可)

**目前這裡大小:69 MB**

---

## 硬體 & 軟體堆疊

```
Xbox controller / 鍵盤 / UDP client
   │
   │ (Python: kbot_deployment/kbot_control/*.py)
   ▼
UDP :10000  JSON {"XVel","YVel","YawRate", ...}
   │
   ▼
faux-rtos (Rust, robot_data/firmware/src/main.rs)
   │
   ├── 讀 kinfer policy (policy/FR01/*.kinfer, ONNX-ish)
   │     → policy_control.rs 執行 inference
   │
   ├── IMU (Hiwonder / imu.rs) 讀姿態
   │
   ▼
CAN bus (socketcan)  can0..can4
   │
   ▼
Robstride 執行器  (10 顆下半身馬達 — 每隻腳 5 顆)
```

## 目錄結構(遠端 `robot_data/` 對應)

| 目錄 | 語言 | 職責 |
|---|---|---|
| **firmware/** | Rust | `faux-rtos` 主 runtime。單執行緒 tokio,零動態配置。src 內 20+ 個 module(actuator, imu, inference, policy_control, udp_command, telemetry, robstride) |
| **kbot_deployment/kbot_control/** | Python | 使用者輸入層 —`joystick.py`(3 DOF)、`joystick16.py`(16 DOF 含手臂 pose)、`keyboard.py`、`Controller.py`(gamepad 抽象),都送 UDP 給 firmware |
| **kbot_deployment/scripts/** | Bash | `run <policy.kinfer>` 是主入口;`set_can.sh` 設 5 條 CAN;`zero_actuator.sh` / `clear_faults.sh` / `set_max_torque.sh` 等維護 |
| **kbot_deployment/powerboard/** | Python | `power_board.py` 用來讀 / 控功率板 |
| **kbot_deployment/misc/99-imu.rules** | udev | 給 IMU device 一個穩定的 `/dev/imu` symlink |
| **kbot_deployment/tty0tty/** | C | 虛擬 serial port 對(diagnostic loopback 用) |
| **policy/FR01/** | .kinfer | 訓練好的 walking policy(5 個),`kbot_zero_position.kinfer`=站立中立,其他是不同的 gait |
| **robstride/** | Rust + Python | CAN 馬達驅動庫(client, protocol, python_bindings via PyO3)、獨立掃描/讀寫 CLI |
| **imu/** | Rust + Python | Hiwonder / 其他 IMU 的獨立驅動 + 校準工具 |
| **klog/** | (Python + web + Rust) | 遙測 / 錄影 / 資料同步框架(`klog-cam`、`klog-robot`、`klog-server`、`klog-sync`) |
| **test_policy/** | | 開發時測 IMU 相關 policy 的隔離環境 |

## 執行流程

**1. 冷啟動**(在 leg RPi 上做)
```bash
cd ~/robot_data/kbot_deployment
scripts/set_can.sh                        # 5 條 CAN 帶起
scripts/run ~/robot_data/policy/FR01/kbot_zero_position.kinfer
```
`run` 腳本會:
- 呼叫 `scripts/reset_max_torques.sh`(安全)
- 建立 `model.kinfer` symlink 指向指定 policy
- `sudo chrt 80 firmware/target/release/faux-rtos --lpf-cutoff-hz 10 --min-jerk-blend-ms 0`

`faux-rtos` 命令列參數:
- `--policy-scale` (default 1.0):policy 輸出動作幅度縮放
- `--kp-scale` / `--kd-scale`:PID gain 縮放
- `--lpf-cutoff-hz` (default 6.0):對 policy output 做低通,除抖
- `--min-jerk-blend-ms`:transition 平滑

**2. 送指令(從外部)**
```bash
# 在同機
python3 kbot_control/joystick.py       # Xbox controller,3-DoF walking
python3 kbot_control/joystick16.py     # 16-DoF 完整全身
python3 kbot_control/keyboard.py       # 鍵盤模式
```
以上都送 UDP `localhost:10000`,JSON payload:
```json
{"XVel":0.5, "YVel":0, "YawRate":0}                                       // 基本 3D
{"XVel":..., "YVel":..., "YawRate":..., "BaseHeight":..., "BaseRoll":...,
 "BasePitch":..., "RShoulderPitch":..., ..., "LWristPitch":...}           // 16D 全身
```
firmware 的 `udp_command.rs` 兩種格式都吃。

**3. Policy 切換**
現有 `policy/FR01/`:
- `kbot_zero_position.kinfer`(956 B) — 站立不動的 zero point,適合冷啟動
- `stand_frozen.kinfer`(781 B) — 站立平衡
- `determined_hellman.kinfer` (4 MB)、`eloquent_ride.kinfer` (4 MB)、`fervent_euler.kinfer` (2 MB) — 三個訓練好的 walking gait,名稱是隨機產出的 code name

## 關鍵設計決策(從程式碼與 README 讀出)

1. **Rust 單執行緒 runtime**:`faux-rtos` 特意設計成單 thread + tokio,無動態配置(避免 malloc jitter)。IO 用 `io_uring`。
2. **UDP 命令通道**:走本機 loopback,firmware 內是 `UnifiedUdpManager` — 500ms timeout 之後自動歸零,防止外部 client 掛掉造成暴衝。
3. **kinfer policy 格式**:官方 K-Scale 的 packaged inference bundle(內含 ONNX + metadata + normalization)— firmware 用 `ort` crate (ONNX Runtime) 執行。
4. **Robstride 驅動器**:5 條 CAN bus 平行,每條 1 Mbps,`socketcan` crate 或 `robstride` 自己的 Rust client。
5. **Systemd-less**:目前是手動 `scripts/run` 起,沒有 service 檔;klog-deploy 有 queue 部署系統但目前空的。

## 未來整合方向(上半身 ↔ 下半身)

**目前**:上半身(Feetech + 頭 + arm + hand)由 head RPi(`robot@172.20.10.7`)控制;下半身(Robstride + 走路)由 leg RPi(`fr01@172.20.10.6`)控制,兩邊完全獨立。

**可以整合的接口:**
- **`joystick16.py` 已有欄位**支援 R/L shoulder/elbow/wrist 5 個 arm DOF — 但目前 firmware 端 policy 可能沒吃(要看 policy input schema)。走這條可以用同一支 Xbox controller 同時控腳 + 手臂。
- **klog-server** 已有 MQTT 遙測 spec (`klog/Mqtt_spec.md`);若要跨機 pub/sub 讓兩個 RPi 對話,klog 是現成的傳輸層。
- **UDP :10000** 是天然的 remote 入口 — launcher 可以直接送 packet 從筆電控腳,不必 SSH。

## 常用 diagnostic 腳本(在 `robot_data/kbot_deployment/scripts/`)

| 腳本 | 用途 |
|---|---|
| `set_can.sh` | 設 can0..can4 up (1M bps) — 每次開機要跑 |
| `ping_actuator.sh <id>` | 探測單顆 robstride 有沒有回應 |
| `read_kp.sh` / `read_max_torque.sh` / `read_param.sh` | 讀馬達參數 |
| `set_max_torque.sh` / `reset_max_torques.sh` | 設限力矩(安全) |
| `zero_actuator.sh <id>` | 逐顆歸零 |
| `set_zero_sta.sh` | 全體站立零位重設 |
| `clear_faults.sh` | 清所有 fault flag(過流/過熱後常需要) |
| `factory_reset.sh` | 危險 — 恢復出廠 |
| `request_feedback.sh` | 讀 telemetry snapshot |
| `run <policy.kinfer>` | **主入口**,啟動 firmware |
