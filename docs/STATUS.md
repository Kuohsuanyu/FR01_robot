# Q-BOT — 目前狀態總覽

## 硬體拓樸

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           【上位機 · 這台 Linux 筆電】                       │
│                             192.168.0.118 (LAN)                          │
│  ── 直連的裝置 ──                                                          │
│    /dev/ttyACM0   Feetech 手臂馬達 bus  (右臂 ID 10–14 已上線)             │
│    /dev/video0/1  筆電內建 webcam                                          │
│    /dev/video2/3  Logitech C310  (可移到 RPi)                             │
└──────────┬──────────────────────────────────────────────────────────┬────┘
           │  Wi-Fi                                                     │
           │  HTTP 8000 / HTTPS 8443 / WebSocket                        │
           │                                                            │
┌──────────┴──────────────────────┐                             ┌───────┴────────┐
│  【樹莓派 5 · 頭部】              │                             │  【手機】       │
│    192.168.0.123 / user=robot   │                             │  (VR Cardboard)│
│  ── 直連的裝置 ──                │                             │  Safari/Chrome │
│    /dev/ttyACM0 Feetech head    │                             │  接感測器      │
│      ID 1 (yaw), 2 (roll),      │                             └────────────────┘
│      3 (pitch)                  │
│    /dev/video0/1 USB 相機        │
│  ── 軟體 ──                      │
│    ~/head_rpi_agent/agent.py    │
│      aiohttp: HTTP 8000 + HTTPS 8443
│      python3 系統套件已裝(apt) │
└─────────────────────────────────┘
```

## 完成的功能(對照真實 workflow)

| # | 場景 | 檔案 | 狀態 |
|---|---|---|---|
| 1 | 逐關節馬達測試(sliders + 遙測) | `upper_body_control/joint_test_gui.py` | ✓ 可用 |
| 2 | 讀 EPROM 限位健診 | `upper_body_control/read_motor_limits.py` | ✓ |
| 3 | FD.exe 替代(Python 版 motor_tool) | `upper_body_control/motor_tool.py` | ✓ |
| 4 | 手臂鬼影 + IK + 2 點線性校準 | `upper_body_control/qbot_ik_gui.py` | ✓(校準 UI 已簡化) |
| 5 | 頭部鬼影 + 手機 IMU(本機串口) | `upper_body_control/head_ghost_gui.py` | ✓ Stage 1-3 |
| 6 | RPi headless agent(馬達 + 相機 + WS) | `deployment/head_rpi_agent/agent.py` | ✓ 已部署到 RPi |
| 7 | PC 上位機(接 RPi、鬼影 + 相機面板) | `deployment/head_pc_client/remote_gui.py` | ✓ |
| 8 | 手機 VR + IMU → 頭 (直接跟 RPi 對話) | `deployment/head_rpi_agent/vr.html` | ✓ HTTPS 8443/vr |

## 一鍵啟動 scripts

在 `scripts/` 資料夾下:

| 腳本 | 做什麼 |
|---|---|
| `launcher.py` | Tk 選單,一目瞭然狀態 + 所有按鈕 |
| `head_start.sh` | SSH 到 RPi 啟動 agent → 起 PC 端 `remote_gui.py` |
| `head_stop.sh` | 停 RPi 上的 agent(釋放 port 8000/8443 + 相機 + 馬達 bus) |
| `arm_start.sh` | 啟動 `qbot_ik_gui.py`(本機直連手臂) |
| `motor_tool_start.sh` | 啟動 register 編輯工具 |
| `lib.sh` | 共用 helper(RPi_HOST / RPi_KEY / colour echo) |

**最快啟動法**: `python3 scripts/launcher.py`

## RPi 連接

| 項目 | 值 |
|---|---|
| IP | `192.168.0.123` |
| user | `robot` |
| SSH 免密登入 | `~/.ssh/qbot_rpi`(公鑰已存 RPi 的 `~/.ssh/authorized_keys`) |
| agent 目錄 | `/home/robot/head_rpi_agent/` |
| log | `/home/robot/agent.log` |
| Python 版本 | 3.11.2(系統 python3-*) |
| 已裝依賴 | aiohttp 3.8 / opencv 4.6 / pyserial 3.5 / numpy 1.24 / cryptography 38 |

**手動 SSH**:`ssh -i ~/.ssh/qbot_rpi robot@192.168.0.123`

## 網路 endpoints(RPi agent 在跑時)

| URL | 用途 |
|---|---|
| `http://192.168.0.123:8000/` | 首頁 (HTML dashboard) |
| `http://192.168.0.123:8000/status` | JSON 健康檢查(motor_ids / cam_frames / port …) |
| `http://192.168.0.123:8000/snapshot` | 單張 JPEG(debug 用) |
| `http://192.168.0.123:8000/mjpeg` | 相機 MJPEG 串流(PC 用) |
| `ws://192.168.0.123:8000/ws` | 馬達控制 + telemetry(PC remote_gui 用) |
| `https://192.168.0.123:8443/vr` | **手機 VR 頁面**(需要 HTTPS 才能讀 IMU) |
| `wss://192.168.0.123:8443/pose_ws` | 手機 IMU → 頭部馬達(agent 內部路由) |

## 延遲實測

| 通道 | 延遲(LAN) |
|---|---|
| PC → RPi WebSocket send | < 5 ms |
| Phone IMU → RPi → motor | ~ 20 ms(50 Hz sample rate) |
| RPi camera → PC 顯示 | 50–80 ms |
| PC slider 50 Hz SyncWrite → 頭 | ~ 10 ms |

## 靈巧手 hands_control/

| 檔案 | 用途 |
|---|---|
| `hands_control/camera_receiver.py --show` | 相機影像模式(推薦) |
| `hands_control/glove_receiver.py`         | 手套 JSON UDP 模式 |
| `hands_control/hand_slider_gui.py`        | 手動滑桿測試 |
| `hands_control/scan_hand_motors.py`       | 掃描手部馬達 ID |

Launcher 已加對應按鈕(相機影像 / 手套 UDP)。

## 手臂 IK GUI 校準流程(最新)

1. **馬達端 2-點線性校準**(你已做完):`[←Lo] [←Hi]` per joint
2. **每個 Joint 的 ON/OFF**(slider 左邊按鈕):
   - ON(綠) = 拖 slider / IK 會送這顆
   - OFF(紅) = 這顆不動(用來單獨測試某軸,其他保持不動)
3. **設為零位** 按鈕:
   - 拖鬼影到「你認定的 home 位置」
   - 按下 → 當前鬼影姿態記為 q=0(zero_offset 存這個 offset)
   - Slider 全部歸零、鬼影不動、馬達不動(因為 offset 吸收了)
   - 之後 IK 拖鬼影,q_display 顯示相對於這個 home 的變化
4. **Save** — 把 cal_points / zero_offset / motor_live 都寫進 `qbot_arm_calibration.json`

## 待辦

- ~~靈巧手整合~~ ✓ 已加 launcher 按鈕
- 手臂遠端化(現在只有頭部走 RPi;手臂還是本機串口)
- 訓練 pipeline 部署到實機(QBOT_TRAIN → ONNX → Rust firmware)
- 頭部 IMU/motor 校準值持久化(現在存在 agent memory,重啟消失)
