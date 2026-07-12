# FR01 Q-BOT 系統架構文件

**版本**:2026-07-05 snapshot
**目的**:單一總覽,列出目前所有元件、資料流、腳本入口、設定檔位置。給接手/回來的人一份「立刻上手」的地圖。

---

## 1. 硬體拓撲

```
                                     [ 頭部相機 (/dev/video0) ]
                                                │
                                                │ v4l2
                                                ▼
┌────────────────┐  Wi-Fi LAN  ┌────────────────────────────┐
│   Laptop (X11) │◀───────────▶│  RPi 192.168.0.123 (robot) │
│  Ubuntu Tk GUI │             │  head_rpi_agent / arm_agent│
└────────────────┘             └──────────┬─────────────────┘
                                          │ /dev/ttyACM0 (CH340 @ 1 Mbps)
                                          ▼
             ┌────────────────────────────────────────────────────┐
             │           一條 daisy-chain (混合 STS + SCS)         │
             │  sid 1-3   HEAD  (STS3215, model 777)              │
             │  sid 10-14 R arm (STS3215, model 777)              │
             │  sid 20-24 L arm (STS3215, model 777)              │
             │  sid 41,46 HAND  (STS3032, model 521)              │
             │  sid 42-45 HAND  (SCS009, model 1029)              │
             └────────────────────────────────────────────────────┘

               [手機] ── https :8443/vr ── phone IMU (Web DeviceOrientation)
                       (WSS)
```

## 2. 軟體元件總表

| # | 元件 | 路徑 | 執行位置 | 通訊 | 角色 |
|---|---|---|---|---|---|
| 1 | **head_rpi_agent** | `deployment/head_rpi_agent/agent.py` | RPi | HTTP 8000 / HTTPS 8443 / WS `/ws` `/pose_ws` / MJPEG `/mjpeg` | 頭部馬達 + 相機 + 手機 IMU 樞紐,現在也走 SCS009(手部)混掛 dispatch |
| 2 | **arm_rpi_agent** | `deployment/arm_rpi_agent/agent.py` | RPi | HTTP 8100 / WS `/ws` | 專門 arm 匯流排 (若要獨立控 arm 時用) |
| 3 | **head_pc_client** | `deployment/head_pc_client/remote_gui.py` | Laptop Tk | WS→agent, MJPEG← | PC 端 head 上位機,幽靈骨架 + slider + 相機 |
| 4 | **qbot_ik_gui** | `upper_body_control/qbot_ik_gui.py` | Laptop Tk | WS→arm_agent | Arm IK GUI (兩支手臂、5-DoF/支、SLSQP IK + 校準) |
| 5 | **motion_replay_gui** | `upper_body_control/motion_imitator/motion_replay_gui.py` | Laptop Tk | WS→arm_agent | 讀 npz/csv 錄製動作 → 幽靈全身 + 手臂馬達 |
| 6 | **pose_imitator** | `deployment/pose_imitator/pose_imitator.py` | RPi | UDP 9999 / 9998 → arm GUI | MediaPipe Pose → arm 5-joint retarget → UDP |
| 7 | **camera_receiver** (右) | `hands_control/camera_receiver.py --side right` | Laptop | 本機 serial | 右手 MediaPipe Hands → 本機 /dev/ttyACM0 |
| 8 | **camera_receiver** (左) | `hands_control/camera_receiver.py --side left` | Laptop | RPi MJPEG in, WS→head_agent out | 左手,吃 RPi 相機串流,馬達寫回 head_agent |
| 9 | **glove_receiver** | `hands_control/glove_receiver.py --side <r|l>` | 任一端 | UDP :6000 in | 手套 JSON → tick,可作為 UDP 反射的馬達 daemon |
| 10 | **hand_slider_gui** | `hands_control/hand_slider_gui.py --hand <r|l>` | 需 X11(RPi ssh -Y) | 本機 serial | 手部逐顆搖桿 + finger 對應校準 |
| 11 | **VR wizard** | `deployment/head_rpi_agent/vr.html` | 手機瀏覽器 | WSS pose_ws | 手機 IMU + 校準精靈(START → forward → yaw_L → center → yaw_R → center → pitch_up → center → pitch_down) |
| 12 | **launcher** | `scripts/launcher.py` | Laptop Tk | subprocess | 一鍵啟停控制板 |

## 3. 腳本入口 (`scripts/`)

啟動類:
- `head_start.sh` — SSH → RPi 啟 head_agent + 本機 remote_gui.py
- `head_stop.sh` — 停 head agent + 上位機
- `head_lock.sh` — 只鎖 sid 1/2/3 目前位置
- `arm_remote_start.sh` / `arm_start.sh` / `arm_remote_stop.sh`
- `pose_imitate_start.sh` — RPi 上跑 pose_imitator.py(auto-restart wrapper)
- `hand_camera_start.sh` — 右手相機模式 (本機 USB)
- `hand_camera_left_start.sh` — 左手模式:RPi mjpeg + head_agent WS 寫馬達
- `hand_glove_start.sh` / `_left`
- `hand_slider_left_start.sh` — SSH X11 forwarding 開左手滑桿校準
- `hand_stop.sh` — 一鍵掃所有手部相關 process(本機 + RPi)
- `motor_tool_start.sh` — 舊 Feetech motor_tool GUI

工具類:
- `lib.sh` — RPI_USER/RPI_HOST/RPI_KEY/rpi_ssh/rpi_ping helper
- `rpi_shutdown.sh` — 遠端關機

## 4. 通訊協定

### 4.1 head_agent WebSocket `/ws`

Ops(client → agent):
```
{"op":"write",  "sid":X, "step":Y, "speed":S, "acc":A}    # 依 model 分派 STS/SCS
{"op":"sync",   "cmds":[[sid,step], ...], "speed":S, "acc":A}
{"op":"torque", "sid":X, "on":true|false}
{"op":"torque_all", "on":true|false}
{"op":"read",   "sid":X, "addr":A, "size":N}
{"op":"scan",   "range":[lo,hi]}
{"op":"set_neck","axis":"yaw|pitch|roll","tick_lo":..,"q_lo":..,"tick_hi":..,"q_hi":..}
{"op":"get_neck"}
{"op":"imu_active","on":true|false}          # 開/關頭部 IMU→馬達 gate
{"op":"tele_hz","hz":50}
```

Agent → client:
```
{"t":"tele", "ts":..., "motors":[{sid,pos,volt,temp}, ...]}    # 週期性 telemetry
{"t":"reply","req_id":..,"ok":true,"value":...}
```

### 4.2 head_agent HTTP

- `GET /status` → JSON:motor_port, motor_ids, motor_models (dict sid→model),
  cam_frames, phone_connected, phone_imu_active, phone_pose_age_ms, neck_q
- `GET /snapshot` → single JPEG
- `GET /mjpeg` → multipart MJPEG
- `GET /vr` → phone 校準頁 (HTTPS 8443)

### 4.3 head_agent phone `/pose_ws` (HTTPS 8443)

```
{"yaw":Y, "pitch":P, "roll":R}                     # phone deviceorientation
{"cmd":"active","on":true|false}                   # IMU_ACTIVE 開關
{"cmd":"calibrate"}                                # 舊快校準
{"cmd":"cal_result","data":{"forward":{y,p,r},"yaw_left":..,"yaw_right":..,
                            "pitch_up":..,"pitch_down":..}}
```

### 4.4 pose_imitator UDP

- `127.0.0.1:9999` — JSON `{"L":{"q":[...5], "conf":..}, "R":{...}, "lm2d":[...]}`
- `127.0.0.1:9998` — annotated JPEG bytes

qbot_ik_gui bind 這兩個 port,收到就更新 pose imitation 目標 + 相機面板。

### 4.5 camera_receiver 內部路徑

三種輸出模式(argparse):
- 預設:本機 `HandBus`(serial),直接寫 sid 對應手指
- `--udp-to HOST:PORT`:UDP JSON `{"closure":{"Right":{Thumb,Index,Middle,Ring,Little}}}`
- `--remote-hand HOST:PORT`:WebSocket→head_agent sync op(左手模式使用)

## 5. 設定檔

### 5.1 右手掌 (`hands_control/config.py`)

```
HAND_SIDE = "right"
SERVO_PORT = "/dev/ttyACM0"
FINGERS = {
  "pinky":  id=31 model=1029 SCS009    open=515 close=780
  "ring":   id=32 model=1029 SCS009    open=518 close=240  (反向)
  "middle": id=33 model=1029 SCS009    open=430 close=700
  "index":  id=34 model=521  STS3032   open=3400 close=2650 (反向)
  "thumb":  id=36 model=521  STS3032   open=2660 close=2200 (反向)
}
FIXED = { "thumb_rotate": id=35 model=1029 hold=650 }
```

### 5.2 左手掌 (`hands_control/config_left.py`)

```
HAND_SIDE = "left"
SERVO_PORT = "/dev/ttyACM0"
FINGERS = {
  "pinky":  id=42 model=1029 SCS009    open=550 close=110   (反向)
  "ring":   id=43 model=1029 SCS009    open=410 close=840
  "middle": id=44 model=1029 SCS009    open=550 close=200   (反向)
  "index":  id=41 model=521  STS3032   open=1800 close=2700
  "thumb":  id=46 model=521  STS3032   open=1300 close=2500
}
FIXED = { "thumb_rotate": id=45 model=1029 hold=180 }
```

### 5.3 頭部 NECK_MAPPING (`head_rpi_agent/agent.py`)

實體接線(2026-07-05 校正):
- sid 1 = yaw
- sid 2 = pitch
- sid 3 = roll(現在**不驅動**,保留舊 tick range)

每軸 `tick_lo/q_lo/tick_hi/q_hi` 為線性 2-point mapping。校準時只更新 q_lo/q_hi。
`AXIS_FLIP` env(HEAD_YAW_FLIP、HEAD_PITCH_FLIP、HEAD_ROLL_FLIP)可翻轉單軸方向。

### 5.4 手臂校準 (`upper_body_control/qbot_arm_calibration.json`)

`L` / `R` 各含 5 個 joint 的 `{tick_lo, q_lo, tick_hi, q_hi}` + `zero_offset` + `motor_live` + `slider_range`。線性內插 q ↔ tick。GUI 有「reload cal」按鈕重讀。

### 5.5 MJCF 模型 (`QBOT_MJCF/qbot.xml`)

完整全身 joints:
- 腿 dof_(right|left)_(hip_pitch_04|hip_roll_03|hip_yaw_03|knee_04|ankle_02)
- 手 ub_(right|left)_(shoulder|lateral_raise|arm_twist|elbow) — 4-DoF/支
- 頸 ub_neck_(yaw|pitch|roll)

## 6. 資料流圖

### 6.1 手臂 IK 控制

```
Laptop qbot_ik_gui slider / IK target
   │
   ▼
q → tick (cal_points)
   │
   ▼  WebSocket op="write" / "sync"
arm_agent (RPi:8100) 或 head_agent (RPi:8000)
   │
   ▼ /dev/ttyACM0
STS3215  sid 10-14 (R) / 20-24 (L)
```

同時 UDP :9999 若有 pose_imitator 在跑 → 覆蓋 arm GUI 的 target。

### 6.2 頭部相機 → 幽靈骨架 → 馬達

```
Phone Safari (HTTPS 8443)
   │  DeviceOrientation → wss /pose_ws
   ▼
head_agent 內部:
   IMU_ZERO 減去 → HEAD_NECK_Q + (若 IMU_ACTIVE) motor sync yaw+pitch
   │
   ├─ HTTP /status → HEAD_NECK_Q
   ▼
head_pc_client remote_gui.py:
   phone_neck_q 每 150ms poll → 覆寫 slider q_deg → ghost update
     ▲ IMU 模式 + phone_driving:client 不寫馬達,由 agent 直接寫
     ▲ IK 模式:phone 忽略,slider/滑鼠拖曳 → client 寫馬達
```

模式切換:PC 上位機頂部 radio buttons `[▶ IMU 模式] [▶ IK 拖曳模式]`。

### 6.3 左手相機控制

```
RPi camera /dev/video0
   │ head_agent /mjpeg 640x480 multipart
   ▼
Laptop camera_receiver.py:
   cv2.VideoCapture(http://RPi:8000/mjpeg)  ← reader thread + 3s stall watchdog + reopen
     │
     ▼
   MediaPipe Hands 21 landmarks → per-finger closure 0..1 (EMA α=0.35)
     │
     ▼  cfg_left.tick_from_norm(name, closure) → per-sid tick
   _RemoteHeadAgentBus  (WS→head_agent :8000)
     │  WS reconnect on error
     ▼
   head_agent sync op → dispatch by model (sms_sts / scscl)
     │
     ▼
   sid 41-46 (混掛 STS3032 + SCS009)
```

### 6.4 動作重現 (motion_imitator)

```
~/下載/health_full_qbot_motion.npz
   │  joint_names + joint_angles(N frames × 18) + fps + base_pose
   ▼
motion_replay_gui.py:
   frame_idx → qpos 寫入 MJCF ghost (全身 18 DOFs)
     │
     ▼ 若「送馬達」勾:
   ub_(l|r)_(shoulder|lateral_raise|arm_twist|elbow) → cal → tick
     │
     ▼  WS write
   arm_agent :8100 → 8 顆 arm 馬達
```

## 7. 已建立的關鍵設計決策

1. **一條 bus 混掛 STS + SCS**:head_agent 用 `_known_models[sid]` 分派
   `sms_sts.WritePosEx` vs `scscl.WritePos` 兩個 handler,共用同一條
   `/dev/ttyACM0`。透過 monkey-inject `serial.Serial` 進 scservo_sdk 的
   `PortHandler` 實現。
2. **多 pose 消費者共享 RPi 相機**:head_agent 一份 frame_jpg,MJPEG endpoint
   對每個 client 廣播;pose_imitator 是獨立 process 直接開 v4l2。
3. **UDP 邊界處理**:pose_imitator 用 SO_SNDBUF=262144 保證大 payload 不
   碎;pose_ws 用 wrap_deg 處理 ±180° 邊界。
4. **stream 讀取穩定性**:HTTP MJPEG 用 reader thread + stall watchdog
   3s 自動 reopen,對 ffmpeg 內部 buffer 死結 self-heal。
5. **motor 校準永續化**:JSON 檔存 tick↔q 兩點,GUI reload 按鈕熱重載。
6. **手機校準精靈**:5s prep + 5s record 每個 anchor,forward → yaw_L →
   center → yaw_R → center → pitch_up → center → pitch_down。取每階段
   samples 中位數。
7. **對稱 & 保守校準**:cal_result 用 `_fixed_mapping` 產生固定 ±MAX_DEG,
   forward 一定對到 0 = 馬達中位。避免手機幅度不夠導致的高敏感度。
8. **雙寫防衝突**:PC client + head_agent 都能寫馬達,`_sender_tick` 用
   mode + phone_driving 決定誰負責。

## 8. 快速對照表

| 我想... | 動哪裡 |
|---|---|
| 加/改頭部 NECK 校準常數 | `head_rpi_agent/agent.py` 的 `NECK_MAPPING` |
| 改手指方向 / 端點 tick | `hands_control/config.py` 或 `config_left.py` |
| 加新的 WS op | `head_rpi_agent/agent.py` `_on_cmd()` |
| 換 arm 校準 | `upper_body_control/qbot_arm_calibration.json` + GUI reload |
| 加/移 launcher 按鈕 | `scripts/launcher.py` `_build()` |
| 換手機 wizard 順序 | `head_rpi_agent/vr.html` `WIZ_SEQ` |
| 加新錄音格式 | `motion_imitator/motion_replay_gui.py` `load_motion()` |
| 新 MediaPipe 手指邏輯 | `hands_control/camera_receiver.py` `compute_closures()` |
| 換 UDP schema | `pose_imitator.py` + `qbot_ik_gui.py` `_pose_udp_loop` |

## 9. 已知未完成 / 需要之後處理

- **head 校準 pitch 方向**:sid 2/3 wiring 更新為 pitch=sid2、roll=sid3;
  head_pc_client 尚未推到 RPi(client 端只影響本地 slider 標籤)。
- **head_agent MJPEG BrokenPipeError**:client 中途斷線會產生 log spam,
  無害但吵。
- **手部拇指 closure 幅度**:MediaPipe MCP 角度只有 ~40° 動態,拇指模仿
  行程小。目前用預設 range,若要放大要調 `FINGER_ANGLE_RANGE` + gain。
- **腳部**:MJCF 有 legs,motion_imitator 可播放,但實體 leg 未接線。
- **arm sid 11-14 現在偶爾漏掃**:電源或接觸問題,head_agent scan 每 30s
  自動 rescan 會補上。
