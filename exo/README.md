# FR01 外骨骼(Exoskeleton)控制系統

**目標**:操作者穿一副與 FR01 上半身「同比例、同自由度」的外骨骼,由第三顆
獨立 RPi 讀外骨骼各關節的實時角度,傳到筆電,由筆電做「趨近 → 進入即時
連動」的兩段流程,最終驅動實體 FR01 上半身跟著操作者動。

參考架構:類似 RX1 / GR-1 human-machine-interface 外骨骼的做法 —— 
「本體控制器把 upper body 的 motion capture 打包成 UDP/WS,PC 端做 kinematic
retargeting 再送到機器人執行器」。這裡不做 mocap,直接用外骨骼各關節的
encoder 讀值(和機器人各關節「一對一」對應,免 IK)。

---

## 1. 三方架構

```
    ┌────────────────────────────┐
    │  Laptop (Ubuntu, Tk GUI)   │
    │                            │
    │   exo_pc_client/exo_gui.py │
    │                            │
    │   ┌───────────────────┐    │
    │   │ 幽靈 A (機器人)    │    │ 藍青色 — 讀 head/arm agent /status
    │   │ 幽靈 B (外骨骼)    │    │ 橘綠色 — 讀 exo agent /status
    │   └───────────────────┘    │
    │                            │
    │   [ 趨近 → 進入即時連動 ]   │
    └───────┬─────────┬──────────┘
            │         │
            │ WS      │ WS
            │         │
    ┌───────▼──┐   ┌──▼───────────────────┐
    │ head RPi │   │ exo RPi              │
    │ arm/hand │   │ (fully independent)  │
    │ port 8000│   │ port 8200            │
    └────┬─────┘   └──────┬───────────────┘
         │                │
         │ /dev/ttyACM0   │ /dev/ttyACM0 (or CAN)
         ▼                ▼
    FR01 上半身        外骨骼各關節 encoder
    18 顆 Feetech      (Feetech STS3215 假設,或 magnetic encoder)
```

**三個節點:**
| 節點 | IP 來源 | 職責 |
|---|---|---|
| head RPi | `.rpi_host` (auto-detect via `find_rpi.sh --head`) | 驅動 FR01 上半身馬達,serve /mjpeg + WS |
| **exo RPi** | `.rpi_host_exo` (auto-detect via `find_rpi.sh --exo`) | **讀外骨骼 encoder,serve /ws + /status** |
| Laptop | 本機 | `exo_gui.py` 顯示兩個 ghost + 控制門檻 |

---

## 2. 兩個幽靈骨架的 UI 設計

**同一個 MJCF (qbot.xml)** 開兩個 `MjData`:
- `data_robot` — 由 head/arm agent 的 telemetry 更新(顯示現在機器人真的在哪)
- `data_exo` — 由 exo agent 的 /status 更新(顯示操作者現在穿的外骨骼在哪)

`mj_forward` 各自跑,`Renderer.update_scene()` 用 additive 模式把兩個 data 疊
在同一畫面。用 `geom_rgba` 幫兩份骨架上色:
- 機器人:半透明藍青(`[0.30, 0.55, 0.85, 0.50]`)
- 外骨骼:半透明橘紅(`[0.90, 0.55, 0.30, 0.55]`)

顯示模式:
- **未連動時**:兩個 ghost 各自更新,可能有明顯落差
- **趨近中**:機器人 ghost 慢慢用 EMA 貼近 exo ghost 的姿勢(視覺看得到藍青
  在追橘紅),但**沒有寫馬達**
- **連動中**:機器人 ghost 應該和 exo ghost 幾乎重疊,同時 WS 寫馬達
  (frame-by-frame)

---

## 3. 兩段狀態機

```
      IDLE
        │  ← 按「開始趨近」
        ▼
     CONVERGE
        │  超過 T=5s 且 max_err < threshold
        ▼
     READY
        │  ← 按「進入即時連動」
        ▼
   LIVE  (WS write @ 50 Hz)
        │  ← 按「解除」/E-stop
        ▼
      IDLE
```

- **IDLE**:兩個 ghost 各自顯示,不動馬達
- **CONVERGE**:對每個 joint,`robot_ghost.q_target = (1-α) * current + α * exo.q`,
  α 每 tick 用 T=5s 的 slew 慢慢從 0 走到 1。到達後 err<閾值才進 READY。
  **這段沒有寫馬達** — 只是幽靈預覽。
- **READY**:提示可以按下一步。此時如果按「送馬達」會**用當前 robot_ghost 的
  q 值送出**(一次 sync 到位),讓實體慢慢跟上 ghost。
- **LIVE**:每個 tick 讀 exo → 直接寫馬達 sync + 更新 robot_ghost。
  E-stop 可以隨時退出。

---

## 4. 對應表(exo joint → robot joint)

外骨骼 encoder 的實體位置和 FR01 一對一,不做 IK,只做:
- **鏡射方向(mirror sign)**:若 encoder 正轉方向和機器人相反 → 翻符號
- **偏移(zero offset)**:操作者穿上時的自然姿勢定義為 q=0

假設外骨骼包含 FR01 完整上半身自由度 (13 joints):

| Exo channel | Robot joint (MJCF) | Motor sid | Sign | Note |
|---|---|---|---|---|
| exo_neck_yaw | ub_neck_yaw | 1 | +1 | |
| exo_neck_pitch | ub_neck_pitch | 2 | +1 | (pitch=sid2 per 2026-07-05 wiring) |
| exo_neck_roll | ub_neck_roll | 3 | +1 | 可選,若不做則 hold 中位 |
| exo_R_shoulder | ub_right_shoulder | 10 | +1 | |
| exo_R_lat_raise | ub_right_lateral_raise | 11 | +1 | |
| exo_R_arm_twist | ub_right_arm_twist | 12 | +1 | |
| exo_R_elbow | ub_right_elbow | 13 | +1 | |
| exo_R_wrist | (virtual) | 14 | +1 | |
| exo_L_shoulder | ub_left_shoulder | 20 | +1 | |
| exo_L_lat_raise | ub_left_lateral_raise | 21 | +1 | |
| exo_L_arm_twist | ub_left_arm_twist | 22 | +1 | |
| exo_L_elbow | ub_left_elbow | 23 | +1 | |
| exo_L_wrist | (virtual) | 24 | +1 | |

(手指/hand 是否做外骨骼版本待定,若做,再擴展一份 `hand_channel` 對應表。)

Sign / offset 由 [`exo_rpi_agent/config.py`](exo_rpi_agent/config.py) 定義。

---

## 5. 資料流

**Exo → Laptop**(WS `/pose` 廣播,50 Hz):
```json
{
  "t": "exo_pose",
  "ts": 1234567890.123,
  "q": {
    "ub_neck_yaw": 0.05,
    "ub_neck_pitch": -0.10,
    "ub_right_shoulder": 0.30,
    ...
  }
}
```
Laptop 端 `exo_gui.py` 訂閱這個 topic,寫進 `data_exo.qpos`。

**Laptop → Head/Arm agent**(WS `/ws`,在 LIVE 狀態下,50 Hz):
```json
{"op": "sync", "cmds": [[sid, tick], ...], "speed": S, "acc": A}
```
每個 exo joint 值透過 [`qbot_arm_calibration.json`](../upper_body_control/qbot_arm_calibration.json)
換成 tick,批次送 sync。

**Laptop ← Head/Arm agent**(/status polling,10 Hz):
用來更新 `data_robot.qpos` — 顯示機器人「實際在哪」。

---

## 6. 檔案結構

```
exo_control/
├── README.md                       # 本文件
├── exo_rpi_agent/
│   ├── agent.py                    # RPi 端 WS + /status,50Hz 廣播 exo_pose
│   ├── config.py                   # exo channel → robot joint 對應
│   ├── fake_encoder.py             # 沒硬體時用來測 GUI(sin wave 假動作)
│   └── requirements.txt
├── exo_pc_client/
│   └── exo_gui.py                  # 雙 ghost + state machine (IDLE/CONVERGE/READY/LIVE)
└── scripts/
    ├── exo_start.sh                # SSH → RPi 啟 agent + 本機起 gui
    ├── exo_stop.sh
    └── (find_rpi.sh 已支援 --exo)
```

---

## 7. 上線流程

**開發(沒真外骨骼硬體):**
1. `exo_rpi_agent/fake_encoder.py --mode sin` — 本機跑 fake agent,產 sine 假姿勢
2. 開 `exo_pc_client/exo_gui.py --exo-host 127.0.0.1:8200` — 看兩個 ghost,一個
   靜態一個擺動,測狀態機
3. 都能動 → 換到真外骨骼硬體

**上線(有外骨骼硬體):**
1. launcher 點「[?] 尋找 Exo RPi IP」(等我把按鈕加進去) → 寫 `.rpi_host_exo`
2. 「▶ 啟動 Exo」 → SSH 啟 `exo_rpi_agent/agent.py` 在 exo RPi,同時本機起 GUI
3. 操作者穿好,雙手自然下垂 → 按「校零」(agent 把當前讀值設為 0 位)
4. 動一下手臂 → GUI 上橘紅 ghost 跟著動(藍青 robot ghost 不動)
5. 按「趨近」→ 藍青 ghost 慢慢從當前姿勢貼過去橘紅 ghost(5 秒)
6. err 收斂 → 進 READY,按「送馬達到當前 ghost」 → 實體慢慢動到藍青 ghost
7. 對齊後按「進入 LIVE」→ 每一 tick 讀 exo → 寫馬達,實體同步動作
8. 有問題按「E-stop」 → 退回 IDLE,馬達停在當前 tick(不 disable)

---

## 8. 待定事項(給你決定)

- [ ] 外骨骼 encoder 用什麼 — Feetech STS(讀 present position 當 encoder)? 磁感應
      encoder(AS5048 之類)? CAN?
- [ ] 外骨骼 RPi 用哪個 model — 沿用 Feetech USB CH340 + head_agent 那份程式碼
      可以直接改寫
- [ ] 外骨骼是否做 5-DoF 手指版本(靈巧手部份)
- [ ] Exo 是否包含頭部 IMU(現在用手機 IMU;若外骨骼有頭套裝 IMU 就取代)
- [ ] Handshake safety:操作者按什麼實體按鈕觸發「進入 LIVE」(避免誤按 GUI 導致
      手臂突然動)

在這些決定之前,我先照 **Feetech STS3215 假設** 建 stub,實體換掉再改一小段
就好。
