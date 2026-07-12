# Q-BOT 巧手控制(獨立模組) — **右手 (right hand)**

沿用 `upper_body_control/` 的分工:硬體 bring-up + 驗證都先在這裡做,
確認無誤才接進即時控制與整合部署。

## 硬體(2026-07-02 首次掃描)
- 串口 `/dev/ttyACM0`, baud `1_000_000`。
- 6 顆 Feetech 伺服,ID 落在 `31..36`,**兩型號混掛,byte order 不同**:

| ID | model | 型號       | byte order | 位置解析度  | min | max | 備註 |
|----|-------|-----------|------------|-------------|-----|-----|------|
| 31 | 1029  | **SCS009** | big-endian    | 10-bit (0..1023) | 515 | 780 | 對到舊 test2.py ID 11 |
| 32 | 1029  | **SCS009** | big-endian    | 10-bit (0..1023) | 240 | 518 | 對到舊 test2.py ID 12 |
| 33 | 1029  | **SCS009** | big-endian    | 10-bit (0..1023) | 430 | 700 | 對到舊 test2.py ID 13 |
| 34 |  521  | STS3032   | little-endian | 12-bit (0..4095) | 2650 | 3400 | 對到舊 test2.py ID 14 |
| 35 | 1029  | **SCS009** | big-endian    | 10-bit (0..1023) | 500 | 750 | 對到舊 test2.py ID 15 |
| 36 |  521  | STS3032   | little-endian | 12-bit (0..4095) | 2200 | 2660 | 對到舊 test2.py ID 16 |

> **踩過的最大坑 — Feetech 兩系列 byte order 相反,同一條匯流排都能掛**:
> - `sms_sts` (SMS/STS/HLS 系) → `scs_end=0`,**little-endian**,位置 12-bit
> - `scscl`   (SCSCL/SCS009 系) → `scs_end=1`,**big-endian**,位置 10-bit
>
> 如果全部用 `sms_sts` handler 讀 SCS009,兩顆 byte 會被顛倒,得出一堆看似「多圈 / 負值」
> 的鬼值(例如 SCS009 min=515 (`0x0203`) 會被 LE 讀成 `0x0302=770`,更糟時最高位
> byte 是 0xF0 就被解成 sign-magnitude 大負值)。
>
> 正確做法:**先 ping 拿到 model → 依 model 用對應 handler**。
> 目前已知 `model 1029 = SCS009`,`model 521 = STS3032 系`;若日後遇到新 model 加進
> `scan_hand_motors.py` 的 `SCSCL_MODELS` set 即可。
>
> mode 都是 0 (position),配合上表 min/max 全在單圈範圍內 → 都是**正常單圈**,
> 之前解讀為「多圈」是錯的,是 byte order 誤讀所致。

## 檔案
| 檔案 | 作用 |
|---|---|
| `config.py` | 右手 5 主動手指 + 1 固定,唯一 source of truth |
| `scan_hand_motors.py` | 掃描所有 Feetech,依 model 派 handler,唯讀印限位/位置/電壓/溫度 |
| `debug_read.py` | Raw byte 讀取 + 多次取樣;byte order 有疑慮時用 |
| `hand_slider_gui.py` | tkinter 逐顆搖桿測試,建 `finger_map.json` 用 |
| `glove_receiver.py` | **JSON UDP 手套接收 → 巧手驅動**,綁 0.0.0.0:6000 |
| `camera_receiver.py` | **相機 MediaPipe 手勢 → 巧手驅動**(不需外接硬體) |
| `vmc_receiver.py` | VMC OSC 通道 (39539) 接收(sender 端限制,不建議)|
| `finger_map.json` | GUI 存出來的手指對應紀錄(config.py 的原料) |

## 用法
```bash
# 預設 /dev/ttyACM0,自動輪詢常見 baud,掃 1..50
python3 scan_hand_motors.py

# 鎖定 baud(推薦,已知本機是 1M)
python3 scan_hand_motors.py --baud 1000000

# 換串口 / 擴大 ID 範圍
python3 scan_hand_motors.py --port /dev/ttyACM1 --max-id 100
```

輸出範例:
```
 ID   model  mode          min_limit         max_limit         present            V    °C
 31    1029  position        +770 (-112.3°)  +3075 ( +90.3°)  -13826 (-1395°)   5.0   24
 34     521  position       +2650 ( +52.9°)  +3400 (+118.8°)   +3015 (  +85°)   5.0   30
```
所有 tick 值都已經 sign-magnitude 解碼,可直接與位置命令比對。

## 三種控制模式

### A. 相機影像手勢(推薦,最簡單)

```bash
# 依賴:pip install opencv-python mediapipe==0.10.13
python3 camera_receiver.py                # camera 0,無畫面
python3 camera_receiver.py --show         # 顯示畫面 + 骨架 debug + closure bar
python3 camera_receiver.py --no-motor     # dry-run
```

- 用 MediaPipe Hands 抓 21 關鍵點,PIP 關節角度算 0..1 closure
- 22-30 fps 穩定,單機不需 Wi-Fi、不用手套硬體、零校準
- 手勢範圍 tune 在 `FINGER_ANGLE_RANGE` 常數(straight/folded 各手指的 degree)
- 追不到手時馬達**保持當下位置**(不會突然張開)

### B. JSON UDP 手套(用專用 sender)

```bash
python3 glove_receiver.py                 # 綁 0.0.0.0:6000,通馬達
python3 glove_receiver.py --no-motor
```

- 對面 Windows 用 `glove_send_gui.py` 送 JSON `{"closure": {"Right": {"Thumb":..., ...}}}`
- 純軟體,格式極簡

### C. VMC OSC(相容 VSeeFace / Warudo 等)

```bash
# 純接收測試(不動馬達),可搭配 pythonosc 送假資料驗證
python3 vmc_receiver.py --no-motor

# 正式跑(會通電馬達 + 送 ID 35 到 hold 650)
python3 vmc_receiver.py
```

- **綁 127.0.0.1:39539** (VMC 官方 port,可用 `--port` 覆寫)
- 訂閱 `/VMC/Ext/Bone/Pos`,對應右手 Proximal(四指彎曲)+ Thumb Intermediate(拇指開合)
- Bone → finger:
  ```
  RightLittleProximal    → pinky   (id 31)
  RightRingProximal      → ring    (id 32)
  RightMiddleProximal    → middle  (id 33)
  RightIndexProximal     → index   (id 34)
  RightThumbIntermediate → thumb   (id 36)
  ```
- 拇指內旋 (ID 35) 不動態控制,啟動時送到 config.FIXED 的 hold_tick=650

啟動後鍵盤指令:
- `o` = 校準張開手 3 秒(抓每根 quaternion baseline)
- `g` = 校準握緊手 3 秒(抓每根 twist 上限)→ **兩個都做完才進 RUN**
- `s` = 看接到哪些骨骼、哪些已校準  `p` = 印目前 RUN 值  `d` = debug 開關  `q` = 退出

## 之後(TODO)
- 左手接上來後:重新跑 `hand_slider_gui.py` 建 finger_map、拆 config 成 `config_right.py` / `config_left.py`,`vmc_receiver.py` 加 `LEFT_BONE_TO_FINGER`。
- 加 `--record` 選項:把每次 RUN 的 0..1 值時序存 JSONL,方便重播/train。
- 上層抽象(FeetechHand 類別)封裝 driver + config 讀取,給 policy 匯入時用。

## 參考
本地已備份 USB 上的關鍵舊資料到 [`reference_from_usb/`](reference_from_usb/README.md):
- `old_hand_gui_range.py` — 舊 6 顆巧手的 `SERVOS = {id: (open_tick, close_tick)}` 校正資料
  (ID 11-16,和目前 ID 31-36 值一致)
- `old_move_single_id.py` / `old_feetech_scan.py` — 原始封包等級的範例
- `scscl_examples/` — SCS009 (model 1029) 用的 SDK 範例(BE / 10-bit)
