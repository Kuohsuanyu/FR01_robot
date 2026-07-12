# Q-BOT 上半身伺服控制(獨立模組)

**刻意與腿部/policy 部署分開**。先在這裡獨立開發 + 驗證上半身 Feetech 伺服控制,
確認無誤後,才匯出模型、接上即時控制,最後才整合上下半身。

## 硬體
- 8 個手臂關節,Feetech **STS3215**(SMS_STS 協定),4096 步/圈。
- **減速**:肩部 3 自由度(shoulder / lateral_raise / arm_twist)各 **4:1 行星減速**;
  手肘 **直驅(1:1)**。與訓練模型 `QBOT_TRAIN/qbot.xml` 一致。
- 因為 `gear × 角度` 會超過一圈(90° × 4 = 360° 伺服行程),STS3215 需設**多圈/步進位置模式**。

## 關節 ↔ 伺服換算(沿用 rx1_motor 方案)
```
servo_steps = round(angle_rad × 651.9 × gear × dir) + 2048 + home
servo_speed = round(joint_speed × gear × 652.6)     # rad/s → Feetech 單位
servo_acc   = round(joint_acc   × gear × 6.526)
```

## 檔案
| 檔案 | 作用 |
|---|---|
| `config.py` | 8 關節設定(id/gear/dir/home/limits)、常數、home 姿態、串口 |
| `feetech_arm.py` | 換算 + `FeetechArm` 控制類別(SDK 或無硬體 MockBus) |
| `test_mapping.py` | **不需硬體**驗證換算/減速/多圈範圍/home 指令 |

## 跑驗證(現在就能跑,不需硬體)
```bash
cd upper_body_control
python test_mapping.py        # 印出換算結果與 home 姿態的伺服指令
```

## ⚠️ Bring-up 待填(硬體專屬,在 `config.py`)
這些**現在是 placeholder**,要在實機上量測/校正:
1. **`id`** — 每個關節的 Feetech 伺服匯流排 ID(用 Feetech 工具掃描)。
2. **`dir`** — ±1,讓關節正方向與模型一致(裝配方向決定;逐軸試)。
3. **`home`** — 關節角 0 時的伺服步數偏移(校正中位)。
4. **`SERVO_PORT`** — 控制板串口(如 `/dev/ttyUSB0`)、baud。
5. Feetech Python SDK:`pip install feetech-servo-sdk`(提供 `scservo_sdk`)。

## 跟得上推論速度?(關鍵驗證)
- 伺服空載 4.72 rad/s ÷ 減速 4 = **關節最高 ~1.18 rad/s**。
- 50Hz 控制下,每步關節位移上限 = 1.18 × 0.02 ≈ **0.024 rad(~1.35°)**。
- 訓練端已用 `damping=3.3` 把肩部壓在 ~0.9 rad/s,所以 policy 命令軌跡會落在範圍內 → 伺服跟得上。

## 之後整合(尚未做)
1. 模型訓練好 → 匯出 ONNX/kinfer。
2. 這個模組接收 policy 的 8 個手臂目標(`set_arm_vector`),即時送伺服。
3. 與腿部(Robstride)+ IMU(H30 橋接)整成完整部署(見 `../deployment/`)。
   - 可維持獨立 Python 程序(類似 IMU 橋接),或日後移植進 Rust 韌體。
