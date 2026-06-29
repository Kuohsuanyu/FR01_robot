# Q-BOT

K-Bot 腿 + Q-BOT 上半身的人形機器人:ksim/JAX 走路訓練 + sim2sim 驗證 + 實機部署
(腿 Robstride / 手臂 Feetech 伺服)。

## 目錄結構
```
.
├── QBOT_TRAIN/          走路訓練(ksim 0.1.3 / JAX / MuJoCo)
│   ├── train_qbot.py        訓練主程式(策略網路、獎勵、致動器)
│   ├── qbot.xml + meshes/   訓練模型(18 DOF,脖子鎖定)
│   ├── retarget/            動作 retarget 工具
│   └── humanoid_walking_task/run_0/policy/   sim2sim 工具(下方)
├── QBOT_MJCF/           完整 MJCF(21 DOF,含脖子)+ view.py 互動檢視器
├── QBOT_URDF/           URDF(由 MJCF 轉出,含關節限位/扭力)
├── upper_body_control/  ★ 上半身 Feetech 伺服控制(Linux/部署端,給 policy 即時控制)
├── windows_motor_ik/    ★ Windows 手臂伺服控制 + IK GUI 工具(rx1 移植,bring-up/測試)
├── deployment/          實機部署
│   ├── deploye_robot/       K-Bot Rust 韌體(策略 ONNX 推論 + Robstride + IMU)
│   └── imu_linux/           H30 IMU 橋接(Python)
└── rx1/                 (gitignore)Red-Rabbit RX1 參考,減速方案參考用
```

## 硬體配置(重點)
- **腿**:Robstride 馬達(CAN),沿用 K-Bot。
- **手臂**:Feetech STS3215 伺服。**肩部 3 自由度有 4:1 行星減速**,手肘直驅。
- **IMU**:WHEELTEC H30 Mini → HiWonder 協定橋接。

## 環境
訓練/模擬用 conda env `ksim`(Python 3.11, ksim 0.1.3, JAX 0.4.38)。
CUDA 套件須固定 **12.8**(驅動 590 / RTX 5070 上限);12.9 會壞 cuSolver。

## 常用指令
```bash
PY=/home/andykuo/miniconda3/envs/ksim/bin/python    # (本機路徑,測試機請改)

# 訓練
cd QBOT_TRAIN && $PY -m train_qbot num_envs=256 batch_size=128

# sim2sim(策略在獨立 MuJoCo 跑 + 關節扭力視窗)
cd QBOT_TRAIN/humanoid_walking_task/run_0/policy
$PY sim2sim.py            # 互動 viewer(需顯示器)
$PY sim2sim.py --video    # 輸出 mp4

# 上半身換算驗證(不需硬體)
cd upper_body_control && $PY test_mapping.py
```

## 狀態 / 進行中
- 走路訓練:慢走(forward clip 0.5 m/s)+ 手臂不舉高 + 肩部減速限速(damping 3.3 ≈ 0.9 rad/s)。
- 上半身伺服控制:獨立開發中(`upper_body_control/`),硬體值待 bring-up 填入。
- 待做:模型匯出 ONNX → 接上半身即時控制 → 整合上下半身部署。

> `humanoid_walking_task/` 的 checkpoint/tensorboard 不入庫(訓練重生)。
> 測試機 clone 後:建立 `ksim` 環境、改各腳本的 Python 路徑與串口、`rx1/` 另行 clone。
