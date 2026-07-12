# Q-BOT 完整機器人 MJCF（獨立、含限位與扭力）

完整機器人：K-Bot 腿 + 腰腿連接件 + Q-BOT 上半身（含脖子）。
自包含 MJCF + mesh，可直接用 MuJoCo 開,或自行轉成 URDF。

## 結構
```
QBOT_MJCF/
├── qbot.xml              # 自包含模型（相對 mesh 路徑、無中文/絕對路徑）
└── meshes/
    ├── legs/       (24)  # K-Bot 腿
    ├── waist/      (31)  # 腰腿連接件
    └── upperbody/  (22)  # Q-BOT 上半身
```
41 bodies、22 joints（1 floating base + 21 可動）、21 actuators、77 meshes。

## 開啟
```bash
python -m mujoco.viewer --mjcf=qbot.xml
# 或
import mujoco; m = mujoco.MjModel.from_xml_path("qbot.xml")
```

## 關節限位 + 扭力（全部 21 個關節都有）

| 部位 | 關節 | 限位 | 扭力(峰值) |
|---|---|---|---|
| 腿 hip_pitch/knee | RS04 | kbot 內建 | ±120 N·m |
| 腿 hip_roll/yaw | RS03 | kbot 內建 | ±60 N·m |
| 腿 ankle | RS02 | kbot 內建 | ±17 N·m |
| 手臂 shoulder/lateral_raise/arm_twist | Feetech STS3215 | ±90° | **±0.98 N·m (10 kg·cm)** |
| 手臂 elbow | Feetech STS3215 | 0~135° | **±0.98 N·m** |
| 脖子 pitch/roll | Feetech STS3215 | ±30° | **±0.98 N·m** |
| 脖子 yaw | Feetech STS3215 | ±90° | **±0.98 N·m** |

- 控制方式：全部**位置控制（position actuator）**。
- 扭力以 actuator `forcerange` 設定（硬上限）；手臂/脖子伺服 = 10 kg·cm = 0.981 N·m。
- 腿用峰值當硬上限（額定 40/20/6 N·m 的軟性懲罰只在訓練腳本內,不在此模型）。

## 轉 URDF 提示
MuJoCo 無內建 MJCF→URDF。可用社群工具或自寫腳本走訪 body/joint/mesh。
本模型 `jnt_pos` 全為 0（關節在 body 原點）、每個可動 body 僅 1 個 hinge,轉換很乾淨。
