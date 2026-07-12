# Q-BOT 訓練包 (ksim walking)

K-Bot 腿 + 腰腿連接件 + Q-BOT 上半身,沿用 ksim-gym 的走路訓練與獎勵。
在 **Linux + NVIDIA GPU** 上訓練。

## 架構

```
QBOT_TRAIN/
├── train_qbot.py        # 訓練程式(由 ksim-gym train.py 移植)
├── qbot.xml             # 自包含模型(脖子鎖定、IMU、扭力上限都已內建)
├── meshes/
│   ├── legs/            # K-Bot 腿 (24)
│   ├── waist/           # 腰腿連接件 (31)
│   └── upperbody/       # Q-BOT 上半身 (22)
├── requirements.txt
└── README.md
```

模型統計:19 joints(1 free base + 18 可動)、18 actuators、16 sensors、77 meshes。

## 關節(18 可動)

| 部位 | 數量 | 馬達 | 扭力 |
|---|---|---|---|
| 腿(左右各 5) | 10 | Robstride | 峰值 ±120/±60/±17,額定 40/20/6 N·m |
| 手臂(左右各 4) | 8 | Feetech STS3215 伺服(位置控制) | ±0.98 N·m(10 kg·cm) |
| 脖子 | 3 | — | **訓練時鎖定**(剛體) |

- 腿:hip_pitch / hip_roll / hip_yaw / knee / ankle(與 K-Bot 完全相同)
- 手臂:shoulder / lateral_raise / arm_twist / elbow(左右對稱)
- IMU 在上胸,感測器名 `imu_site_quat` / `imu_acc` / `imu_gyro`

## 扭力模型(腿)

峰值是硬上限(可短暫爆發);超過**額定**(RS04=40 / RS03=20 / RS02=6 N·m)由
`OverRatedTorquePenalty` 線性扣分,且持續越久累積越多(thermal,慢慢長大)。

## 安裝(Linux GPU)

```bash
python -m venv venv && source venv/bin/activate     # Python >= 3.11
pip install -r requirements.txt
pip install "jax[cuda12]"
python -c "import jax; print(jax.default_backend())"   # 應印 gpu
```

## 訓練

```bash
# 1. 先確認模型載入正確(開互動視窗)
python -m train_qbot run_mode=view

# 2. 開始訓練
python -m train_qbot

# 3. 儀表板(TensorBoard)
tensorboard --logdir humanoid_walking_task
# 瀏覽器開 http://localhost:6006
```

## ⚠️ 在 GPU 機上要驗證/可能要微調的點

1. **`num_critic_inputs`**(train_qbot.py `get_model`):critic 觀測含 com_inertia /
   com_vel,長度隨 body 數變(本模型 body 比 kbot 多)。若啟動報 critic shape
   mismatch,錯誤訊息會印出期望長度,填進去即可。目前暫填 446。
2. **`get_actuators`**:用模型內建的 `<position>` actuator。若你的 ksim 版本沒有
   `MITPositionActuators`,程式會 fallback 到 `PositionActuators(metadata=...)`。
3. **`OverRatedTorquePenalty`**:用 `trajectory.obs["actuator_force_observation"]`
   + `jax.lax.scan`,邏輯正確但請確認與你安裝的 ksim Reward API 相容。

## 與原版 K-Bot 的差異

見原始開發資料夾的 `DIFF.md`。摘要:腿完全沿用;手臂改 8 DOF 對稱伺服、無手腕;
脖子訓練鎖定;IMU 移到上胸;新增腿部額定扭力懲罰。獎勵與 PPO/網路架構全部沿用。
