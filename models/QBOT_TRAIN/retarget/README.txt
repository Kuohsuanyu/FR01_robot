KBOT_PETG MuJoCo 模擬模型
使用說明 (建立日期: 2026-05-11)
============================================================

【環境需求】
  conda activate mujoco_env
  Python 套件: mujoco, trimesh, numpy

【互動檢視】
  python -m mujoco.viewer --mjcf KBOT_PETG.mjcf
  或執行: scripts\viewer\run_viewer.bat

  快捷鍵:
    F1       顯示/隱藏說明
    Ctrl+A   顯示碰撞幾何體 (橘色腳板膠囊)
    2        切換視覺層(group 2 = 外觀網格)
    Space    暫停/繼續模擬

【模型結構: KBOT_PETG.mjcf】
  主模型檔，包含完整機器人定義。

  自由度 (nq=17):
    7  = 浮動基座 (freejoint: 3 位移 + 4 四元數)
    10 = 馬達關節 (見下方)

  10 個馬達 (nu=10) ── RL 訓練動作空間:
    [0] dof_right_hip_pitch_04  RS04  ±120 Nm
    [1] dof_right_hip_roll_03   RS03  ±60 Nm
    [2] dof_right_hip_yaw_03    RS03  ±60 Nm
    [3] dof_right_knee_04       RS04  ±120 Nm
    [4] dof_right_ankle_02      RS02  ±17 Nm
    [5] dof_left_hip_pitch_04   RS04  ±120 Nm
    [6] dof_left_hip_roll_03    RS03  ±60 Nm
    [7] dof_left_hip_yaw_03     RS03  ±60 Nm
    [8] dof_left_knee_04        RS04  ±120 Nm
    [9] dof_left_ankle_02       RS02  ±17 Nm

【IMU 設定】
  位置: upper_torso frame  pos=(0.093, 0.0, -0.046)
        軀幹正中心、手臂高度略上方
  顯示: 亮綠色小方塊 (3cm×3cm×3cm, group=2 視覺層)
  感測器 (sensor section):
    imu_acc   加速度計 (noise=0.01)
    imu_gyro  陀螺儀   (noise=0.01)
    imu_mag   磁力計   (noise=0.05)
    imu_site_pos / imu_site_quat / imu_site_linvel / imu_site_angvel

【RL 觀測空間 (nsensor=37)】
  感測器索引:
  [0-4]   base_site: pos, quat, linvel, angvel, vel
  [5-12]  imu: acc, gyro, mag, pos, quat, linvel, angvel, vel
  [13-22] 10 關節位置 (right: pitch/roll/yaw/knee/ankle, left: same)
  [23-32] 10 關節速度 (同上)
  [33-34] right/left_foot_force
  [35-36] right/left_foot_touch

【碰撞設定】
  只有腳底板啟用碰撞 (conaffinity=1):
    KB_D_501R/L: 各 2 個膠囊幾何體 (capsule)
  其餘所有幾何體: contype=0, conaffinity=0 (無碰撞)
  地面: contype=1, conaffinity=1 → 與腳板接觸 ✓

【實測質量 (2026-05-11 更新)】
  upper_torso  : 9.640 kg  (6.8 kg 軀幹 + 2×RS04 髖仰俯 1.42 kg)
  KC_D_102R/L  : 0.163 kg  (×2, 髖軛 PETG)
  RS03_4/5     : 2.577 kg  (×2, 髖側滾+偏轉 RS03×2 + PETG)
  KC_D_301R/L  : 2.212 kg  (×2, 股骨 + RS04 膝)
  KC_D_401R/L  : 0.938 kg  (×2, 小腿 + RS02 踝)
  KB_D_501R/L  : 0.190 kg  (×2, 腳板 PETG)
  arm_L/arm_R  : 1.400 kg  (×2, 單臂實測)
  --------------------------
  總計          : 24.60 kg

  馬達重量參考: RS04=1420g  RS03=880g  RS02=405g

【STL 網格位置: meshes/】
  meshes/legs/          ── 腿部零件 STL (原始)
  meshes/torso/         ── 軀幹 STL (torse_FR01.stl)
  meshes/arms/
    arm_left.stl        ── 合併左臂 (~6.1 MB, 122k 三角形)
    arm_right.stl       ── 合併右臂 (~6.1 MB, 122k 三角形)
    right_components/   ── 右臂原始零件 (merge_arm_stl.py 的輸入來源)
  meshes/head/          ── 頭部 (保留備用)

【腳本: scripts/】
  viewer/
    run_viewer.bat      ── 一鍵啟動 viewer
    view_robot.py       ── Python 啟動腳本
    validate_mjcf.py    ── 驗證 MJCF 載入 (顯示 nbody/nu/nsensor)
    test_joints.py      ── 關節測試

  tools/
    merge_arm_stl.py    ── 合併 right_components/ → arm_right.stl,
                           並 Y 軸鏡像生成 arm_left.stl
    merge_torso_stl.py  ── 合併軀幹 STL
    step_to_stl.py      ── STEP → STL 轉換工具

  training/
    (待填入 RL 訓練腳本)

【訓練政策: Policies/】
  *.kinfer 格式，可直接部署到硬體
  現有政策:
    kbot_zero_position.kinfer  ── 歸零姿勢
    kbot_sine_motion.kinfer    ── 正弦波動作測試
    stand_frozen.kinfer        ── 靜止站立
    determined_hellman.kinfer  ── 訓練結果
    eloquent_ride.kinfer       ── 訓練結果
    fervent_euler.kinfer       ── 訓練結果

【參考資料】
  FR-1_v3.step           ── 原始完整 CAD (63 MB)

============================================================
