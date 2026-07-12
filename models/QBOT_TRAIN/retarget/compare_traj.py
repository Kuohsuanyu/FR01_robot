"""
並排比較 H1 vs KBOT 軌跡播放
左視窗：UnitreeH1（參考動作）
右視窗：KBOT（我們的機器人）

執行方式：
    conda activate loco_new
    python compare_traj.py
"""

import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# ── 軌跡路徑 ──────────────────────────────────────────────────────────
TRAJ_NPZ = "/home/andykuo/.loco-mujoco-caches/lafan1_converted/KBotEnv/walk1_subject1.npz"
H1_XML   = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1/h1.xml"
KBOT_XML = "/home/andykuo/ksim-gym/MIMIC/kbot/KBOT_loco.xml"


def load_trajectory(npz_path):
    """載入軌跡 NPZ，回傳 qpos、joint_names、fps"""
    data   = np.load(npz_path, allow_pickle=True)
    qpos   = data["qpos"]          # (T, 26)
    jnames = list(data["joint_names"])  # ['root', 'hip_rotation_l', ...]
    fps    = int(data["frequency"])
    return qpos, jnames, fps


def build_qpos_map(model, traj_jnames):
    """
    建立「軌跡 joint index → model qpos index」的映射。
    回傳 list of (traj_col, model_qpos_start, ndof)
    """
    mapping = []
    # freejoint 固定在最前面
    free_col = traj_jnames.index("root") if "root" in traj_jnames else None
    if free_col is not None:
        mapping.append((free_col, 0, 7))   # freejoint 佔 7 個 qpos

    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname == "root":
            continue
        if jname in traj_jnames:
            traj_col = traj_jnames.index(jname)
            qpos_start = model.jnt_qposadr[jid]
            ndof = 1  # hinge joint
            mapping.append((traj_col, qpos_start, ndof))
    return mapping


def set_qpos_from_traj(data_mj, traj_qpos_row, mapping, x_offset=0.0):
    """把軌跡的一幀 qpos 寫入 MjData，並加上 x 方向偏移"""
    for traj_col, qpos_start, ndof in mapping:
        data_mj.qpos[qpos_start:qpos_start+ndof] = traj_qpos_row[traj_col:traj_col+ndof]
    # 把 freejoint 的 x 座標加上偏移
    data_mj.qpos[0] += x_offset


def main():
    # ── 載入軌跡 ──────────────────────────────────────────────────────
    print("Loading trajectory...")
    traj_qpos, traj_jnames, fps = load_trajectory(TRAJ_NPZ)
    dt = 1.0 / fps
    n_steps = len(traj_qpos)
    print(f"Trajectory: {n_steps} steps @ {fps} Hz ({n_steps/fps:.1f} sec)")

    # ── 載入模型 ──────────────────────────────────────────────────────
    print("Loading models...")
    h1_model   = mujoco.MjModel.from_xml_path(H1_XML)
    kbot_model = mujoco.MjModel.from_xml_path(KBOT_XML)
    h1_data    = mujoco.MjData(h1_model)
    kbot_data  = mujoco.MjData(kbot_model)

    # ── 建立 joint 映射 ───────────────────────────────────────────────
    h1_map   = build_qpos_map(h1_model, traj_jnames)
    kbot_map = build_qpos_map(kbot_model, traj_jnames)

    print(f"H1   qpos mapped: {len(h1_map)} joints")
    print(f"KBOT qpos mapped: {len(kbot_map)} joints")

    # ── 啟動兩個被動視窗 ─────────────────────────────────────────────
    print("\nOpening viewers (arrange side by side)...")
    print("  Left window  = UnitreeH1  (reference)")
    print("  Right window = KBOT       (your robot)")

    h1_viewer   = mujoco.viewer.launch_passive(h1_model,   h1_data,
                                               show_left_ui=False, show_right_ui=False)
    kbot_viewer = mujoco.viewer.launch_passive(kbot_model, kbot_data,
                                               show_left_ui=False, show_right_ui=False)

    h1_viewer.cam.azimuth   = 90
    h1_viewer.cam.elevation = -15
    h1_viewer.cam.distance  = 3.5
    kbot_viewer.cam.azimuth   = 90
    kbot_viewer.cam.elevation = -15
    kbot_viewer.cam.distance  = 3.5

    # ── 播放迴圈（可重複） ────────────────────────────────────────────
    episode = 0
    while h1_viewer.is_running() and kbot_viewer.is_running():
        episode += 1
        print(f"\n▶ Episode {episode}")
        t0 = time.time()

        for step in range(n_steps):
            if not h1_viewer.is_running() or not kbot_viewer.is_running():
                break

            row = traj_qpos[step]

            # 寫入 H1
            set_qpos_from_traj(h1_data, row, h1_map, x_offset=0.0)
            mujoco.mj_forward(h1_model, h1_data)
            h1_viewer.sync()

            # 寫入 KBOT（相同動作，只映射腿部 joint）
            set_qpos_from_traj(kbot_data, row, kbot_map, x_offset=0.0)
            mujoco.mj_forward(kbot_model, kbot_data)
            kbot_viewer.sync()

            # 控制播放速度
            elapsed  = time.time() - t0
            expected = (step + 1) * dt
            sleep_t  = expected - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        print(f"   Done in {time.time()-t0:.1f}s")
        time.sleep(0.5)

    print("Viewers closed.")


if __name__ == "__main__":
    main()
