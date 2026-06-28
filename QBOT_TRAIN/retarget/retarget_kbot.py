"""
KBOT FK→IK Retargeting — Damped Jacobian IK
將 UnitreeH1 的 LAFAN1 軌跡重定向到 KBOT

方法：
  1. H1 Forward Kinematics → 取得骨盆/髖/膝/踝世界座標
  2. 按 KBOT 腿長縮放（以骨盆為原點）
  3. Damped Least-Squares Jacobian IK → 穩定求解 KBOT 關節角度（含關節限位）

執行：
    conda activate loco_new
    python retarget_kbot.py walk1_subject1

輸出：
    ~/.loco-mujoco-caches/lafan1_converted/KBotEnv_retargeted/{motion}.npz
"""

import sys, os, time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

H1_XML   = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1/h1.xml"
KBOT_XML = "/home/andykuo/ksim-gym/MIMIC/kbot/KBOT_loco.xml"
TRAJ_DIR = "/home/andykuo/.loco-mujoco-caches/lafan1_converted"
OUT_DIR  = f"{TRAJ_DIR}/KBotEnv_retargeted"

MOTION    = sys.argv[1] if len(sys.argv) > 1 else "walk1_subject1"
TRAJ_NPZ  = f"{TRAJ_DIR}/KBotEnv/{MOTION}.npz"

# H1 → KBOT 的 site 對應（骨盆單獨處理）
H1_SITES  = ["pelvis_mimic",
             "left_hip_mimic",  "left_knee_mimic",  "left_foot_mimic",
             "right_hip_mimic", "right_knee_mimic", "right_foot_mimic"]
IK_SITES  = H1_SITES[1:]   # 不含骨盆

# IK 超參數
IK_MAX_ITER = 200           # 最大迭代次數
IK_DAMPING  = 0.05          # 阻尼係數（防止奇異）
IK_ALPHA    = 0.5           # 步長
IK_TOL      = 5e-3          # 收斂閾值（公尺）
SKIP_EVERY  = 1


def get_sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def compute_scale(h1_model, kbot_model):
    hd = mujoco.MjData(h1_model)
    kd = mujoco.MjData(kbot_model)
    mujoco.mj_forward(h1_model, hd)
    mujoco.mj_forward(kbot_model, kd)

    def bp(model, data, bname):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        return data.xpos[bid].copy()

    h1_thigh  = np.linalg.norm(bp(h1_model,   hd, "left_knee_link")  - bp(h1_model,   hd, "left_hip_yaw_link"))
    h1_shin   = np.linalg.norm(bp(h1_model,   hd, "left_ankle_link") - bp(h1_model,   hd, "left_knee_link"))
    kb_thigh  = np.linalg.norm(bp(kbot_model, kd, "left_knee_link")  - bp(kbot_model, kd, "left_hip_yaw_link"))
    kb_shin   = np.linalg.norm(bp(kbot_model, kd, "left_ankle_link") - bp(kbot_model, kd, "left_knee_link"))

    print(f"H1   thigh={h1_thigh:.4f}  shin={h1_shin:.4f}")
    print(f"KBOT thigh={kb_thigh:.4f}  shin={kb_shin:.4f}")
    print(f"Scale thigh={kb_thigh/h1_thigh:.4f}  shin={kb_shin/h1_shin:.4f}")
    return kb_thigh / h1_thigh, kb_shin / h1_shin


def damped_ik(model, data, q_adr, q_limits, target_positions, site_ids,
              max_iter=IK_MAX_ITER, damping=IK_DAMPING, alpha=IK_ALPHA, tol=IK_TOL):
    """
    Damped least-squares Jacobian IK.
    q_adr    : list of qpos addresses for the leg joints (10 joints)
    q_limits : (n_joints, 2) array of [lo, hi]
    target_positions : (n_sites, 3) target world positions
    site_ids : list of site IDs in the model
    """
    n_joints = len(q_adr)
    n_sites  = len(site_ids)
    n_tasks  = n_sites * 3

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        # 計算位置誤差
        error = np.zeros(n_tasks)
        for k, sid in enumerate(site_ids):
            error[k*3:(k+1)*3] = target_positions[k] - data.site_xpos[sid]

        if np.linalg.norm(error) < tol:
            break

        # 建立雅可比矩陣 (n_tasks × nv)
        J_full = np.zeros((n_tasks, model.nv))
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        for k, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J_full[k*3:(k+1)*3, :] = jacp

        # 只取腿部關節的欄（freejoint 的 vel index 是前 6 個）
        # KBOT freejoint = dof 0-5 (3 trans + 3 rot), leg joints = dof 6-15
        # 找每個 qpos address 對應的 dof index
        J = np.zeros((n_tasks, n_joints))
        for i, adr in enumerate(q_adr):
            jid = np.where(model.jnt_qposadr == adr)[0]
            if len(jid) > 0:
                dof_adr = model.jnt_dofadr[jid[0]]
                J[:, i] = J_full[:, dof_adr]

        # Damped pseudo-inverse: J_pinv = J^T (J J^T + λ²I)^-1
        JJt   = J @ J.T + (damping**2) * np.eye(n_tasks)
        delta_q = J.T @ np.linalg.solve(JJt, error)
        delta_q *= alpha

        # 更新關節角度 + 限位 clamp
        for i, adr in enumerate(q_adr):
            data.qpos[adr] += delta_q[i]
            data.qpos[adr]  = np.clip(data.qpos[adr], q_limits[i, 0], q_limits[i, 1])

    return data.qpos.copy()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{MOTION}.npz")

    print(f"Motion: {MOTION}")
    print(f"Input:  {TRAJ_NPZ}")
    print(f"Output: {out_path}")

    # 載入軌跡
    npz = np.load(TRAJ_NPZ, allow_pickle=True)
    traj_qpos   = npz["qpos"]
    traj_jnames = list(npz["joint_names"])
    fps = int(npz["frequency"])
    T   = len(traj_qpos)
    print(f"Trajectory: {T} steps @ {fps} Hz")

    traj_col = {}; c = 0
    for n in traj_jnames:
        traj_col[n] = c; c += 7 if n == "root" else 1

    # 載入模型
    print("Loading models...")
    h1_model  = mujoco.MjModel.from_xml_path(H1_XML)
    h1_data   = mujoco.MjData(h1_model)
    kbot_model = mujoco.MjModel.from_xml_path(KBOT_XML)
    kbot_data  = mujoco.MjData(kbot_model)
    mujoco.mj_forward(kbot_model, kbot_data)

    thigh_scale, shin_scale = compute_scale(h1_model, kbot_model)
    leg_scales = np.array([thigh_scale, thigh_scale, shin_scale,
                           thigh_scale, thigh_scale, shin_scale])

    # H1 site IDs
    h1_site_ids = [get_sid(h1_model, s) for s in H1_SITES]
    # KBOT site IDs（IK 目標）
    kbot_site_ids = [get_sid(kbot_model, s) for s in IK_SITES]

    # KBOT 腿部關節 qpos address 和限位（跳過 freejoint）
    leg_q_adr    = []
    leg_q_limits = []
    for jid in range(kbot_model.njnt):
        name = mujoco.mj_id2name(kbot_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and name != "root":
            leg_q_adr.append(kbot_model.jnt_qposadr[jid])
            leg_q_limits.append(kbot_model.jnt_range[jid])
    leg_q_limits = np.array(leg_q_limits)
    print(f"Leg joints: {len(leg_q_adr)} DOF")
    print(f"IK: damping={IK_DAMPING}  alpha={IK_ALPHA}  max_iter={IK_MAX_ITER}  tol={IK_TOL}m")

    # KBOT freejoint qpos address
    kbot_root_adr = kbot_model.jnt_qposadr[0]

    # ── 逐幀重定向 ────────────────────────────────────────────────────────
    indices   = list(range(0, T, SKIP_EVERY))
    n_out     = len(indices)
    kbot_qpos_list = []
    kbot_qvel_list = []

    print(f"\nRetargeting {n_out} frames...")
    t0 = time.time()

    for i_frame, t_idx in enumerate(indices):
        row = traj_qpos[t_idx]

        # H1 FK
        rc = traj_col["root"]
        h1_data.qpos[0:7] = row[rc:rc+7]
        for jn in traj_jnames[1:]:
            jid = mujoco.mj_name2id(h1_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                h1_data.qpos[h1_model.jnt_qposadr[jid]] = row[traj_col[jn]]
        mujoco.mj_forward(h1_model, h1_data)

        h1_site_pos = np.array([h1_data.site_xpos[s].copy() for s in h1_site_ids])
        h1_site_mat = h1_data.site_xmat[h1_site_ids[0]].reshape(3, 3)
        h1_pelvis_pos = h1_site_pos[0]

        # 骨盆：直接設 freejoint
        kbot_data.qpos[kbot_root_adr:kbot_root_adr+3] = h1_pelvis_pos
        quat = Rotation.from_matrix(h1_site_mat).as_quat()
        kbot_data.qpos[kbot_root_adr+3] = quat[3]   # qw
        kbot_data.qpos[kbot_root_adr+4] = quat[0]   # qx
        kbot_data.qpos[kbot_root_adr+5] = quat[1]   # qy
        kbot_data.qpos[kbot_root_adr+6] = quat[2]   # qz

        # 縮放腿部目標位置
        targets = np.zeros((len(IK_SITES), 3))
        for k in range(len(IK_SITES)):
            rel = h1_site_pos[k+1] - h1_pelvis_pos
            targets[k] = h1_pelvis_pos + rel * leg_scales[k]

        # Damped Jacobian IK
        damped_ik(kbot_model, kbot_data, leg_q_adr, leg_q_limits, targets, kbot_site_ids)

        mujoco.mj_forward(kbot_model, kbot_data)
        kbot_qpos_list.append(kbot_data.qpos[:kbot_model.nq].copy())
        kbot_qvel_list.append(kbot_data.qvel[:kbot_model.nv].copy())

        if (i_frame + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i_frame + 1) / elapsed
            eta  = (n_out - i_frame - 1) / rate
            # 誤差報告
            err = np.mean([np.linalg.norm(kbot_data.site_xpos[sid] - targets[k])
                           for k, sid in enumerate(kbot_site_ids)])
            print(f"  {i_frame+1}/{n_out}  {rate:.1f} f/s  ETA {eta:.0f}s  avg_err={err:.4f}m")

    elapsed = time.time() - t0
    print(f"Done: {n_out} frames in {elapsed:.1f}s ({n_out/elapsed:.1f} fps)")

    # 關節名稱
    joint_names = []
    for jid in range(kbot_model.njnt):
        joint_names.append(mujoco.mj_id2name(kbot_model, mujoco.mjtObj.mjOBJ_JOINT, jid))

    kbot_qpos_arr = np.array(kbot_qpos_list, dtype=np.float32)
    kbot_qvel_arr = np.array(kbot_qvel_list, dtype=np.float32)

    np.savez(out_path,
             qpos=kbot_qpos_arr, qvel=kbot_qvel_arr,
             joint_names=np.array(joint_names),
             frequency=np.int64(fps // SKIP_EVERY),
             split_points=np.array([0, n_out], dtype=np.int32),
             xpos=np.zeros(0, dtype=np.float32),
             xquat=np.zeros(0, dtype=np.float32),
             cvel=np.zeros(0, dtype=np.float32),
             subtree_com=np.zeros(0, dtype=np.float32),
             site_xpos=np.zeros(0, dtype=np.float32),
             site_xmat=np.zeros(0, dtype=np.float32),
             metadata=np.array({}),
             njnt=np.int32(kbot_model.njnt),
             jnt_type=np.array([kbot_model.jnt_type[i] for i in range(kbot_model.njnt)]),
             nbody=np.array({}), body_rootid=np.zeros(0, dtype=np.float32),
             body_weldid=np.zeros(0, dtype=np.float32),
             body_mocapid=np.zeros(0, dtype=np.float32),
             body_pos=np.zeros(0, dtype=np.float32), body_quat=np.zeros(0, dtype=np.float32),
             body_ipos=np.zeros(0, dtype=np.float32), body_iquat=np.zeros(0, dtype=np.float32),
             nsite=np.array({}), site_bodyid=np.zeros(0, dtype=np.float32),
             site_pos=np.zeros(0, dtype=np.float32), site_quat=np.zeros(0, dtype=np.float32),
             )
    print(f"Saved: {out_path}  shape={kbot_qpos_arr.shape}")

    # ── 快速驗證 ──────────────────────────────────────────────────────────
    print("\n=== Validation (frame 500) ===")
    f = 500
    kbot_data.qpos[:] = kbot_qpos_arr[f]
    mujoco.mj_forward(kbot_model, kbot_data)
    for k, sid in enumerate(kbot_site_ids):
        achieved = kbot_data.site_xpos[sid]
        row2 = traj_qpos[f]
        h1_data.qpos[0:7] = row2[traj_col["root"]:traj_col["root"]+7]
        for jn in traj_jnames[1:]:
            jid2 = mujoco.mj_name2id(h1_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid2 >= 0: h1_data.qpos[h1_model.jnt_qposadr[jid2]] = row2[traj_col[jn]]
        mujoco.mj_forward(h1_model, h1_data)
        h1_p = h1_data.site_xpos[h1_site_ids[0]]
        rel  = h1_data.site_xpos[h1_site_ids[k+1]] - h1_p
        target = h1_p + rel * leg_scales[k]
        err = np.linalg.norm(achieved - target)
        print(f"  {IK_SITES[k]:25s} err={err:.4f}m")

    print("\nJoint values:")
    for jid in range(kbot_model.njnt):
        name = mujoco.mj_id2name(kbot_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not name or name == "root": continue
        adr = kbot_model.jnt_qposadr[jid]
        lo, hi = kbot_model.jnt_range[jid]
        val = kbot_qpos_arr[f, adr]
        flag = " *** OUT ***" if val < lo or val > hi else ""
        print(f"  {name:30s}: {val:+7.4f}  [{lo:.3f}, {hi:.3f}]{flag}")


if __name__ == "__main__":
    main()
