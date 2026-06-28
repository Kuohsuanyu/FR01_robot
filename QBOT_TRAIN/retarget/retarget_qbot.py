"""
QBOT FK→IK Retargeting — Damped Jacobian IK（腿 + 手臂）

由 retarget_kbot.py 改寫，差異：
  1. 目標模型 = qbot_retarget.xml（含新加的 *_mimic site）
  2. freejoint 名為 'floating_base'（KBOT 是 'root'）
  3. 新增「手臂」重定向：H1 肩/肘/腕 → QBOT 手臂 8 DOF
     - 腿：目標相對 H1 骨盆（骨盆已直接對齊 root）
     - 手臂：目標相對 QBOT「自己的肩膀」世界座標（肩膀沒對齊 H1），
            上臂/前臂分段縮放，貼合 QBOT 較短的前臂

方法：
  1. H1 Forward Kinematics → 取得骨盆/髖/膝/踝 + 肩/肘/腕世界座標
  2. 骨盆直接寫入 freejoint；腿、手臂分別做 Damped LS Jacobian IK
  3. 各關節含限位 clamp

執行：
    conda activate loco_new
    python retarget_qbot.py dance1_subject1

輸出：
    ~/.loco-mujoco-caches/lafan1_converted/QBotEnv_retargeted/{motion}.npz
"""

import sys, os, time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

H1_XML   = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1/h1.xml"
_HERE    = os.path.dirname(os.path.abspath(__file__))
QBOT_XML = os.path.join(_HERE, "..", "qbot_retarget.xml")   # 含 mimic site 的模型（在 QBOT_TRAIN/ 下）
TRAJ_DIR = "/home/andykuo/.loco-mujoco-caches/lafan1_converted"
OUT_DIR  = f"{TRAJ_DIR}/QBotEnv_retargeted"

MOTION    = sys.argv[1] if len(sys.argv) > 1 else "dance1_subject1"
# 輸入軌跡：沿用 KBotEnv 轉好的 H1-LAFAN1 軌跡（含手臂關節）
TRAJ_NPZ  = f"{TRAJ_DIR}/KBotEnv/{MOTION}.npz"

# ── site 對應（H1 與 QBOT 同名）─────────────────────────────────────────────
# 統一公式：肢段目標相對「QBOT 自己的肢根（髖 / 肩）」分段縮放，
# IK 只追中段+末端（髖、肩是鏈根、由骨盆/軀幹固定，無法平移，不列入 IK）。
PELVIS_SITE = "pelvis_mimic"
# 每側 (根, 中, 末)
LEG_SIDES   = [("left_hip_mimic",  "left_knee_mimic",  "left_foot_mimic"),
               ("right_hip_mimic", "right_knee_mimic", "right_foot_mimic")]
LEG_IK_SITES = ["left_knee_mimic", "left_foot_mimic",
                "right_knee_mimic", "right_foot_mimic"]
ARM_SIDES   = [("left_shoulder_mimic",  "left_elbow_mimic",  "left_hand_mimic"),
               ("right_shoulder_mimic", "right_elbow_mimic", "right_hand_mimic")]
ARM_IK_SITES = ["left_elbow_mimic", "left_hand_mimic",
                "right_elbow_mimic", "right_hand_mimic"]

ROOT_JOINT = "floating_base"
LEG_JOINTS = ["dof_right_hip_pitch_04", "dof_right_hip_roll_03", "dof_right_hip_yaw_03",
              "dof_right_knee_04", "dof_right_ankle_02",
              "dof_left_hip_pitch_04", "dof_left_hip_roll_03", "dof_left_hip_yaw_03",
              "dof_left_knee_04", "dof_left_ankle_02"]
ARM_JOINTS = ["ub_right_shoulder", "ub_right_lateral_raise", "ub_right_arm_twist", "ub_right_elbow",
              "ub_left_shoulder",  "ub_left_lateral_raise",  "ub_left_arm_twist",  "ub_left_elbow"]

# IK 超參數
IK_MAX_ITER = 200
IK_DAMPING  = 0.05
IK_ALPHA    = 0.5
IK_TOL      = 5e-3
SKIP_EVERY  = 1

# 姿態正則化權重（對應 LEG_JOINTS / ARM_JOINTS 順序）：
# 旋轉自由度（hip_yaw 第 2,7 個；arm_twist 第 2,6 個）給強正則，避免冗餘亂轉把肢體扭歪；
# 其餘關節給很小值維持平滑、不影響位置追蹤。
LEG_REG = np.array([0.02, 0.05, 0.40, 0.02, 0.02,
                    0.02, 0.05, 0.40, 0.02, 0.02])
ARM_REG = np.array([0.03, 0.03, 0.20, 0.03,
                    0.03, 0.03, 0.20, 0.03])


def get_sid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def site_pos(model, data, name):
    return data.site_xpos[get_sid(model, name)].copy()


def compute_leg_scale(h1_model, qbot_model):
    """以 body 量測大腿 / 小腿縮放比（沿用 KBOT 版邏輯，body 名同 QBOT 腿鏈）。"""
    hd = mujoco.MjData(h1_model);   mujoco.mj_forward(h1_model, hd)
    kd = mujoco.MjData(qbot_model); mujoco.mj_forward(qbot_model, kd)

    def bp(model, data, bname):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        return data.xpos[bid].copy()

    # 用 mimic site 量（兩模型同名），比依賴 body 名更穩
    def seg(model, data, a, b):
        return np.linalg.norm(site_pos(model, data, a) - site_pos(model, data, b))

    h1_thigh = seg(h1_model, hd, "left_hip_mimic",  "left_knee_mimic")
    h1_shin  = seg(h1_model, hd, "left_knee_mimic", "left_foot_mimic")
    kb_thigh = seg(qbot_model, kd, "left_hip_mimic",  "left_knee_mimic")
    kb_shin  = seg(qbot_model, kd, "left_knee_mimic", "left_foot_mimic")
    print(f"H1   thigh={h1_thigh:.4f}  shin={h1_shin:.4f}")
    print(f"QBOT thigh={kb_thigh:.4f}  shin={kb_shin:.4f}")
    print(f"Leg scale thigh={kb_thigh/h1_thigh:.4f}  shin={kb_shin/h1_shin:.4f}")
    return kb_thigh / h1_thigh, kb_shin / h1_shin


def compute_arm_scale(h1_model, qbot_model):
    """上臂(肩→肘) / 前臂(肘→腕) 縮放比，取左右平均。"""
    hd = mujoco.MjData(h1_model);   mujoco.mj_forward(h1_model, hd)
    kd = mujoco.MjData(qbot_model); mujoco.mj_forward(qbot_model, kd)

    def arm_lens(model, data):
        up, fo = [], []
        for sh, el, ha in ARM_SIDES:
            up.append(np.linalg.norm(site_pos(model, data, el) - site_pos(model, data, sh)))
            fo.append(np.linalg.norm(site_pos(model, data, ha) - site_pos(model, data, el)))
        return np.mean(up), np.mean(fo)

    h1_up, h1_fo = arm_lens(h1_model, hd)
    kb_up, kb_fo = arm_lens(qbot_model, kd)
    print(f"H1   upper={h1_up:.4f}  fore={h1_fo:.4f}")
    print(f"QBOT upper={kb_up:.4f}  fore={kb_fo:.4f}")
    print(f"Arm scale upper={kb_up/h1_up:.4f}  fore={kb_fo/h1_fo:.4f}")
    return kb_up / h1_up, kb_fo / h1_fo


def damped_ik(model, data, q_adr, q_limits, target_positions, site_ids,
              q_rest=None, reg_w=None,
              max_iter=IK_MAX_ITER, damping=IK_DAMPING, alpha=IK_ALPHA, tol=IK_TOL):
    """Damped least-squares Jacobian IK（位置）＋姿態正則化（把指定關節拉回 q_rest）。

    位置任務只約束 site 位置 → 冗餘自由度（尤其 hip_yaw 旋轉）會亂轉把腿扭歪。
    reg_w[i] 對第 i 個關節加一條軟約束 q→q_rest，避免該關節衝到限位 / 扭曲。
    """
    n_joints = len(q_adr)
    n_sites  = len(site_ids)
    n_pos    = n_sites * 3
    if q_rest is None: q_rest = np.zeros(n_joints)
    if reg_w  is None: reg_w  = np.zeros(n_joints)
    Wp = np.diag(reg_w)

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)

        e_pos = np.zeros(n_pos)
        for k, sid in enumerate(site_ids):
            e_pos[k*3:(k+1)*3] = target_positions[k] - data.site_xpos[sid]
        if np.linalg.norm(e_pos) < tol and not reg_w.any():
            break

        J_full = np.zeros((n_pos, model.nv))
        jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
        for k, sid in enumerate(site_ids):
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            J_full[k*3:(k+1)*3, :] = jacp

        J = np.zeros((n_pos, n_joints))
        for i, adr in enumerate(q_adr):
            jid = np.where(model.jnt_qposadr == adr)[0]
            if len(jid) > 0:
                J[:, i] = J_full[:, model.jnt_dofadr[jid[0]]]

        # 堆疊：位置任務 + 姿態正則化軟約束
        qcur = np.array([data.qpos[a] for a in q_adr])
        A = np.vstack([J, Wp])
        b = np.concatenate([e_pos, reg_w * (q_rest - qcur)])
        AtA = A.T @ A + (damping**2) * np.eye(n_joints)
        delta_q = alpha * np.linalg.solve(AtA, A.T @ b)

        for i, adr in enumerate(q_adr):
            data.qpos[adr] = np.clip(data.qpos[adr] + delta_q[i], q_limits[i, 0], q_limits[i, 1])

    return data.qpos.copy()


def joint_addr_limits(model, names):
    adr, lim = [], []
    for n in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid < 0:
            raise RuntimeError(f"joint not found in model: {n}")
        adr.append(model.jnt_qposadr[jid])
        lim.append(model.jnt_range[jid])
    return adr, np.array(lim)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{MOTION}.npz")
    print(f"Motion: {MOTION}\nInput:  {TRAJ_NPZ}\nOutput: {out_path}")

    npz = np.load(TRAJ_NPZ, allow_pickle=True)
    traj_qpos   = npz["qpos"]
    traj_jnames = list(npz["joint_names"])
    fps = int(npz["frequency"]); T = len(traj_qpos)
    print(f"Trajectory: {T} steps @ {fps} Hz, joints={len(traj_jnames)}")

    traj_col = {}; c = 0
    for n in traj_jnames:
        traj_col[n] = c; c += 7 if n == "root" else 1

    print("Loading models...")
    h1_model = mujoco.MjModel.from_xml_path(H1_XML)
    h1_data  = mujoco.MjData(h1_model)
    qbot_model = mujoco.MjModel.from_xml_path(QBOT_XML)
    qbot_data  = mujoco.MjData(qbot_model)
    mujoco.mj_forward(qbot_model, qbot_data)

    thigh_s, shin_s = compute_leg_scale(h1_model, qbot_model)
    upper_s, fore_s = compute_arm_scale(h1_model, qbot_model)

    # site ids
    h1_pelvis_sid = get_sid(h1_model, PELVIS_SITE)
    qbot_leg_sids = [get_sid(qbot_model, s) for s in LEG_IK_SITES]
    qbot_arm_sids = [get_sid(qbot_model, s) for s in ARM_IK_SITES]

    leg_q_adr, leg_q_lim = joint_addr_limits(qbot_model, LEG_JOINTS)
    arm_q_adr, arm_q_lim = joint_addr_limits(qbot_model, ARM_JOINTS)
    print(f"Leg joints: {len(leg_q_adr)} DOF   Arm joints: {len(arm_q_adr)} DOF")
    print(f"IK: damping={IK_DAMPING} alpha={IK_ALPHA} max_iter={IK_MAX_ITER} tol={IK_TOL}m")

    root_adr = qbot_model.jnt_qposadr[mujoco.mj_name2id(qbot_model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT)]

    # 肘是單純鉸鏈：用「3D 彎曲角度」映射比純位置 IK 冗餘求解可靠、左右對稱、與動作無關。
    # 預計算 QBOT 每側肘 (joint值 → 上臂/前臂 3D 夾角) 的單調對應，之後用 H1 肘角反查。
    elbow_jidx = {"right": 3, "left": 7}   # 對應 ARM_JOINTS 順序
    def build_elbow_map(side):
        sid = lambda n: get_sid(qbot_model, n)
        tmp = mujoco.MjData(qbot_model)
        ea  = arm_q_adr[elbow_jidx[side]]
        qs  = np.linspace(0.0, 2.35619, 40)
        angs = []
        for qv in qs:
            tmp.qpos[:] = 0.0; tmp.qpos[ea] = qv
            mujoco.mj_forward(qbot_model, tmp)
            u = tmp.site_xpos[sid(f"{side}_elbow_mimic")] - tmp.site_xpos[sid(f"{side}_shoulder_mimic")]
            f = tmp.site_xpos[sid(f"{side}_hand_mimic")]  - tmp.site_xpos[sid(f"{side}_elbow_mimic")]
            angs.append(np.degrees(np.arccos(np.clip(u@f/(np.linalg.norm(u)*np.linalg.norm(f)+1e-9), -1, 1))))
        return np.array(angs), qs    # angs 單調遞增 → np.interp(theta, angs, qs)
    elbow_map = {s: build_elbow_map(s) for s in ("left", "right")}

    def h1_elbow_angle(h1_sh, h1_el, h1_ha):
        u = h1_el - h1_sh; f = h1_ha - h1_el
        return np.degrees(np.arccos(np.clip(u@f/(np.linalg.norm(u)*np.linalg.norm(f)+1e-9), -1, 1)))

    # 手臂正則（搭配「q_rest = 上一幀」做時間平滑）：
    #   肘 1.5（鎖到 H1 映射角，q_rest 另外覆寫）；arm_twist 0.15（抑制奇異點 ±π 翻轉跳動，
    #   但仍讓它跟著前臂朝向慢慢轉）；肩/lateral 0.05 輕度平滑。
    ARM_REG_E = np.array([0.15, 0.15, 0.15, 1.5,
                          0.15, 0.15, 0.15, 1.5])

    indices = list(range(0, T, SKIP_EVERY)); n_out = len(indices)
    qpos_list, qvel_list = [], []
    print(f"\nRetargeting {n_out} frames...")
    t0 = time.time()

    for i_frame, t_idx in enumerate(indices):
        row = traj_qpos[t_idx]

        # ── H1 FK ──
        rc = traj_col["root"]
        h1_data.qpos[0:7] = row[rc:rc+7]
        for jn in traj_jnames[1:]:
            jid = mujoco.mj_name2id(h1_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                h1_data.qpos[h1_model.jnt_qposadr[jid]] = row[traj_col[jn]]
        mujoco.mj_forward(h1_model, h1_data)

        h1_pelvis = h1_data.site_xpos[h1_pelvis_sid].copy()
        h1_pelmat = h1_data.site_xmat[h1_pelvis_sid].reshape(3, 3)

        # ── 骨盆 → freejoint ──
        qbot_data.qpos[root_adr:root_adr+3] = h1_pelvis
        q = Rotation.from_matrix(h1_pelmat).as_quat()    # [x,y,z,w]
        qbot_data.qpos[root_adr+3:root_adr+7] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(qbot_model, qbot_data)

        # ── 腿 IK：目標相對 QBOT 自己的髖，分段縮放（大腿/小腿）──
        leg_tgt = []
        for (hip, knee, foot) in LEG_SIDES:
            hip_q  = site_pos(qbot_model, qbot_data, hip)   # QBOT 髖世界座標（root+forward 後固定）
            h1_hip = site_pos(h1_model, h1_data, hip)
            h1_kn  = site_pos(h1_model, h1_data, knee)
            h1_ft  = site_pos(h1_model, h1_data, foot)
            kn_t = hip_q + (h1_kn - h1_hip) * thigh_s
            ft_t = kn_t  + (h1_ft - h1_kn) * shin_s
            leg_tgt.append(kn_t); leg_tgt.append(ft_t)
        leg_tgt = np.array(leg_tgt)
        damped_ik(qbot_model, qbot_data, leg_q_adr, leg_q_lim, leg_tgt, qbot_leg_sids, reg_w=LEG_REG)

        # ── 手臂 IK：目標相對 QBOT 自己的肩膀，分段縮放 ──
        mujoco.mj_forward(qbot_model, qbot_data)
        arm_tgt = []
        # q_rest 預設拉回「上一幀」的手臂姿態 → 時間平滑，抑制 arm_twist 在奇異點 ±π 翻轉
        arm_rest = np.array([qbot_data.qpos[a] for a in arm_q_adr])
        for (sh, el, ha) in ARM_SIDES:
            sh_q  = site_pos(qbot_model, qbot_data, sh)         # QBOT 肩世界座標
            h1_sh = site_pos(h1_model, h1_data, sh)
            h1_el = site_pos(h1_model, h1_data, el)
            h1_ha = site_pos(h1_model, h1_data, ha)
            el_t = sh_q + (h1_el - h1_sh) * upper_s
            ha_t = el_t + (h1_ha - h1_el) * fore_s
            arm_tgt.append(el_t); arm_tgt.append(ha_t)
            # 肘：由 H1 肘 3D 角度反查 QBOT 肘關節值，強正則鎖定（肩 3-DOF 仍由位置 IK 解）
            side = sh.split("_")[0]                              # "left" / "right"
            angs, qs = elbow_map[side]
            arm_rest[elbow_jidx[side]] = np.interp(h1_elbow_angle(h1_sh, h1_el, h1_ha), angs, qs)
        damped_ik(qbot_model, qbot_data, arm_q_adr, arm_q_lim, np.array(arm_tgt), qbot_arm_sids,
                  q_rest=arm_rest, reg_w=ARM_REG_E)

        mujoco.mj_forward(qbot_model, qbot_data)
        qpos_list.append(qbot_data.qpos[:qbot_model.nq].copy())
        qvel_list.append(qbot_data.qvel[:qbot_model.nv].copy())

        if (i_frame + 1) % 200 == 0:
            el = time.time() - t0; rate = (i_frame+1)/el
            leg_err = np.mean([np.linalg.norm(qbot_data.site_xpos[s]-leg_tgt[k]) for k,s in enumerate(qbot_leg_sids)])
            arm_err = np.mean([np.linalg.norm(qbot_data.site_xpos[s]-arm_tgt[k]) for k,s in enumerate(qbot_arm_sids)])
            print(f"  {i_frame+1}/{n_out}  {rate:.1f} f/s  ETA {(n_out-i_frame-1)/rate:.0f}s  leg_err={leg_err:.4f}m arm_err={arm_err:.4f}m")

    elapsed = time.time() - t0
    print(f"Done: {n_out} frames in {elapsed:.1f}s ({n_out/elapsed:.1f} fps)")

    joint_names = [mujoco.mj_id2name(qbot_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(qbot_model.njnt)]
    qpos_arr = np.array(qpos_list, dtype=np.float32)
    qvel_arr = np.array(qvel_list, dtype=np.float32)

    np.savez(out_path,
             qpos=qpos_arr, qvel=qvel_arr,
             joint_names=np.array(joint_names),
             frequency=np.int64(fps // SKIP_EVERY),
             split_points=np.array([0, n_out], dtype=np.int32),
             xpos=np.zeros(0, dtype=np.float32), xquat=np.zeros(0, dtype=np.float32),
             cvel=np.zeros(0, dtype=np.float32), subtree_com=np.zeros(0, dtype=np.float32),
             site_xpos=np.zeros(0, dtype=np.float32), site_xmat=np.zeros(0, dtype=np.float32),
             metadata=np.array({}),
             njnt=np.int32(qbot_model.njnt),
             jnt_type=np.array([qbot_model.jnt_type[i] for i in range(qbot_model.njnt)]),
             nbody=np.array({}), body_rootid=np.zeros(0, dtype=np.float32),
             body_weldid=np.zeros(0, dtype=np.float32), body_mocapid=np.zeros(0, dtype=np.float32),
             body_pos=np.zeros(0, dtype=np.float32), body_quat=np.zeros(0, dtype=np.float32),
             body_ipos=np.zeros(0, dtype=np.float32), body_iquat=np.zeros(0, dtype=np.float32),
             nsite=np.array({}), site_bodyid=np.zeros(0, dtype=np.float32),
             site_pos=np.zeros(0, dtype=np.float32), site_quat=np.zeros(0, dtype=np.float32))
    print(f"Saved: {out_path}  shape={qpos_arr.shape}")

    # ── 驗證 ──
    f = min(500, n_out - 1)
    print(f"\n=== Validation (frame {f}) ===")
    qbot_data.qpos[:] = qpos_arr[f]; mujoco.mj_forward(qbot_model, qbot_data)
    print("Joint values:")
    for j in range(qbot_model.njnt):
        n = mujoco.mj_id2name(qbot_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not n or n == ROOT_JOINT: continue
        adr = qbot_model.jnt_qposadr[j]; lo, hi = qbot_model.jnt_range[j]
        v = qpos_arr[f, adr]; flag = " *** OUT ***" if v < lo or v > hi else ""
        print(f"  {n:26s}: {v:+7.4f}  [{lo:.3f}, {hi:.3f}]{flag}")


if __name__ == "__main__":
    main()
