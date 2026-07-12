"""
三方並排比較：真人骨架（SMPL）| H1 機器人 | FR01 機器人
同步播放同一個 AMASS 動作

執行：
    conda activate loco_new
    JAX_PLATFORM_NAME=cpu python compare_three_way.py [dataset/subject/motion]

範例：
    JAX_PLATFORM_NAME=cpu python compare_three_way.py CMU/01/01_01_poses
    JAX_PLATFORM_NAME=cpu python compare_three_way.py DanceDB/DanceDB/.../Capoeira_poses
"""

import sys, os, time, re
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import mujoco, mujoco.viewer
import xml.etree.ElementTree as ET

sys.path.insert(0, '/home/andykuo/ksim-gym/MIMIC/loco_mujoco')

# ── 路徑設定 ───────────────────────────────────────────────────────────
H1_DIR    = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1"
H1_XML    = os.path.join(H1_DIR, "h1.xml")
KBOT_DIR  = "/home/andykuo/ksim-gym/MIMIC/kbot"
KBOT_XML  = os.path.join(KBOT_DIR, "KBOT_loco.xml")
AMASS_DIR = "/media/andykuo/T7/IMU_dataset/AMASS"
SMPL_MODEL= "/home/andykuo/.loco-mujoco-caches/smpl/SMPLH_NEUTRAL.pkl"
CACHE_DIR = "/home/andykuo/.loco-mujoco-caches/amass_converted/UnitreeH1"

MOTION    = sys.argv[1] if len(sys.argv) > 1 else "CMU/01/01_01_poses"
X_H1      = 0.0    # H1 在 x=0
X_SMPL    = -2.5   # 真人骨架在左邊
X_FR01    =  2.5   # FR01 在右邊

# SMPL 22 個骨骼的連接關係（父→子）
SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
SMPL_NAMES   = ["Pelvis","L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2",
                "L_Ankle","R_Ankle","Spine3","L_Foot","R_Foot","Neck",
                "L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
                "L_Elbow","R_Elbow","L_Wrist","R_Wrist"]

# KBOT → H1 關節映射
KBOT_TO_H1 = {
    "kb_hip_flexion_r":"hip_flexion_r","kb_hip_adduction_r":"hip_adduction_r",
    "kb_hip_rotation_r":"hip_rotation_r","kb_knee_angle_r":"knee_angle_r",
    "kb_ankle_angle_r":"ankle_angle_r","kb_hip_flexion_l":"hip_flexion_l",
    "kb_hip_adduction_l":"hip_adduction_l","kb_hip_rotation_l":"hip_rotation_l",
    "kb_knee_angle_l":"knee_angle_l","kb_ankle_angle_l":"ankle_angle_l",
}
NEGATE = {"kb_knee_angle_r", "kb_hip_flexion_l", "kb_ankle_angle_l"}

KBOT_CLASS_CTRLRANGE = {
    "robstride_04":"-120.0 120.0","robstride_03":"-60.0 60.0",
    "robstride_02":"-17.0 17.0","robstride_00":"-14.0 14.0"
}


def load_amass_and_retarget(motion_path):
    """載入 AMASS 並 retarget 到 H1（有快取則跳過）"""
    npz_path   = os.path.join(AMASS_DIR, motion_path + ".npz")
    cache_path = os.path.join(CACHE_DIR, motion_path + ".npz")

    if os.path.exists(cache_path):
        print(f"快取命中: {cache_path}")
        return np.load(cache_path, allow_pickle=True)

    print(f"執行 AMASS retargeting: {motion_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf
    env = ImitationFactory.make("UnitreeH1",
        amass_dataset_conf=AMASSDatasetConf([motion_path]))
    # 找到快取
    return np.load(cache_path, allow_pickle=True)


def get_smpl_joints(motion_path, traj_length, fps):
    """從 AMASS 原始檔取得 SMPL 關節世界座標"""
    npz_path = os.path.join(AMASS_DIR, motion_path + ".npz")
    raw = np.load(npz_path, allow_pickle=True)
    poses   = raw["poses"]    # (T_raw, 156)
    betas   = raw["betas"][:16] if len(raw["betas"]) >= 16 else raw["betas"]
    trans   = raw["trans"]    # (T_raw, 3)
    raw_fps = float(raw["mocap_framerate"])

    # 從 SMPL 取關節位置（只用 body 部分：前 22 joints × 3 = 66 params）
    import torch
    import smplx
    model = smplx.create(SMPL_MODEL, model_type="smplh",
                         gender="neutral", ext="pkl",
                         num_betas=16, use_pca=False,
                         batch_size=1).eval()

    # 降採樣到 fps
    step = max(1, int(round(raw_fps / fps)))
    T_sub = min(traj_length, (len(poses) - 1) // step + 1)

    joints_list = []
    with torch.no_grad():
        for i in range(T_sub):
            idx = min(i * step, len(poses) - 1)
            pose_b = torch.tensor(poses[idx:idx+1, :66], dtype=torch.float32)
            rhand  = torch.zeros(1, 45, dtype=torch.float32)
            lhand  = torch.zeros(1, 45, dtype=torch.float32)
            trans_b= torch.tensor(trans[idx:idx+1], dtype=torch.float32)
            beta_b = torch.tensor(betas[:16][None], dtype=torch.float32)
            out = model(body_pose=pose_b[:, 3:66],
                        global_orient=pose_b[:, :3],
                        betas=beta_b,
                        transl=trans_b,
                        right_hand_pose=rhand,
                        left_hand_pose=lhand)
            joints_list.append(out.joints[0, :22].numpy())

    return np.array(joints_list)  # (T, 22, 3)


def build_smpl_xml(n_joints=22):
    """建立 SMPL 骨架的 MuJoCo XML（球體+連線）"""
    bones_xml = []
    # 骨骼節點（彩色球體）
    colors = {
        "Pelvis":"0.9 0.5 0.1 1","L_Hip":"0.1 0.7 0.3 1","R_Hip":"0.1 0.3 0.9 1",
        "Spine1":"0.8 0.8 0.1 1","L_Knee":"0.1 0.7 0.3 1","R_Knee":"0.1 0.3 0.9 1",
    }
    default_color = "0.85 0.85 0.85 1"
    for i, name in enumerate(SMPL_NAMES):
        color = colors.get(name, default_color)
        bones_xml.append(
            f'<body name="smpl_{name}" pos="0 0 0" mocap="true">'
            f'  <geom type="sphere" size="0.04" rgba="{color}" contype="0" conaffinity="0"/>'
            f'</body>'
        )
    # 骨骼連線（膠囊體）
    for child_idx, parent_idx in enumerate(SMPL_PARENTS):
        if parent_idx < 0: continue
        cname = SMPL_NAMES[child_idx]
        pname = SMPL_NAMES[parent_idx]
        bones_xml.append(
            f'<body name="smpl_bone_{pname}_{cname}" pos="0 0 0" mocap="true">'
            f'  <geom name="sbone_{child_idx}" type="capsule" size="0.015" '
            f'       fromto="0 0 0 0 0 0.001" rgba="0.7 0.7 0.7 0.8" '
            f'       contype="0" conaffinity="0"/>'
            f'</body>'
        )
    return "\n".join(bones_xml)


def build_combined_xml():
    """合併 H1 + KBOT + SMPL 骨架"""
    kbot_tree = ET.parse(KBOT_XML)
    kbot_rr   = kbot_tree.getroot()

    kbot_bodies = []
    for child in kbot_rr.find("worldbody"):
        if (child.tag in ("geom","light") and child.get("name","").startswith("floor")) or child.tag=="light":
            continue
        s = ET.tostring(child, encoding="unicode")
        s = re.sub(r'file="(meshes/[^"]+)"',
                   lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"', s)
        s = re.sub(r'\s*childclass="[^"]*"', '', s)
        s = re.sub(r'\s*class="(?!visual|collision|h1)[^"]*"', '', s)
        s = re.sub(r'name="(?!kb_)([^"]+)"', r'name="kb_\1"', s)
        s = re.sub(r'joint="(?!kb_)([^"]+)"', r'joint="kb_\1"', s)
        s = re.sub(r'mesh="(?!kb_)([^"]+)"',  r'mesh="kb_\1"',  s)
        s = re.sub(r'material="(?!kb_)([^"]+)"', r'material="kb_\1"', s)
        kbot_bodies.append(s)

    kbot_assets = []
    for elem in kbot_rr.find("asset"):
        s = ET.tostring(elem, encoding="unicode")
        if elem.tag == "material": s = re.sub(r'name="([^"]+)"', r'name="kb_\1"', s)
        elif elem.tag == "mesh":
            s = re.sub(r'name="([^"]+)"', r'name="kb_\1"', s)
            s = re.sub(r'file="(meshes/[^"]+)"',
                       lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"', s)
        kbot_assets.append(s)

    kbot_acts = []
    for elem in kbot_rr.find("actuator"):
        s = ET.tostring(elem, encoding="unicode")
        s = re.sub(r'name="([^"]+)"',  r'name="kb_\1"', s)
        s = re.sub(r'joint="([^"]+)"', r'joint="kb_\1"', s)
        cls = re.search(r'class="([^"]+)"', s)
        if cls and cls.group(1) in KBOT_CLASS_CTRLRANGE and 'ctrlrange' not in s:
            s = s.replace(f'class="{cls.group(1)}"',
                          f'ctrlrange="{KBOT_CLASS_CTRLRANGE[cls.group(1)]}"')
        else:
            s = re.sub(r'\s*class="[^"]*"', '', s)
        kbot_acts.append(s)

    smpl_xml = build_smpl_xml()

    combined = f"""<mujoco model="three_way">
  <include file="h1.xml"/>
  <asset>{''.join(kbot_assets)}</asset>
  <worldbody>
    {''.join(kbot_bodies)}
    {smpl_xml}
  </worldbody>
  <actuator>{''.join(kbot_acts)}</actuator>
</mujoco>"""

    tmp = os.path.join(H1_DIR, "_three_way.xml")
    with open(tmp, "w") as f: f.write(combined)
    return tmp


def main():
    # ── 載入 H1 軌跡（AMASS retargeted）────────────────────────────────
    print(f"動作: {MOTION}")
    h1_npz   = load_amass_and_retarget(MOTION)
    traj_qpos= h1_npz["qpos"]
    jnames   = list(h1_npz["joint_names"])
    fps      = int(float(h1_npz["frequency"]))
    T        = len(traj_qpos)
    dt       = 1.0 / fps
    print(f"H1 軌跡: {T} frames @ {fps}Hz")

    col = {}; c = 0
    for n in jnames: col[n] = c; c += 7 if n == "root" else 1

    # ── 取得 SMPL 關節座標 ─────────────────────────────────────────────
    print("載入 SMPL 骨架座標...")
    smpl_joints = get_smpl_joints(MOTION, T, fps)  # (T, 22, 3)
    T = min(T, len(smpl_joints))
    print(f"SMPL 骨架: {T} frames, {smpl_joints.shape[1]} joints")

    # ── 建立並載入合併模型 ─────────────────────────────────────────────
    print("建立合併模型...")
    tmp_xml = build_combined_xml()
    model   = mujoco.MjModel.from_xml_path(tmp_xml)
    data_mj = mujoco.MjData(model)
    os.unlink(tmp_xml)
    print(f"Combined: nq={model.nq}, nmocap={model.nmocap}")

    # joint 映射
    jmap = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name: jmap[name] = model.jnt_qposadr[jid]

    # mocap body ID 映射（SMPL 骨骼節點）
    def mid(bname):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        return model.body_mocapid[bid] if bid >= 0 else -1

    smpl_mids = [mid(f"smpl_{n}") for n in SMPL_NAMES]
    bone_mids = [mid(f"smpl_bone_{SMPL_NAMES[p]}_{SMPL_NAMES[c]}")
                 for c, p in enumerate(SMPL_PARENTS) if p >= 0]
    bone_pairs = [(c, p) for c, p in enumerate(SMPL_PARENTS) if p >= 0]

    # 計算 SMPL 骨架的偏移（讓它站在左邊 x=X_SMPL）
    smpl_pelvis0 = smpl_joints[0, 0]   # 第一幀骨盆位置
    smpl_x_offset = X_SMPL - smpl_pelvis0[0]

    def apply_frame(step):
        row = traj_qpos[step]

        # H1（x=0）
        if "root" in jmap:
            adr = jmap["root"]
            data_mj.qpos[adr:adr+7] = row[col["root"]:col["root"]+7]
        for jn, adr in jmap.items():
            if jn.startswith("kb_") or jn == "root": continue
            if jn in col: data_mj.qpos[adr] = row[col[jn]]

        # FR01（x=X_FR01）
        if "kb_root" in jmap:
            adr = jmap["kb_root"]
            data_mj.qpos[adr]   = 0.0
            data_mj.qpos[adr+1:adr+7] = row[col["root"]+1:col["root"]+7]
        for kb, h1 in KBOT_TO_H1.items():
            if kb in jmap and h1 in col:
                val = row[col[h1]]
                if kb in NEGATE: val = -val
                data_mj.qpos[jmap[kb]] = val

        # SMPL 骨架（x=X_SMPL，mocap bodies）
        joints = smpl_joints[step]
        for i, (name, mocap_id) in enumerate(zip(SMPL_NAMES, smpl_mids)):
            if mocap_id < 0: continue
            pos = joints[i].copy()
            pos[0] += smpl_x_offset
            data_mj.mocap_pos[mocap_id] = pos
            data_mj.mocap_quat[mocap_id] = [1, 0, 0, 0]

        # 骨骼連線（capsule from parent to child）
        for k, (child_idx, parent_idx) in enumerate(bone_pairs):
            if k >= len(bone_mids) or bone_mids[k] < 0: continue
            mid_id = bone_mids[k]
            p_pos = joints[parent_idx].copy(); p_pos[0] += smpl_x_offset
            c_pos = joints[child_idx].copy();  c_pos[0] += smpl_x_offset
            center = (p_pos + c_pos) / 2
            data_mj.mocap_pos[mid_id] = center
            data_mj.mocap_quat[mid_id] = [1, 0, 0, 0]

        # FR01 跟 H1 同向移動（從各自的初始位置出發）
        if "kb_root" in jmap:
            adr = jmap["kb_root"]
            data_mj.qpos[adr] = X_FR01 + (row[col["root"]] - traj_x0)

        mujoco.mj_forward(model, data_mj)

    # 記錄軌跡的初始 x，讓 FR01 從 X_FR01 出發往相同方向移動
    traj_x0 = traj_qpos[0, col["root"]]

    apply_frame(0)

    print(f"\n{'='*55}")
    print(f"  左 (x={X_SMPL})  = 真人骨架 (SMPL)")
    print(f"  中 (x={X_H1 })   = UnitreeH1")
    print(f"  右 (x={X_FR01})  = FR01")
    print(f"  動作: {MOTION}  |  {T} frames @ {fps}Hz")
    print(f"{'='*55}\n")

    with mujoco.viewer.launch_passive(
        model, data_mj,
        show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.azimuth    = 90
        viewer.cam.elevation  = -18
        viewer.cam.distance   = 8.0

        ep = 0
        while viewer.is_running():
            ep += 1; print(f"▶ Episode {ep}")
            t0 = time.time()
            for step in range(T):
                if not viewer.is_running(): break
                apply_frame(step); viewer.sync()
                wait = (step + 1) * dt - (time.time() - t0)
                if wait > 0.001: time.sleep(wait)
            print(f"   {time.time()-t0:.1f}s")
            time.sleep(0.3)


if __name__ == "__main__":
    main()
