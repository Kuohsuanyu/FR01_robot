"""
單一視窗並排比較：UnitreeH1（左）vs KBOT（右）
同步播放相同的 LAFAN1 走路軌跡

執行：
    conda activate loco_new
    python compare_side_by_side.py
"""

import sys, time, os, re
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

MOTIONS = {
    "walk1":  "walk1_subject1",
    "walk2":  "walk2_subject3",
    "run1":   "run1_subject2",
    "run2":   "run1_subject5",
    "dance1": "dance1_subject1",
    "dance2": "dance2_subject2",
    "sprint": "sprint1_subject2",
}

motion_key  = sys.argv[1] if len(sys.argv) > 1 else "walk1"
# 用 --retargeted 旗標切換到 retargeted 版本
use_retargeted = "--retargeted" in sys.argv or "-r" in sys.argv
motion_name = MOTIONS.get(motion_key, motion_key)

if use_retargeted:
    TRAJ_NPZ = f"/home/andykuo/.loco-mujoco-caches/lafan1_converted/KBotEnv_retargeted/{motion_name}.npz"
    print(f"Motion: {motion_key} → {motion_name}  [RETARGETED]")
else:
    TRAJ_NPZ = f"/home/andykuo/.loco-mujoco-caches/lafan1_converted/KBotEnv/{motion_name}.npz"
    print(f"Motion: {motion_key} → {motion_name}  [direct mapping]")
H1_DIR   = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1"
H1_XML   = os.path.join(H1_DIR, "h1.xml")
KBOT_XML = "/home/andykuo/ksim-gym/MIMIC/kbot/KBOT_loco.xml"
KBOT_DIR = os.path.dirname(KBOT_XML)
X_OFFSET = 2.2   # KBOT 向右偏移距離（公尺）


# ── KBOT joint 名稱對應 H1 trajectory joint 名稱 ──────────────────────
KBOT_TO_H1_JOINT = {
    "kb_hip_flexion_r":   "hip_flexion_r",
    "kb_hip_adduction_r": "hip_adduction_r",
    "kb_hip_rotation_r":  "hip_rotation_r",
    "kb_knee_angle_r":    "knee_angle_r",
    "kb_ankle_angle_r":   "ankle_angle_r",
    "kb_hip_flexion_l":   "hip_flexion_l",
    "kb_hip_adduction_l": "hip_adduction_l",
    "kb_hip_rotation_l":  "hip_rotation_l",
    "kb_knee_angle_l":    "knee_angle_l",
    "kb_ankle_angle_l":   "ankle_angle_l",
}


def build_combined_model(h1_xml_path, kbot_xml_path, x_offset):
    """
    建立合併模型：
    - H1 在 x=0（使用 include 引入）
    - KBOT 在 x=x_offset（inline，mesh 路徑改絕對路徑）
    臨時檔寫在 H1 目錄，讓 H1 的 include 正常解析。
    """
    kbot_tree = ET.parse(kbot_xml_path)
    kbot_root = kbot_tree.getroot()
    kbot_wb   = kbot_root.find("worldbody")

    # 把 KBOT worldbody 所有子元素複製，並加前綴 + 絕對路徑
    kbot_bodies_xml = []
    for child in kbot_wb:
        # 跳過地板、燈光等場景元素（H1 的場景已包含）
        if child.tag in ("geom", "light") and child.get("name", "").startswith("floor") or child.tag == "light":
            continue
        child_str = ET.tostring(child, encoding="unicode")
        # 把相對 mesh/material 路徑改成絕對路徑
        child_str = re.sub(
            r'file="(meshes/[^"]+)"',
            lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"',
            child_str
        )
        # 移除所有 class/childclass 屬性（避免 H1 沒有對應 default class 衝突）
        child_str = re.sub(r'\s*childclass="[^"]*"', '', child_str)
        child_str = re.sub(r'\s*class="(?!visual|collision|h1)[^"]*"', '', child_str)
        # 加 kb_ 前綴到 name 屬性（避免與 H1 衝突）
        child_str = re.sub(r'name="(?!kb_)([^"]+)"', r'name="kb_\1"', child_str)
        child_str = re.sub(r'joint="(?!kb_)([^"]+)"', r'joint="kb_\1"', child_str)
        child_str = re.sub(r'mesh="(?!kb_)([^"]+)"',  r'mesh="kb_\1"',  child_str)
        child_str = re.sub(r'site="(?!kb_)([^"]+)"',  r'site="kb_\1"',  child_str)
        child_str = re.sub(r'material="(?!kb_)([^"]+)"', r'material="kb_\1"', child_str)
        kbot_bodies_xml.append(child_str)

    # KBOT assets（mesh/material）加前綴 + 絕對路徑
    kbot_assets_xml = []
    kbot_asset = kbot_root.find("asset")
    if kbot_asset is not None:
        for elem in kbot_asset:
            elem_str = ET.tostring(elem, encoding="unicode")
            # material 加前綴
            if elem.tag == "material":
                elem_str = re.sub(r'name="([^"]+)"', r'name="kb_\1"', elem_str)
            elif elem.tag == "mesh":
                elem_str = re.sub(r'name="([^"]+)"', r'name="kb_\1"', elem_str)
                elem_str = re.sub(
                    r'file="(meshes/[^"]+)"',
                    lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"',
                    elem_str
                )
            kbot_assets_xml.append(elem_str)

    # KBOT actuators — 從 class 展開 ctrlrange，再移除 class 屬性
    KBOT_CLASS_CTRLRANGE = {
        "robstride_04": "-120.0 120.0",
        "robstride_03": "-60.0 60.0",
        "robstride_02": "-17.0 17.0",
        "robstride_00": "-14.0 14.0",
    }
    kbot_actuators_xml = []
    kbot_act = kbot_root.find("actuator")
    if kbot_act is not None:
        for elem in kbot_act:
            elem_str = ET.tostring(elem, encoding="unicode")
            elem_str = re.sub(r'name="([^"]+)"',  r'name="kb_\1"',  elem_str)
            elem_str = re.sub(r'joint="([^"]+)"', r'joint="kb_\1"', elem_str)
            # 把 class 屬性展開為 ctrlrange
            cls_match = re.search(r'class="([^"]+)"', elem_str)
            if cls_match:
                cls_name = cls_match.group(1)
                if cls_name in KBOT_CLASS_CTRLRANGE and 'ctrlrange' not in elem_str:
                    elem_str = elem_str.replace('class="' + cls_name + '"',
                                                f'ctrlrange="{KBOT_CLASS_CTRLRANGE[cls_name]}"')
                else:
                    elem_str = re.sub(r'\s*class="[^"]*"', '', elem_str)
            kbot_actuators_xml.append(elem_str)

    # KBOT defaults（前綴）
    kbot_defaults_xml = []
    kbot_def = kbot_root.find("default")
    if kbot_def is not None:
        for elem in kbot_def:
            elem_str = ET.tostring(elem, encoding="unicode")
            kbot_defaults_xml.append(elem_str)

    # 組合成完整 XML
    combined = f"""<mujoco model="h1_kbot_combined">
  <!-- H1 include（使用相對路徑，因此此檔必須在 H1 目錄） -->
  <include file="h1.xml"/>

  <!-- KBOT assets -->
  <asset>
    {''.join(kbot_assets_xml)}
  </asset>

  <!-- KBOT worldbody（直接加入，freejoint x 在 apply_frame 加偏移） -->
  <worldbody>
    {''.join(kbot_bodies_xml)}
  </worldbody>

  <!-- KBOT actuators -->
  <actuator>
    {''.join(kbot_actuators_xml)}
  </actuator>
</mujoco>"""

    # 儲存在 H1 目錄（讓 include 路徑正常解析）
    tmp_path = os.path.join(H1_DIR, "_combined_temp.xml")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return tmp_path


def main():
    # ── 載入軌跡 ──────────────────────────────────────────────────────
    print("Loading trajectory...")
    npz = np.load(TRAJ_NPZ, allow_pickle=True)
    traj_qpos   = npz["qpos"]
    traj_jnames = list(npz["joint_names"])
    fps  = int(npz["frequency"])
    dt   = 1.0 / fps
    T    = len(traj_qpos)
    print(f"Trajectory: {T} steps @ {fps} Hz ({T/fps:.1f}s)")

    # ── 建立合併模型 ──────────────────────────────────────────────────
    print("Building combined model...")
    tmp_xml = build_combined_model(H1_XML, KBOT_XML, X_OFFSET)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_xml)
    except Exception as e:
        print("Error:", e)
        os.unlink(tmp_xml)
        return
    data = mujoco.MjData(model)
    print(f"Combined model: nq={model.nq}, njnt={model.njnt}")

    # ── 建立 joint 名稱 → qpos address 映射 ──────────────────────────
    jmap = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jmap[name] = model.jnt_qposadr[jid]

    # ── 正確的 qpos column 映射（freejoint 佔 7 個 slot）──────────────
    # joint_names 有 20 項，qpos 有 26 欄：freejoint(7) + 19 joints
    traj_name_to_col = {}
    col = 0
    for name in traj_jnames:
        traj_name_to_col[name] = col
        col += 7 if name == "root" else 1  # freejoint=7, hinge=1

    def apply_frame(step_idx):
        row = traj_qpos[step_idx]

        # H1 freejoint（root，7個 qpos）
        if "root" in jmap and "root" in traj_name_to_col:
            adr = jmap["root"]
            col = traj_name_to_col["root"]
            data.qpos[adr:adr+7] = row[col:col+7]

        # H1 其他關節（各 1 個 qpos）
        for jname, adr in jmap.items():
            if jname.startswith("kb_") or jname == "root":
                continue
            if jname in traj_name_to_col:
                data.qpos[adr] = row[traj_name_to_col[jname]]

        # KBOT freejoint（kb_root，x 加偏移讓 KBOT 出現在 x=X_OFFSET）
        if "kb_root" in jmap and "root" in traj_name_to_col:
            adr = jmap["kb_root"]
            col = traj_name_to_col["root"]
            data.qpos[adr]   = 0.0              # x_rel=0 → world_x=X_OFFSET
            data.qpos[adr+1] = row[col+1]       # y
            data.qpos[adr+2] = row[col+2]       # z
            data.qpos[adr+3] = row[col+3]       # qw
            data.qpos[adr+4] = row[col+4]       # qx
            data.qpos[adr+5] = row[col+5]       # qy
            data.qpos[adr+6] = row[col+6]       # qz

        # KBOT 腿部關節（直接映射 + 符號修正）
        # Jacobian 分析結論：
        #   knee_angle_r : H1正=腳後, KBOT正=腳前 → NEGATE
        #   hip_flexion_l: H1正=腳後, KBOT正=腳前 → NEGATE (軸向不對稱)
        #   ankle_angle_l: H1正=腳下, KBOT正=腳上 → NEGATE (軸向不對稱)
        NEGATE_JOINTS = {"kb_knee_angle_r", "kb_hip_flexion_l", "kb_ankle_angle_l"}
        for kb_jname, h1_jname in KBOT_TO_H1_JOINT.items():
            if kb_jname in jmap and h1_jname in traj_name_to_col:
                val = row[traj_name_to_col[h1_jname]]
                if kb_jname in NEGATE_JOINTS:
                    val = -val
                data.qpos[jmap[kb_jname]] = val

        mujoco.mj_forward(model, data)

    apply_frame(0)

    print()
    print("視窗說明")
    print("  左邊 = UnitreeH1（參考動作）")
    print("  右邊 = KBOT（你的機器人，相同腿部動作）")
    print("  滾輪縮放、拖拉旋轉、ESC 關閉")
    print()

    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,
        show_right_ui=False
    ) as viewer:
        # 攝影機看向兩隻機器人中間
        viewer.cam.lookat[:] = [X_OFFSET / 2, 0.0, 0.8]
        viewer.cam.azimuth    = 90
        viewer.cam.elevation  = -18
        viewer.cam.distance   = 5.5

        ep = 0
        while viewer.is_running():
            ep += 1
            print(f"▶ Episode {ep}")
            t0 = time.time()
            for step in range(T):
                if not viewer.is_running():
                    break
                apply_frame(step)
                viewer.sync()
                wait = (step + 1) * dt - (time.time() - t0)
                if wait > 0.001:
                    time.sleep(wait)
            print(f"   {time.time()-t0:.1f}s")
            time.sleep(0.3)

    os.unlink(tmp_xml)
    print("Done.")


if __name__ == "__main__":
    main()
