"""
單一視窗並排比較：UnitreeH1（左，參考動作）vs QBOT（右，retarget 結果）
左：H1 播原始 LAFAN1 軌跡；右：QBOT 播 retarget_qbot.py 算出的 qpos（腿+手臂）

執行：
    conda activate loco_new
    python compare_qbot.py [motion]
    例: python compare_qbot.py dance1     |     python compare_qbot.py dance2_subject2

由 compare_side_by_side.py（H1 vs KBOT）改寫；QBOT 直接吃 retargeted npz，不做關節 hack。
"""

import sys, time, os, re
import numpy as np
import mujoco
import mujoco.viewer
import xml.etree.ElementTree as ET

MOTIONS = {
    "walk1": "walk1_subject1", "walk2": "walk2_subject3",
    "run1": "run1_subject2", "run2": "run1_subject5",
    "dance1": "dance1_subject1", "dance2": "dance2_subject2",
    "sprint": "sprint1_subject2",
}
motion_key  = sys.argv[1] if len(sys.argv) > 1 else "dance1"
motion_name = MOTIONS.get(motion_key, motion_key)

HERE      = os.path.dirname(os.path.abspath(__file__))
H1_DIR    = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1"
H1_XML    = os.path.join(H1_DIR, "h1.xml")
QBOT_XML  = os.path.join(HERE, "..", "qbot_retarget.xml")
QBOT_DIR  = os.path.dirname(os.path.abspath(QBOT_XML))   # QBOT_TRAIN/（mesh 相對於此）
H1_NPZ    = os.path.expanduser(f"~/.loco-mujoco-caches/lafan1_converted/KBotEnv/{motion_name}.npz")
QBOT_NPZ  = os.path.expanduser(f"~/.loco-mujoco-caches/lafan1_converted/QBotEnv_retargeted/{motion_name}.npz")
X_OFFSET  = 1.5   # QBOT 向右偏移（公尺）


def build_combined_model():
    """H1 用 include 引入；QBOT inline，加 qb_ 前綴 + 絕對 mesh 路徑，丟掉 actuator（純運動學）。"""
    tree = ET.parse(QBOT_XML); root = tree.getroot()

    def absmesh(s):
        return re.sub(r'file="(meshes/[^"]+)"',
                      lambda m: f'file="{os.path.join(QBOT_DIR, m.group(1))}"', s)

    def prefix(s):
        s = re.sub(r'\s*childclass="[^"]*"', '', s)
        s = re.sub(r'\s*class="(?!visual|collision|h1)[^"]*"', '', s)
        for attr in ("name", "joint", "mesh", "site", "material"):
            s = re.sub(rf'{attr}="(?!qb_)([^"]+)"', rf'{attr}="qb_\1"', s)
        return s

    # worldbody（只有一個 base body）
    bodies = []
    for child in root.find("worldbody"):
        if child.tag in ("light",) or (child.tag == "geom" and child.get("name", "").startswith("floor")):
            continue
        bodies.append(prefix(absmesh(ET.tostring(child, encoding="unicode"))))

    # assets（mesh/material 加前綴 + 絕對路徑）
    assets = []
    for elem in (root.find("asset") or []):
        es = ET.tostring(elem, encoding="unicode")
        if elem.tag in ("mesh", "material"):
            es = re.sub(r'name="([^"]+)"', r'name="qb_\1"', es)
        if elem.tag == "mesh":
            es = absmesh(es)
        assets.append(es)

    combined = f"""<mujoco model="h1_qbot_combined">
  <include file="h1.xml"/>
  <asset>
    {''.join(assets)}
  </asset>
  <worldbody>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    tmp = os.path.join(H1_DIR, "_combined_qbot_temp.xml")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(combined)
    return tmp


def col_map(jnames):
    m = {}; c = 0
    for n in jnames:
        m[n] = c; c += 7 if (n == "root" or n == "floating_base") else 1
    return m


def main():
    print(f"Motion: {motion_key} → {motion_name}")
    h1 = np.load(H1_NPZ, allow_pickle=True)
    qb = np.load(QBOT_NPZ, allow_pickle=True)
    h1_q, h1_jn = h1["qpos"], list(h1["joint_names"])
    qb_q, qb_jn = qb["qpos"], list(qb["joint_names"])
    fps = int(h1["frequency"]); dt = 1.0 / fps
    T = min(len(h1_q), len(qb_q))
    print(f"H1 frames={len(h1_q)}  QBOT frames={len(qb_q)}  play={T} @ {fps}Hz")

    tmp = build_combined_model()
    try:
        model = mujoco.MjModel.from_xml_path(tmp)
    except Exception as e:
        print("BUILD ERROR:", e); os.unlink(tmp); return
    data = mujoco.MjData(model)
    print(f"Combined: nq={model.nq} njnt={model.njnt}")

    jmap = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): model.jnt_qposadr[j]
            for j in range(model.njnt)}
    h1_col = col_map(h1_jn)
    qb_col = col_map(qb_jn)

    def apply(step):
        # ── H1（左）──
        r = h1_q[step]
        if "root" in jmap:
            a = jmap["root"]; c = h1_col["root"]; data.qpos[a:a+7] = r[c:c+7]
        for jn, c in h1_col.items():
            if jn == "root": continue
            if jn in jmap: data.qpos[jmap[jn]] = r[c]
        # ── QBOT（右，retargeted qpos）──
        rq = qb_q[step]
        for jn, c in qb_col.items():
            adr = jmap.get("qb_" + jn)
            if adr is None: continue
            if jn == "floating_base":
                data.qpos[adr:adr+7] = rq[c:c+7]
                data.qpos[adr] += X_OFFSET          # 右移分開
            else:
                data.qpos[adr] = rq[c]
        mujoco.mj_forward(model, data)

    apply(0)
    print("\n左 = H1（參考）   右 = QBOT（retarget 腿+手臂）   滾輪縮放/拖拉旋轉/ESC 關閉\n")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        v.opt.sitegroup[3] = 1
        v.cam.lookat[:] = [X_OFFSET / 2, 0.0, 0.8]
        v.cam.azimuth = 90; v.cam.elevation = -15; v.cam.distance = 4.5
        ep = 0
        while v.is_running():
            ep += 1; print(f"▶ Episode {ep}"); t0 = time.time()
            for step in range(T):
                if not v.is_running(): break
                apply(step); v.sync()
                wait = (step + 1) * dt - (time.time() - t0)
                if wait > 0.001: time.sleep(wait)
            time.sleep(0.3)
    os.unlink(tmp)
    print("Done.")


if __name__ == "__main__":
    main()
