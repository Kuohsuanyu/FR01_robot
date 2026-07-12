"""
快速切換 AMASS 資料播放：H1 + FR01 並排（osmesa 離線渲染，影片嵌入右下角）

使用方式：
    JAX_PLATFORM_NAME=cpu python compare_amass.py [motion_id]

motion_id 範例：
    CMU/01/01_01_poses        ← AMASS CMU 子集走路
    CMU/02/02_01_poses        ← CMU 跑步
    BMLmovi/Subject_1/F_PG1_poses
    ACCAD/Male1Walking_c3d/Walk1_poses

快捷鍵（視窗內按）：
    Q / ESC : 離開
    SPACE   : 暫停 / 繼續
    R       : 從頭重播
    S       : 速度 x0.5
    F       : 速度 x2

影片模式（含原始影片嵌入）：
    JAX_PLATFORM_NAME=cpu python compare_amass.py --video \\
        /path/to/frames_dir  video_motion_name
"""
import sys, os, time, re, argparse
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# 不設 MUJOCO_GL → 使用 passive viewer
sys.path.insert(0, '/home/andykuo/ksim-gym/MIMIC/loco_mujoco')

import numpy as np
import mujoco
import mujoco.viewer
import cv2
from pathlib import Path
import xml.etree.ElementTree as ET

# ── 常數 ─────────────────────────────────────────────────────────────────
H1_DIR   = "/home/andykuo/miniconda3/envs/loco_new/lib/python3.11/site-packages/loco_mujoco_models/unitree_h1"
KBOT_DIR = "/home/andykuo/ksim-gym/MIMIC/kbot"
KBOT_XML = os.path.join(KBOT_DIR, "KBOT_loco.xml")
AMASS_CACHE = "/home/andykuo/.loco-mujoco-caches/amass_converted/UnitreeH1"

X_H1, X_FR01 = 0.0, 2.5
KBOT_TO_H1 = {
    "kb_hip_flexion_r":"hip_flexion_r","kb_hip_adduction_r":"hip_adduction_r",
    "kb_hip_rotation_r":"hip_rotation_r","kb_knee_angle_r":"knee_angle_r",
    "kb_ankle_angle_r":"ankle_angle_r","kb_hip_flexion_l":"hip_flexion_l",
    "kb_hip_adduction_l":"hip_adduction_l","kb_hip_rotation_l":"hip_rotation_l",
    "kb_knee_angle_l":"knee_angle_l","kb_ankle_angle_l":"ankle_angle_l",
}
NEGATE = {"kb_knee_angle_r","kb_hip_flexion_l","kb_ankle_angle_l"}
KBOT_CTRL = {"robstride_04":"-120.0 120.0","robstride_03":"-60.0 60.0","robstride_02":"-17.0 17.0"}
WIN_W, WIN_H = 1280, 720
VID_W, VID_H = 360, 202

# ── 解析引數 ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("motion_id", nargs="?", default="CMU/01/01_01_poses",
                    help="AMASS motion ID, e.g. CMU/01/01_01_poses")
parser.add_argument("--video", action="store_true",
                    help="Video mode: embed frames from IMG_DIR")
parser.add_argument("img_dir", nargs="?", default=None,
                    help="(video mode) path to frame images")
parser.add_argument("motion_name", nargs="?", default=None,
                    help="(video mode) cache name for the motion")
args = parser.parse_args()

VIDEO_MODE = args.video
IMG_DIR    = args.img_dir
MOTION_ID  = args.motion_id

# ── 載入 H1 軌跡 ───────────────────────────────────────────────────────
def load_trajectory(motion_id):
    """從 AMASS 快取載入 H1 軌跡，若無則從 AMASS 原始資料 retarget"""
    # 快取路徑
    safe = motion_id.replace("/", "_")
    cache = os.path.join(AMASS_CACHE, safe + ".npz")
    if not os.path.exists(cache):
        # 嘗試從 loco-mujoco 的原始快取結構找
        parts = motion_id.split("/")
        if len(parts) >= 2:
            sub = "/".join(parts[:-1])
            name = parts[-1]
            cache2 = os.path.join(AMASS_CACHE, sub, name + ".npz")
            if os.path.exists(cache2):
                cache = cache2
    if not os.path.exists(cache):
        print(f"快取不存在: {cache}")
        print(f"請先跑 retargeting：")
        print(f"  JAX_PLATFORM_NAME=cpu python -c \"")
        print(f"  from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory")
        print(f"  load_retargeted_amass_trajectory('UnitreeH1', '{motion_id}')\"")
        return None, None, None
    d = np.load(cache, allow_pickle=True)
    return d["qpos"], list(d["joint_names"]), int(float(d["frequency"]))

# ── 建立 H1+FR01 合併 XML ─────────────────────────────────────────────
def build_model():
    kbot_tree = ET.parse(KBOT_XML)
    root = kbot_tree.getroot()
    bodies, assets, acts = [], [], []
    for ch in root.find("worldbody"):
        if ch.tag == "light": continue
        s = ET.tostring(ch, encoding="unicode")
        s = re.sub(r'file="(meshes/[^"]+)"',
                   lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"', s)
        s = re.sub(r'\s*childclass="[^"]*"', '', s)
        s = re.sub(r'\s*class="(?!visual|collision|h1)[^"]*"', '', s)
        # 把 collision geom 移到 group=3（MuJoCo 預設不顯示），避免橘色遮蓋視覺
        s = re.sub(r'class="collision"', 'class="collision" group="3"', s)
        for tag in ("name", "joint", "mesh", "material"):
            s = re.sub(rf'{tag}="(?!kb_)([^"]+)"', rf'{tag}="kb_\1"', s)
        bodies.append(s)
    for el in root.find("asset"):
        s = ET.tostring(el, encoding="unicode")
        if el.tag == "material":
            s = re.sub(r'name="([^"]+)"', r'name="kb_\1"', s)
        elif el.tag == "mesh":
            s = re.sub(r'name="([^"]+)"', r'name="kb_\1"', s)
            s = re.sub(r'file="(meshes/[^"]+)"',
                       lambda m: f'file="{os.path.join(KBOT_DIR, m.group(1))}"', s)
        assets.append(s)
    for el in root.find("actuator"):
        s = ET.tostring(el, encoding="unicode")
        s = re.sub(r'name="([^"]+)"',  r'name="kb_\1"', s)
        s = re.sub(r'joint="([^"]+)"', r'joint="kb_\1"', s)
        cls = re.search(r'class="([^"]+)"', s)
        if cls and cls.group(1) in KBOT_CTRL and "ctrlrange" not in s:
            s = s.replace(f'class="{cls.group(1)}"',
                          f'ctrlrange="{KBOT_CTRL[cls.group(1)]}"')
        else:
            s = re.sub(r'\s*class="[^"]*"', '', s)
        acts.append(s)

    xml = f"""<mujoco model="h1_fr01">
  <visual><global offwidth="{WIN_W}" offheight="{WIN_H}"/></visual>
  <include file="h1.xml"/>
  <asset>{''.join(assets)}</asset>
  <worldbody>{''.join(bodies)}</worldbody>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""
    tmp = os.path.join(H1_DIR, "_compare_tmp.xml")
    with open(tmp, "w") as f:
        f.write(xml)
    return tmp

# ── 主程式 ────────────────────────────────────────────────────────────
traj_qpos, jnames, fps = load_trajectory(MOTION_ID)
if traj_qpos is None:
    sys.exit(1)

T  = len(traj_qpos)
dt = 1.0 / fps
print(f"Motion: {MOTION_ID} | {T} 幀 @ {fps}Hz ({T/fps:.1f}s)")

col = {}; c = 0
for n in jnames:
    col[n] = c
    c += 7 if n == "root" else 1

traj_x0 = traj_qpos[0, col["root"]]

tmp_xml = build_model()
model_mj = mujoco.MjModel.from_xml_path(tmp_xml)
data_mj  = mujoco.MjData(model_mj)
os.unlink(tmp_xml)

jmap = {}
for jid in range(model_mj.njnt):
    name = mujoco.mj_id2name(model_mj, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if name:
        jmap[name] = model_mj.jnt_qposadr[jid]

def apply_frame(step):
    row = traj_qpos[step]
    # H1 freejoint（原點）
    if "root" in jmap:
        adr = jmap["root"]
        data_mj.qpos[adr:adr+7] = row[col["root"]:col["root"]+7]
    # H1 關節
    for jn, adr in jmap.items():
        if jn.startswith("kb_") or jn == "root": continue
        if jn in col:
            data_mj.qpos[adr] = row[col[jn]]
    # FR01 位置（X 偏移 +2.5）
    if "kb_root" in jmap:
        adr = jmap["kb_root"]
        data_mj.qpos[adr]     = X_FR01 + (row[col["root"]] - traj_x0)
        data_mj.qpos[adr+1:adr+7] = row[col["root"]+1:col["root"]+7]
    # FR01 關節角度
    for kb, h1 in KBOT_TO_H1.items():
        if kb in jmap and h1 in col:
            v = row[col[h1]]
            if kb in NEGATE:
                v = -v
            data_mj.qpos[jmap[kb]] = v
    mujoco.mj_forward(model_mj, data_mj)

apply_frame(0)

# ── 影片幀（video mode 時）────────────────────────────────────────────
vid_frames, vid_map = [], []
if VIDEO_MODE and IMG_DIR and os.path.isdir(IMG_DIR):
    for p in sorted(Path(IMG_DIR).glob("*.jpg")) + sorted(Path(IMG_DIR).glob("*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            vid_frames.append(img)
    if vid_frames:
        vid_map = [int(i * (len(vid_frames)-1) / max(T-1, 1)) for i in range(T)]
        print(f"原始影片: {len(vid_frames)} 幀 → 對應 {T} 幀")
        cv2.namedWindow("原始影片", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("原始影片", VID_W, VID_H)
        cv2.moveWindow("原始影片", 1550, 830)   # 螢幕右下角

print(f"\n顯示中: {MOTION_ID}  |  {T} 幀 @ {fps}Hz  ({T/fps:.1f}s)")
print("  左 = H1 (參考)    右 = FR01 (你的機器人)")
print("  視窗內：滾輪縮放、右鍵拖拉旋轉、Esc 關閉\n")

with mujoco.viewer.launch_passive(model_mj, data_mj,
                                   show_left_ui=False, show_right_ui=False) as v:
    v.cam.lookat[:] = [1.25, 0.0, 0.9]
    v.cam.azimuth   = 90
    v.cam.elevation = -18
    v.cam.distance  = 6.0
    ep = 0
    while v.is_running():
        ep += 1
        print(f"▶ Episode {ep}  [{MOTION_ID}]")
        t0 = time.time()
        for step in range(T):
            if not v.is_running():
                break
            apply_frame(step)
            v.sync()
            # 同步顯示原始影片
            if vid_frames:
                thumb = cv2.resize(vid_frames[vid_map[step]], (VID_W, VID_H))
                cv2.putText(thumb, f"{step+1}/{T}", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imshow("原始影片", thumb)
                cv2.waitKey(1)
            wait = (step + 1) * dt - (time.time() - t0)
            if wait > 0.001:
                time.sleep(wait)
        print(f"   {time.time()-t0:.1f}s")
        time.sleep(0.2)

if vid_frames:
    cv2.destroyAllWindows()
