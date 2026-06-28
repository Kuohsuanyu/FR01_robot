"""
回放 retarget 結果（純運動學，不跑物理）。
    conda activate loco_new
    python play_qbot.py [motion]            # 預設 dance1_subject1

操作：滑鼠轉視角；Space 由 viewer 控制；關視窗結束。
動作會循環播放。按 's' 看 mimic site（group 3）。
"""
import sys, os, time
import numpy as np
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
XML  = os.path.join(HERE, "..", "qbot_retarget.xml")
MOTION = sys.argv[1] if len(sys.argv) > 1 else "dance1_subject1"
NPZ = os.path.expanduser(f"~/.loco-mujoco-caches/lafan1_converted/QBotEnv_retargeted/{MOTION}.npz")

d = np.load(NPZ, allow_pickle=True)
qpos = d["qpos"]
fps = int(d["frequency"])
T = len(qpos)
dt = 1.0 / fps
print(f"Motion: {MOTION}  frames={T}  fps={fps}")

m = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, data) as viewer:
    # 顯示 mimic site (group 3)
    viewer.opt.sitegroup[3] = 1
    t = 0
    while viewer.is_running():
        data.qpos[:] = qpos[t % T]
        mujoco.mj_forward(m, data)
        viewer.sync()
        t += 1
        time.sleep(dt)
