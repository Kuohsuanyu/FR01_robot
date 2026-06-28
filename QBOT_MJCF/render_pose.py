# -*- coding: utf-8 -*-
"""Offscreen-render the start pose to PNGs (front + side) for confirmation."""
import os, math, numpy as np, mujoco
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "qbot.xml"))
m.vis.global_.offwidth = 700
m.vis.global_.offheight = 900

# same visibility tweaks as view.py
m.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
m.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
m.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
G = 0.30
for i in range(m.nmat):
    a = m.mat_rgba[i, 3]; m.mat_rgba[i] = [G, G, G, a]
    m.mat_specular[i] = 0.5; m.mat_shininess[i] = 0.4
for i in range(m.ngeom):
    a = m.geom_rgba[i, 3]; m.geom_rgba[i] = [G, G, G, a]

d = mujoco.MjData(m)
act_j = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
         for i in range(m.nu)]
POSE = {
    "dof_right_hip_pitch_04": -20.0, "dof_right_knee_04": -50.0, "dof_right_ankle_02": 30.0,
    "dof_left_hip_pitch_04": 20.0, "dof_left_knee_04": 50.0, "dof_left_ankle_02": -30.0,
    "ub_right_lateral_raise": 80.0, "ub_right_arm_twist": 80.0, "ub_right_elbow": 67.5,
    "ub_left_lateral_raise": -80.0, "ub_left_arm_twist": 80.0, "ub_left_elbow": 67.5,
}
for i, jn in enumerate(act_j):
    if jn in POSE:
        d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]] = math.radians(POSE[jn])
mujoco.mj_forward(m, d)

ren = mujoco.Renderer(m, height=900, width=700)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(m, cam)
cam.distance = 2.6
cam.lookat[:] = [0, 0, 0.8]
for name, az, el in [("front", 90, -10), ("side", 0, -10), ("threeq", 45, -15)]:
    cam.azimuth = az; cam.elevation = el
    ren.update_scene(d, cam)
    Image.fromarray(ren.render()).save(os.path.join(HERE, f"pose_{name}.png"))
    print("saved pose_%s.png" % name)
print("done")
