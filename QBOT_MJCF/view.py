# -*- coding: utf-8 -*-
"""View qbot.xml with working slider control.

The MuJoCo Control-panel sliders set each joint's ANGLE directly (kinematic:
slider value -> qpos, then mj_forward). No physics is stepped, so dragging a
slider moves that joint immediately and the robot holds the pose (never falls).

Run:
  python view.py
"""
import os
import math
import time
import mujoco
import mujoco.viewer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "qbot.xml")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)

    # --- make the all-black model visible in the viewer ---
    # The model materials are pure black (rgba 0,0,0) on disk. Ambient/diffuse
    # light is MULTIPLIED by the material colour, so pure black reflects nothing
    # and the robot looks like a void no matter how bright the light. To see it
    # WITHOUT changing the model file, we (1) raise the ambient/diffuse/specular
    # headlight, (2) lift the *displayed* colour to a dark grey, and (3) add a
    # specular highlight + shininess so curvature/edges catch the light.
    model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]

    DISPLAY_GREY = 0.22   # shows as near-black but is actually visible
    for i in range(model.nmat):
        a = model.mat_rgba[i, 3]            # keep original alpha
        model.mat_rgba[i] = [DISPLAY_GREY, DISPLAY_GREY, DISPLAY_GREY, a]
        model.mat_specular[i] = 0.5
        model.mat_shininess[i] = 0.4
    # Lift EVERY geom's rgba (not just material-less ones). The leg geoms carry
    # BOTH a material and an explicit dark rgba, and in MuJoCo geom rgba takes
    # precedence over the material colour -- so without this the legs stayed at
    # the model's 0.05 black while the body (no material) showed the lifted grey,
    # making the legs look darker than the body.
    for i in range(model.ngeom):
        a = model.geom_rgba[i, 3]
        model.geom_rgba[i] = [DISPLAY_GREY, DISPLAY_GREY, DISPLAY_GREY, a]

    data = mujoco.MjData(model)

    # map each actuator (a Control slider) -> its joint's qpos address + name
    act_qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]]
                for i in range(model.nu)]
    act_jname = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   model.actuator_trnid[i, 0])
                 for i in range(model.nu)]

    # --- starting pose (degrees) shown when the viewer opens ---
    # Arms raised to the side + rotated, elbows at half their range; legs in a
    # slightly-bent stable stance; neck neutral. Right arm mirrors the left
    # (lateral_raise sign flips; arm_twist/elbow keep sign -- verified by FK).
    START_POSE_DEG = {
        # legs (slightly-bent stable stance)
        "dof_right_hip_pitch_04": -20.0, "dof_right_hip_roll_03": 0.0,
        "dof_right_hip_yaw_03": 0.0, "dof_right_knee_04": -50.0,
        "dof_right_ankle_02": 30.0,
        "dof_left_hip_pitch_04": 20.0, "dof_left_hip_roll_03": 0.0,
        "dof_left_hip_yaw_03": 0.0, "dof_left_knee_04": 50.0,
        "dof_left_ankle_02": -30.0,
        # arms
        "ub_right_shoulder": 0.0, "ub_right_lateral_raise": 80.0,
        "ub_right_arm_twist": 80.0, "ub_right_elbow": 67.5,
        "ub_left_shoulder": 0.0, "ub_left_lateral_raise": -80.0,
        "ub_left_arm_twist": 80.0, "ub_left_elbow": 67.5,
        # neck
        "ub_neck_pitch": 0.0, "ub_neck_roll": 0.0, "ub_neck_yaw": 0.0,
    }
    for i, jn in enumerate(act_jname):
        if jn in START_POSE_DEG:
            data.ctrl[i] = math.radians(START_POSE_DEG[jn])  # slider start value
    mujoco.mj_forward(model, data)

    POSE_FILE = os.path.join(HERE, "current_pose.txt")

    def save_pose():
        """Dump the current joint angles (deg) + a ready-to-paste ZEROS list."""
        lines, zeros = [], []
        for i, adr in enumerate(act_qadr):
            deg = math.degrees(float(data.qpos[adr]))
            lines.append(f"{act_jname[i]:<26} {deg:8.2f} deg")
            zeros.append(f'    ("{act_jname[i]}", math.radians({deg:.1f})),')
        block = ("\n".join(lines) + "\n\n# ZEROS snippet:\nZEROS = [\n"
                 + "\n".join(zeros) + "\n]\n")
        with open(POSE_FILE, "w") as f:
            f.write(block)
        print("\n===== POSE SAVED to %s =====" % POSE_FILE)
        print(block)

    def key_callback(keycode):
        if keycode in (ord("P"), ord("p")):
            save_pose()

    print("Loaded qbot.xml: %d joints, %d actuators." % (model.njnt, model.nu))
    print("Open the left Control panel and drag sliders to pose each joint.")
    print(">>> Press  P  in the viewer window to save the current pose. <<<")
    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_callback) as viewer:
        while viewer.is_running():
            for i, adr in enumerate(act_qadr):
                data.qpos[adr] = data.ctrl[i]   # slider -> joint angle
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
