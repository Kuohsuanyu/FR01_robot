#!/usr/bin/env python3
"""在這台電腦用 MuJoCo 跑一顆腿部 .kinfer,做 sim2sim 走路測試。

.kinfer(init_fn.onnx + step_fn.onnx + metadata)用 onnxruntime 依 kinfer 慣例
驅動,MuJoCo 用本地全身模型 models/QBOT_MJCF/qbot.xml(上半身固定成站姿,只讓
policy 控制 10 個腿部關節 —— 對應你「上半身尚未 sim2real」的現況)。

觀測(step_fn 輸入,依 onnx 介面):
  joint_angles[10], joint_angular_velocities[10], projected_gravity[3],
  imu_acc[3], imu_gyro[3], time[1], carry[N]
動作:actions[10] 當作位置目標(絕對 rad),經位置致動器(PD)套用。

用法:
  python3 leg/policies/sim2sim.py leg/policies/verified/walk.kinfer          # 互動視窗
  python3 leg/policies/sim2sim.py <policy.kinfer> --seconds 8 --headless     # 無視窗+量化評分
  python3 leg/policies/sim2sim.py <policy.kinfer> --video /tmp/walk.mp4       # 輸出影片
可調(慣例收斂用):--time-unit s|us  --action-offset none|nominal  --kp/--kd
"""
from __future__ import annotations
import argparse, json, math, os, tarfile, tempfile, time as _time
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_MODEL = os.path.join(REPO, "models", "QBOT_MJCF", "qbot.xml")

# 訓練起始姿勢(腿部 nominal,來自舊 sim2sim ZEROS;上半身固定成這組)
NOMINAL = {
    "dof_right_hip_pitch_04": math.radians(-20.0), "dof_right_hip_roll_03": 0.0,
    "dof_right_hip_yaw_03": 0.0, "dof_right_knee_04": math.radians(-50.0),
    "dof_right_ankle_02": math.radians(30.0),
    "dof_left_hip_pitch_04": math.radians(20.0), "dof_left_hip_roll_03": 0.0,
    "dof_left_hip_yaw_03": 0.0, "dof_left_knee_04": math.radians(50.0),
    "dof_left_ankle_02": math.radians(-30.0),
    "ub_right_shoulder": 0.0, "ub_right_lateral_raise": math.radians(80.0),
    "ub_right_arm_twist": math.radians(80.0), "ub_right_elbow": math.radians(67.5),
    "ub_left_shoulder": 0.0, "ub_left_lateral_raise": math.radians(-80.0),
    "ub_left_arm_twist": math.radians(80.0), "ub_left_elbow": math.radians(67.5),
}


def ensure_floor(model_path):
    """裸機器人 MJCF 沒地板會直接墜落/數值爆炸。若無 plane geom,
    在模型同目錄產生含地板+光源的 scene 包裝(mesh 相對路徑才解析得到)。"""
    m = mujoco.MjModel.from_xml_path(model_path)
    if any(m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE for i in range(m.ngeom)):
        return model_path
    d = os.path.dirname(os.path.abspath(model_path))
    scene = os.path.join(d, "_sim2sim_scene.xml")
    with open(scene, "w") as f:
        f.write(f'''<mujoco model="sim2sim_scene">
  <include file="{os.path.basename(model_path)}"/>
  <statistic center="0 0 0.6" extent="1.8"/>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="_floor" type="plane" size="0 0 0.05" pos="0 0 0"
          contype="1" conaffinity="1" friction="1 0.005 0.0001" rgba="0.5 0.55 0.6 1"/>
  </worldbody>
</mujoco>''')
    return scene


def load_kinfer(path):
    t = tarfile.open(path)
    meta = json.load(t.extractfile("metadata.json"))
    sess = {}
    for n in ("init_fn.onnx", "step_fn.onnx"):
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
            tf.write(t.extractfile(n).read()); p = tf.name
        sess[n] = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        os.unlink(p)
    return meta, sess["init_fn.onnx"], sess["step_fn.onnx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kinfer")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--headless", action="store_true", help="不開視窗,只印評分")
    ap.add_argument("--video", default="", help="輸出 mp4(需 headless 渲染)")
    ap.add_argument("--time-unit", choices=["s", "us"], default="s")
    ap.add_argument("--action-offset", choices=["none", "nominal"], default="none",
                    help="none: ctrl=action(絕對);nominal: ctrl=nominal+action(相對)")
    ap.add_argument("--kp", type=float, default=-1.0)
    ap.add_argument("--kd", type=float, default=-1.0)
    args = ap.parse_args()

    meta, init_fn, step_fn = load_kinfer(args.kinfer)
    jnames = meta["joint_names"]
    n = len(jnames)
    print(f"[sim2sim] policy: {n} 關節, carry={meta.get('carry_size')}, "
          f"num_commands={meta.get('num_commands')}, dt={meta.get('dt')}")

    m = mujoco.MjModel.from_xml_path(ensure_floor(args.model))
    d = mujoco.MjData(m)

    # 關節/致動器映射
    def jid(name): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    leg_qpos = [m.jnt_qposadr[jid(j)] for j in jnames]
    leg_dof  = [m.jnt_dofadr[jid(j)] for j in jnames]
    # joint_id → actuator_id
    j2act = {m.actuator_trnid[i, 0]: i for i in range(m.nu)}
    leg_act = [j2act[jid(j)] for j in jnames]
    # 自由關節(base)
    base_qadr = next(m.jnt_qposadr[i] for i in range(m.njnt)
                     if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
    base_dof = next(m.jnt_dofadr[i] for i in range(m.njnt)
                    if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)

    # 可選 kp/kd(位置致動器)
    if args.kp >= 0:
        for a in range(m.nu): m.actuator_gainprm[a, 0] = args.kp
    if args.kd >= 0:
        for a in range(m.nu):
            if m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_AFFINE:
                m.actuator_biasprm[a, 2] = -args.kd

    # 初始姿勢:所有關節設 nominal,直立
    mujoco.mj_resetData(m, d)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm in NOMINAL:
            d.qpos[m.jnt_qposadr[i]] = NOMINAL[nm]
    d.qpos[base_qadr + 3:base_qadr + 7] = [1, 0, 0, 0]     # 直立四元數
    d.qpos[base_qadr + 2] = 0.95
    for k, nm in enumerate(NOMINAL):                        # 全關節目標 = nominal
        d.ctrl[j2act[jid(nm)]] = NOMINAL[nm]
    mujoco.mj_forward(m, d)

    nominal_leg = np.array([NOMINAL[j] for j in jnames], np.float32)

    # 落地穩定:保持 nominal,讓腳踩到地面到達平衡,再交給 policy
    for _ in range(int(0.8 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    print(f"[sim2sim] 落地穩定後 base_z={d.qpos[base_qadr + 2]:.2f}")

    # kinfer 初始 carry
    carry = init_fn.run(None, {})[0]

    ctrl_dt = float(meta.get("dt", 0.02))
    phys_dt = m.opt.timestep
    steps_per_ctrl = max(1, round(ctrl_dt / phys_dt))

    def observe(t_now):
        qw, qx, qy, qz = d.qpos[base_qadr + 3:base_qadr + 7]
        # base 旋轉矩陣(world←body)
        R = np.array([
            [1 - 2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw),   1 - 2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1 - 2*(qx*qx+qy*qy)]])
        proj_g = R.T @ np.array([0, 0, -1.0])                 # 重力於 body 系
        gyro = d.qvel[base_dof + 3:base_dof + 6].copy()       # body 角速度
        acc = R.T @ np.array([0, 0, 9.81])                    # 靜態重力(近似)
        ja = np.array([d.qpos[a] for a in leg_qpos], np.float32)
        jv = np.array([d.qvel[a] for a in leg_dof], np.float32)
        tval = t_now * (1e6 if args.time_unit == "us" else 1.0)
        return {
            "joint_angles": ja, "joint_angular_velocities": jv,
            "projected_gravity": proj_g.astype(np.float32),
            "imu_acc": acc.astype(np.float32), "imu_gyro": gyro.astype(np.float32),
            "time": np.array([tval], np.float32), "carry": carry,
        }

    def apply_action(a):
        tgt = a if args.action_offset == "none" else (nominal_leg + a)
        for k, act in enumerate(leg_act):
            lo, hi = m.actuator_ctrlrange[act]
            d.ctrl[act] = float(np.clip(tgt[k], lo, hi)) if hi > lo else float(tgt[k])

    # 評分
    def score():
        h = d.qpos[base_qadr + 2]
        up = observe(0)["projected_gravity"][2]     # 直立時 ≈ -1
        return h, up

    def run_loop(render_cb=None):
        nonlocal carry
        t0 = 0.0
        h_min, tilt_max, fell_at = 1e9, 0.0, None
        x0 = d.qpos[base_qadr]
        nsteps = int(args.seconds / ctrl_dt)
        for s in range(nsteps):
            obs = observe(t0)
            out = step_fn.run(None, obs)
            actions, carry = out[0], out[1]
            apply_action(np.asarray(actions, np.float32))
            for _ in range(steps_per_ctrl):
                mujoco.mj_step(m, d)
            t0 += ctrl_dt
            h, up = score()
            h_min = min(h_min, h); tilt_max = max(tilt_max, 1.0 + up)  # up→-1 理想
            if h < 0.4 and fell_at is None:
                fell_at = t0
            if render_cb: render_cb()
        dx = d.qpos[base_qadr] - x0
        return h_min, tilt_max, fell_at, dx

    print(f"[sim2sim] ctrl {1/ctrl_dt:.0f}Hz, phys {1/phys_dt:.0f}Hz, "
          f"time-unit={args.time_unit}, action-offset={args.action_offset}")

    if args.video:
        import mediapy as media  # optional
        rn = mujoco.Renderer(m, 480, 640); frames = []
        def cb():
            if len(frames) < args.seconds * 30:
                rn.update_scene(d); frames.append(rn.render())
        r = run_loop(cb); media.write_video(args.video, frames, fps=30)
        print(f"[sim2sim] 影片 → {args.video}")
    elif args.headless:
        r = run_loop()
    else:
        with mujoco.viewer.launch_passive(m, d) as v:
            def cb():
                v.sync(); _time.sleep(ctrl_dt)
            r = run_loop(cb)

    h_min, tilt_max, fell_at, dx = r
    print("──── sim2sim 評分 ────")
    print(f"  最低身高 base_z_min = {h_min:.2f} m   (太低=跌倒)")
    print(f"  最大傾斜 = {tilt_max:.2f}          (0=完全直立)")
    print(f"  前進距離 Δx = {dx:+.2f} m")
    print(f"  跌倒時間 = {'沒跌倒 ✓' if fell_at is None else f'{fell_at:.1f}s ✗'}")
    verdict = ("站穩/前進 ✓" if fell_at is None and h_min > 0.5
               else "疑似跌倒/不穩 ✗ — 試調 --time-unit / --action-offset / --kp")
    print(f"  判定:{verdict}")


if __name__ == "__main__":
    main()
