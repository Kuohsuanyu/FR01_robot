# -*- coding: utf-8 -*-
"""Standalone sim2sim deployment of the trained Q-BOT walking policy.

Self-contained: loads `policy.bin` + local `qbot.xml`/`meshes` and runs the
policy in an INDEPENDENT plain-MuJoCo rollout (no ksim training loop). The
observation pipeline mirrors train_qbot.run_actor exactly:
    [sin t, cos t, joint_pos(18), joint_vel(18), proj_grav(3), imu_acc(3), imu_gyro(3)]
Control at 50 Hz (ctrl_dt 0.02), physics at 500 Hz (dt 0.002), action passed
straight through to the position actuators (clipped to ctrlrange).

Modes:
  python sim2sim.py            # LIVE: interactive 3D window + live joint-torque bars
  python sim2sim.py --video    # headless: render mp4 (robot + torque bars baked in)

Modelled on ksim's `run_mode=view` (load checkpoint -> roll out in a viewer).
"""
import os
import math
import argparse
import numpy as np
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import ksim
import xax
from jaxtyping import Array, PRNGKeyArray
from xax.task.mixins.checkpointing import load_ckpt

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- start/held pose bias (MUST match the training ZEROS, model joint order) --
ZEROS = [
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
    ("ub_right_shoulder", 0.0),
    ("ub_right_lateral_raise", math.radians(80.0)),
    ("ub_right_arm_twist", math.radians(80.0)),
    ("ub_right_elbow", math.radians(67.5)),
    ("ub_left_shoulder", 0.0),
    ("ub_left_lateral_raise", math.radians(-80.0)),
    ("ub_left_arm_twist", math.radians(80.0)),
    ("ub_left_elbow", math.radians(67.5)),
]
ZEROS_BIAS = np.array([v for _, v in ZEROS], dtype=np.float32)

# Robstride RATED / PEAK leg torque [N*m] by motor-class suffix.
RATED = {"_04": 40.0, "_03": 20.0, "_02": 6.0}
PEAK = {"_04": 120.0, "_03": 60.0, "_02": 17.0}
SERVO_NM = 0.981  # arm/neck Feetech


# ----------------------------- network (copied from train_qbot) ---------------
class Actor(eqx.Module):
    input_proj: eqx.nn.Linear
    rnns: tuple
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(self, key, *, num_inputs, num_outputs, min_std, max_std,
                 var_scale, hidden_size, num_mixtures, depth):
        key, ik = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(num_inputs, hidden_size, key=ik)
        key, rk = jax.random.split(key)
        self.rnns = tuple(eqx.nn.GRUCell(hidden_size, hidden_size, key=k)
                          for k in jax.random.split(rk, depth))
        self.output_proj = eqx.nn.Linear(hidden_size, num_outputs * 3 * num_mixtures, key=key)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n, carry):
        x = self.input_proj(obs_n)
        outs = []
        for i, rnn in enumerate(self.rnns):
            x = rnn(x, carry[i])
            outs.append(x)
        out = self.output_proj(x)
        sl = self.num_outputs * self.num_mixtures
        mean = out[..., :sl].reshape(self.num_outputs, self.num_mixtures)
        std = out[..., sl:sl * 2].reshape(self.num_outputs, self.num_mixtures)
        logits = out[..., sl * 2:].reshape(self.num_outputs, self.num_mixtures)
        std = jnp.clip((jax.nn.softplus(std) + self.min_std) * self.var_scale, max=self.max_std)
        mean = mean + jnp.array(ZEROS_BIAS)[:, None]
        dist = ksim.MixtureOfGaussians(means_nm=mean, stds_nm=std, logits_nm=logits)
        return dist, jnp.stack(outs, axis=0)


class Critic(eqx.Module):
    input_proj: eqx.nn.Linear
    rnns: tuple
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(self, key, *, num_inputs, hidden_size, depth):
        key, ik = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(num_inputs, hidden_size, key=ik)
        key, rk = jax.random.split(key)
        self.rnns = tuple(eqx.nn.GRUCell(hidden_size, hidden_size, key=k)
                          for k in jax.random.split(rk, depth))
        self.output_proj = eqx.nn.Linear(hidden_size, 1, key=key)
        self.num_inputs = num_inputs

    def forward(self, obs_n, carry):
        x = self.input_proj(obs_n)
        outs = []
        for i, rnn in enumerate(self.rnns):
            x = rnn(x, carry[i])
            outs.append(x)
        return self.output_proj(x), jnp.stack(outs, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(self, key, *, num_actor_inputs, num_actor_outputs, num_critic_inputs,
                 min_std, max_std, var_scale, hidden_size, num_mixtures, depth):
        ak, ck = jax.random.split(key)
        self.actor = Actor(ak, num_inputs=num_actor_inputs, num_outputs=num_actor_outputs,
                           min_std=min_std, max_std=max_std, var_scale=var_scale,
                           hidden_size=hidden_size, num_mixtures=num_mixtures, depth=depth)
        self.critic = Critic(ck, num_inputs=num_critic_inputs, hidden_size=hidden_size, depth=depth)


HIDDEN, DEPTH, MIX = 128, 5, 5
NJ = 18
NUM_ACTOR_IN = 2 + 2 * NJ + 3 + 6  # 47


def load_policy(ckpt_path):
    template = Model(jax.random.PRNGKey(0), num_actor_inputs=NUM_ACTOR_IN,
                     num_actor_outputs=NJ, num_critic_inputs=712, min_std=0.001,
                     max_std=1.0, var_scale=0.5, hidden_size=HIDDEN,
                     num_mixtures=MIX, depth=DEPTH)
    model = load_ckpt(ckpt_path, part="model", model_templates=[template])[0]
    return model.actor


@eqx.filter_jit
def policy_step(actor, t, qpos_j, qvel_j, quat, acc, gyro, carry):
    gravity = jnp.array([0.0, 0.0, -9.81])
    proj_grav = xax.rotate_vector_by_quat(gravity, quat, inverse=True)
    obs = jnp.concatenate([jnp.sin(t)[None], jnp.cos(t)[None], qpos_j, qvel_j,
                           proj_grav, acc, gyro])
    dist, carry = actor.forward(obs, carry)
    return dist.mode(), carry


def make_env():
    m = mujoco_scenes.mjcf.load_mjmodel(os.path.join(HERE, "qbot.xml"), scene="smooth")
    m.opt.timestep = 0.002
    d = mujoco.MjData(m)
    # init to the start pose
    mujoco.mj_resetData(m, d)
    d.qpos[7:7 + NJ] = ZEROS_BIAS
    mujoco.mj_forward(m, d)
    return m, d


def sensor_adr(m, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return m.sensor_adr[sid], m.sensor_dim[sid]


def torque_limits(m):
    """Per-actuator (rated, peak) torque for plotting."""
    rated, peak = [], []
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        base = name.replace("_ctrl", "")
        if name.startswith("dof_"):
            r = p = None
            for suf in RATED:
                if base.endswith(suf):
                    r, p = RATED[suf], PEAK[suf]
            rated.append(r or SERVO_NM); peak.append(p or SERVO_NM)
        else:
            rated.append(SERVO_NM); peak.append(SERVO_NM)
    return np.array(rated), np.array(peak)


def joint_short_names(m):
    out = []
    for i in range(m.nu):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").replace("_ctrl", "")
        n = n.replace("dof_", "").replace("ub_", "")
        out.append(n)
    return out


def _reset(m, d):
    mujoco.mj_resetData(m, d)
    d.qpos[7:7 + NJ] = ZEROS_BIAS
    mujoco.mj_forward(m, d)


def _make_control_step(m, d, actor, ctrl_lo, ctrl_hi):
    acc_a, _ = sensor_adr(m, "imu_acc")
    gyro_a, _ = sensor_adr(m, "imu_gyro")
    quat_a, _ = sensor_adr(m, "imu_site_quat")
    n_sub = 10  # ctrl_dt 0.02 / dt 0.002

    def control_step(carry):
        t = jnp.array(d.time, dtype=jnp.float32)
        qpos_j = jnp.array(d.qpos[7:7 + NJ], dtype=jnp.float32)
        qvel_j = jnp.array(d.qvel[6:6 + NJ], dtype=jnp.float32)
        quat = jnp.array(d.sensordata[quat_a:quat_a + 4], dtype=jnp.float32)
        acc = jnp.array(d.sensordata[acc_a:acc_a + 3], dtype=jnp.float32)
        gyro = jnp.array(d.sensordata[gyro_a:gyro_a + 3], dtype=jnp.float32)
        action, carry = policy_step(actor, t, qpos_j, qvel_j, quat, acc, gyro, carry)
        d.ctrl[:] = np.clip(np.array(action), ctrl_lo, ctrl_hi)
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        return carry

    return control_step


def _make_torque_fig(m, rated, peak, names, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    if getattr(fig.canvas, "manager", None):
        fig.canvas.manager.set_window_title(title)
    yy = np.arange(m.nu)
    bars = ax.barh(yy, np.zeros(m.nu), color="#44cc88")
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 130); ax.set_xlabel("|torque| (N·m)")
    for i in range(m.nu):
        ax.plot([rated[i], rated[i]], [i - 0.4, i + 0.4], color="orange", lw=1.2)
        ax.plot([peak[i], peak[i]], [i - 0.4, i + 0.4], color="red", lw=1.2)
    ax.set_title("joint torque | orange=rated  red=peak", fontsize=10)
    fig.tight_layout()
    return fig, bars


def _update_bars(d, m, bars, rated, peak):
    tau = np.abs(d.actuator_force[:m.nu])
    for i, b in enumerate(bars):
        b.set_width(tau[i])
        b.set_color("#ee4444" if tau[i] > peak[i]
                    else ("#ffbb33" if tau[i] > rated[i] + 1e-6 else "#44cc88"))
    return tau


def run_live(m, d, control_step, carry, rated, peak, names):
    """Native MuJoCo interactive viewer (rotate/zoom) + live joint-torque window."""
    import time
    import matplotlib.pyplot as plt
    import mujoco.viewer

    fig, bars = _make_torque_fig(m, rated, peak, names, "Q-BOT joint torque")
    plt.ion(); plt.show(block=False)
    print("MuJoCo viewer: drag = rotate, scroll = zoom, ctrl+drag = apply force. "
          "Close the 3D window to exit.")

    step = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        wall0 = time.time() - d.time
        while viewer.is_running():
            carry = control_step(carry)
            viewer.sync()
            step += 1
            if step % 2 == 0:  # ~25 Hz plot refresh
                tau = _update_bars(d, m, bars, rated, peak)
                fig.suptitle(f"t={d.time:5.2f}s  x={d.qpos[0]:+5.2f}m  "
                             f"vx={d.qvel[0]:+4.2f} m/s  max τ={tau.max():4.1f} N·m",
                             fontsize=9)
                fig.canvas.draw_idle(); fig.canvas.flush_events()
            if d.qpos[2] < 0.5:  # fell -> auto-reset and keep going
                _reset(m, d); carry = jnp.zeros((DEPTH, HIDDEN)); wall0 = time.time()
            # pace to real time
            sleep = (wall0 + d.time) - time.time()
            if sleep > 0:
                time.sleep(sleep)
    plt.close(fig)
    print("viewer closed.")


def run_video(m, d, control_step, carry, rated, peak, names, seconds, out_path):
    """Headless: render robot + torque bars baked into an mp4."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    renderer = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultFreeCamera(m, cam); cam.distance = 3.5
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.4),
                                         gridspec_kw={"width_ratios": [1.25, 1]})
    ax_img.axis("off")
    im = ax_img.imshow(np.zeros((480, 640, 3), np.uint8))
    yy = np.arange(m.nu)
    bars = ax_bar.barh(yy, np.zeros(m.nu), color="#44cc88")
    ax_bar.set_yticks(yy); ax_bar.set_yticklabels(names, fontsize=7); ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 130); ax_bar.set_xlabel("|torque| (N·m)")
    for i in range(m.nu):
        ax_bar.plot([rated[i], rated[i]], [i - 0.4, i + 0.4], color="orange", lw=1)
        ax_bar.plot([peak[i], peak[i]], [i - 0.4, i + 0.4], color="red", lw=1)
    ax_bar.set_title("torque | orange=rated  red=peak", fontsize=9)
    title = fig.suptitle("")
    frames = []
    for step in range(int(seconds / 0.02)):
        carry = control_step(carry)
        if step % 2 == 0:
            cam.lookat[:] = d.qpos[:3]
            renderer.update_scene(d, cam); im.set_data(renderer.render())
            tau = _update_bars(d, m, bars, rated, peak)
            title.set_text(f"t={d.time:5.2f}s   x={d.qpos[0]:+5.2f}m   "
                           f"vx={d.qvel[0]:+4.2f} m/s   max τ={tau.max():5.1f} N·m")
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if d.qpos[2] < 0.5:
            print(f"fell at t={d.time:.2f}s, x={d.qpos[0]:.2f}m"); break
    renderer.close()
    out = out_path or os.path.join(HERE, "rollout.mp4")
    try:
        imageio.mimsave(out, frames, fps=25, macro_block_size=1)
    except Exception:
        out = out.replace(".mp4", ".gif"); imageio.mimsave(out, frames, fps=25)
    print("saved", out)


def run(live=True, seconds=15.0, out_path=None):
    m, d = make_env()
    actor = load_policy(os.path.join(HERE, "policy.bin"))
    ctrl_lo = m.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = m.actuator_ctrlrange[:, 1].copy()
    rated, peak = torque_limits(m)
    names = joint_short_names(m)
    carry = jnp.zeros((DEPTH, HIDDEN))
    control_step = _make_control_step(m, d, actor, ctrl_lo, ctrl_hi)
    if live:
        run_live(m, d, control_step, carry, rated, peak, names)
    else:
        run_video(m, d, control_step, carry, rated, peak, names, seconds, out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="headless render to mp4 instead of live viewer")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    run(live=not args.video, seconds=args.seconds)
