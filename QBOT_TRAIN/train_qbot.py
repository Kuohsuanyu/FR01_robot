"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

# These are in the order of the neural network outputs, which MUST match the
# model's actuator / qpos[7:] order (the Actor adds these as a per-index position
# bias and the action is passed straight through to data.ctrl). For qbot.xml that
# order is: 10 legs first (K-Bot, Robstride), then 8 upper-body arm joints
# (Feetech servos: shoulder, lateral_raise, arm_twist, elbow). Neck is locked.
ZEROS: list[tuple[str, float]] = [
    # --- legs (K-Bot, Robstride) ---
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
    # --- upper body / arms (Feetech STS3215 servos) ---
    # Start/held pose from the interactive viewer (QBOT_MJCF/view.py): arms down
    # at the sides, elbows bent ~half range. lateral_raise signs mirror L/R;
    # arm_twist + elbow keep the same sign. lateral_raise/arm_twist are kept at
    # 80 deg (NOT the 90 deg joint limit) so the arms sit just off the hard stop
    # and the policy can explore/sway them both ways instead of being pinned.
    ("ub_right_shoulder", 0.0),
    ("ub_right_lateral_raise", math.radians(80.0)),
    ("ub_right_arm_twist", math.radians(80.0)),
    ("ub_right_elbow", math.radians(67.5)),
    ("ub_left_shoulder", 0.0),
    ("ub_left_lateral_raise", math.radians(-80.0)),
    ("ub_left_arm_twist", math.radians(80.0)),
    ("ub_left_elbow", math.radians(67.5)),
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in ZEROS}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ArmPosturePenalty(JointPositionPenalty):
    """Firmly hold lateral_raise / arm_twist / elbow at the start pose, so the
    arms stay down and the robot can't raise a hand for momentum. (Distinct class
    name from ShoulderSwingPenalty so ksim keys them separately — rewards are
    stored in a dict by class name, so two same-named penalties would collide.)"""

    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -0.25,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "ub_right_lateral_raise",
                "ub_right_arm_twist",
                "ub_right_elbow",
                "ub_left_lateral_raise",
                "ub_left_arm_twist",
                "ub_left_elbow",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ShoulderSwingPenalty(JointPositionPenalty):
    """Only lightly anchor the shoulders to neutral, so they can swing back and
    forth and produce a natural alternating arm swing while walking."""

    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -0.01,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=["ub_right_shoulder", "ub_left_shoulder"],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


# Robstride RATED (continuous) torque per leg motor class [N*m]. Peak torque
# (120/60/17) stays as the actuator's hard force limit; this penalty discourages
# operating ABOVE the rated value, and the longer it stays over, the more it
# costs (a slowly-growing "thermal" term so brief bursts are cheap).
RATED_TORQUE_NM = {"_04": 40.0, "_03": 20.0, "_02": 6.0}


@attrs.define(frozen=True, kw_only=True)
class OverRatedTorquePenalty(ksim.Reward):
    """Linear penalty on leg torque above rated, with a slow time accumulation.

    excess_t = max(0, |tau_t| - rated)             (per joint)
    heat_t   = decay * heat_{t-1} + excess_t        (accumulates while over rated)
    penalty  = sum_j ( excess + accum_rate * excess * heat )
    """

    rated_nm: tuple[float, ...]
    accum_rate: float = 0.02   # small: cost grows slowly the longer it's over
    decay: float = 0.97        # heat cools each step (so it isn't permanent)

    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -0.01,
        accum_rate: float = 0.02,
        decay: float = 0.97,
    ) -> Self:
        # rated torque per actuator, in model actuator (ctrl) order; non-leg
        # (arm) -> inf. get_rewards is called with the *mjx* physics model, which
        # has no mj_id2name, so resolve actuator names via ksim's helper (works
        # on both mujoco.MjModel and mjx.Model).
        from ksim.utils.mujoco import get_ctrl_data_idx_by_name

        name_to_idx = get_ctrl_data_idx_by_name(physics_model)
        rated = [float("inf")] * len(name_to_idx)
        for name, i in name_to_idx.items():
            if name.startswith("dof_"):  # leg (Robstride)
                for suf, val in RATED_TORQUE_NM.items():
                    if name.replace("_ctrl", "").endswith(suf):
                        rated[i] = val
                        break
        return cls(rated_nm=tuple(rated), scale=scale, accum_rate=accum_rate, decay=decay)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        tau = jnp.abs(trajectory.obs["actuator_force_observation"])  # [T, nu]
        rated = jnp.array(self.rated_nm)
        excess = jnp.maximum(tau - rated, 0.0)                       # [T, nu]

        def _scan(heat: Array, e: Array) -> tuple[Array, Array]:
            heat = self.decay * heat + e
            return heat, heat

        _, heat = jax.lax.scan(_scan, jnp.zeros(excess.shape[-1]), excess)
        penalty = excess.sum(-1) + self.accum_rate * (excess * heat).sum(-1)  # [T]
        # Return the POSITIVE penalty magnitude. ksim multiplies this by the
        # (negative) scale to get the final negative reward contribution. The
        # original code returned -penalty, which combined with the negative scale
        # flipped the sign and ended up *rewarding* over-rated torque.
        return penalty


class PassthroughPositionActuators(ksim.Actuators):
    """Self-contained position actuators for qbot.xml.

    qbot.xml already defines one ``<general>`` affine position servo per joint,
    with kp/kv and the per-motor torque (force) limits baked into the MJCF, so
    the policy's action *is* the desired joint position: we pass it straight
    through to ``data.ctrl`` (clipped to the actuator ctrlrange) and let MuJoCo's
    built-in affine transmission compute the PD torque and enforce the force
    limits. This is unlike ksim's ``PositionActuators``, which expects torque
    actuators and recomputes the PD itself from K-Scale metadata (which only
    covers kbot, not this self-contained qbot model).
    """

    def __init__(self, physics_model: ksim.PhysicsModel) -> None:
        ctrl_range = jnp.array(physics_model.actuator_ctrlrange)
        self.ctrl_min = ctrl_range[:, 0]
        self.ctrl_max = ctrl_range[:, 1]

    def get_ctrl(self, action: Array, physics_data: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        return jnp.clip(action, self.ctrl_min, self.ctrl_max)

    def get_default_action(self, physics_data: ksim.PhysicsData) -> Array:
        # Hold the reset pose on the first (and any action-latency) step.
        return physics_data.qpos[7:]


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        # Self-contained flat model: K-Bot legs + waist-leg connection + Q-BOT
        # upper body. Neck locked, IMU on the chest, position actuators with
        # per-joint torque limits baked in. mujoco_scenes adds the ground scene.
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        return mujoco_scenes.mjcf.load_mjmodel(os.path.join(here, "qbot.xml"), scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        # qbot is a self-contained model, NOT in the K-Scale registry, so we can
        # not fetch metadata via ksim.get_mujoco_model_metadata("qbot"). Build a
        # minimal Metadata straight from the MJCF actuator gains so ksim's
        # joint-config logging table is satisfied. These kp/kd are descriptive
        # only: actual control uses the MJCF's own affine PD (see get_actuators).
        joint_name_to_metadata: dict[str, ksim.JointMetadata] = {}
        for i in range(int(mj_model.nu)):
            jid = int(mj_model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if jname is None:
                continue
            kp = float(mj_model.actuator_gainprm[i, 0])     # gainprm[0]
            kd = float(-mj_model.actuator_biasprm[i, 2])    # -biasprm[2]
            joint_name_to_metadata[jname] = ksim.JointMetadata(
                kp=kp, kd=kd, actuator_type="position"
            )
        return ksim.Metadata(
            joint_name_to_metadata=joint_name_to_metadata,
            actuator_type_to_metadata={"position": ksim.ActuatorMetadata(actuator_type="position")},
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        # The MJCF already defines one <general> affine position servo per joint
        # with the correct kp/kv and force (torque) limits, so drive them
        # directly (passthrough: action -> ctrl) instead of rebuilding the PD
        # from K-Scale metadata (which only covers kbot, and would double up the
        # PD on top of the MJCF's built-in affine transmission).
        return PassthroughPositionActuators(physics_model)

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.3,
                force_range=(0.5, 1.0),
                x_angular_force=0.0,
                y_angular_force=0.0,
                z_angular_force=0.0,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            # Slow, steady walk: forward-velocity reward saturates at 0.5 m/s, so
            # there is no incentive to run faster than a gentle walking pace.
            ksim.NaiveForwardReward(clip_max=0.5, in_robot_frame=False, scale=3.0),
            ksim.NaiveForwardOrientationReward(scale=1.0),
            ksim.StayAliveReward(scale=1.0),
            ksim.UprightReward(scale=0.5),
            # Avoid movement penalties.
            ksim.AngularVelocityPenalty(index=("x", "y"), scale=-0.1),
            ksim.LinearVelocityPenalty(index=("z"), scale=-0.1),
            # Normalization penalties.
            ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            ksim.ActionAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # Bespoke rewards.
            # Arm posture: keep lateral_raise / arm_twist / elbow FIRMLY at the
            # start pose so the arms stay down and the robot can't "raise a hand"
            # for momentum (strong -0.25 anchor). Leave the shoulders only lightly
            # anchored (-0.01) so they can swing back-and-forth — this is the joint
            # that produces a natural alternating arm swing while walking.
            ArmPosturePenalty.create_penalty(physics_model, scale=-0.25),
            ShoulderSwingPenalty.create_penalty(physics_model, scale=-0.01),
            StraightLegPenalty.create_penalty(physics_model, scale=-0.1),
            # Leg motors may burst to peak, but sustained torque above the rated
            # value (RS04=40, RS03=20, RS02=6 N*m) is penalised, growing slowly
            # the longer it stays over (small accum_rate so it builds gradually).
            # scale reduced -0.01 -> -0.001: at -0.01 (with the sign now fixed)
            # this term was ~-0.2, swamping the forward reward and stopping the
            # robot from using the torque it needs to walk. -0.001 keeps it a
            # gentle nudge that grows with sustained over-torque.
            OverRatedTorquePenalty.create_penalty(
                physics_model, scale=-0.001, accum_rate=0.02, decay=0.97
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level_steps=5,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        # Actor obs = sin,cos(2) + joint_pos(NJ) + joint_vel(NJ) + proj_grav(3)
        #             [+ imu_acc(3) + imu_gyro(3) if use_acc_gyro].
        nj = len(ZEROS)  # 18 for Q-BOT (was 20 for K-Bot)
        num_actor_inputs = 2 + 2 * nj + 3 + (6 if self.config.use_acc_gyro else 0)
        # Critic adds com_inertia + com_vel (both scale with body count, which
        # differs from K-Bot). If ksim raises a shape mismatch on the critic, it
        # prints the expected length — set num_critic_inputs to that value.
        # Base (non-com) terms: 2 + 2*nj + 3+3+3 + nj + 3 + 4 = 3*nj + 18.
        # Verified for this Q-BOT model (nbody=41): base 3*nj+18 = 72, plus
        # com_inertia (nbody-1)*10 = 400 and com_vel (nbody-1)*6 = 240 -> 712.
        num_critic_inputs = 712
        return Model(
            key,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=nj,
            num_critic_inputs=num_critic_inputs,
            min_std=0.001,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        time_1 = observations["timestep_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]

        obs = [
            jnp.sin(time_1),
            jnp.cos(time_1),
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        time_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.sin(time_1),
                jnp.cos(time_1),
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_track_body_id=0,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
