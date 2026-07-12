# -*- coding: utf-8 -*-
"""Measure the joint speeds (rad/s) the trained policy actually commands,
to compare against the Feetech STS3215 servo limit.

No-load: 0.222 s / 60deg  ->  1.0472/0.222 = 4.72 rad/s. Conservative (loaded,
not always ideal) ~3.0 rad/s.
"""
import os
import numpy as np
import jax.numpy as jnp
import mujoco
import sim2sim as S

NOLOAD = 1.0472 / 0.222   # 4.72 rad/s
CONSERV = 3.0

m, d = S.make_env()
actor = S.load_policy(os.path.join(S.HERE, "policy.bin"))
carry = jnp.zeros((S.DEPTH, S.HIDDEN))
acc_a, _ = S.sensor_adr(m, "imu_acc")
gyro_a, _ = S.sensor_adr(m, "imu_gyro")
quat_a, _ = S.sensor_adr(m, "imu_site_quat")
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]) for i in range(m.nu)]

maxv = np.zeros(m.nu)
allv = []
for step in range(750):  # 15 s
    t = jnp.array(d.time, jnp.float32)
    qpos_j = jnp.array(d.qpos[7:7 + S.NJ], jnp.float32)
    qvel_j = jnp.array(d.qvel[6:6 + S.NJ], jnp.float32)
    quat = jnp.array(d.sensordata[quat_a:quat_a + 4], jnp.float32)
    acc = jnp.array(d.sensordata[acc_a:acc_a + 3], jnp.float32)
    gyro = jnp.array(d.sensordata[gyro_a:gyro_a + 3], jnp.float32)
    action, carry = S.policy_step(actor, t, qpos_j, qvel_j, quat, acc, gyro, carry)
    d.ctrl[:] = np.clip(np.array(action), lo, hi)
    for _ in range(10):
        mujoco.mj_step(m, d)
        v = np.abs(d.qvel[6:6 + S.NJ])
        maxv = np.maximum(maxv, v)
        allv.append(v.copy())
    if d.qpos[2] < 0.5:
        print(f"fell at t={d.time:.2f}s")
        break

allv = np.array(allv)
print(f"\nservo no-load = {NOLOAD:.2f} rad/s   conservative cap = {CONSERV:.2f} rad/s\n")
print(f"{'joint':24} {'max':>6} {'p95':>6} {'p99':>6}   (rad/s)")
for i in range(m.nu):
    tag = "ARM " if names[i].startswith("ub_") else "leg "
    flag = ""
    if names[i].startswith("ub_"):
        if maxv[i] > NOLOAD:
            flag = "  <-- EXCEEDS no-load!"
        elif maxv[i] > CONSERV:
            flag = "  <-- over conservative cap"
    print(f"{tag}{names[i]:20} {maxv[i]:6.2f} {np.percentile(allv[:, i], 95):6.2f} "
          f"{np.percentile(allv[:, i], 99):6.2f}{flag}")
arm = [i for i in range(m.nu) if names[i].startswith("ub_")]
print(f"\nARM peak  = {maxv[arm].max():.2f} rad/s")
print(f"ARM p99   = {np.percentile(allv[:, arm], 99):.2f} rad/s")
