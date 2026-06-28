"""合併多個 chunk 的 retarget_poses_g1.h5 和環境點雲，用於完整播放"""
import h5py, numpy as np, sys, os
from pathlib import Path

REAL2SIM   = sys.argv[1]
FULL_NAME  = sys.argv[2]
VIMO_START = int(sys.argv[3])
VIMO_END   = int(sys.argv[4])
CHUNK      = int(sys.argv[5])
MERGE_DIR  = sys.argv[6]

os.makedirs(MERGE_DIR, exist_ok=True)
os.chdir(REAL2SIM)

sys.path.insert(0, REAL2SIM)
from stage2_optimization.megahunter_utils import load_dict_from_hdf5, save_dict_to_hdf5

# ─── 合併 retarget_poses_g1.h5 ────────────────────────────────────────────
print("合併 retarget_poses_g1.h5...")
all_joints, all_root_pos, all_root_quat = [], [], []
all_lf, all_rf = [], []

for chunk_start in range(VIMO_START, VIMO_END + 1, CHUNK):
    chunk_end = min(chunk_start + CHUNK - 1, VIMO_END)
    chunk_name = f"{FULL_NAME}_chunk{chunk_start}"
    retarget = (f"demo_data/output_calib_mesh/"
                f"megahunter_megasam_reconstruction_results_{chunk_name}_cam01"
                f"_frame_0_{CHUNK}_subsample_1/retarget_poses_g1.h5")
    if not os.path.exists(retarget):
        print(f"  跳過（未找到）: {retarget}")
        continue
    with h5py.File(retarget, 'r') as f:
        all_joints.append(f['joints'][:])
        all_root_pos.append(f['root_pos'][:])
        all_root_quat.append(f['root_quat'][:])
        if 'contacts/left_foot' in f:
            all_lf.append(f['contacts/left_foot'][:])
            all_rf.append(f['contacts/right_foot'][:])
    print(f"  chunk {chunk_start}: {all_joints[-1].shape[0]} 幀")

if not all_joints:
    print("沒有找到任何 chunk！")
    sys.exit(1)

joints_all = np.concatenate(all_joints)
root_pos_all = np.concatenate(all_root_pos)
root_quat_all = np.concatenate(all_root_quat)
T_total = joints_all.shape[0]
print(f"合計: {T_total} 幀")

out_retarget = os.path.join(MERGE_DIR, 'retarget_poses_g1.h5')
with h5py.File(out_retarget, 'w') as f:
    f.create_dataset('joints', data=joints_all)
    f.create_dataset('root_pos', data=root_pos_all)
    f.create_dataset('root_quat', data=root_quat_all)
    if all_lf:
        g = f.create_group('contacts')
        g.create_dataset('left_foot', data=np.concatenate(all_lf))
        g.create_dataset('right_foot', data=np.concatenate(all_rf))
print(f"retarget_poses_g1.h5 → {out_retarget}")

# ─── 合併環境點雲 h5 ──────────────────────────────────────────────────────
print("\n合併環境點雲...")
all_frames = {}
for chunk_start in range(VIMO_START, VIMO_END + 1, CHUNK):
    chunk_name = f"{FULL_NAME}_chunk{chunk_start}"
    env_h5 = (f"demo_data/input_megasam/"
              f"megasam_reconstruction_results_{chunk_name}_cam01"
              f"_frame_0_{CHUNK}_subsample_1.h5")
    if not os.path.exists(env_h5):
        print(f"  跳過環境: {env_h5}")
        continue
    with h5py.File(env_h5, 'r') as f:
        d = load_dict_from_hdf5(f)
    world = d['monst3r_ga_output']
    fkeys = sorted(world.keys())
    for i, fk in enumerate(fkeys):
        global_idx = chunk_start + i
        all_frames[f'{global_idx:05d}'] = world[fk]
    print(f"  chunk {chunk_start}: {len(fkeys)} 幀")

out_env = os.path.join(MERGE_DIR, 'env_merged.h5')
with h5py.File(out_env, 'w') as f:
    save_dict_to_hdf5(f, {'monst3r_ga_output': all_frames})
print(f"env_merged.h5 → {out_env} ({len(all_frames)} 幀)")

# ─── 合併 gravity_calibrated_megahunter.h5 ────────────────────────────────
# 需要 retargeting_visualization 讀取 SMPL mesh
print("\n合併 gravity_calibrated_megahunter.h5...")
merged_world = {}
merged_smpl  = {}
pid_global   = None

for chunk_start in range(VIMO_START, VIMO_END + 1, CHUNK):
    chunk_name = f"{FULL_NAME}_chunk{chunk_start}"
    calib_h5 = (f"demo_data/output_calib_mesh/"
                f"megahunter_megasam_reconstruction_results_{chunk_name}_cam01"
                f"_frame_0_{CHUNK}_subsample_1/gravity_calibrated_megahunter.h5")
    if not os.path.exists(calib_h5):
        continue
    with h5py.File(calib_h5, 'r') as f:
        d = load_dict_from_hdf5(f)
    
    # world cameras
    world = d.get('our_pred_world_cameras_and_structure', {})
    for i, fk in enumerate(sorted(world.keys())):
        gfk = f'{chunk_start+i:05d}'
        merged_world[gfk] = world[fk]
    
    # SMPL params
    smpl = d.get('our_pred_humans_smplx_params', {})
    for pid, params in smpl.items():
        if pid_global is None:
            pid_global = pid
            merged_smpl[pid] = {k: [] for k in params if isinstance(params[k], np.ndarray)}
            merged_smpl[pid]['betas'] = params['betas']
        for k in merged_smpl[pid]:
            if k == 'betas': continue
            merged_smpl[pid][k].append(params[k])

if pid_global and merged_smpl[pid_global]:
    for k in merged_smpl[pid_global]:
        if k == 'betas': continue
        arrs = merged_smpl[pid_global][k]
        merged_smpl[pid_global][k] = np.concatenate(arrs, axis=0)
    
    frames_all = sorted(merged_world.keys())
    T_smpl = len(merged_smpl[pid_global]['body_pose'])
    pfi = np.array(frames_all[:T_smpl], dtype='S20')
    
    out_calib = os.path.join(MERGE_DIR, 'gravity_calibrated_megahunter.h5')
    merged_data = {
        'our_pred_world_cameras_and_structure': merged_world,
        'our_pred_humans_smplx_params': merged_smpl,
        'person_frame_info_list': {pid_global: pfi},
    }
    with h5py.File(out_calib, 'w') as f:
        save_dict_to_hdf5(f, merged_data)
    print(f"gravity_calibrated_megahunter.h5 → {out_calib} ({T_smpl} 幀)")

# ─── 複製 gravity_calibrated_keypoints.h5 ────────────────────────────────
import shutil
for chunk_start in range(VIMO_START, VIMO_END + 1, CHUNK):
    chunk_name = f"{FULL_NAME}_chunk{chunk_start}"
    kp_h5 = (f"demo_data/output_calib_mesh/"
              f"megahunter_megasam_reconstruction_results_{chunk_name}_cam01"
              f"_frame_0_{CHUNK}_subsample_1/gravity_calibrated_keypoints.h5")
    if os.path.exists(kp_h5):
        shutil.copy(kp_h5, os.path.join(MERGE_DIR, 'gravity_calibrated_keypoints.h5'))
        print(f"gravity_calibrated_keypoints.h5 복사 완료")
        break

bg = os.path.join(MERGE_DIR, 'background_mesh.obj')
if not os.path.exists(bg):
    with open(bg, 'w') as f:
        f.write("# Flat ground\nv -5 -5 0\nv 5 -5 0\nv 5 5 0\nv -5 5 0\nf 1 2 3\nf 1 3 4\n")

print("\n✅ 合併完成！")
print(f"視覺化目錄: {MERGE_DIR}")
