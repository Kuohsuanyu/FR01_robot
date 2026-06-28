"""合併多個 retarget_poses_g1.h5 chunk 成一個連續序列"""
import h5py, numpy as np, sys, os
from pathlib import Path

CHUNKS_DIR = sys.argv[1]
OUTPUT_H5  = sys.argv[2]

chunk_files = sorted(Path(CHUNKS_DIR).glob("chunk*.h5"))
if not chunk_files:
    print("沒有找到 chunk 檔案")
    sys.exit(1)

print(f"合併 {len(chunk_files)} 個 chunk...")

all_joints, all_root_pos, all_root_quat = [], [], []
all_link_pos, all_link_quat = [], []
all_lf_contact, all_rf_contact = [], []

for cf in chunk_files:
    with h5py.File(cf,'r') as f:
        all_joints.append(f['joints'][:])
        all_root_pos.append(f['root_pos'][:])
        all_root_quat.append(f['root_quat'][:])
        if 'link_pos' in f: all_link_pos.append(f['link_pos'][:])
        if 'link_quat' in f: all_link_quat.append(f['link_quat'][:])
        if 'contacts/left_foot' in f:
            all_lf_contact.append(f['contacts/left_foot'][:])
            all_rf_contact.append(f['contacts/right_foot'][:])
    print(f"  {cf.name}: {all_joints[-1].shape[0]} 幀")

joints_concat = np.concatenate(all_joints, axis=0)
root_pos_concat = np.concatenate(all_root_pos, axis=0)
root_quat_concat = np.concatenate(all_root_quat, axis=0)
T_total = joints_concat.shape[0]
print(f"合計: {T_total} 幀")

os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
with h5py.File(OUTPUT_H5,'w') as f:
    f.create_dataset('joints', data=joints_concat)
    f.create_dataset('root_pos', data=root_pos_concat)
    f.create_dataset('root_quat', data=root_quat_concat)
    if all_link_pos:
        f.create_dataset('link_pos', data=np.concatenate(all_link_pos, axis=0))
        f.create_dataset('link_quat', data=np.concatenate(all_link_quat, axis=0))
    if all_lf_contact:
        f.create_group('contacts')
        f['contacts'].create_dataset('left_foot', data=np.concatenate(all_lf_contact))
        f['contacts'].create_dataset('right_foot', data=np.concatenate(all_rf_contact))

print(f"儲存完成: {OUTPUT_H5}")
