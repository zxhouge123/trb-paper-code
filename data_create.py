import numpy as np 
from  creat_missingpoint import inject_point_missing, inject_block_missing



X = np.load("speed_tensor.npz", allow_pickle=True)
truth = np.nan_to_num(X["speed"].astype(np.float64), nan=0.0)
obs_mask = X["obs_mask"].astype(np.float32)

missing_rate=0.3

# 分别生成两种场景的mask,各自存一份
eval_mask_pt, binary_mask_pt = inject_point_missing(obs_mask, missing_rate=missing_rate, seed=42)
eval_mask_bk, binary_mask_bk = inject_block_missing(obs_mask, missing_rate=missing_rate, block_len=48, seed=42)


np.savez_compressed(f"./mask/masks_point_{int(missing_rate*100)}.npz", eval_mask=eval_mask_pt, binary_mask=binary_mask_pt,
                     truth=truth, obs_mask=obs_mask)
np.savez_compressed(f"./mask/masks_block_{int(missing_rate*100)}.npz", eval_mask=eval_mask_bk, binary_mask=binary_mask_bk,
                     truth=truth, obs_mask=obs_mask)