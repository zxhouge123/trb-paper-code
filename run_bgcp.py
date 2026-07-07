# -*- coding: utf-8 -*-
"""离线BGCP批量实验入口: 跑不同mask场景(point/block)/不同rank,结果汇总到一张csv。
与因果版(run_causal_bgcp.py)输出列完全一致,共用同一份summary.csv不会错位。"""
import os, re, json, time, argparse
import numpy as np
import pandas as pd
from bgcp_gpu import BGCP_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="speed_tensor.npz")
ap.add_argument("--mask_npz", default="./mask/masks_block_30.npz",
                 help="point/block等挖洞mask文件,和因果版共用同一批")
ap.add_argument("--rank", type=int, default=10)
ap.add_argument("--burn_iter", type=int, default=1000)
ap.add_argument("--gibbs_iter", type=int, default=500)
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", default="float64")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="", help="场景标签,留空则从mask_npz文件名自动推导")
ap.add_argument("--out_dir", default="results_interp")
ap.add_argument("--summary_csv", default="results_interp/summary.csv")
args = ap.parse_args()

if not args.tag:
    fname = os.path.splitext(os.path.basename(args.mask_npz))[0]
    m = re.search(r"(point|block)[_-]?(\d+)", fname, flags=re.IGNORECASE)
    args.tag = f"{m.group(1).lower()}_{m.group(2)}" if m else fname
print(f"[tag] 自动识别为: {args.tag}  (来自文件: {args.mask_npz})")

D0 = np.load(args.data, allow_pickle=True)
truth = np.nan_to_num(D0["speed"].astype(np.float64), nan=0.0)
obs_mask = D0["obs_mask"].astype(np.float64)

M = np.load(args.mask_npz)
binary = M["binary_mask"].astype(np.float64)
eval_mask = M["eval_mask"].astype(bool)
assert (binary <= obs_mask).all(), "binary必须是obs_mask的子集"

sparse = truth * binary

t0 = time.time()
hat = BGCP_gpu(sparse, binary, rank=args.rank, burn_iter=args.burn_iter,
               gibbs_iter=args.gibbs_iter, device=args.device, dtype=args.dtype,
               seed=args.seed, verbose=True)
dt = time.time() - t0


def evaluate(pred, truth, mask):
    err = (pred - truth)[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(truth[mask]) + 1e-6)) * 100)
    return rmse, mae, mape


r = evaluate(hat, truth, eval_mask)
print(f"\n==== BGCP(offline) [{args.tag}] DONE ({dt:.0f}s) ====")
print(f"RMSE={r[0]:.4f}  MAE={r[1]:.4f}  MAPE={r[2]:.4f}%")

os.makedirs(args.out_dir, exist_ok=True)
out_path = f"{args.out_dir}/bgcp_offline_{args.tag}_r{args.rank}_seed{args.seed}.npz"
np.savez_compressed(out_path, completion=hat, truth=truth, obs_mask=obs_mask,
                     binary_mask=binary, eval_mask=eval_mask,
                     config=json.dumps(vars(args), ensure_ascii=False), elapsed_sec=dt)
print("saved:", out_path)

# ---- 追加汇总表: 与因果版完全相同的列 ----
os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
params_common = dict(burn_iter=args.burn_iter, gibbs_iter=args.gibbs_iter)
row = dict(method="bgcp_offline", tag=args.tag, rank=args.rank, seed=args.seed,
           rmse=r[0], mae=r[1], mape=r[2],
           elapsed_sec=round(dt, 1), params=json.dumps(params_common, ensure_ascii=False),
           out_path=out_path, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
COLS = ["method", "tag", "rank", "seed", "rmse", "mae", "mape",
        "elapsed_sec", "params", "out_path", "timestamp"]
pd.DataFrame([row])[COLS].to_csv(args.summary_csv, mode="a",
                                   header=not os.path.exists(args.summary_csv), index=False)
print("汇总已写入:", args.summary_csv)