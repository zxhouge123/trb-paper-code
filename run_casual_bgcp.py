# -*- coding: utf-8 -*-
"""因果BGCP批量实验入口: 跑不同mask场景(point/block)/不同rank,结果汇总到一张csv。"""
import os, re, json, time, argparse
import numpy as np
import pandas as pd
from causal_bgcp import CausalBGCP, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="speed_tensor.npz")
ap.add_argument("--mask_npz", default="./mask/masks_point_30.npz",
                 help="point/block等挖洞mask文件")
ap.add_argument("--rank", type=int, default=50)
ap.add_argument("--warmup", type=int, default=14)
ap.add_argument("--warm_burn", type=int, default=300)
ap.add_argument("--warm_sample", type=int, default=100)
ap.add_argument("--daily_refresh", type=int, default=8)
ap.add_argument("--n_samples", type=int, default=20)
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

model = CausalBGCP(sparse, binary, rank=args.rank, warmup_days=args.warmup,
                    warm_burn=args.warm_burn, warm_sample=args.warm_sample,
                    daily_refresh=args.daily_refresh, n_samples=args.n_samples,
                    seed=args.seed, verbose=True)
t0 = time.time()
res = model.run()
dt = time.time() - t0

day_ok = np.zeros_like(eval_mask)
day_ok[:, args.warmup:, :] = True
em = eval_mask & day_ok
r_strict = evaluate(res["completion_strict"], truth, em)
r_online = evaluate(res["completion_online"], truth, em)

print(f"\n==== CausalBGCP [{args.tag}] DONE ({dt:.0f}s) ====")
print(f"strict RMSE={r_strict[0]:.4f} MAE={r_strict[1]:.4f} MAPE={r_strict[2]:.4f}%")
print(f"online RMSE={r_online[0]:.4f} MAE={r_online[1]:.4f} MAPE={r_online[2]:.4f}%")

os.makedirs(args.out_dir, exist_ok=True)
out_path = f"{args.out_dir}/causal_bgcp_{args.tag}_r{args.rank}_seed{args.seed}.npz"
np.savez_compressed(out_path, **res, truth=truth, obs_mask=obs_mask,
                     binary_mask=binary, eval_mask=eval_mask,
                     config=json.dumps(vars(args), ensure_ascii=False), elapsed_sec=dt)
print("saved:", out_path)

# ---- 追加汇总表: 固定列,方法专属超参数打包进params列 ----
os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)
params_common = dict(warmup=args.warmup, warm_burn=args.warm_burn,
                      warm_sample=args.warm_sample, daily_refresh=args.daily_refresh,
                      n_samples=args.n_samples)
ts = time.strftime("%Y-%m-%d %H:%M:%S")
rows = [
    dict(method="causal_bgcp_strict", tag=args.tag, rank=args.rank, seed=args.seed,
         rmse=r_strict[0], mae=r_strict[1], mape=r_strict[2],
         elapsed_sec=round(dt, 1), params=json.dumps(params_common, ensure_ascii=False),
         out_path=out_path, timestamp=ts),
    dict(method="causal_bgcp_online", tag=args.tag, rank=args.rank, seed=args.seed,
         rmse=r_online[0], mae=r_online[1], mape=r_online[2],
         elapsed_sec=round(dt, 1), params=json.dumps(params_common, ensure_ascii=False),
         out_path=out_path, timestamp=ts),
]
COLS = ["method", "tag", "rank", "seed", "rmse", "mae", "mape",
        "elapsed_sec", "params", "out_path", "timestamp"]
pd.DataFrame(rows)[COLS].to_csv(args.summary_csv, mode="a",
                                  header=not os.path.exists(args.summary_csv), index=False)
print("汇总已写入:", args.summary_csv)