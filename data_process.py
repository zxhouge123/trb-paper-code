# -*- coding: utf-8 -*-
"""原始长表(每行=一个路段+一个时间点) -> 路段x天数x时段 三维张量。
不做插值,自然缺失保留NaN,同时输出obs_mask标记哪些位置是真实观测。"""
import pandas as pd
import numpy as np

# ---------------- 参数 ----------------
FP = r"D:\桌面\trb论文代码\G20_上行_294.385-321.885.csv"          # 原始长表路径
OUT = r"D:\桌面\trb论文代码\speed_tensor.npz"      # 输出路径
WIN_START = "2021-10-20"               # 起始日期
WIN_DAYS = 73                          # 覆盖天数
SLOT_MIN = 30                          # 每个时段的分钟数(30分钟一个槽 -> 每天48个槽)

# ---------------- 读取原始数据 ----------------
df = pd.read_csv(FP)
df["time"] = pd.to_datetime(df["time"])

start = pd.Timestamp(WIN_START)
end = start + pd.Timedelta(days=WIN_DAYS)
df = df[(df["time"] >= start) & (df["time"] < end)].copy()

# ---------------- 计算 天序号(day) / 槽序号(slot) ----------------
df["day"] = (df["time"].dt.normalize() - start).dt.days
slots_per_day = 24 * 60 // SLOT_MIN
df["slot"] = (df["time"].dt.hour * 60 + df["time"].dt.minute) // SLOT_MIN

# ---------------- 路段编号 -> 张量下标 ----------------
segs = sorted(df["ldbh"].unique().tolist())          # 固定顺序,后续和邻接矩阵对齐
seg2idx = {s: i for i, s in enumerate(segs)}
df["seg_idx"] = df["ldbh"].map(seg2idx)

n_seg = len(segs)

# ---------------- 构建张量 (路段, 天, 时段) ----------------
X = np.full((n_seg, WIN_DAYS, slots_per_day), np.nan, dtype=np.float64)

# 用向量化索引一次性写入,比逐行iterrows快很多
valid = df["day"].between(0, WIN_DAYS - 1) & df["slot"].between(0, slots_per_day - 1)
d = df[valid]
X[d["seg_idx"].to_numpy(), d["day"].to_numpy(), d["slot"].to_numpy()] = d["Speed"].to_numpy(dtype=np.float64)

# ---------------- 自然缺失标记 ----------------
obs_mask = (~np.isnan(X)).astype(np.float32)   # 1=真实观测, 0=自然缺失(未插值,保持NaN)

n_missing = int(np.isnan(X).sum())
print(f"张量形状: {X.shape} (路段 x 天 x 时段)")
print(f"自然缺失点: {n_missing} / {X.size} ({100*n_missing/X.size:.3f}%)")

# ---------------- 保存 ----------------
np.savez_compressed(
    OUT,
    speed=X.astype(np.float32),
    obs_mask=obs_mask,
    segs=np.array(segs),
)
print("已保存:", OUT)