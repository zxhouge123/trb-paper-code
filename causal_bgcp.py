# -*- coding: utf-8 -*-
"""因果 BGCP (Causal Bayesian Gaussian CP decomposition) 道路速度插值。

与离线 BGCP (bgcp.py / bgcp_gpu.py) 的区别:
    离线版: 一次性用全部 73 天数据(含"未来")拟合 -> 插值任意位置 => 未来信息泄露
    因果版: 沿时间轴滚动。插值 (d 天, s 槽) 的缺失时, 模型只见过全局时刻
            早于 (d,s) 的观测 => 无未来泄露, 管线可真实部署。

同一次滚动同时输出两种记录模式:
    strict : 插值 (d,s) 只用 < (d,s) 的观测        —— "只用前 t-1 时刻"
    online : 额外并入 (d,s) 同时刻其他路段的观测    —— 与 IDW/克里金信息集对齐
             ((d,s) 待插位置自身无观测, 不构成泄露)

流程:
    1) warm-up: 前 warmup_days 天全量 Gibbs (burn+sample) 收敛;
       warm-up 期缺失用 sample 段后验均值回填(只用 <=warm-up 的信息, 合法)
    2) 逐天 d: 用全部 <d 天观测热启动刷新 U/W/V[:d]/tau (daily_refresh 轮批量 Gibbs)
    3) 天内逐槽 s: 天因子 v_d 的条件后验是解析高斯(增量 QtQ/Qty), 每槽抽
       n_samples 个后验样本重构缺失 => 均值(插值) + 标准差(不确定性) + 样本矩阵
       (样本供分析插值波动、后续作为不确定性通道输入 ST-LLM)

说明: 槽内记录的 std 是"条件后验"波动(U/W/tau 取当前链状态), 是总不确定性的
下界估计; 跨天的 U/W 重采样部分弥补了因子不确定性。

纯算法模块 + 命令行入口(读 data_056, 复用 models_bgcp 的挖洞 mask)。
"""
import sys
import time
import json
import argparse

import numpy as np
import scipy.linalg as sla
from numpy.linalg import inv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------
# 基础采样工具 (与 bgcp.py / bgcp_gpu.py 同族, numpy 向量化)
# ---------------------------------------------------------------
def _khatri_rao(A, B):
    """A(I,r),B(J,r) -> (I*J,r); 行序 i 慢 j 快, 匹配 C-order reshape 展开。"""
    I, r = A.shape
    J = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, r)


def _sample_wishart(rng, df, scale):
    """Bartlett 分解采样 W ~ Wishart(df, scale)"""
    r = scale.shape[0]
    L = np.linalg.cholesky(scale)
    A = np.zeros((r, r))
    A[np.diag_indices(r)] = np.sqrt(rng.chisquare(df - np.arange(r)))
    A[np.tril_indices(r, -1)] = rng.standard_normal(r * (r - 1) // 2)
    M = L @ A
    return M @ M.T


def _mvn_prec(rng, mu, prec):
    """从 N(mu, prec^{-1}) 采样单个向量"""
    L = np.linalg.cholesky(prec)
    z = rng.standard_normal(mu.shape[0])
    return mu + sla.solve_triangular(L.T, z, lower=False)


def _gw_hyper(rng, Uk, beta0, nu0, mu0, W0_inv):
    """因子矩阵行向量的 Gaussian-Wishart 后验超参 (mu_k, Lambda_k) 采样"""
    m = Uk.shape[0]
    Ubar = Uk.mean(axis=0)
    S = (Uk - Ubar).T @ (Uk - Ubar)
    beta_post = beta0 + m
    nu_post = nu0 + m
    mu_post = (beta0 * mu0 + m * Ubar) / beta_post
    diff = mu0 - Ubar
    W_post = inv(W0_inv + S + (beta0 * m / beta_post) * np.outer(diff, diff))
    W_post = (W_post + W_post.T) / 2
    Lambda_k = _sample_wishart(rng, nu_post, W_post)
    mu_k = _mvn_prec(rng, mu_post, beta_post * Lambda_k)
    return mu_k, Lambda_k


def _batch_rows(rng, Tk, Mk, KR, Lambda_k, mu_k, tau, jit):
    """批量采样因子矩阵所有行 (照 bgcp_gpu 的批量思路翻成 numpy)。
    Tk (m,P) 展开值, Mk (m,P) 0/1 掩码, KR (P,r) Khatri-Rao 基。"""
    m = Tk.shape[0]
    P, r = KR.shape
    KRo = np.einsum("pr,ps->prs", KR, KR).reshape(P, r * r)
    prec = Lambda_k[None] + tau * (Mk @ KRo).reshape(m, r, r) + jit[None]
    rhs = (Lambda_k @ mu_k)[None] + tau * ((Mk * Tk) @ KR)
    mean = np.linalg.solve(prec, rhs[..., None])[..., 0]
    L = np.linalg.cholesky(prec)
    z = rng.standard_normal((m, r, 1))
    pert = np.linalg.solve(np.transpose(L, (0, 2, 1)), z)[..., 0]
    return mean + pert


# ---------------------------------------------------------------
# 因果 BGCP 主类
# ---------------------------------------------------------------
class CausalBGCP:
    def __init__(self, sparse, binary, rank=50, warmup_days=14,
                 warm_burn=300, warm_sample=100, daily_refresh=8,
                 n_samples=20, seed=0, verbose=True):
        """
        sparse : (N,D,S) 缺失处0, 观测处真实值
        binary : (N,D,S) 1=训练可见观测, 0=缺失(含人工挖洞+自然缺失)
        """
        assert sparse.shape == binary.shape and sparse.ndim == 3
        self.sparse = sparse.astype(np.float64)
        self.binary = binary.astype(np.float64)
        self.N, self.D, self.S = sparse.shape
        self.r = rank
        self.warmup_days = warmup_days
        self.warm_burn = warm_burn
        self.warm_sample = warm_sample
        self.daily_refresh = daily_refresh
        self.n_samples = n_samples
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        # 先验超参 (与 bgcp.py 完全一致)
        self.beta0, self.nu0 = 1.0, rank
        self.mu0 = np.zeros(rank)
        self.W0_inv = np.eye(rank)
        self.a0 = self.b0 = 1e-6
        self.tau = 1.0
        self.JIT = 1e-6 * np.eye(rank)

        # 因子矩阵
        self.U = 0.1 * self.rng.standard_normal((self.N, rank))
        self.V = 0.1 * self.rng.standard_normal((self.D, rank))
        self.W = 0.1 * self.rng.standard_normal((self.S, rank))
        self.mu_v = np.zeros(rank)          # 天因子先验 (日刷新时更新)
        self.Lambda_v = np.eye(rank)

    # ---------------- 全量 Gibbs 一轮 (数据 = 前 d_lim 天) ----------------
    def _sweep(self, d_lim):
        sp = self.sparse[:, :d_lim, :]
        bi = self.binary[:, :d_lim, :]
        rng, jit = self.rng, self.JIT

        # mode-0: 路段因子 U
        mu_k, Lam_k = _gw_hyper(rng, self.U, self.beta0, self.nu0, self.mu0, self.W0_inv)
        KR = _khatri_rao(self.V[:d_lim], self.W)
        self.U = _batch_rows(rng, sp.reshape(self.N, -1), bi.reshape(self.N, -1),
                             KR, Lam_k, mu_k, self.tau, jit)

        # mode-1: 天因子 V[:d_lim] (其超参即新一天 v_d 的先验)
        self.mu_v, self.Lambda_v = _gw_hyper(rng, self.V[:d_lim], self.beta0,
                                             self.nu0, self.mu0, self.W0_inv)
        KR = _khatri_rao(self.U, self.W)
        Tk = np.moveaxis(sp, 1, 0).reshape(d_lim, -1)
        Mk = np.moveaxis(bi, 1, 0).reshape(d_lim, -1)
        self.V[:d_lim] = _batch_rows(rng, Tk, Mk, KR, self.Lambda_v, self.mu_v,
                                     self.tau, jit)

        # mode-2: 槽因子 W
        mu_k, Lam_k = _gw_hyper(rng, self.W, self.beta0, self.nu0, self.mu0, self.W0_inv)
        KR = _khatri_rao(self.U, self.V[:d_lim])
        Tk = np.moveaxis(sp, 2, 0).reshape(self.S, -1)
        Mk = np.moveaxis(bi, 2, 0).reshape(self.S, -1)
        self.W = _batch_rows(rng, Tk, Mk, KR, Lam_k, mu_k, self.tau, jit)

        # 精度 tau
        hat = np.einsum("ir,jr,kr->ijk", self.U, self.V[:d_lim], self.W)
        obs = bi.astype(bool)
        err = (sp - hat)[obs]
        n_obs = obs.sum()
        self.tau = rng.gamma(self.a0 + 0.5 * n_obs,
                             1.0 / (self.b0 + 0.5 * np.sum(err ** 2)))
        return float(np.sqrt(np.mean(err ** 2)))

    # ---------------- warm-up ----------------
    def _warmup(self):
        dW = self.warmup_days
        t0 = time.time()
        hat_sum = np.zeros((self.N, dW, self.S))
        hat_sq = np.zeros_like(hat_sum)
        for it in range(self.warm_burn + self.warm_sample):
            rmse = self._sweep(dW)
            if it + 1 > self.warm_burn:
                hat = np.einsum("ir,jr,kr->ijk", self.U, self.V[:dW], self.W)
                hat_sum += hat
                hat_sq += hat ** 2
            if self.verbose and (it + 1) % 100 == 0:
                print(f"  [warm-up] iter {it+1}/{self.warm_burn+self.warm_sample}"
                      f"  obs-RMSE={rmse:.4f}  tau={self.tau:.4f}  {time.time()-t0:.0f}s",
                      flush=True)
        self.warm_mean = hat_sum / self.warm_sample
        var = np.maximum(hat_sq / self.warm_sample - self.warm_mean ** 2, 0.0)
        self.warm_std = np.sqrt(var)

    # ---------------- 天因子条件后验采样 (解析高斯) ----------------
    def _vd_posterior_samples(self, QtQ, Qty, n):
        prec = self.Lambda_v + self.tau * QtQ + self.JIT
        rhs = self.Lambda_v @ self.mu_v + self.tau * Qty
        L = np.linalg.cholesky(prec)
        mean = sla.cho_solve((L, True), rhs)
        Z = self.rng.standard_normal((self.r, n))
        pert = sla.solve_triangular(L.T, Z, lower=False)
        return mean[None, :] + pert.T          # (n, r)

    # ---------------- 主流程 ----------------
    def run(self):
        N, D, S, r = self.N, self.D, self.S, self.r
        dW = self.warmup_days
        if self.verbose:
            print(f"CausalBGCP: 张量 {self.sparse.shape}  rank={r}  warm-up {dW} 天"
                  f"  daily_refresh={self.daily_refresh}  n_samples={self.n_samples}",
                  flush=True)
        self._warmup()

        # 输出容器 (观测处=原值, 缺失处填估计)
        comp_strict = self.sparse.copy()
        comp_online = self.sparse.copy()
        std_strict = np.zeros((N, D, S))
        std_online = np.zeros((N, D, S))
        # warm-up 期缺失回填 (使用 warm-up 后验, 只含 <=warm-up 的信息)
        miss_w = self.binary[:, :dW, :] == 0
        comp_strict[:, :dW, :][miss_w] = self.warm_mean[miss_w]
        comp_online[:, :dW, :][miss_w] = self.warm_mean[miss_w]
        std_strict[:, :dW, :][miss_w] = self.warm_std[miss_w]
        std_online[:, :dW, :][miss_w] = self.warm_std[miss_w]

        samples_list = []      # 逐缺失点后验样本 (strict)
        idx_list = []          # 对应 (n, d, s)
        tau_trace = []

        t0 = time.time()
        rmse = float("nan")
        for d in range(dW, D):
            # ---- 日刷新: 只用 <d 天的数据 ----
            for _ in range(self.daily_refresh):
                rmse = self._sweep(d)
            tau_trace.append(self.tau)

            # ---- 天内逐槽滚动 ----
            QtQ = np.zeros((r, r))
            Qty = np.zeros(r)
            for s in range(S):
                miss = self.binary[:, d, s] == 0
                obs = ~miss

                if miss.any():
                    # strict: 条件于 槽<s (QtQ 尚未纳入槽 s)
                    Vs = self._vd_posterior_samples(QtQ, Qty, self.n_samples)
                    Xhat = (Vs * self.W[s][None, :]) @ self.U.T   # (n_samples, N)
                    xm = Xhat[:, miss]
                    comp_strict[miss, d, s] = xm.mean(axis=0)
                    std_strict[miss, d, s] = xm.std(axis=0)
                    samples_list.append(xm.T.astype(np.float32))  # (n_miss, n_samples)
                    nn = np.where(miss)[0]
                    idx_list.append(np.stack([nn, np.full_like(nn, d),
                                              np.full_like(nn, s)], axis=1))

                # 纳入槽 s 观测 (时间推进过槽 s)
                if obs.any():
                    Q = self.U[obs] * self.W[s][None, :]
                    y = self.sparse[obs, d, s]
                    QtQ += Q.T @ Q
                    Qty += Q.T @ y

                if miss.any():
                    # online: 条件于 槽<=s (含同时刻其他路段)
                    Vs = self._vd_posterior_samples(QtQ, Qty, self.n_samples)
                    Xhat = (Vs * self.W[s][None, :]) @ self.U.T
                    xm = Xhat[:, miss]
                    comp_online[miss, d, s] = xm.mean(axis=0)
                    std_online[miss, d, s] = xm.std(axis=0)

            # 天末: 用全天信息采样 v_d 写回 V (供后续日刷新热启动)
            self.V[d] = self._vd_posterior_samples(QtQ, Qty, 1)[0]

            if self.verbose and ((d - dW + 1) % 10 == 0 or d == D - 1):
                el = time.time() - t0
                eta = el / (d - dW + 1) * (D - 1 - d)
                print(f"  [rolling] day {d+1}/{D}  obs-RMSE={rmse:.4f}"
                      f"  tau={self.tau:.4f}  已用{el:.0f}s  预计还需{eta:.0f}s",
                      flush=True)

        samples = (np.concatenate(samples_list, axis=0)
                   if samples_list else np.zeros((0, self.n_samples), np.float32))
        samples_idx = (np.concatenate(idx_list, axis=0).astype(np.int32)
                       if idx_list else np.zeros((0, 3), np.int32))
        return dict(completion_strict=comp_strict, std_strict=std_strict,
                    completion_online=comp_online, std_online=std_online,
                    samples_strict=samples, samples_idx=samples_idx,
                    tau_trace=np.array(tau_trace), warmup_days=dW)


# ---------------------------------------------------------------
# 命令行入口: 真实数据 (data_056 + 复用已有挖洞 mask)
# ---------------------------------------------------------------
def evaluate(hat, truth, mask):
    err = (hat - truth)[mask]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(truth[mask]) + 1e-6)) * 100)
    return rmse, mae, mape


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="因果BGCP滚动插值 (无未来泄露)")
    ap.add_argument("--data", default=r"D:\桌面\ST-LLM-BG\data_056\speed_tensor.npz")
    ap.add_argument("--mask_npz", default=r"D:\桌面\ST-LLM-BG\models_bgcp\bgcp_RM_20_r50.npz",
                    help="复用该文件的 binary_mask/eval_mask, 保证与离线BGCP同一评估集")
    ap.add_argument("--out", default=r"D:\桌面\ST-LLM-BG\results_interp\causal_bgcp_RM_20_r50.npz")
    ap.add_argument("--rank", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=14)
    ap.add_argument("--warm_burn", type=int, default=300)
    ap.add_argument("--warm_sample", type=int, default=100)
    ap.add_argument("--daily_refresh", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    D0 = np.load(args.data, allow_pickle=True)
    truth = np.nan_to_num(D0["speed"].astype(np.float64), nan=0.0)
    obs_mask = D0["obs_mask"].astype(np.float64)

    M = np.load(args.mask_npz)
    binary = M["binary_mask"].astype(np.float64)
    eval_mask = M["eval_mask"].astype(bool)
    assert binary.shape == truth.shape
    assert (binary <= obs_mask).all(), "binary 必须是 obs_mask 的子集"
    assert np.allclose(np.nan_to_num(M["truth"], nan=0.0), truth), "truth 与 mask_npz 不一致"

    sparse = truth * binary
    model = CausalBGCP(sparse, binary, rank=args.rank, warmup_days=args.warmup,
                       warm_burn=args.warm_burn, warm_sample=args.warm_sample,
                       daily_refresh=args.daily_refresh, n_samples=args.n_samples,
                       seed=args.seed, verbose=True)
    t0 = time.time()
    res = model.run()
    dt = time.time() - t0

    # 快速评估 (只在 warm-up 之后的人工挖洞点, 有真值)
    day_ok = np.zeros_like(eval_mask)
    day_ok[:, args.warmup:, :] = True
    em = eval_mask & day_ok
    r1 = evaluate(res["completion_strict"], truth, em)
    r2 = evaluate(res["completion_online"], truth, em)
    print(f"\n==== CausalBGCP DONE ({dt:.0f}s) ====")
    print(f"评估点(day>={args.warmup}): {int(em.sum())}")
    print(f"strict(≤t-1)  RMSE={r1[0]:.4f}  MAE={r1[1]:.4f}  MAPE={r1[2]:.4f}%")
    print(f"online(≤t)    RMSE={r2[0]:.4f}  MAE={r2[1]:.4f}  MAPE={r2[2]:.4f}%")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfg = json.dumps(vars(args), ensure_ascii=False)
    np.savez_compressed(args.out, **res, truth=truth, obs_mask=obs_mask,
                        binary_mask=binary, eval_mask=eval_mask,
                        config=cfg, elapsed_sec=dt)
    print("saved:", args.out)
