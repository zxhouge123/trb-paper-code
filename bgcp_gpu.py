# -*- coding: utf-8 -*-
"""BGCP GPU 版 (PyTorch): 向量化批量行采样 + batched Cholesky, 支持 CUDA/CPU。
与 bgcp.py(numpy) 数值等价, 用于服务器 GPU 加速。被 04_全表复现.py 调用。
纯算法模块, 不读写数据文件。

加速核心: 原 numpy 版对每个因子矩阵逐行 Gibbs 采样(Python for 循环),
本版把同一 mode 所有行的精度矩阵/右端项批量算出, 用 batched cholesky 一次采样。
"""
import numpy as np
import torch


def _khatri_rao(A, B):
    """A(I,r),B(J,r)->(I*J,r); 行序 i 慢 j 快, 匹配 reshape(m,-1) 的 C-order 展开"""
    I, r = A.shape
    J = B.shape[0]
    return (A.reshape(I, 1, r) * B.reshape(1, J, r)).reshape(I * J, r)


def cp_combine(U):
    return torch.einsum('ir,jr,kr->ijk', U[0], U[1], U[2])


def _sample_wishart(df, scale_tril):
    """Bartlett 分解采样 W ~ Wishart(df, V), V = scale_tril @ scale_tril^T"""
    r = scale_tril.shape[0]
    dev, dt = scale_tril.device, scale_tril.dtype
    idx = torch.arange(r, device=dev, dtype=dt)
    chi2 = torch.distributions.Gamma((df - idx) / 2.0, 0.5).sample()  # chi2(df-i)
    A = torch.zeros(r, r, device=dev, dtype=dt)
    A[torch.arange(r), torch.arange(r)] = torch.sqrt(chi2)
    til = torch.tril_indices(r, r, offset=-1, device=dev)
    A[til[0], til[1]] = torch.randn(til.shape[1], device=dev, dtype=dt)
    L = scale_tril @ A
    return L @ L.T


def _mvn_from_precision(mu, prec):
    """从 N(mu, prec^{-1}) 采样单个向量"""
    L = torch.linalg.cholesky(prec)
    z = torch.randn(mu.shape[0], device=mu.device, dtype=mu.dtype)
    x = torch.linalg.solve_triangular(L.transpose(-1, -2), z.unsqueeze(-1), upper=True).squeeze(-1)
    return mu + x


def BGCP_gpu(sparse_tensor, binary_tensor, rank=50, burn_iter=1000, gibbs_iter=500,
             device='cuda', dtype='float64', seed=None, verbose=False, return_factors=False):
    """
    sparse_tensor, binary_tensor : numpy 数组(缺失处0 / 0-1掩码)
    device : 'cuda' 或 'cpu'; dtype : 'float64'(与numpy对齐) 或 'float32'(GPU更快)
    return_factors: True 时额外返回因子矩阵 [U0,U1,U2] (最后一次采样, numpy)
    返回: tensor_hat (numpy, 后验采样均值); 或 (tensor_hat, factors)
    """
    if seed is not None:
        torch.manual_seed(seed)
    dev = torch.device(device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    dt = torch.float64 if dtype == 'float64' else torch.float32
    sparse = torch.as_tensor(sparse_tensor, dtype=dt, device=dev)
    binary = torch.as_tensor(binary_tensor, dtype=dt, device=dev)
    dim = sparse.shape
    d = len(dim)

    beta0, nu0 = 1.0, rank
    mu0 = torch.zeros(rank, dtype=dt, device=dev)
    W0 = torch.eye(rank, dtype=dt, device=dev)
    a0 = b0 = 1e-6
    tau = 1.0
    U = [0.1 * torch.randn(dim[k], rank, dtype=dt, device=dev) for k in range(d)]
    obs = binary.bool()
    n_obs = int(obs.sum())
    jit = torch.eye(rank, dtype=dt, device=dev) * 1e-6

    hat_sum = torch.zeros(dim, dtype=dt, device=dev)
    for it in range(burn_iter + gibbs_iter):
        for k in range(d):
            Uk = U[k]
            m = dim[k]
            # ---- Gaussian-Wishart 后验超参数 ----
            Ubar = Uk.mean(0)
            cen = Uk - Ubar
            S = cen.T @ cen
            beta_post = beta0 + m
            nu_post = nu0 + m
            mu_post = (beta0 * mu0 + m * Ubar) / beta_post
            diff = mu0 - Ubar
            Wpost = torch.linalg.inv(W0 + S + (beta0 * m / beta_post) * torch.outer(diff, diff))
            Wpost = (Wpost + Wpost.T) / 2
            Lambda_k = _sample_wishart(nu_post, torch.linalg.cholesky(Wpost))
            mu_k = _mvn_from_precision(mu_post, beta_post * Lambda_k)
            # ---- 批量行采样 ----
            a, b = [j for j in range(d) if j != k]
            KR = _khatri_rao(U[a], U[b])                       # (P, r)
            P = KR.shape[0]
            Tk = torch.movedim(sparse, k, 0).reshape(m, -1)    # (m, P)
            Mk = torch.movedim(binary, k, 0).reshape(m, -1)    # (m, P)
            KRouter = torch.einsum('pr,ps->prs', KR, KR).reshape(P, rank * rank)
            prec = Lambda_k.reshape(1, rank, rank) + tau * (Mk @ KRouter).reshape(m, rank, rank) + jit
            rhs = (Lambda_k @ mu_k).reshape(1, rank) + tau * ((Mk * Tk) @ KR)   # (m, r)
            L = torch.linalg.cholesky(prec)                    # (m, r, r)
            mean = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)       # (m, r)
            z = torch.randn(m, rank, dtype=dt, device=dev)
            samp = torch.linalg.solve_triangular(L.transpose(-1, -2), z.unsqueeze(-1), upper=True).squeeze(-1)
            U[k] = mean + samp
        # ---- 精度 tau ~ Gamma ----
        hat = cp_combine(U)
        err = (sparse - hat)[obs]
        ss = torch.sum(err ** 2).item()
        tau = float(torch.distributions.Gamma(a0 + 0.5 * n_obs, b0 + 0.5 * ss).sample())
        if it + 1 > burn_iter:
            hat_sum += hat
        if verbose and (it + 1) % 200 == 0:
            print(f"  iter {it+1}/{burn_iter+gibbs_iter}  obs-RMSE={np.sqrt(ss/n_obs):.4f}  tau={tau:.4f}")
    tensor_hat = (hat_sum / gibbs_iter).cpu().numpy()
    if return_factors:
        return tensor_hat, [u.cpu().numpy() for u in U]
    return tensor_hat
