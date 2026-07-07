# -*- coding: utf-8 -*-
"""贝叶斯高斯张量分解 BGCP (Bayesian Gaussian CANDECOMP/PARAFAC)
复现亓航博士论文第4章 / 陈新宇 transdim 同源方法。
三阶张量 CP 分解 + Gibbs 采样, Gaussian-Wishart 因子超先验 + Gamma 精度先验。

被 03_插值复现.py 调用; 纯算法模块, 不读写数据文件。
"""
import numpy as np
from numpy.linalg import solve, cholesky, inv
from numpy.random import normal
from scipy.stats import wishart


def cp_combine(U):
    """从因子矩阵列表重构三阶张量: X_ijk = sum_r U0[i,r]U1[j,r]U2[k,r]"""
    return np.einsum('ir,jr,kr->ijk', U[0], U[1], U[2])


def mvnrnd(mu, Lambda):
    """从 N(mu, Lambda^{-1}) 采样 (Lambda=精度矩阵)"""
    L = cholesky(Lambda)
    z = normal(size=mu.shape[0])
    return mu + solve(L.T, z)


def _sample_hyperparams(Uk, beta0, nu0, mu0, W0_inv):
    """采样因子矩阵 Uk 行向量的 Gaussian-Wishart 后验超参数 (mu_k, Lambda_k)"""
    m, r = Uk.shape
    Ubar = Uk.mean(axis=0)
    S = (Uk - Ubar).T @ (Uk - Ubar)
    beta_post = beta0 + m
    nu_post = nu0 + m
    mu_post = (beta0 * mu0 + m * Ubar) / beta_post
    diff = mu0 - Ubar
    W_post = inv(W0_inv + S + (beta0 * m / beta_post) * np.outer(diff, diff))
    W_post = (W_post + W_post.T) / 2  # 数值对称化
    Lambda_k = wishart.rvs(df=nu_post, scale=W_post)
    mu_k = mvnrnd(mu_post, beta_post * Lambda_k)
    return mu_k, Lambda_k


def BGCP(sparse_tensor, binary_tensor, rank=50, burn_iter=1000, gibbs_iter=500,
         seed=None, verbose=False, return_factors=False):
    """
    sparse_tensor : 含缺失的张量, 缺失处为0, 观测处为真实值
    binary_tensor : 0/1 掩码, 1=观测, 0=缺失
    return_factors: True 时额外返回因子矩阵 [U0,U1,U2] (最后一次采样)
    返回: tensor_hat (后验采样均值重构的完整张量); 或 (tensor_hat, factors)
    """
    if seed is not None:
        np.random.seed(seed)
    dim = sparse_tensor.shape
    d = len(dim)
    # 先验超参数
    beta0, nu0 = 1.0, rank
    mu0 = np.zeros(rank)
    W0_inv = np.eye(rank)
    a0 = b0 = 1e-6
    tau = 1.0
    # 初始化因子矩阵
    U = [0.1 * normal(size=(dim[k], rank)) for k in range(d)]
    obs_idx = binary_tensor.astype(bool)
    n_obs = int(obs_idx.sum())

    tensor_hat_sum = np.zeros(dim)
    for it in range(burn_iter + gibbs_iter):
        for k in range(d):
            mu_k, Lambda_k = _sample_hyperparams(U[k], beta0, nu0, mu0, W0_inv)
            a, b = [j for j in range(d) if j != k]
            Q = np.einsum('ir,jr->ijr', U[a], U[b]).reshape(-1, rank)  # (dim_a*dim_b, r)
            Tk = np.moveaxis(sparse_tensor, k, 0).reshape(dim[k], -1)   # (dim_k, dim_a*dim_b)
            Mk = np.moveaxis(binary_tensor, k, 0).reshape(dim[k], -1).astype(bool)
            Lmu = Lambda_k @ mu_k
            for i in range(dim[k]):
                m_i = Mk[i]
                Qm = Q[m_i]
                ym = Tk[i][m_i]
                var_inv = Lambda_k + tau * (Qm.T @ Qm)
                rhs = Lmu + tau * (Qm.T @ ym)
                U[k][i] = mvnrnd(solve(var_inv, rhs), var_inv)
        # 采样精度 tau ~ Gamma
        tensor_hat = cp_combine(U)
        err = (sparse_tensor - tensor_hat)[obs_idx]
        tau = np.random.gamma(a0 + 0.5 * n_obs, 1.0 / (b0 + 0.5 * np.sum(err ** 2)))
        if it + 1 > burn_iter:
            tensor_hat_sum += tensor_hat
        if verbose and (it + 1) % 200 == 0:
            rmse_obs = np.sqrt(np.mean(err ** 2))
            print(f"  iter {it+1}/{burn_iter+gibbs_iter}  obs-RMSE={rmse_obs:.4f}  tau={tau:.4f}")
    tensor_hat = tensor_hat_sum / gibbs_iter
    if return_factors:
        return tensor_hat, [u.copy() for u in U]
    return tensor_hat
