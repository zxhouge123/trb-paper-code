import numpy as np

# ──────────────────────────────────────────────
# 以下两个函数保持与师兄论文完全一致的风格:
#   固定 seed → 一次性生成随机矩阵 → 阈值切分
#   保证不同缺失率的评估集是嵌套的,比较公平
# ──────────────────────────────────────────────

def inject_point_missing(obs_mask, missing_rate=0.2, seed=42):
    """随机点缺失: 在真实观测点(obs_mask==1)上随机挖散点。

    与师兄 gen_mask_RM 完全等价:
        binary = (rand > rate).astype(float)
        返回 (eval_mask, binary_mask) 两套掩码便于后续使用。

    返回:
        eval_mask  : 1=本次挖的洞(有真值,用于算插值误差)
        binary_mask: 1=插值时可见的观测 = obs_mask 且 未被挖洞
    """
    rng = np.random.RandomState(seed)
    obs = obs_mask > 0
    rand = rng.random(obs.shape)          # 一次性生成全部随机数
    eval_mask = (obs & (rand < missing_rate)).astype(np.float32)
    binary_mask = (obs & (rand >= missing_rate)).astype(np.float32)
    return eval_mask, binary_mask


def inject_block_missing(obs_mask, missing_rate=0.2, block_len=None, seed=42):
    """纤维/结构缺失: 沿时段维整条(路段,天)缺失, 模拟传感器故障。

    与师兄 gen_mask_NRM 完全等价:
        对每个(路段,天), 以 rate 概率缺失整天的所有时段,
        保证不同缺失率下评估集嵌套, 公平比较。

    block_len: 保留参数但不使用(为兼容旧调用), 统一按 fiber 缺失处理

    返回:
        eval_mask  : 1=本次挖的洞(有真值,用于算插值误差)
        binary_mask: 1=插值时可见的观测 = obs_mask 且 未被挖洞
    """
    rng = np.random.RandomState(seed)
    N, D, S = obs_mask.shape
    obs = obs_mask > 0

    fiber_obs = rng.rand(N, D) > missing_rate                   # (N, D) bool
    binary_mask = np.repeat(fiber_obs[:, :, None], S, axis=2)  # (N, D, S) bool

    binary_mask = (binary_mask & obs).astype(np.float32)
    eval_mask = (obs & (binary_mask == 0)).astype(np.float32)

    return eval_mask, binary_mask