# -*- coding: utf-8 -*-
"""三项传输代价（论文 3.5-3.6 节，式 (13)-(22)）。

C = C_app + λ·C_sem + μ·C_str                        （式 (4)）

三项代价的量纲可比性由命题 1（式 (18)）保证：外观与结构两项各自除以最大值
归一化到 [0,1]，语义项本身即落在 [0,1] 内。
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 外观代价（式 (4) 首项）
# --------------------------------------------------------------------------
def appearance_cost(fc: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
    """C^app_ij = ‖f_i^c − f_j^s‖²，返回 [B, N, M]。

    用 (a−b)² = a² + b² − 2ab 展开计算，避免构造 [B,N,M,d] 中间张量。
    """
    a2 = (fc ** 2).sum(-1, keepdim=True)                # [B, N, 1]
    b2 = (fs ** 2).sum(-1, keepdim=True)                # [B, M, 1]
    ab = fc @ fs.transpose(1, 2)                        # [B, N, M]
    c = a2 + b2.transpose(1, 2) - 2.0 * ab              # [B, N, M]
    eps = torch.finfo(c.dtype).eps
    return c + eps                                       # 数值下界，避免出现负值


# --------------------------------------------------------------------------
# 联合聚类与语义不相容代价（式 (13)-(17)）
# --------------------------------------------------------------------------
def spherical_kmeans(
    x: torch.Tensor, k: int, iters: int = 20, seed: int = 0
) -> tuple:
    """球面 K 均值（式 (14)）。x: [B, T, d]，返回 (centers [B,k,d], assign [B,T])。

    约束 ‖c_k‖₂ = 1，等价于在归一化特征上做余弦簇分配、中心归一化。
    中心更新不参与梯度反传（3.5.5 节）。
    """
    with torch.no_grad():
        b, t, d = x.shape
        xn = F.normalize(x, dim=-1)                     # f̄_t，式 (14) 的约束载体
        # k-means++ 的简化版：随机取 k 个互异点作为初始中心
        gen = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(t, generator=gen)[:k].to(xn.device)
        centers = xn[:, idx, :].clone()                 # [B, k, d]

        assign = None
        for _ in range(iters):
            sim = xn @ centers.transpose(1, 2)           # [B, T, k]，余弦相似度
            new_assign = sim.argmax(dim=-1)              # 式 (15)
            if assign is not None and torch.equal(new_assign, assign):
                assign = new_assign
                break
            assign = new_assign
            onehot = F.one_hot(assign, k).to(xn.dtype)   # [B, T, k]
            counts = onehot.sum(dim=1)                   # [B, k]
            sums = onehot.transpose(1, 2) @ xn           # [B, k, d]
            # 空簇回退：保留原中心，避免除零
            empty = counts < 1
            new_centers = torch.where(
                empty.unsqueeze(-1),
                centers,
                sums / counts.clamp_min(1).unsqueeze(-1),
            )
            centers = F.normalize(new_centers, dim=-1)   # 重新投影回单位球
    return centers, assign


def semantic_cost(
    fc: torch.Tensor, fs: torch.Tensor, k: int, sigma_d2: float,
    iters: int = 20, seed: int = 0,
) -> torch.Tensor:
    """语义不相容代价 D̃（式 (16)-(17)），返回 [B, N, M]。

    步骤：两侧特征拼接（式 (13)）→ 球面 K 均值（式 (14)）→ 簇归属（式 (15)）
         → 连续化的不相容代价（式 (17)）。
    """
    b, n, d = fc.shape
    m = fs.shape[1]
    x = torch.cat([fc, fs], dim=1)                      # F = [F^c; F^s]，式 (13)
    centers, assign = spherical_kmeans(x, k, iters=iters, seed=seed)

    x_n = F.normalize(x, dim=-1)
    y_c, y_s = assign[:, :n], assign[:, n:]             # 式 (15) 两侧归属

    # 取风格位置 j 所属簇中心 c_{y_j^s}：在归一化特征上即 x 自身簇心的邻域中心
    c_ys = F.normalize(
        torch.gather(centers, 1, y_s.unsqueeze(-1).expand(-1, -1, d)), dim=-1
    )                                                   # [B, M, d]

    fn_c = x_n[:, :n, :]                                # f̄_i^c，[B,N,d]
    # ‖f̄_i^c − c_{y_j^s}‖² 展开为 [B,N,M]
    a2 = (fn_c ** 2).sum(-1, keepdim=True)              # [B,N,1]
    b2 = (c_ys ** 2).sum(-1, keepdim=True)              # [B,M,1]
    dist2 = a2 + b2.transpose(1, 2) - 2.0 * (fn_c @ c_ys.transpose(1, 2))
    dist2 = dist2.clamp_min(0.0)                        # 数值误差可能给出微负值

    d_tilde = 1.0 - torch.exp(-dist2 / (2.0 * sigma_d2))  # 式 (17)
    return d_tilde.clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# 结构偏离代价（式 (19)-(22)）
# --------------------------------------------------------------------------
def self_similarity(
    f: torch.Tensor, p: torch.Tensor, sigma_s2: float, radius: int
) -> torch.Tensor:
    """局部自相似矩阵 Ŝ（式 (19)-(20)），返回 [B, T, T]。

    先按特征距离算全局自相似，再用空间邻域掩码稀疏化为带状（式 (20)）。
    """
    fn = F.normalize(f, dim=-1)
    a2 = (fn ** 2).sum(-1, keepdim=True)
    dist2 = (a2 + a2.transpose(1, 2) - 2.0 * (fn @ fn.transpose(1, 2))).clamp_min(0.0)
    s = torch.exp(-dist2 / (2.0 * sigma_s2))            # 式 (19)

    # 空间邻域掩码：把 token 网格由 [B,T,2] 的坐标还原为网格，用切比雪夫距离判邻域
    t = p.shape[1]
    g = int(round(t ** 0.5))
    if g * g != t:
        raise ValueError("token 数 %d 非完全平方，无法按方形网格构造邻域掩码" % t)
    u = p[:, :, 0].reshape(-1, g, g)                    # 归一化坐标 → 网格下标
    v = p[:, :, 1].reshape(-1, g, g)
    # 坐标是 (i+0.5)/g 形式，乘 g 取整即得网格下标
    gi = (v * g).long().clamp(0, g - 1)
    gj = (u * g).long().clamp(0, g - 1)
    flat_i = gi.reshape(-1, t)                          # [B, T]
    flat_j = gj.reshape(-1, t)
    di = (flat_i.unsqueeze(2) - flat_i.unsqueeze(1)).abs()   # [B,T,T]
    dj = (flat_j.unsqueeze(2) - flat_j.unsqueeze(1)).abs()
    mask = ((di <= radius) & (dj <= radius)).to(s.dtype)     # 切比雪夫邻域，式 (20)

    return s * mask                                     # 式 (20)：邻域外置 0


def band_neighbors(p: torch.Tensor, radius: int):
    """由归一化坐标 p [B,T,2] 导出固定宽度的邻域索引表。

    返回 (idx, valid, count)：
      idx   [T, K] —— 每行的 K 个邻域下标，K = (2r+1)²；越界项填自身下标
      valid [T, K] —— 该项是否落在网格内的布尔掩码
      count [T]    —— 每行实际的有效邻域数（≤ K）

    几何只与网格有关、与批次无关，故按单个样本计算后共享。

    注意 count 的必要性：式 (20) 的邻域在图像边界处被截断，此时若把越界项
    直接 clamp 到自身下标，同一位置会被重复计数，使「邻域内权重和」偏大。
    `structure_cost` 用 valid 掩码把越界项剔除（见该函数），count 用于自检。
    """
    t = p.shape[1]
    g = int(round(t ** 0.5))
    if g * g != t:
        raise ValueError("token 数 %d 非完全平方，无法按方形网格构造邻域" % t)
    v = p[0, :, 1].reshape(g, g)                        # 归一化坐标 → 网格下标
    u = p[0, :, 0].reshape(g, g)
    gi = (v * g).long().clamp(0, g - 1).reshape(-1)
    gj = (u * g).long().clamp(0, g - 1).reshape(-1)

    self_idx = gi * g + gj
    idx, valid = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                idx.append(self_idx)
                valid.append(torch.ones_like(self_idx, dtype=torch.bool))
                continue
            inside = ((gi + dy >= 0) & (gi + dy < g) & (gj + dx >= 0) & (gj + dx < g))
            ni = (gi + dy).clamp(0, g - 1)
            nj = (gj + dx).clamp(0, g - 1)
            # 越界项填自身下标，但由 valid=False 标记，求和时被剔除
            idx.append(torch.where(inside, ni * g + nj, self_idx))
            valid.append(inside)
    idx = torch.stack(idx, dim=1)                       # [T, K]
    valid = torch.stack(valid, dim=1)                   # [T, K]
    return idx, valid, valid.sum(dim=1)                 # count: [T]


def structure_cost(
    t0: torch.Tensor, sc: torch.Tensor, ss: torch.Tensor,
    c_idx: torch.Tensor, c_valid: torch.Tensor,
    s_idx: torch.Tensor, s_valid: torch.Tensor,
) -> torch.Tensor:
    """结构偏离代价 Φ（式 (22)），返回 [B, N, M]。

        Φ_ij = Σ_{i'} Σ_{j'} T⁰_{i'j'} · |Ŝ^c_{i i'} − Ŝ^s_{j j'}|

    T⁰ 为仅含外观代价的参考传输计划，作为常量使用（3.6.3 节），
    因此本函数输出对上游参数不可微，符合式 (22) 的定位。

    直接按双重求和计算是 O(N²M²)（N=M=256 时约 4×10⁹），不可用。
    这里利用式 (20) 的带状稀疏性做等价改写。Ŝ^s 在邻域外恒为 0，
    于是按 j' 是否落在 j 的邻域内拆分：

        Σ_{j'} T⁰[i',j']·|a − Ŝ^s_{j j'}|,   a = Ŝ^c_{i i'}
          = Σ_{j'∈邻域} T⁰[i',j']·|a − Ŝ^s_{j j'}|  +  Σ_{j'∉邻域} T⁰[i',j']·a

    第二项中 |a − 0| = a 可提出求和号。又因 T⁰ 行和为 μ^c = 1/N（式 (2)），
    Σ_{j'∉邻域} T⁰[i',j'] = 1/N − Σ_{j'∈邻域} T⁰[i',j']，只需在 K = 25 项上求和。
    两项合计把每轮 i' 的代价从 O(NM) 降到 O(NM·K/… )，总量降约一个数量级。

    越界的邻域项以 valid 掩码剔除：它们不属于 j'∈邻域集合，其贡献已由第二项承担。
    """
    b, n, _ = sc.shape
    m = ss.shape[1]
    t0 = t0.detach()
    sc = sc.detach()
    ss = ss.detach()

    s_idx = s_idx.to(ss.device)
    s_valid = s_valid.to(ss.device).to(ss.dtype)
    ss_band = ss[:, :, s_idx]                            # [B, M, K]
    ss_band = ss_band * s_valid.unsqueeze(0)             # 越界项归零

    b, n, _ = sc.shape
    m = ss.shape[1]
    t0 = t0.detach()
    sc = sc.detach()
    ss = ss.detach()

    s_idx = s_idx.to(ss.device)
    s_valid = s_valid.to(ss.device).to(ss.dtype)
    # 把风格自相似矩阵按邻域索引表取出：[B, M, K]，越界项置零
    ss_band = ss.gather(2, s_idx.unsqueeze(0).expand(b, -1, -1)) * s_valid.unsqueeze(0)

    phi = t0.new_zeros(b, n, m)
    row_w = t0.sum(dim=2)                                # 每行权重和，即 μ^c（式 (2)）

    for i1 in range(n):
        a = sc[:, :, i1]                                 # [B, N]，i' 固定，列为 i
        # 权重按 j' 位置取出。t0[:, i1, :] 为 [B, M]，其第 1 维即风格位置 j'，
        # 故沿 dim=1 gather；先展成 [B, M, K] 再按邻域下标取值。
        w_band = t0[:, i1, :].unsqueeze(-1).expand(-1, -1, s_idx.shape[1])
        w_band = w_band.gather(1, s_idx.unsqueeze(0).expand(b, -1, -1)) * s_valid.unsqueeze(0)

        # 邻域内项：Σ_{j'∈邻域} T⁰[i',j']·|Ŝ^c[i,i'] − Ŝ^s[j,j']|
        diff = (a.unsqueeze(-1).unsqueeze(-1) - ss_band.unsqueeze(1)).abs()
        phi = phi + (diff * w_band.unsqueeze(1)).sum(-1)
        # 邻域外项：Ŝ^s[j,j'] = 0，故 |a − 0| = a，权重和 = 行和 − 邻域内权重和
        w_out = row_w[:, i1].reshape(b, 1) - w_band.sum(-1)        # [B, M]
        phi = phi + a.unsqueeze(-1) * w_out.unsqueeze(1)
    return phi


# --------------------------------------------------------------------------
# 归一化与合成（式 (18)、式 (4)）
# --------------------------------------------------------------------------
def normalize_cost(c: torch.Tensor) -> torch.Tensor:
    """式 (18)：除以全局最大值，归一化到 [0,1]。"""
    return c / c.amax(dim=(1, 2), keepdim=True).clamp_min(torch.finfo(c.dtype).eps)


def compose_cost(
    c_app: torch.Tensor, c_sem: torch.Tensor, c_str: torch.Tensor,
    lam: torch.Tensor, mu: torch.Tensor,
) -> torch.Tensor:
    """式 (4)：C = Ĉ^app + λ·D̃ + μ·Φ̂。

    三项均归一化后相加，λ、μ 由命题 1 约束到 [0,1]（式 (18) 之后的说明）。
    lam/mu 可为标量张量（可学习）或 python 浮点数。
    """
    c = normalize_cost(c_app) + lam * c_sem + mu * normalize_cost(c_str)
    return c
