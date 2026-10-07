# -*- coding: utf-8 -*-
"""熵正则最优传输的 Sinkhorn 求解与隐函数求导（论文 3.7 节，式 (23)-(31)）。

对应论文中的三处关键设计：
1. 核矩阵下界截断（3.7.6 节、附录 A.2）：K_ij ≥ κ = 1e-6，防下溢为 0；
2. 每轮对 u、v 归一化（3.7.7 节）：抑制数值漂移；
3. 式 (27) 的隐函数求导：反向不展开迭代图，而是「求解一个线性系统」（3.7.3 节原文）。

关于式 (27) 的实现（`SinkhornOT.backward` 内的完整推导见该函数文档字符串）：
反向的完整导数是

    ∂L/∂C = (T*/ε) ⊙ (p ⊕ q − G)

其中 G = ∂L/∂T*，而 (p, q) 由一组与 Sinkhorn 同形的**伴随迭代**解出。
只取 (T*/ε)⊙(−G) 一项（即把收敛点的 u、v 当作与 C 无关的常数）会漏掉
对偶变量随代价变化的那部分，实测偏差约 13%（见 `selfcheck.py` 第 3 项）。
式 (27) 中的 (J − P)† 正是这一伴随线性系统的矩阵形式，本文件用矩阵无关的
迭代法求解它，避免构造 (N+M)² 规模的雅可比。
"""

import torch


def kernel_matrix(cost: torch.Tensor, eps: float, floor: float) -> torch.Tensor:
    """核矩阵 K = exp(−C/ε)，按附录 A.2 取下界 κ 截断。

    截断只改变数值实现：对未触底的元素 dK/dC = −K/ε，
    对触底的元素梯度置零（截断的语义即「此处不再随代价变化」）。
    """
    k = torch.exp(-cost / eps)
    floored = (k < floor)
    return torch.where(floored, torch.full_like(k, floor), k), floored


class SinkhornOT(torch.autograd.Function):
    """熵正则最优传输求解 + 式 (27) 的隐函数反传。

    前向：按式 (26) 迭代到收敛，返回 T* = diag(u)·K·diag(v)（式 (24)）。
    反向：见 `backward`。包裹成 Function 的目的是让反向不经过 50 轮迭代的
    计算图，从而与 3.10.3 节「反向代价与前向迭代次数无关」一致。
    """

    @staticmethod
    def forward(ctx, cost, eps, mu_c, mu_s, iters, tol, floor, check_every):
        with torch.no_grad():
            b, n, m = cost.shape
            k, floored = kernel_matrix(cost, eps, floor)
            mass_tot = mu_c.sum(dim=1).reshape(-1, 1, 1)      # 应为 1（3.1.2 节均匀分布）

            u = torch.full((b, n), 1.0 / n, device=cost.device, dtype=cost.dtype)
            v = torch.full((b, m), 1.0 / m, device=cost.device, dtype=cost.dtype)

            prev_t = None
            used = iters
            for it in range(iters):
                # 式 (26)：交替更新；分母加极小量防除零（3.7.7 节）
                u = mu_c / ((k @ v.unsqueeze(-1)).squeeze(-1) + 1e-30)
                v = mu_s / ((k.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1) + 1e-30)

                # 附录 A.3 式 (A3)：按最大值归一化，防止 u、v 量级失控。
                # 该操作等价于对 (u, v) 施加互逆缩放，T 只差一个整体尺度（gauge），
                # 故下面的收敛判据与返回值都先按总质量把 gauge 固定回来。
                u = u / u.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)
                v = v / v.abs().max(dim=1, keepdim=True).values.clamp_min(1e-30)

                cur_t = u.unsqueeze(-1) * k * v.unsqueeze(1)  # 式 (24)
                cur_t = cur_t * (mass_tot / cur_t.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30))

                # 收敛判据：相邻两轮传输计划的 ℓ1 变化 < tol（4.4 节、式 (A4)）
                if prev_t is not None and (it + 1) % check_every == 0:
                    if (cur_t - prev_t).abs().sum(dim=(1, 2)).max() < tol:
                        used = it + 1
                        prev_t = cur_t
                        break
                prev_t = cur_t

            # 返回前对 u 做一次半步更新。此时 v 已收敛，按式 (26) 重算 u 使
            # 行约束 T·1 = μ^c 精确成立（而非仅近似），列约束则保持收敛精度。
            u = mu_c / ((k @ v.unsqueeze(-1)).squeeze(-1) + 1e-30)
            t = u.unsqueeze(-1) * k * v.unsqueeze(1)          # 式 (24)
            t = t * (mass_tot / t.sum(dim=(1, 2), keepdim=True).clamp_min(1e-30))

        ctx.save_for_backward(t, mu_c, mu_s, floored)
        ctx.eps = eps
        return t, used

    @staticmethod
    def backward(ctx, grad_t, grad_used):
        """式 (27) 的矩阵无关实现。

        推导（f, g 为使 T*_ij = exp((f_i+g_j−C_ij)/ε) 的对偶变量；
        a = μ^c，b = μ^s）：
        记 G = ∂L/∂T*。约束 Σ_j T*_ij = a_i 与 Σ_i T*_ij = b_j 对 C_kl 求导，得

            a_i·∂f_i/∂C_kl + Σ_j T*_ij·∂g_j/∂C_kl = δ_ik·T*_il          (A)
            b_j·∂g_j/∂C_kl + Σ_i T*_ij·∂f_i/∂C_kl = δ_jl·T*_kl          (B)

        欲求 S = Σ_i (∂L/∂f_i)(∂f_i/∂C_kl) + Σ_j (∂L/∂g_j)(∂g_j/∂C_kl)，
        其中 ∂L/∂f_i = Σ_j G_ij T*_ij、∂L/∂g_j = Σ_i G_ij T*_ij。
        对 (A)(B) 取伴随（转置）得到关于 (p, q) 的系统

            a_i·p_i + Σ_j T*_ji·q_j = Σ_j G_ij T*_ij
            Σ_i T*_ij·p_i + b_j·q_j = Σ_i G_ij T*_ij

        整理后即下面迭代的两式。解出 (p, q) 后 S = p_k·T*_kl + q_l·T*_kl，
        与直接项 −G_kl·T*_kl 合并，得

            ∂L/∂C = (T*/ε) ⊙ (p ⊕ q − G)，  (p ⊕ q)_kl = p_k + q_l

        该式即式 (27) 的等价形式：(J − P)† 对应的正是 (p, q) 的线性系统。
        把 p = q = 0 代回即得「u、v 视为常数」的近似 —— 缺的正是这一修正项。
        """
        if grad_t is None:
            return None, None, None, None, None, None, None, None

        t, mu_c, mu_s, floored = ctx.saved_tensors
        eps = ctx.eps
        g = grad_t

        a = mu_c.sum(dim=-1, keepdim=True) if mu_c.dim() == 3 else mu_c
        b = mu_s.sum(dim=-1, keepdim=True) if mu_s.dim() == 3 else mu_s

        with torch.no_grad():
            # 伴随迭代（与 Sinkhorn 同形，几何收敛）
            p = torch.zeros_like(mu_c)
            q = torch.zeros_like(mu_s)
            for _ in range(200):
                p_new = ((g - q.unsqueeze(1)) * t).sum(dim=-1) / a.clamp_min(1e-12)
                q_new = ((g - p_new.unsqueeze(-1)) * t).sum(dim=1) / b.clamp_min(1e-12)
                if (p_new - p).abs().max() < 1e-10 and (q_new - q).abs().max() < 1e-10:
                    p, q = p_new, q_new
                    break
                p, q = p_new, q_new

            correction = p.unsqueeze(-1) + q.unsqueeze(1)      # p ⊕ q
            grad_c = (t / eps) * (correction - g)
            # 核矩阵触底处 dK/dC = 0，对应位置的梯度一并置零（附录 A.2 的截断语义）
            grad_c = torch.where(floored, torch.zeros_like(grad_c), grad_c)

        return grad_c, None, None, None, None, None, None, None


def sinkhorn(
    cost: torch.Tensor,
    eps: float,
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    iters: int = 50,
    tol: float = 1e-3,
    kernel_floor: float = None,
    check_every: int = 1,
    return_log: bool = False,
):
    """求解式 (23)，返回 (T*, 实际迭代轮数)。

    T* 与 cost 处于同一计算图中，梯度按式 (27) 流动。
    kernel_floor 为 None 时取下界 exp(−3/ε)（论文 3.7.6 节）。
    """
    if kernel_floor is None:
        import math
        kernel_floor = math.exp(-3.0 / float(eps))
    t, used = SinkhornOT.apply(
        cost, float(eps), mu_c, mu_s, int(iters), float(tol),
        float(kernel_floor), int(check_every),
    )
    if return_log:
        return t, used, {"iters": used}
    return t, used


def sinkhorn_blocked(
    cost: torch.Tensor,
    eps: float,
    mu_c: torch.Tensor,
    mu_s: torch.Tensor,
    max_elems: int = 4_000_000,
    **kw,
):
    """分块传输（式 (30)-(31)）。

    当 N·M 超过 max_elems 时，把内容位置切成 B = ⌈N·M / C_max⌉ 块，逐块求解，
    每块的内容边缘分布取 μ^c 在该块上的**限制** μ^c|_{B_b}（式 (30) 的块划分），
    风格边缘分布仍取完整的 μ^s —— 每个块都从全部风格位置取质量。

    【与论文式 (31) 的差异，重要】
    论文 3.7.5 节把块内容边缘分布写成 μ^c_b = (|B_b| / N)·μ^c。该式与式 (2) 的
    行约束不相容：设 μ^c 为式 (3.1.2) 的均匀分布（各元 1/N），则 (|B_b|/N)·μ^c
    各元为 |B_b|/N²，块内合计 |B_b|²/N²，而块内 |B_b| 个位置按式 (2) 应合计
    |B_b|/N，二者仅在 |B_b| = N 时相等，即只在不分块时成立。
    本文的实现取限制口径 μ^c_b = μ^c|_{B_b}（各元仍为 1/N，合计 |B_b|/N），
    它同时满足式 (2) 与式 (31) 想表达的「按块大小分配质量」之意。
    该项差异已在 `code/README.md` 的「与论文的差异」一节登记。

    3.7.5 节另说明：分块后块与块之间的结构关系不再被约束；而式 (20) 的自相似
    邻域不因分块而截断 —— 邻域在代价构造阶段完成（`costs.band_neighbors`），
    与本函数无关。
    """
    n = cost.shape[1]
    if n * cost.shape[2] <= max_elems:
        t, used = sinkhorn(cost, eps, mu_c, mu_s, **kw)
        return t, used, {"blocks": 1}

    n_blocks = (n * cost.shape[2] + max_elems - 1) // max_elems
    n_per = (n + n_blocks - 1) // n_blocks              # N_B = N / B
    outs, used_all = [], []
    for bi in range(n_blocks):
        lo, hi = bi * n_per, min((bi + 1) * n_per, n)
        if lo >= hi:
            continue
        t_b, used_b = sinkhorn(
            cost[:, lo:hi, :], eps, mu_c[:, lo:hi], mu_s, **kw
        )                                               # 式 (30)，见上方说明
        outs.append(t_b)
        used_all.append(used_b)
    return torch.cat(outs, dim=1), max(used_all), {"blocks": len(outs)}


def transport_features(t: torch.Tensor, fs: torch.Tensor, n: int) -> torch.Tensor:
    """重心映射（式 (28)）：f̃_i^s = N · Σ_j T*_ij · f̄_j^s。"""
    fs_n = fs / fs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return n * (t @ fs_n)
