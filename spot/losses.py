# -*- coding: utf-8 -*-
"""损失函数（论文 3.9 节，式 (37)-(40)）。

    L_total = L_cont + γ·L_sty + η·L_ent                            （式 (40)）

论文 3.9.4 节明确说明：结构约束已前移到代价矩阵，因此损失中**不含**显式结构项。
风格损失用通道均值与协方差而非 Gram 矩阵（式 (38) 之后说明）。
"""

import torch


def content_loss(enc_feat_out: torch.Tensor, enc_feat_c: torch.Tensor) -> torch.Tensor:
    """式 (37)：L_cont = (1/N) Σ ‖E(I_cs)_i − E(I_c)_i‖²。"""
    n = enc_feat_out.shape[1]
    return ((enc_feat_out - enc_feat_c) ** 2).sum(dim=-1).mean() / 1.0 * (n / n)


def style_loss(enc_feat_out: torch.Tensor, enc_feat_s: torch.Tensor) -> torch.Tensor:
    """式 (38)：L_sty = ‖μ(I_cs) − μ(I_s)‖² + ‖Σ(I_cs) − Σ(I_s)‖_F²。

    μ 为通道均值，Σ 为通道协方差矩阵。
    """
    def stats(f):
        mu = f.mean(dim=1)                                  # [B, d]，对位置取均值
        x = f - mu.unsqueeze(1)
        n = f.shape[1]
        cov = (x.transpose(1, 2) @ x) / max(n - 1, 1)       # [B, d, d]
        return mu, cov

    mu_o, cov_o = stats(enc_feat_out)
    mu_s, cov_s = stats(enc_feat_s)
    l_mu = ((mu_o - mu_s) ** 2).sum(dim=-1)
    l_cov = ((cov_o - cov_s) ** 2).sum(dim=(1, 2))
    return (l_mu + l_cov).mean()


def entropy_loss(t: torch.Tensor) -> torch.Tensor:
    """式 (39)：L_ent = Σ_{i,j} T*_ij log T*_ij（抑制传输计划趋于均匀）。"""
    eps = torch.finfo(t.dtype).eps
    return (t * torch.log(t.clamp_min(eps))).sum(dim=(1, 2)).mean()


def total_loss(
    enc_feat_out, enc_feat_c, enc_feat_s, t,
    gamma: float = 10.0, eta: float = 0.01,
):
    """式 (40)，返回 (总损失, 分量字典)。"""
    l_cont = content_loss(enc_feat_out, enc_feat_c)
    l_sty = style_loss(enc_feat_out, enc_feat_s)
    l_ent = entropy_loss(t)
    total = l_cont + gamma * l_sty + eta * l_ent
    return total, {"cont": l_cont.item(), "sty": l_sty.item(), "ent": l_ent.item()}
