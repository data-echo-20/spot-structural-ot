# -*- coding: utf-8 -*-
"""SPOT 模型：把编码、代价构造、传输求解、解码串成完整前向（论文算法 1）。

    I_cs = D( R(F^c, T*·F̄^s) ),  T* = Sinkhorn(C(F^c, F^s))        （式 (5)）

λ、μ 是本模型的可学习参数（4.4 节），每步更新后按命题 1 裁剪到 [0,1]（式 (18)）。
"""

import torch
import torch.nn as nn

from .config import SPOTConfig
from .encoder import PositionAwareViT, freeze_module
from .costs import (
    appearance_cost, semantic_cost, self_similarity, structure_cost,
    compose_cost, band_neighbors, normalize_cost,
)
from .sinkhorn import sinkhorn_blocked, transport_features
from .decoder import ScaleModulatedDecoder


class SPOT(nn.Module):
    def __init__(self, cfg: SPOTConfig = None):
        super().__init__()
        self.cfg = cfg or SPOTConfig()
        self.encoder = PositionAwareViT(self.cfg)
        if self.cfg.freeze_encoder:
            freeze_module(self.encoder)

        self.decoder = ScaleModulatedDecoder(self.cfg, self.cfg.embed_dim)
        # 融合层参数在解码器首层（式 (29)），此处仅保留引用便于说明

        # 代价权重：以 logit 参数化以保证取值始终落在 [0,1]（命题 1 的约束）
        self.lambda_logit = nn.Parameter(torch.tensor(_inv_sigmoid(self.cfg.lambda_init)))
        self.mu_logit = nn.Parameter(torch.tensor(_inv_sigmoid(self.cfg.mu_init)))

    # ------------------------------------------------------------------
    @property
    def lam(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_logit)              # λ ∈ (0,1)

    @property
    def mu(self) -> torch.Tensor:
        return torch.sigmoid(self.mu_logit)                  # μ ∈ (0,1)

    def clamp_weights_(self):
        """命题 1：λ, μ ∈ [0,1]。logit 参数化下天然满足，此方法保留为显式约束接口。"""
        with torch.no_grad():
            lo, hi = _inv_sigmoid(0.0) + 1e-6, _inv_sigmoid(1.0) - 1e-6
            self.lambda_logit.clamp_(lo, hi)
            self.mu_logit.clamp_(lo, hi)

    # ------------------------------------------------------------------
    def encode(self, content: torch.Tensor, style: torch.Tensor):
        """（式 (6)-(12)）两侧共用同一冻结编码器，输出特征与坐标集合。"""
        fc, pc = self.encoder(content)
        fs, ps = self.encoder(style)
        return fc, pc, fs, ps

    def build_cost(self, fc, pc, fs, ps, seed: int = 0):
        """构造式 (4) 的三项代价并合成，返回 (C, 分量字典)。"""
        cfg = self.cfg
        # 式 (4) 首项。**必须在这里就归一化**：T⁰（式 22）要用它求解，
        # 而 compose_cost 内部也会归一化。若此处传未归一化的 c_app，
        # T⁰ 与最终合成就会用两种尺度 —— 实测未归一化时 T⁰ 根本收敛不了
        # （2000 次迭代后行残差仍 2.5e-02，50 次与 2000 次迭代的 T⁰ 相差 77%），
        # 式 (22) 的结构项因此建立在一个未收敛的中间值上。
        c_app = normalize_cost(appearance_cost(fc, fs))       # 式 (4) 首项、式 (18)

        c_sem = semantic_cost(                        # 式 (13)-(17)
            fc, fs, cfg.num_clusters, cfg.sigma_d2,
            iters=cfg.kmeans_iters, seed=seed,
        )

        sc = self_similarity(fc, pc, cfg.sigma_s2, cfg.selfsim_radius)   # 式 (19)-(20)
        ss = self_similarity(fs, ps, cfg.sigma_s2, cfg.selfsim_radius)

        # T⁰：仅含外观代价的参考传输计划（式 (22) 的常量），不需要梯度
        with torch.no_grad():
            t0, _, _ = sinkhorn_blocked(
                c_app, cfg.eps, self._uniform(fc.shape[1], fc),
                self._uniform(fs.shape[1], fs), max_elems=cfg.max_cost_elems,
            )
        c_idx, c_val, _ = band_neighbors(pc, cfg.selfsim_radius)
        s_idx, s_val, _ = band_neighbors(ps, cfg.selfsim_radius)
        c_str = structure_cost(t0, sc, ss, c_idx, c_val, s_idx, s_val)   # 式 (22)

        c = compose_cost(c_app, c_sem, c_str, self.lam, self.mu)         # 式 (4)、(18)
        parts = {"app": c_app, "sem": c_sem, "str": c_str, "T0": t0}
        return c, parts

    @staticmethod
    def _uniform(n: int, ref: torch.Tensor) -> torch.Tensor:
        """均匀质量分布 μ = 1/n · 1（3.1.2 节）。"""
        return torch.full((ref.shape[0], n), 1.0 / n, device=ref.device, dtype=ref.dtype)

    # ------------------------------------------------------------------
    def forward(self, content: torch.Tensor, style: torch.Tensor, seed: int = 0,
                return_parts: bool = False):
        """算法 1 的完整前向。"""
        cfg = self.cfg
        fc, pc, fs, ps = self.encode(content, style)          # 步骤 1
        c, parts = self.build_cost(fc, pc, fs, ps, seed=seed) # 步骤 2-5

        # 步骤 6：传输计划求解（含分块，式 (30)-(31)）
        t, n_iters, blk = sinkhorn_blocked(
            c, cfg.eps, self._uniform(fc.shape[1], fc),
            self._uniform(fs.shape[1], fs), max_elems=cfg.max_cost_elems,
        )

        # 步骤 7：重心映射（式 (28)）与融合（式 (29)）
        f_tilde = transport_features(t, fs, fc.shape[1])
        f_fuse = torch.cat([fc, f_tilde], dim=-1)

        # 步骤 8：解码
        out = self.decoder(f_fuse, fs)                        # 式 (35)-(36)

        parts.update({"T": t, "sinkhorn_iters": n_iters, "blocks": blk["blocks"],
                      "fc": fc, "fs": fs, "f_fuse": f_fuse})
        return (out, parts) if return_parts else out


def _inv_sigmoid(p: float) -> float:
    import math
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))
