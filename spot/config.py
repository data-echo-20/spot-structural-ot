# -*- coding: utf-8 -*-
"""SPOT 超参数配置。

本文件中的每个字段都对应论文中的一个具体取值，出处见行内注释。
凡论文未明确给出的取值，一律在注释中标为「论文未给出」，不做臆造。
"""

from dataclasses import dataclass, asdict, fields
from typing import Tuple


@dataclass
class SPOTConfig:
    # ---------------- 数据与分辨率（4.1 节、附录 D.2） ----------------
    content_size: int = 256          # 内容图 256×256
    style_size: int = 512            # 风格图 512×512

    # ---------------- 编码器（4.4 节、附录 D.3） ----------------
    patch_size: int = 16             # 图块边长 P = 16（式 (6)）
    embed_dim: int = 384             # 特征维度 d = 384
    depth: int = 12                  # 层数 L = 12
    num_heads: int = 6               # 注意力头数 6
    mlp_ratio: float = 4.0           # 论文未给出，取 ViT 通行值
    freeze_encoder: bool = True      # 编码器冻结（4.4 节）

    # ---------------- 代价参数（4.4 节、附录 D.3） ----------------
    num_clusters: int = 8            # 簇数 K = 8（式 (14)）
    sigma_d2: float = 0.25           # 语义带宽 σ_D² = 0.25（式 (17)）
    sigma_s2: float = 0.5            # 自相似带宽 σ_S² = 0.5（式 (19)）
    selfsim_radius: int = 2          # 局部自相似邻域 r：5×5 token → 半径 2（式 (20)）
    kmeans_iters: int = 20           # 球面 K 均值迭代上限（3.10.3 节取 I = 20）

    # ---------------- 熵正则传输（3.7 节、附录 A/D） ----------------
    eps: float = 0.05                # 熵正则系数 ε = 0.05（式 (23)）
    sinkhorn_iters: int = 50         # 迭代上限 50 轮（式 (26)）
    sinkhorn_tol: float = 1e-3       # 收敛判据：相邻两轮 ℓ1 变化 < 1e-3
    # 核矩阵截断下界。论文 3.7.6 节明确规定取下界 exp(−3/ε)：
    # 「由于式 (18) 保证 Ĉ_ij ∈ [0,3]，核矩阵元素下界为 exp(−3/ε)，
    #   该值在浮点运算中可能下溢为 0，导致对数运算出现 −∞。
    #   实现时对核矩阵取下界 exp(−3/ε) 做截断，避免数值问题。」
    # 故该值由 ε 决定而非独立常数。留 None 表示按此公式自动取值；
    # 显式给数则覆盖自动值（仅自检用）。
    # 【重要】早期版本此处误取固定 1e-6，远大于 ε=0.05 时的 exp(−60)≈8.8e−27，
    # 导致全体核元素被截断为同一常数、T* 退化为均匀分布，λ 与 μ 的梯度
    # 因此恒为零 —— 论文 3.7.7 节「λ、μ 可与主干网络联合优化」将不成立。
    kernel_floor: float = None       # None → exp(−3/ε)（3.7.6 节）
    implicit_diff: bool = True       # True 用式 (27) 的隐函数反传；False 用展开反传（仅用于自检）

    @property
    def kernel_floor_eff(self) -> float:
        """实际使用的核矩阵下界：exp(−3/ε)，见 3.7.6 节。"""
        import math
        if self.kernel_floor is not None:
            return float(self.kernel_floor)
        return math.exp(-3.0 / self.eps)

    # ---------------- 分块传输（3.7.5 节、式 (30)-(31)） ----------------
    max_cost_elems: int = 4_000_000  # C_max = 4×10^6

    # ---------------- 损失权重（4.4 节、附录 D.3） ----------------
    gamma: float = 10.0              # 风格损失权重 γ = 10（式 (40)）
    eta: float = 0.01                # 熵惩罚权重 η = 0.01（式 (39)-(40)）
    lambda_init: float = 0.5         # λ 初值 0.5（式 (4)）
    mu_init: float = 0.3             # μ 初值 0.3（式 (4)）

    # ---------------- 解码器（3.8 节） ----------------
    decoder_channels: Tuple[int, ...] = (384, 256, 128, 64)   # 4.4 节：384,256,128,64
    style_window_sizes: Tuple[int, ...] = (16, 8, 4, 2)        # 3.8.2 节：w1..w4
    leaky_slope: float = 0.2         # 3.8.5 节：LeakyReLU 负斜率 0.2

    # ---------------- 训练（4.4 节、附录 D.4） ----------------
    epochs: int = 200
    batch_size: int = 8
    lr: float = 1e-4
    lr_min: float = 1e-6             # 余弦退火至 1e-6
    weight_decay: float = 1e-4
    early_stop_patience: int = 10    # 验证集 LPIPS 连续 10 轮未下降
    seeds: Tuple[int, ...] = (42, 123, 456, 789, 1024)   # 5 次独立运行

    # ---------------- 运行 ----------------
    device: str = "cpu"
    num_workers: int = 0

    def grid_shape(self, n_tokens: int) -> Tuple[int, int]:
        """由 token 数反推方阵网格边数。"""
        g = int(round(n_tokens ** 0.5))
        if g * g != n_tokens:
            raise ValueError("token 数 %d 不是完全平方数，无法构成方阵网格" % n_tokens)
        return g, g

    def tokens_per_side(self, image_size: int) -> int:
        return image_size // self.patch_size

    def to_dict(self):
        return asdict(self)

    def describe(self):
        """列出全部配置项的取值与出处，供 README 与日志引用。"""
        out = []
        for f in fields(self):
            out.append("%-22s = %s" % (f.name, getattr(self, f.name)))
        return "\n".join(out)


DEFAULT = SPOTConfig()
