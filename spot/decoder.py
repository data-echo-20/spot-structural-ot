# -*- coding: utf-8 -*-
"""逐尺度风格调制解码器（论文 3.8 节，式 (32)-(36)）。

四个上采样块，通道 384→256→128→64（4.4 节）；
风格窗口尺寸 w1..w4 = 16, 8, 4, 2（3.8.2 节），随尺度递减。
每块内依次为：最近邻上采样 → 风格调制（式 (33)-(34)）→ 两层 3×3 卷积（式 (35)）。

【论文未给出、本实现自行确定的细节】
式 (32) 定义 μ^s_k, σ^s_k ∈ R^d（d = 384，即编码器维度），而 4.4 节给出的
解码器各块通道数为 384/256/128/64。除第一块外二者通道数不一致，逐通道调制
无法直接施加。本实现为每块加一个可学习的通道投影 W_s: R^d → R^{C_in}，
把风格统计量投到该块调制发生的通道空间（即 in_ch）后再参与式 (33)。论文
正文未涉及这一投影，故在此显式标注；它不改变式 (33)-(34) 的形式，只补足
通道数对齐所需的映射。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_grid(x: torch.Tensor):
    """[B, T, d] → [B, d, g, g]，T 须为完全平方数。"""
    b, t, d = x.shape
    g = int(round(t ** 0.5))
    if g * g != t:
        raise ValueError("token 数 %d 非完全平方，无法还原为方形网格" % t)
    return x.transpose(1, 2).reshape(b, d, g, g)


def from_grid(z: torch.Tensor):
    """[B, d, g, g] → [B, g*g, d]"""
    b, d, g, _ = z.shape
    return z.reshape(b, d, g * g).transpose(1, 2)


def local_stats(x: torch.Tensor, window: int):
    """按窗口计算逐通道均值与标准差（式 (32) 的统计口径）。

    x: [B, T, d] → (mean, std)，形状均为 [B, T, d]。窗口以 token 为单位，
    采用奇数核的对称平均池化实现，等价于对窗口内位置取平均。
    """
    xg = to_grid(x)
    g = xg.shape[-1]
    k = max(1, min(window, g))
    if k % 2 == 0:
        k -= 1                                   # 池化核取奇数，保证窗口对称
    pad = k // 2
    mean = F.avg_pool2d(xg, k, stride=1, padding=pad)
    sq = F.avg_pool2d(xg * xg, k, stride=1, padding=pad)
    std = torch.sqrt((sq - mean * mean).clamp_min(0.0) + 1e-6)
    return from_grid(mean), from_grid(std)


def resize_stats(stats: torch.Tensor, size: int) -> torch.Tensor:
    """把 token 网格上的统计量双线性重采样到目标空间尺寸（尺寸对齐，非式 (32) 内容）。"""
    z = to_grid(stats)
    z = F.interpolate(z, size=(size, size), mode="bilinear", align_corners=False)
    return from_grid(z)


class UpBlock(nn.Module):
    """一个上采样块：Up → Mod → Conv → Conv（式 (35)）。"""

    def __init__(self, in_ch: int, out_ch: int, style_dim: int, window: int, slope: float = 0.2):
        super().__init__()
        self.window = window
        self.in_ch, self.out_ch = in_ch, out_ch
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.LeakyReLU(slope)                        # 3.8.5 节：斜率 0.2
        # 风格统计量的通道投影（见文件头【论文未给出】说明）。
        # 投影目标是 in_ch 而非 out_ch：式 (35) 的括号层级为
        #     Conv(LeakyReLU(Conv(Mod(Up(h^{(s-1)})))))
        # 即 Mod 作用在 Up(h^{(s-1)}) 上、卷积之前；而上采样不改变通道数，
        # 故 Mod 所处的通道空间是 in_ch，通道变换由随后的 conv1 完成。
        self.style_proj_mu = nn.Linear(style_dim, in_ch)
        self.style_proj_sg = nn.Linear(style_dim, in_ch)
        # α 经 Sigmoid 参数化到 [0,1]（式 (34)）。初值 0 对应 α = 0.5，
        # 与算法 2 第 1 步「α_1..α_4 ← 0.5」一致。
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x: torch.Tensor, mu_s: torch.Tensor, sigma_s: torch.Tensor):
        """x: [B, in_ch, h, w]；mu_s/sigma_s: [B, T_s, style_dim]（风格侧同尺度统计量）。"""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")   # 上采样
        b, c, hh, ww = x.shape
        t = hh * ww

        mu = self.style_proj_mu(resize_stats(mu_s, hh))           # [B, t, in_ch]
        sg = F.softplus(self.style_proj_sg(resize_stats(sigma_s, hh))) + 1e-4   # 标准差须为正

        h_flat = x.reshape(b, c, t).transpose(1, 2)               # [B, t, c]
        mu_c, sigma_c = local_stats(h_flat, window=max(1, hh // 8))
        mod = sg * (h_flat - mu_c) / sigma_c + mu                 # 式 (33)
        h_flat = self.alpha * mod + (1.0 - self.alpha) * h_flat   # 式 (34)
        x = h_flat.transpose(1, 2).reshape(b, c, hh, ww)

        x = self.act(self.conv1(x))                               # 式 (35)
        x = self.act(self.conv2(x))
        return x


class ScaleModulatedDecoder(nn.Module):
    """四层上采样解码器 + 输出头（式 (29)、(35)-(36)）。"""

    def __init__(self, cfg, style_dim: int):
        super().__init__()
        self.cfg = cfg
        chans = list(cfg.decoder_channels)                        # 384, 256, 128, 64
        # 式 (29)：F_fuse = W_f [F^c; F̃^s] + b_f，输入通道 2d → 第一块通道
        self.fuse = nn.Conv2d(2 * style_dim, chans[0], 1)
        blocks, prev = [], chans[0]
        for i, ch in enumerate(chans):
            blocks.append(UpBlock(prev, ch, style_dim, cfg.style_window_sizes[i], cfg.leaky_slope))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(prev, 3, 3, padding=1)              # 输出头 W_o, b_o

    def forward(self, f_fuse: torch.Tensor, fs: torch.Tensor, f_fuse_hw=None):
        """f_fuse: [B, T, 2d]（式 (29) 的融合特征）；fs: [B, M, d] 风格特征。"""
        b, t, d2 = f_fuse.shape
        g = int(round(t ** 0.5))
        x = f_fuse.transpose(1, 2).reshape(b, d2, g, g)
        x = self.fuse(x)                                          # [B, 384, g, g]

        for blk in self.blocks:
            # 式 (32)：按该尺度的窗口 w_s 从风格特征统计局部均值与标准差
            mu_s, sg_s = local_stats(fs, blk.window)
            x = blk(x, mu_s, sg_s)                                # 式 (35)
        return torch.tanh(self.head(x))                           # 式 (36)
