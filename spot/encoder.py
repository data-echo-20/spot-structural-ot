# -*- coding: utf-8 -*-
"""位置感知特征编码器（论文 3.4 节，式 (6)-(12)）。

设计要点：
- 每个 token 对应原图一个 P×P 图块，位置编码显式记录坐标（式 (7)-(8)）；
- 输出除特征矩阵 F 外，还给出每个 token 的归一化坐标集合 P（式 (12)）；
- 坐标集合不参与前向梯度，仅供结构代价使用（3.4.4 节）；
- 编码器在训练中冻结（4.4 节）。

实现说明：本模块自带一个纯 PyTorch 的 ViT 实现，不依赖 timm。
若环境中有预训练权重，用 `load_pretrained` 载入；否则用随机初始化权重，
此时**只能**用于形状/梯度自检，不能用于复现论文数值。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 二维正弦位置编码（式 (8)）
# --------------------------------------------------------------------------
def sincos_2d_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """构造二维正弦位置编码，返回 [grid_h*grid_w, dim]。

    式 (8) 按坐标 x_i 生成正弦/余弦分量。本文把特征维度 d 对半分给
    横纵两个方向，每个方向内部再按 10000^{2k/d} 的频率展开，
    这与 3.4.2 节「二维正弦位置编码」的描述一致。
    """
    if dim % 4 != 0:
        raise ValueError("embed dim 需被 4 整除以平分给两个方向的 sin/cos，当前 %d" % dim)
    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32) / (half / 2.0)
    omega = 1.0 / (10000 ** omega)                      # [half/2]

    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    # 归一化到 [0,1]，与式 (12) 的归一化坐标口径一致
    y = grid_y.reshape(-1) / max(grid_h - 1, 1)
    x = grid_x.reshape(-1) / max(grid_w - 1, 1)

    out_x = x[:, None] * omega[None, :]                 # [N, half/2]
    out_y = y[:, None] * omega[None, :]
    emb = torch.cat(
        [torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1
    )
    return emb                                          # [N, dim]


# --------------------------------------------------------------------------
# 最小可用的 Transformer 编码块（式 (9)-(10)）
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, N, d]。式 (10)：先注意力残差、再 MLP 残差。"""
        h = self.norm1(z)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        z = z + attn_out                                # 式 (10) 第一项
        z = z + self.mlp(self.norm2(z))                 # 式 (10) 第二项
        return z


# --------------------------------------------------------------------------
# 位置感知 ViT 编码器
# --------------------------------------------------------------------------
class PositionAwareViT(nn.Module):
    """输出 (F, P)：特征矩阵与归一化坐标集合。

    F: [B, N, d]   —— 式 (11)
    P: [B, N, 2]   —— 式 (12)，(u, v) ∈ [0,1]²
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, P = cfg.embed_dim, cfg.patch_size
        self.patch_embed = nn.Conv2d(3, d, kernel_size=P, stride=P)   # 式 (6)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(d)

    def _coords(self, grid_h: int, grid_w: int, device, dtype) -> torch.Tensor:
        gy, gx = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=dtype),
            torch.arange(grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        u = (gx.reshape(-1) + 0.5) / grid_w             # token 中心坐标
        v = (gy.reshape(-1) + 0.5) / grid_h
        return torch.stack([u, v], dim=-1)              # [N, 2]

    def forward(self, x: torch.Tensor):
        """x: [B, 3, H, W] → (F, P)"""
        b = x.shape[0]
        z = self.patch_embed(x)                         # [B, d, gh, gw]
        grid_h, grid_w = z.shape[-2], z.shape[-1]
        z = z.flatten(2).transpose(1, 2)                # [B, N, d]
        n = z.shape[1]

        pos = sincos_2d_pos_embed(grid_h, grid_w, self.cfg.embed_dim).to(z.device, z.dtype)
        z = z + pos.unsqueeze(0)                        # 式 (7)

        for blk in self.blocks:
            z = blk(z)
        f = self.norm(z)                                # 式 (11)

        p = self._coords(grid_h, grid_w, x.device, z.dtype)
        p = p.unsqueeze(0).expand(b, n, 2)              # 式 (12)
        return f, p

    @torch.no_grad()
    def load_pretrained(self, state_dict_or_path, strict: bool = False):
        """载入预训练权重。传入路径时用 torch.load。"""
        if isinstance(state_dict_or_path, str):
            sd = torch.load(state_dict_or_path, map_location="cpu")
            sd = sd.get("model", sd.get("state_dict", sd))
        else:
            sd = state_dict_or_path
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        return missing, unexpected


def freeze_module(m: nn.Module) -> nn.Module:
    """冻结全部参数（4.4 节：编码器训练中完全冻结）。"""
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m
