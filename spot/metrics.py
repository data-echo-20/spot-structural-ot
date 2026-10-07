# -*- coding: utf-8 -*-
"""评测指标（论文 4.2 节、附录 D.5）。

| 指标 | 定义 | 出处 |
|---|---|---|
| LPIPS ↓ | 预训练网络特征空间的加权距离 | 4.2 节 |
| SSIM ↑ | 亮度、对比度、结构 | 4.2 节 |
| 内容损失 ↓ | 编码器特征空间的 ℓ2 距离，式 (37) | 4.2 节 |
| 风格损失 ↓ | 通道均值与协方差之差，式 (38) | 4.2 节 |
| 结构相似保持率 SSP ↑ | 输出图与内容图的自相似矩阵差异，式 (41) | 4.2 节 |

其中内容/风格损失直接复用 `losses.py` 的式 (37)/(38) 实现。
LPIPS 依赖外部权重（`lpips` 包不在本环境），本模块给出**未安装时明确报错**
而非静默用替代量，以免把非论文口径的数字写进结果文件。
"""

import torch
import torch.nn.functional as F

from .costs import self_similarity
from .losses import content_loss, style_loss


# --------------------------------------------------------------------------
# SSIM（4.2 节）
# --------------------------------------------------------------------------
def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, c2: float = 1e-4) -> torch.Tensor:
    """结构相似度，返回 [B] 的均值（按标准 SSIM 的窗口统计口径）。"""
    c1 = 1e-4
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size ** 2)
    ch = x.shape[1]
    kernel = kernel.repeat(ch, 1, 1, 1)

    def filt(z):
        return F.conv2d(z, kernel, padding=pad, groups=ch)

    mu_x, mu_y = filt(x), filt(y)
    sx = filt(x * x) - mu_x * mu_x
    sy = filt(y * y) - mu_y * mu_y
    sxy = filt(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2)
    return (num / den).mean(dim=(1, 2, 3))


# --------------------------------------------------------------------------
# 结构相似保持率 SSP（式 (41)，4.2 节）
# --------------------------------------------------------------------------
def structural_preservation(
    f_out: torch.Tensor, f_c: torch.Tensor, p_c: torch.Tensor,
    sigma_s2: float = 0.5, radius: int = 2,
) -> torch.Tensor:
    """式 (41)：SSP = 1 − 归一化的自相似矩阵差异，返回 [B]。

    取输出图与内容图在**编码器特征空间**的自相似矩阵（式 (19)-(20) 同一构造），
    差异用 Frobenius 范数比上内容图自相似矩阵自身的范数，取 1 减去该比值。
    """
    s_out = self_similarity(f_out, p_c, sigma_s2, radius)
    s_c = self_similarity(f_c, p_c, sigma_s2, radius)
    num = (s_out - s_c).flatten(1).norm(dim=1)
    den = s_c.flatten(1).norm(dim=1).clamp_min(1e-8)
    return (1.0 - num / den).clamp(-1.0, 1.0)


# --------------------------------------------------------------------------
# LPIPS（4.2 节）
# --------------------------------------------------------------------------
class LPIPS:
    """按 4.2 节口径的 LPIPS。依赖 `lpips` 包与预训练权重。

    未安装时抛 ImportError，并说明原因 —— 不返回替代指标。
    """

    def __init__(self, net: str = "alex", device: str = "cpu"):
        try:
            import lpips  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LPIPS 需要外部包与预训练权重（论文 4.2 节的评测口径）。"
                "当前环境未安装 `lpips`，请先 `pip install lpips`。"
                "为避免写出与论文口径不符的数字，此处不提供替代实现。"
            ) from e
        import lpips as _lp
        self.fn = _lp.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fn(x.to(self.device) * 2 - 1, y.to(self.device) * 2 - 1).flatten()


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------
def evaluate_pair(
    out: torch.Tensor, content: torch.Tensor, style: torch.Tensor,
    fc, fs, f_out, pc, lpips_fn: LPIPS = None,
):
    """对一组 (输出, 内容, 风格) 计算全部指标，返回字典。"""
    res = {
        "ssim": ssim(out, content).mean().item(),
        "content_loss": content_loss(f_out, fc).item(),
        "style_loss": style_loss(f_out, fs).item(),
        "ssp": structural_preservation(f_out, fc, pc).mean().item(),
    }
    if lpips_fn is not None:
        res["lpips"] = lpips_fn(out, content).mean().item()
    return res
