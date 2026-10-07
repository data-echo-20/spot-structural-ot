# -*- coding: utf-8 -*-
"""SPOT —— 基于结构保持最优传输的艺术风格迁移（论文 v8 配套实现）。

模块与论文章节的对应关系：

    spot/config.py     —— 4.4 节、附录 D 的全部超参数
    spot/encoder.py    —— 3.4 节  位置感知特征编码与支撑点构造（式 (6)-(12)）
    spot/costs.py      —— 3.5-3.6 节 三项传输代价（式 (13)-(22)）
    spot/sinkhorn.py   —— 3.7 节  熵正则传输求解与不动点求导（式 (23)-(31)）
    spot/decoder.py    —— 3.8 节  逐尺度风格调制解码器（式 (32)-(36)）
    spot/losses.py     —— 3.9 节  损失函数（式 (37)-(40)）
    spot/model.py      —— 3.10 节 算法 1/2 的整体前向与训练
    spot/metrics.py    —— 4.2 节  评测指标
"""

from .config import SPOTConfig, DEFAULT
from .model import SPOT

__all__ = ["SPOTConfig", "DEFAULT", "SPOT"]
__version__ = "1.0.0"
