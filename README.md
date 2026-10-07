# SPOT: Structure-Preserving Optimal Transport for Artistic Style Transfer

Reference implementation of the method in

> **Structure-Preserving Artistic Style Transfer via Semantic and Structural Transport Costs**
> Bin Wang, Zhen Li, Jie Liu
> *The Visual Computer* (submitted)

The code corresponds module by module to Section 3 of the manuscript and to its
equations. `selfcheck.py` verifies code–paper consistency item by item.

---

## 中文介绍

### 方法概述

艺术风格迁移要在采用风格图纹理的同时保住内容图的结构。既有方法在中间特征层上
刻画风格—内容的对应关系，传输计划完全由外观相似度决定，语义无关的区域因此常被
错误配对，内容结构随之被破坏。

SPOT 把语义与结构约束**从损失函数移到传输代价矩阵**——即真正决定传输计划形状的
地方。代价矩阵分解为三项：

```
C_ij = |f_i^c - f_j^s|_2^2  +  λ · D_ij  +  μ · Φ_ij        (式 4)
        外观代价              语义不相容     结构偏离
```

其中 `D_ij` 是式 (16) 的 0/1 语义不相容指示量，来自对内容与风格特征做的**联合球面
K-means 聚类**——两侧共用同一组聚类中心，因而可直接比较；`Φ_ij` 是式 (22) 的结构偏离
代价，由两个局部自相似矩阵之差构造，在相似变换下不变。传输计划由 Sinkhorn 迭代
（式 24）求解，并按不动点条件（式 27）反传，使 λ 与 μ 能与主干网络联合优化，而
不必作为人工设定的超参数。

### 代码结构

```
spot/
├── config.py     — 4.4 节的全部超参数
├── encoder.py    — 3.3 节  位置感知特征编码与支撑点构造（式 6–12）
├── costs.py      — 3.4–3.5 节  三项传输代价（式 13–22）
├── sinkhorn.py   — 3.6 节  熵正则传输求解与不动点求导（式 23–31）
├── decoder.py    — 3.7 节  逐尺度风格调制解码器（式 32–36）
├── losses.py     — 3.8 节  损失函数与总目标（式 37–40）
├── model.py      — 3.2 节 / 算法 1  整体前向
└── metrics.py    — 4.2 节  评测指标（SSIM、SSP、LPIPS）
selfcheck.py      — 代码 ↔ 论文一致性自检
```

### 运行

```bash
python selfcheck.py
```

环境：Python 3.9、PyTorch（CPU 即可）、NumPy、SciPy。不需要 GPU、`timm` 或 `lpips`
包；`metrics.LPIPS` 在缺包时显式抛 `ImportError`，不做静默降级。

---

## Layout

```
spot/
├── config.py     — all hyperparameters of Section 4.4
├── encoder.py    — Section 3.3  Position-aware feature encoding and support points   (Eqs. 6–12)
├── costs.py      — Sections 3.4–3.5  The three transport costs                        (Eqs. 13–22)
├── sinkhorn.py   — Section 3.6  Entropy-regularized transport and implicit diff       (Eqs. 23–31)
├── decoder.py    — Section 3.7  Multi-scale style-modulated decoder                   (Eqs. 32–36)
├── losses.py     — Section 3.8  Loss function and overall objective                   (Eqs. 37–40)
├── model.py      — Section 3.2 / Algorithm 1  End-to-end forward pass
└── metrics.py    — Section 4.2  Evaluation metrics (SSIM, SSP, LPIPS)
selfcheck.py      — code ↔ paper consistency checks
```

The three transport costs, in the notation of the paper:

```
C_ij = |f_i^c - f_j^s|_2^2  +  λ · D_ij  +  μ · Φ_ij        (Eq. 4)
        appearance            semantic     structural
```

`D_ij` is the 0/1 semantic-incompatibility indicator of Eq. (16), derived from the
joint spherical *K*-means clustering that gives both sides one shared set of cluster
centers. `Φ_ij` is the structural deviation of Eq. (22), built from the difference of
the two local self-similarity matrices. The plan is solved by Sinkhorn iterations
(Eq. 24) and differentiated through the fixed-point conditions (Eq. 27) so that λ and
μ are learned jointly with the backbone.

## Running

```bash
python selfcheck.py
```

Environment: Python 3.9, PyTorch (CPU is sufficient), NumPy, SciPy. No GPU, `timm`,
or `lpips` package is required; `metrics.LPIPS` raises an explicit `ImportError`
rather than degrading silently.

## License

Released for academic and research use.
