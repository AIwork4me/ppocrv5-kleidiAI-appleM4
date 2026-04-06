<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 基于 ONNX Runtime + Arm KleidiAI | 与 PaddleOCR 100% 精度对齐 | Apple M4 基准测试**

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.21-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_SME2-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

生产就绪的单文件 PP-OCRv5 推理流水线，使用 ONNX Runtime，在 Apple M4 上相比 PaddleOCR 原生推理实现 **1.51 倍加速**（ORT 1.24.3, 2 线程, KleidiAI SME2），并达到 **100% 文本级精度对齐** —— 在 7 张图片的 228 个文本区域上验证，零误差。

## 亮点

- **比 Paddle 原生推理快 1.51 倍**，Apple M4 平台（ORT 1.24.3, threads=2）
- **100% 精度匹配** —— 228/228 文本完全一致，置信度差异 < 0.00002
- **单文件部署** —— `ppocrv5_onnx.py`（约 720 行），可直接复制到任何 ARM 应用中
- **可复现的基准测试** —— 测试了 5 种配置，3 条命令即可在你的平台上运行
- **KleidiAI SME2 分析** —— 与 ARM 在 [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) 中共同调查

## 基准测试结果 (Apple M4)

<table>
<tr>
<th>配置</th>
<th>平均延迟</th>
<th>vs Paddle</th>
<th>文本匹配</th>
</tr>
<tr>
<td>Paddle 3.3.0 (t=8)</td>
<td>9,567 ms</td>
<td>1.00x</td>
<td>基准</td>
</tr>
<tr>
<td>ORT 1.21.1 (t=8, 无 KleidiAI)</td>
<td>6,497 ms</td>
<td>快 1.47 倍</td>
<td>228/228 ✓</td>
</tr>
<tr>
<td><b>ORT 1.24.3 (t=2, KleidiAI SME2)</b></td>
<td><b>6,332 ms</b></td>
<td><b>快 1.51 倍</b></td>
<td><b>228/228 ✓</b></td>
</tr>
<tr>
<td>ORT 1.24.3 (t=8, KleidiAI SME2)</td>
<td>7,096 ms</td>
<td>快 1.35 倍</td>
<td>228/228 ✓</td>
</tr>
<tr>
<td>ORT 1.24.3 (t=8, KleidiAI 禁用)</td>
<td>7,155 ms</td>
<td>快 1.34 倍</td>
<td>228/228 ✓</td>
</tr>
</table>

> 所有基准测试：每张图片 3 次运行，1 次预热。所有后端产生 100% 相同的文本，置信度差异 < 0.00002。
> 复现：`python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2`

**为什么 ORT 1.24.3 在 t=2 时比 t=8 更快？** ORT 1.24.3 中的 KleidiAI 使用 ARM **SME2**（可扩展矩阵扩展）来处理 GEMM 和 Conv 运算。在 Apple M4 上，SME 是**共享协处理器**（每个集群 2 个设备），而非像 NEON 那样的逐核设计。当线程数 > 2 时，所有线程争抢这 2 个设备 —— Conv 密集的 det 模型从 1,328 ms（ORT 1.21.1 NEON, t=8）退化到 4,312 ms。当 threads=2 时，SME 争抢最小化，rec/textline_ori 从 SME2 加速中大幅受益。完整分析见 [Apple Silicon 上的 SME 线程扩展](#apple-silicon-上的-sme-线程扩展)。

<details>
<summary><b>各模型推理耗时明细 (ms, 7 张图片平均值)</b></summary>

| 模型 | Paddle 3.3.0 (t=8) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) |
|-------|-------------------:|------------------:|------------------:|------------------:|
| doc_ori | 9.65 | 6.63 | **3.41** | 4.82 |
| det | 4,296.21 | **1,328.30** | 4,246.58 | 4,311.64 |
| textline_ori | 444.61 | 222.09 | **80.84** | 98.81 |
| rec | 4,728.09 | 4,850.45 | **1,931.17** | 2,579.15 |

关键发现：ORT 1.21.1（NEON, t=8）在单独的 det 模型上最快，但 ORT 1.24.3（SME2, t=2）在整体流水线上胜出，因为 rec 和 textline_ori 通过 SME2 分别快了 **2.5 倍**和 **2.7 倍**。

</details>

<details>
<summary><b>各图片延迟明细</b></summary>

| 图片 | 文本数 | Paddle (t=8) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) |
|-------|:-----:|------------:|------------------:|------------------:|------------------:|
| ancient_demo.png | 12 | 2,707 ms | 2,246 ms | **1,022 ms** | 1,340 ms |
| handwrite_ch_demo.png | 10 | 1,824 ms | 1,144 ms | **673 ms** | 1,175 ms |
| handwrite_en_demo.png | 11 | 2,121 ms | 1,429 ms | **828 ms** | 2,215 ms |
| japan_demo.png | 28 | 19,706 ms | **7,971 ms** | 18,425 ms | 14,549 ms |
| magazine.png | 65 | 17,947 ms | 14,025 ms | **11,601 ms** | 16,118 ms |
| magazine_vetical.png | 65 | 17,549 ms | 14,341 ms | **9,732 ms** | 11,599 ms |
| pinyin_demo.png | 37 | 5,113 ms | 4,323 ms | **2,039 ms** | 2,676 ms |

</details>

## 流水线架构

```
┌─────────┐     ┌──────────┐     ┌───────┐     ┌──────────────┐     ┌───────┐
│  Image   │────▶│ doc_ori  │────▶│  det  │────▶│ textline_ori │────▶│  rec  │────▶ Results
│ (BGR)    │     │ 4-class  │     │  DB   │     │   2-class    │     │  CTC  │     [{text,
└─────────┘     │ rotation │     │ boxes │     │  rotation    │     │ decode│      conf,
                └──────────┘     └───────┘     └──────────────┘     └───────┘      bbox}]
                  LCNet           PP-OCRv5       LCNet              PP-OCRv5
                  224×224         HxW→stride32   160×80              48×W
```

详见 [docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) 了解预处理参数和实现细节。

## Apple Silicon 上的 SME 线程扩展

> 在 [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) 中调查 —— 已获 ORT 维护者确认。

ORT 1.24.x 使用 KleidiAI **SME2 内核**（SGEMM、IGEMM Conv、Dynamic QGemm）来利用 ARM 的可扩展矩阵扩展。在 Apple M4 上，SME 是**共享协处理器**（共 2 个设备），而非像 NEON 那样的逐核设计。这形成了线程扩展的权衡：

| 线程数 | det (Conv 密集) | rec (GEMM 密集) | 流水线总耗时 |
|:-------:|:-----------------:|:-----------------:|:--------------:|
| 2 | 4,247 ms | **1,931 ms** | **6,332 ms** |
| 8 | 4,312 ms | 2,579 ms | 7,096 ms |
| 8 (无 KleidiAI) | 4,266 ms | 2,677 ms | 7,155 ms |
| ORT 1.21.1, 8 (NEON) | **1,328 ms** | 4,850 ms | 6,497 ms |

**推荐：在 Apple M4 上使用 ORT >= 1.24 时设置 `threads=2`**。这样可以最小化 SME 争抢，同时在 GEMM 密集模型上充分受益于 SME2 加速。

```python
# 推荐配置：Apple M4 + ORT >= 1.24
pipeline = PPOCRv5Pipeline(model_dir, dict_path=dict_path, threads=2)
```

完整分析、实验数据和背景知识详见 [docs/SME_THREAD_SCALING.md](docs/SME_THREAD_SCALING.md)。

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/AIwork4me/ppocrv5-kleidiAI-appleM4.git
cd ppocrv5-kleidiAI-appleM4
pip install onnxruntime>=1.21.0 opencv-python-headless numpy pyclipper
```

### 2. 下载模型

从[百度网盘](https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw)下载（密码：`uepw`），放置到 `models/` 目录下。模型目录结构见 [models/README.md](models/README.md)。

```bash
python scripts/download_models.py  # 验证模型文件是否就位
```

### 3. 运行 OCR

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline("models", dict_path="data/dict/ppocrv5_dict.txt", threads=2)
results = pipeline.predict("image.png")

for r in results:
    print(f"{r['text']}  ({r['confidence']:.4f})")
```

## 集成到你的应用

`ppocrv5_onnx.py` 是一个**单文件模块**（约 720 行），依赖极少。可直接复制到你的项目中：

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=2,  # Apple M4 推荐配置。详见 docs/SME_THREAD_SCALING.md
)
results = pipeline.predict(bgr_image_array)  # 接受文件路径或 BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**依赖项**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## 复现 Benchmark

```bash
# ORT 1.24.3, threads=2（推荐，Apple M4 上流水线吞吐量最佳）
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2

# ORT 1.24.3, threads=8（展示 det 上的 SME 争抢）
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# ORT 1.24.3, threads=8, KleidiAI 禁用（回退到 NEON）
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8 --disable-kleidiai

# ORT 1.21.1, threads=8（NEON 基准，无 KleidiAI）
pip install onnxruntime==1.21.1
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# Paddle 基准测试（可选）
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3

# 对比 results/ 中的所有结果
python benchmarks/compare_results.py
```

结果保存至 `results/*.json`，可跨平台对比。

## 精度对齐

ONNX 流水线与 PaddleOCR/PaddleX 3.4.x 原生推理产生 **100% 完全一致**的文本输出，通过 6 轮系统性调试实现：

| 轮次 | 修复内容 | 匹配率 |
|:-----:|-----|:----------:|
| 1 | CTC 解码、归一化、框排序等 | 65.6% → 71.8% |
| 3 | **det resize 参数**（Pipeline 运行时覆盖 inference.yml） | → 90.8% |
| 5 | **裁剪坐标精度**（int16 → minAreaRect float32） | → 93.3% |
| 6 | **rec 批处理填充**（batch_size=6, 比例排序, 按批填充） | → **100.0%** |

完整过程和关键发现详见 [docs/ACCURACY_ALIGNMENT.md](docs/ACCURACY_ALIGNMENT.md)。

## 项目结构

```
ppocrv5-kleidiAI-appleM4/
├── ppocrv5_onnx.py                 # 核心：单文件推理流水线
├── benchmarks/
│   ├── benchmark_unified.py        # 统一基准测试（--backend paddle|ort）
│   └── compare_results.py          # 多后端对比报告
├── results/                        # 参考基准数据（Apple M4）
│   ├── paddle_3.3.0.json
│   ├── ort_1.21.1.json
│   ├── ort_1.24.3.json             # KleidiAI SME2, threads=8
│   ├── ort_1.24.3_t2.json          # KleidiAI SME2, threads=2（推荐）
│   └── ort_1.24.3_no_kleidiai.json # KleidiAI 禁用, threads=8
├── data/
│   ├── dict/ppocrv5_dict.txt      # 字符字典（18,383 字符）
│   └── images/                     # 7 张测试图片
├── models/                         # ONNX 模型（需单独下载，约 180 MB）
├── docs/
│   ├── ACCURACY_ALIGNMENT.md       # 6 轮精度对齐过程
│   ├── BENCHMARK_RESULTS.md        # 完整基准测试表格
│   ├── PIPELINE_ARCHITECTURE.md    # 4 模型流水线详解
│   └── SME_THREAD_SCALING.md       # KleidiAI SME 线程扩展分析
├── scripts/download_models.py      # 模型验证工具
└── examples/quickstart.py          # 最小用法示例
```

## 文档

| 文档 | 描述 |
|----------|-------------|
| [流水线架构](docs/PIPELINE_ARCHITECTURE.md) | 4 模型流水线、预处理参数、批处理策略 |
| [精度对齐](docs/ACCURACY_ALIGNMENT.md) | 从 65.6% 到 100% 的 6 轮调试历程 |
| [基准测试结果](docs/BENCHMARK_RESULTS.md) | 完整速度/精度表格，逐模型 KleidiAI 分析 |
| [SME 线程扩展](docs/SME_THREAD_SCALING.md) | Apple Silicon 上的 KleidiAI SME 争抢，线程调优指南 |

## 环境要求

| 包 | 版本 | 备注 |
|---------|---------|-------|
| Python | >= 3.10 | |
| onnxruntime | >= 1.21.0 | >= 1.24 可启用 KleidiAI SME2 |
| opencv-python-headless | >= 4.8.0 | |
| numpy | >= 1.24.0 | |
| pyclipper | >= 1.3.0 | DB 后处理 |

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) —— PP-OCRv5 模型及原始推理流水线
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) —— 跨平台推理引擎
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) —— Arm CPU 微内核库，用于加速 ML 推理

## 引用

如果你在工作中使用了 PP-OCRv5 模型，请引用 PaddleOCR 3.0 技术报告：

```bibtex
@article{cui2025paddleocr,
  title={PaddleOCR 3.0 Technical Report},
  author={Cui, Cheng and Sun, Ting and Lin, Manhui and Gao, Tingquan and Zhang, Yubo and Liu, Jiaxuan and Wang, Xueqing and Zhang, Zelun and Zhou, Changda and Liu, Hongen and Zhang, Yue and Lv, Wenyu and Huang, Kui and Zhang, Yichao and Zhang, Jing and Zhang, Jun and Liu, Yi and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2507.05595},
  year={2025}
}
```

- 论文: <https://arxiv.org/abs/2507.05595>
- 源码: <https://github.com/PaddlePaddle/PaddleOCR>
- 文档: <https://paddlepaddle.github.io/PaddleOCR>
- 模型和在线 Demo: <https://huggingface.co/PaddlePaddle>

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。
