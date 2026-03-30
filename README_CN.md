<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 + ONNX Runtime + Arm KleidiAI | 100% 精度对齐 PaddleOCR | Apple M4 基准测试**

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.22-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_I8MM-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

面向生产环境的单文件 PP-OCRv5 推理流水线，基于 ONNX Runtime，在 Apple M4 上比 PaddleOCR 原生推理**快 1.72 倍**，同时保持 **100% 文本级精度对齐** — 在 7 张图片 228 条文本上零误差验证。

## 亮点

- **1.72x 加速** — 相比 Paddle 原生推理（Apple M4，KleidiAI 自动启用）
- **100% 精度一致** — 228/228 条文本完全匹配，置信度差异 < 0.00002
- **单文件部署** — `ppocrv5_onnx.py`（~550 行），可直接 copy-paste 到任何 ARM 应用
- **可复现基准** — 3 条命令即可在你自己的平台上运行
- **KleidiAI 零精度损失** — ORT 1.21 vs 1.23 置信度差异：0.000000

## 基准测试结果 (Apple M4)

<table>
<tr>
<th>后端</th>
<th>平均延迟</th>
<th>相对 Paddle</th>
<th>文本匹配</th>
<th>置信度差异</th>
</tr>
<tr>
<td>Paddle 3.3.0</td>
<td>9,451 ms</td>
<td>1.00x</td>
<td>基准</td>
<td>—</td>
</tr>
<tr>
<td>ORT 1.21.1（无 KleidiAI）</td>
<td>6,407 ms</td>
<td>1.48x 快</td>
<td>228/228 (100%)</td>
<td>0.000019</td>
</tr>
<tr>
<td><b>ORT 1.23.2（KleidiAI）</b></td>
<td><b>5,486 ms</b></td>
<td><b>1.72x 快</b></td>
<td><b>228/228 (100%)</b></td>
<td><b>0.000019</b></td>
</tr>
</table>

> 测试环境：Apple M4, macOS ARM64, 8 线程, 7 张图片, 每张运行 3 次。
> 复现命令：`python benchmarks/benchmark_unified.py --backend ort --num-runs 3`

<details>
<summary><b>KleidiAI 各模型加速效果（ORT 1.21.1 → 1.23.2）</b></summary>

| 模型 | ORT 1.21.1 (ms) | ORT 1.23.2 (ms) | 加速比 | 功能 |
|------|---------------:|---------------:|:------:|------|
| doc_ori | 2.57 | 1.34 | **1.91x** | 文档方向分类（4 类） |
| textline_ori | 67.88 | 36.51 | **1.86x** | 文本行方向分类（2 类） |
| rec | 1,599.89 | 1,319.33 | **1.21x** | 文本识别（CTC） |
| det | 3,779.37 | 3,788.58 | 1.00x | 文本检测（DB，大核卷积主导） |

KleidiAI 对 GEMM友好(GEMM-friendly)的模型加速效果好；对于使用了大核卷积等非GEMM友好的模型，未见加速效果。

</details>

<details>
<summary><b>逐图延迟明细</b></summary>

| 图片 | 文本数 | Paddle 3.3.0 | ORT 1.21.1 | ORT 1.23.2 (KleidiAI) |
|------|:-----:|------------:|----------:|---------------------:|
| ancient_demo.png | 12 | 2,958 ms | 2,379 ms | 2,086 ms |
| handwrite_ch_demo.png | 10 | 1,834 ms | 1,230 ms | 1,064 ms |
| handwrite_en_demo.png | 11 | 2,422 ms | 1,620 ms | 1,395 ms |
| japan_demo.png | 28 | 14,017 ms | 8,606 ms | 7,313 ms |
| magazine.png | 65 | 24,095 ms | 14,625 ms | 12,519 ms |
| magazine_vertical.png | 65 | 17,279 ms | 14,803 ms | 12,803 ms |
| pinyin_demo.png | 37 | 3,553 ms | 1,585 ms | 1,220 ms |

</details>

## 流水线架构

```
┌─────────┐     ┌──────────┐     ┌───────┐     ┌──────────────┐     ┌───────┐
│  输入图片 │────▶│ doc_ori  │────▶│  det  │────▶│ textline_ori │────▶│  rec  │────▶ 结果
│ (BGR)    │     │ 方向分类  │     │ 文本   │     │  行方向分类   │     │ 文本   │   [{text,
└─────────┘     │ 4类旋转  │     │ 检测   │     │  2类旋转     │     │ 识别   │    conf,
                └──────────┘     └───────┘     └──────────────┘     └───────┘    bbox}]
                  LCNet           PP-OCRv5       LCNet              PP-OCRv5
                  224×224         HxW→stride32   160×80              48×W
```

详见 [docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md)。

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/user/ppocrv5-kleidiAI-appleM4.git
cd ppocrv5-kleidiAI-appleM4
pip install onnxruntime>=1.22.0 opencv-python-headless numpy pyclipper
```

### 2. 下载模型

从[百度网盘](https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw)（提取码: `uepw`）下载，放入 `models/` 目录。详见 [models/README.md](models/README.md)。

```bash
python scripts/download_models.py  # 验证模型就位
```

### 3. 运行 OCR

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline("models", dict_path="data/dict/ppocrv5_dict.txt")
results = pipeline.predict("image.png")

for r in results:
    print(f"{r['text']}  ({r['confidence']:.4f})")
```

## 集成到你的应用

`ppocrv5_onnx.py` 是**单文件模块**（~550 行），依赖极少。直接复制到你的项目即可：

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=4,
)
results = pipeline.predict(bgr_image_array)  # 支持文件路径或 BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**依赖**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## 复现 Benchmark

```bash
# ORT benchmark（推荐 ORT >= 1.22 以启用 KleidiAI）
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# Paddle benchmark（可选）
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3

# 对比 results/ 下的所有结果
python benchmarks/compare_results.py
```

结果保存为 `results/*.json`，可跨平台对比。

## 精度对齐

ONNX 流水线输出与 PaddleOCR/PaddleX 3.4.x 原生推理 **100% 一致**，历经 6 轮系统化调试：

| 轮次 | 修复内容 | 匹配率 |
|:----:|---------|:------:|
| 1 | CTC 解码、归一化、box 排序 ... | 65.6% → 71.8% |
| 3 | **det resize 参数**（Pipeline 运行时覆盖 inference.yml） | → 90.8% |
| 5 | **crop 坐标精度**（int16 → minAreaRect float32） | → 93.3% |
| 6 | **rec batch padding**（batch_size=6, ratio 排序, 批内 pad） | → **100.0%** |

详见 [docs/ACCURACY_ALIGNMENT.md](docs/ACCURACY_ALIGNMENT.md)。

## 项目结构

```
ppocrv5-kleidiAI-appleM4/
├── ppocrv5_onnx.py                 # 核心：单文件推理流水线
├── benchmarks/
│   ├── benchmark_unified.py        # 统一 benchmark（--backend paddle|ort）
│   └── compare_results.py          # 多后端对比报告
├── results/                        # 参考 benchmark 数据（Apple M4）
│   ├── paddle_3.3.0.json
│   ├── ort_1.21.1.json
│   └── ort_1.23.2.json
├── data/
│   ├── dict/ppocrv5_dict.txt      # 字符字典（18,383 字符）
│   └── images/                     # 7 张测试图片
├── models/                         # ONNX 模型（需单独下载，~180 MB）
├── docs/
│   ├── ACCURACY_ALIGNMENT.md       # 6 轮精度对齐过程
│   ├── BENCHMARK_RESULTS.md        # 完整 benchmark 表格
│   └── PIPELINE_ARCHITECTURE.md    # 4 模型流水线详解
├── scripts/download_models.py      # 模型验证工具
└── examples/quickstart.py          # 最小使用示例
```

## 文档

| 文档 | 内容 |
|------|------|
| [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | 4 模型流水线、预处理参数、batch 策略 |
| [Accuracy Alignment](docs/ACCURACY_ALIGNMENT.md) | 从 65.6% 到 100% 的 6 轮调试之旅 |
| [Benchmark Results](docs/BENCHMARK_RESULTS.md) | 完整速度/精度表格、逐模型 KleidiAI 分析 |

## 环境要求

| 包 | 版本 | 说明 |
|----|------|------|
| Python | >= 3.10 | |
| onnxruntime | >= 1.21.0 | >= 1.22.0 启用 KleidiAI |
| opencv-python-headless | >= 4.8.0 | |
| numpy | >= 1.24.0 | |
| pyclipper | >= 1.3.0 | DB 后处理 |

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv5 模型和原始推理流水线
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — 跨平台推理引擎
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) — Arm CPU 微内核加速库

## 引用

如果你使用了 PP-OCRv5 模型，请引用 PaddleOCR 3.0 技术报告：

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
- 模型 & 在线体验: <https://huggingface.co/PaddlePaddle>

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
