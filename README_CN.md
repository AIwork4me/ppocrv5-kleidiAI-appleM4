<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 + ONNX Runtime + Arm KleidiAI | 100% 精度对齐 PaddleOCR | Apple M4 基准测试**

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.22-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_SME+I8MM-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

面向生产环境的单文件 PP-OCRv5 推理流水线，基于 ONNX Runtime，在 Apple M4 上比 PaddleOCR 原生推理**快 1.72 倍**（ORT 1.23.2，8 线程），同时保持 **100% 文本级精度对齐** — 在 7 张图片 228 条文本上零误差验证。

## 亮点

- **1.72x 加速** — 相比 Paddle 原生推理（Apple M4，ORT 1.23.2，KleidiAI I8MM GEMM）
- **100% 精度一致** — 228/228 条文本完全匹配，置信度差异 < 0.00002
- **单文件部署** — `ppocrv5_onnx.py`（~720 行），可直接 copy-paste 到任何 ARM 应用
- **可复现基准** — 3 条命令即可在你自己的平台上运行
- **KleidiAI 零精度损失** — ORT 1.21 vs 1.23 置信度差异：0.000000

## 基准测试结果 (Apple M4, 8 线程)

<table>
<tr>
<th>后端</th>
<th>KleidiAI 内核</th>
<th>平均延迟</th>
<th>相对 Paddle</th>
<th>文本匹配</th>
</tr>
<tr>
<td>Paddle 3.3.0</td>
<td>—</td>
<td>9,451 ms</td>
<td>1.00x</td>
<td>基准</td>
</tr>
<tr>
<td>ORT 1.21.1</td>
<td>无</td>
<td>6,407 ms</td>
<td>1.48x 快</td>
<td>228/228 ✓</td>
</tr>
<tr>
<td><b>ORT 1.23.2</b></td>
<td><b>仅 I8MM GEMM</b></td>
<td><b>5,486 ms</b></td>
<td><b>1.72x 快</b></td>
<td><b>228/228 ✓</b></td>
</tr>
<tr>
<td>ORT 1.24.3 *</td>
<td>I8MM GEMM + SME Conv</td>
<td>7,842 ms</td>
<td>1.21x 快</td>
<td>228/228 ✓</td>
</tr>
</table>

> \* ORT 1.24.3 使用 1 次运行、0 次预热（其他版本：3 次运行、1 次预热）。
> 所有后端输出 100% 相同文本，置信度差异 < 0.00002。
> 复现命令：`python benchmarks/benchmark_unified.py --backend ort --num-runs 3`

**为什么 ORT 1.24.3 反而比 1.23.2 更慢？** ORT 1.24.3 新增了 KleidiAI 的 **SME Conv 内核**，使用 ARM 的可扩展矩阵扩展（SME）协处理器。在 Apple M4 上，SME 是**全芯片共享资源**（所有核心共享仅 2 个 SME 设备），而非像 NEON 一样每核独立。在 8 线程下，Conv 密集的 det 模型遭遇严重争抢（1,312 ms → 5,147 ms）。ORT 1.23.2 仅使用运行在 NEON 上的 I8MM GEMM 内核，线性扩展，因此在 8 线程下最快。详见 [SME 线程扩展](#apple-silicon-上的-sme-线程扩展)。

<details>
<summary><b>KleidiAI 逐模型分析：为什么 1.23.2 在 8 线程下胜过 1.24.3</b></summary>

| 模型 | ORT 1.21.1 | ORT 1.23.2 | ORT 1.24.3 * | 1.23.2 内核路径 | 1.24.3 内核路径 |
|------|----------:|----------:|----------:|:---:|:---:|
| doc_ori | 6.16 ms | 3.22 ms | 3.90 ms | I8MM GEMM | I8MM GEMM |
| det | 1,305.97 ms | 1,311.58 ms | **5,147.35 ms** | NEON Conv | **SME Conv**（争抢！） |
| textline_ori | 219.99 ms | 118.23 ms | 92.68 ms | I8MM GEMM | I8MM GEMM + SME Conv |
| rec | 4,786.54 ms | 3,962.88 ms | 2,497.21 ms | I8MM GEMM | I8MM GEMM + SME Conv |

关键发现：det 模型使用大核 Conv（9×9），在 ORT 1.24.3 中命中 SME Conv 路径。8 个线程竞争 2 个 SME 设备，导致 det **退化 3.9 倍**。而 rec 和 textline_ori 反而变快，因为它们的负载混合了 GEMM 和较小的 Conv 运算。

</details>

<details>
<summary><b>逐图延迟明细</b></summary>

| 图片 | 文本数 | Paddle 3.3.0 | ORT 1.21.1 | ORT 1.23.2 (KleidiAI) |
|------|:-----:|------------:|----------:|---------------------:|
| ancient_demo.png | 12 | 2,722 ms | 2,239 ms | 1,913 ms |
| handwrite_ch_demo.png | 10 | 1,816 ms | 1,146 ms | 985 ms |
| handwrite_en_demo.png | 11 | 2,122 ms | 1,397 ms | 1,223 ms |
| japan_demo.png | 28 | 18,994 ms | 7,916 ms | 6,933 ms |
| magazine.png | 65 | 18,494 ms | 13,843 ms | 11,661 ms |
| magazine_vetical.png | 65 | 16,924 ms | 14,073 ms | 12,138 ms |
| pinyin_demo.png | 37 | 5,088 ms | 4,235 ms | 3,552 ms |

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

## Apple Silicon 上的 SME 线程扩展

> 调研过程见 [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) — 已获 ORT 维护者确认。

ORT >= 1.24 新增了 KleidiAI **FP32 SME Conv 内核**，使用 ARM 的可扩展矩阵扩展（SME）。在 Apple M4 上，SME 是**共享协处理器**（共 2 个设备），而非像 NEON 一样每核独立。这带来了线程扩展的权衡：

| 线程数 | NEON (ORT < 1.22) | SME (ORT >= 1.24) |
|:------:|:------------------:|:------------------:|
| 1 | 基准 | **快 2.8 倍** |
| 2 | 约快 2 倍 | **约快 2.4 倍**（SME 最优） |
| 8 | **约快 4 倍**（NEON 最优） | 约快 1.2 倍（争抢） |

**Apple M4 推荐配置：**

| 目标 | ORT 版本 | 线程数 | 禁用 KleidiAI？ |
|------|:-------:|:-----:|:--------------:|
| 最佳 pipeline 吞吐 | **1.23.2** | **8** | N/A（无 SME Conv） |
| 最低单模型延迟 | >= 1.24 | 2 | 否 |
| 最新 ORT + 完整线程扩展 | >= 1.24 | 8 | 是（`mlas.disable_kleidiai=1`） |

```python
# ORT >= 1.24：禁用 KleidiAI 以在 8 线程下使用 NEON
opts = ort.SessionOptions()
opts.add_session_config_entry("mlas.disable_kleidiai", "1")
opts.intra_op_num_threads = 8
```

详见 [docs/SME_THREAD_SCALING.md](docs/SME_THREAD_SCALING.md) 获取完整分析、实验数据和背景。

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

`ppocrv5_onnx.py` 是**单文件模块**（~720 行），依赖极少。直接复制到你的项目即可：

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=4,  # Apple Silicon 线程调优详见 docs/SME_THREAD_SCALING.md
)
results = pipeline.predict(bgr_image_array)  # 支持文件路径或 BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**依赖**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## 复现 Benchmark

```bash
# ORT benchmark（推荐 ORT 1.23.2，8 线程可获最佳 pipeline 吞吐）
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# ORT benchmark 禁用 KleidiAI（ORT >= 1.24，回退 NEON）
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --disable-kleidiai

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
│   ├── ort_1.23.2.json
│   └── ort_1.24.3.json
├── data/
│   ├── dict/ppocrv5_dict.txt      # 字符字典（18,383 字符）
│   └── images/                     # 7 张测试图片
├── models/                         # ONNX 模型（需单独下载，~180 MB）
├── docs/
│   ├── ACCURACY_ALIGNMENT.md       # 6 轮精度对齐过程
│   ├── BENCHMARK_RESULTS.md        # 完整 benchmark 表格
│   ├── PIPELINE_ARCHITECTURE.md    # 4 模型流水线详解
│   └── SME_THREAD_SCALING.md       # KleidiAI SME 线程扩展分析
├── scripts/download_models.py      # 模型验证工具
└── examples/quickstart.py          # 最小使用示例
```

## 文档

| 文档 | 内容 |
|------|------|
| [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | 4 模型流水线、预处理参数、batch 策略 |
| [Accuracy Alignment](docs/ACCURACY_ALIGNMENT.md) | 从 65.6% 到 100% 的 6 轮调试之旅 |
| [Benchmark Results](docs/BENCHMARK_RESULTS.md) | 完整速度/精度表格、逐模型 KleidiAI 分析 |
| [SME Thread Scaling](docs/SME_THREAD_SCALING.md) | Apple Silicon 上 KleidiAI SME 争抢分析与线程调优指南 |

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
