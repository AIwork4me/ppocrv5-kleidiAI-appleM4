<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 on ONNX Runtime + Arm KleidiAI | 100% Accuracy Aligned with PaddleOCR | Apple M4 Benchmark | ORT Version Comparison**

English | [中文](README_CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.21-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_SME2-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

A production-ready, single-file PP-OCRv5 inference pipeline using ONNX Runtime, featuring **up to 2.0x pipeline speedup** via KleidiAI SME2 (ORT 1.21.1 → 1.24.3 on Apple M4) with **100% text-level accuracy alignment** with PaddleOCR — verified on 228 text regions across 7 images with zero mismatch.

## Highlights

- **Up to 2.0x pipeline speedup** via KleidiAI SME2 (ORT 1.21.1 → 1.24.3 at t=1 on Apple M4); rec model **4.4x faster**
- **100% accuracy match** with PaddleOCR — 228/228 texts identical, confidence diff < 0.00002
- **Single-file deployment** — `ppocrv5_onnx.py` (~720 lines), copy-paste into any ARM app
- **Reproducible benchmarks** — ORT 1.21.1 vs 1.24.3 across t=1, t=2, t=8; run on your own platform in 3 commands
- **KleidiAI SME2 analysis** — investigated with ARM in [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633)

## Benchmark Results (Apple M4)

### Speed: ORT 1.21.1 vs ORT 1.24.3

<table>
<tr>
<th>Threads</th>
<th>ORT 1.21.1 (NEON)</th>
<th>ORT 1.24.3 (KleidiAI SME2)</th>
<th>Speedup</th>
</tr>
<tr>
<td>t=1</td>
<td>16,909 ms</td>
<td>8,295 ms</td>
<td><b>2.04x faster</b></td>
</tr>
<tr>
<td><b>t=2 (recommended)</b></td>
<td>9,346 ms</td>
<td><b>6,332 ms</b></td>
<td><b>1.48x faster</b></td>
</tr>
<tr>
<td>t=8</td>
<td>6,497 ms</td>
<td>7,096 ms</td>
<td>0.92x (det regression)</td>
</tr>
</table>

### Accuracy Alignment with PaddleOCR

| Comparison | Texts | Match Rate | Avg Confidence Diff |
|:---|:---:|:---:|:---:|
| PaddleOCR 3.3.0 vs ORT 1.21.1 | 228 | **100.0%** | 0.000019 |
| PaddleOCR 3.3.0 vs ORT 1.24.3 | 228 | **100.0%** | 0.000019 |

> All benchmarks: 3 runs/image, 1 warmup. All ORT configurations produce 100% identical text output to PaddleOCR native inference.
> Reproduce: `python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2`

**Why does ORT 1.24.3 at t=2 beat t=8?** ORT 1.24.3's KleidiAI SME2 kernels dramatically accelerate rec (2.5x) and textline_ori (2.7x), but the det model regresses on high-resolution images due to a **large-kernel Conv regression** in ORT 1.24.x (kernel ≥ 7×7 on large spatial inputs). This regression is NOT caused by SME contention — it persists at t=2 and with `disable-kleidiai`. Factor decomposition: the 3.2x det gap at t=2 (ORT 1.24.3 vs 1.21.1) = 1.94x (thread count: t=2 vs t=8) × 1.65x (Conv kernel regression). The pipeline still wins overall because rec's 2.5x speedup outweighs det's regression. At t=1, the speedup is most dramatic (**2.04x**) because NEON can't parallelize at all, while KleidiAI SME2 still delivers full acceleration. See [SME Thread Scaling](#sme-thread-scaling-on-apple-silicon) for the full analysis.

<details>
<summary><b>Per-model inference breakdown (ms, averaged across 7 images)</b></summary>

| Model | ORT 1.21.1 (t=1) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=1) | ORT 1.24.3 (t=2) | KleidiAI Speedup (t=1) |
|-------|-------------------:|------------------:|------------------:|------------------:|:---:|
| doc_ori | 6.53 | 6.63 | 6.68 | **3.41** | 0.98x |
| det | 4,876.13 | **1,328.30** | 5,440.39 | 4,246.58 | 0.90x (regression) |
| textline_ori | 264.21 | 222.09 | 123.32 | **80.84** | **2.14x** |
| rec | 11,703.89 | 4,850.45 | 2,647.07 | **1,931.17** | **4.42x** |

Key insight: KleidiAI SME2 delivers massive acceleration on rec (4.4x at t=1) and textline_ori (2.1x at t=1). The det model regresses at high resolution, but rec dominates total pipeline time, so the net effect is a **2.04x pipeline speedup at t=1** and **1.48x at t=2**.

</details>

<details>
<summary><b>Per-image latency breakdown</b></summary>

| Image | Texts | ORT 1.21.1 (t=1) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=1) | ORT 1.24.3 (t=2) |
|-------|:-----:|------------------:|------------------:|------------------:|------------------:|
| ancient_demo.png | 12 | 5,882 ms | 2,246 ms | 1,391 ms | **1,022 ms** |
| handwrite_ch_demo.png | 10 | 3,125 ms | 1,144 ms | 1,096 ms | **673 ms** |
| handwrite_en_demo.png | 11 | 3,907 ms | 1,429 ms | 2,296 ms | **828 ms** |
| japan_demo.png | 28 | 22,953 ms | **7,971 ms** | 19,560 ms | 18,425 ms |
| magazine.png | 65 | 36,143 ms | 14,025 ms | 15,990 ms | **11,601 ms** |
| magazine_vetical.png | 65 | 36,115 ms | 14,341 ms | 14,984 ms | **9,732 ms** |
| pinyin_demo.png | 37 | 10,235 ms | 4,323 ms | 2,749 ms | **2,039 ms** |

</details>

## Pipeline Architecture

```
┌─────────┐     ┌──────────┐     ┌───────┐     ┌──────────────┐     ┌───────┐
│  Image   │────▶│ doc_ori  │────▶│  det  │────▶│ textline_ori │────▶│  rec  │────▶ Results
│ (BGR)    │     │ 4-class  │     │  DB   │     │   2-class    │     │  CTC  │     [{text,
└─────────┘     │ rotation │     │ boxes │     │  rotation    │     │ decode│      conf,
                └──────────┘     └───────┘     └──────────────┘     └───────┘      bbox}]
                  LCNet           PP-OCRv5       LCNet              PP-OCRv5
                  224×224         HxW→stride32   160×80              48×W
```

See [docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) for preprocessing parameters and implementation details.

## SME Thread Scaling on Apple Silicon

> Investigated in [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) — resolved with ORT maintainer confirmation.

ORT 1.24.x uses KleidiAI **SME2 kernels** (SGEMM, IGEMM Conv, Dynamic QGemm) that leverage ARM's Scalable Matrix Extension. On Apple M4, SME is a **shared coprocessor** (2 devices total), not per-core like NEON. Combined with a large-kernel Conv regression in ORT 1.24.x, this creates a per-model trade-off:

| Threads | det (Conv-heavy) | rec (GEMM-heavy) | Pipeline Total |
|:-------:|:-----------------:|:-----------------:|:--------------:|
| ORT 1.21.1, 1 (NEON) | 4,876 ms | 11,704 ms | 16,909 ms |
| ORT 1.21.1, 2 (NEON) | 2,571 ms | 6,523 ms | 9,346 ms |
| ORT 1.21.1, 8 (NEON) | **1,328 ms** | 4,850 ms | 6,497 ms |
| ORT 1.24.3, 1 (SME2) | 5,440 ms | 2,647 ms | 8,295 ms |
| **ORT 1.24.3, 2 (SME2)** | 4,247 ms | **1,931 ms** | **6,332 ms** |
| ORT 1.24.3, 8 (SME2) | 4,312 ms | 2,579 ms | 7,096 ms |

At t=1: rec is **4.4x faster** on ORT 1.24.3, giving a **2.04x pipeline speedup**. At t=2: pipeline is **1.48x faster**. At t=8: det benefits from NEON parallelism but rec's SME2 advantage is diluted by contention.

**Recommended: `threads=2`** for ORT >= 1.24 on Apple M4.

```python
# Recommended for Apple M4 with ORT >= 1.24
pipeline = PPOCRv5Pipeline(model_dir, dict_path=dict_path, threads=2)
```

See [docs/SME_THREAD_SCALING.md](docs/SME_THREAD_SCALING.md) for the full analysis, experimental data, and background.

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/AIwork4me/ppocrv5-kleidiAI-appleM4.git
cd ppocrv5-kleidiAI-appleM4
pip install onnxruntime>=1.21.0 opencv-python-headless numpy pyclipper
```

### 2. Download models

Download from [Baidu Pan](https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw) (password: `uepw`), place under `models/`. See [models/README.md](models/README.md) for expected layout.

```bash
python scripts/download_models.py  # verify models are in place
```

### 3. Run OCR

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline("models", dict_path="data/dict/ppocrv5_dict.txt", threads=2)
results = pipeline.predict("image.png")

for r in results:
    print(f"{r['text']}  ({r['confidence']:.4f})")
```

## Integration

`ppocrv5_onnx.py` is a **single-file module** (~720 lines) with minimal dependencies. Copy it directly into your project:

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=2,  # Recommended for Apple M4. See docs/SME_THREAD_SCALING.md
)
results = pipeline.predict(bgr_image_array)  # accepts file path or BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**Dependencies**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## Reproduce Benchmarks

```bash
# ORT 1.24.3, threads=2 (recommended, best pipeline throughput on Apple M4)
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2

# ORT 1.24.3, threads=1 (shows maximum KleidiAI SME2 advantage)
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 1

# ORT 1.24.3, threads=8 (shows det regression at high thread count)
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# ORT 1.24.3, threads=8, KleidiAI disabled (NEON fallback)
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8 --disable-kleidiai

# ORT 1.21.1 (NEON baseline)
pip install onnxruntime==1.21.1
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 1
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# Paddle (for accuracy verification only — not for speed comparison)
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3

# Compare all results in results/
python benchmarks/compare_results.py
```

Results are saved to `results/*.json` and can be compared across platforms.

## Accuracy Alignment

The ONNX pipeline produces **100% identical** text output to PaddleOCR/PaddleX 3.4.x native inference, achieved through 6 rounds of systematic debugging:

| Round | Fix | Match Rate |
|:-----:|-----|:----------:|
| 1 | CTC decode, normalize, box sorting, ... | 65.6% → 71.8% |
| 3 | **det resize params** (Pipeline runtime overrides inference.yml) | → 90.8% |
| 5 | **crop coordinate precision** (int16 → minAreaRect float32) | → 93.3% |
| 6 | **rec batch padding** (batch_size=6, ratio sort, per-batch pad) | → **100.0%** |

See [docs/ACCURACY_ALIGNMENT.md](docs/ACCURACY_ALIGNMENT.md) for the full story and key insights.

## Project Structure

```
ppocrv5-kleidiAI-appleM4/
├── ppocrv5_onnx.py                 # Core: single-file inference pipeline
├── benchmarks/
│   ├── benchmark_unified.py        # Unified benchmark (--backend paddle|ort)
│   └── compare_results.py          # Multi-backend comparison report
├── results/                        # ORT version comparison data (Apple M4)
│   ├── ort_1.21.1_t1.json
│   ├── ort_1.21.1_t2.json
│   ├── ort_1.21.1.json
│   ├── ort_1.24.3_t1.json
│   ├── ort_1.24.3_t2.json          # KleidiAI SME2, threads=2 (recommended)
│   ├── ort_1.24.3.json             # KleidiAI SME2, threads=8
│   ├── ort_1.24.3_no_kleidiai.json # KleidiAI disabled, threads=8
│   └── paddle_3.3.0.json          # For accuracy verification only
├── data/
│   ├── dict/ppocrv5_dict.txt      # Character dictionary (18,383 chars)
│   └── images/                     # 7 test images
├── models/                         # ONNX models (download separately, ~180 MB)
├── docs/
│   ├── ACCURACY_ALIGNMENT.md       # 6-round alignment process
│   ├── BENCHMARK_RESULTS.md        # Full benchmark tables
│   ├── PIPELINE_ARCHITECTURE.md    # 4-model pipeline details
│   └── SME_THREAD_SCALING.md       # KleidiAI SME thread scaling analysis
├── scripts/download_models.py      # Model verification tool
└── examples/quickstart.py          # Minimal usage example
```

## Documentation

| Document | Description |
|----------|-------------|
| [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | 4-model pipeline, preprocessing parameters, batch strategy |
| [Accuracy Alignment](docs/ACCURACY_ALIGNMENT.md) | 6-round debugging journey from 65.6% to 100% |
| [Benchmark Results](docs/BENCHMARK_RESULTS.md) | Full speed/accuracy tables, per-model KleidiAI analysis |
| [SME Thread Scaling](docs/SME_THREAD_SCALING.md) | KleidiAI SME contention on Apple Silicon, thread tuning guide |

## Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | >= 3.10 | |
| onnxruntime | >= 1.21.0 | >= 1.24 for KleidiAI SME2 |
| opencv-python-headless | >= 4.8.0 | |
| numpy | >= 1.24.0 | |
| pyclipper | >= 1.3.0 | DB post-processing |

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCRv5 models and the original inference pipeline
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Cross-platform inference engine
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) — Arm CPU micro-kernel library for accelerated ML inference

## Citation

If you use PP-OCRv5 models in your work, please cite the PaddleOCR 3.0 Technical Report:

```bibtex
@article{cui2025paddleocr,
  title={PaddleOCR 3.0 Technical Report},
  author={Cui, Cheng and Sun, Ting and Lin, Manhui and Gao, Tingquan and Zhang, Yubo and Liu, Jiaxuan and Wang, Xueqing and Zhang, Zelun and Zhou, Changda and Liu, Hongen and Zhang, Yue and Lv, Wenyu and Huang, Kui and Zhang, Yichao and Zhang, Jing and Zhang, Jun and Liu, Yi and Yu, Dianhai and Ma, Yanjun},
  journal={arXiv preprint arXiv:2507.05595},
  year={2025}
}
```

- Paper: <https://arxiv.org/abs/2507.05595>
- Source Code: <https://github.com/PaddlePaddle/PaddleOCR>
- Document: <https://paddlepaddle.github.io/PaddleOCR>
- Models & Online Demo: <https://huggingface.co/PaddlePaddle>

## License

This project is licensed under the [Apache License 2.0](LICENSE).
