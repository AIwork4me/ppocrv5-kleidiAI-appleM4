<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 on ONNX Runtime + Arm KleidiAI | 100% Accuracy Aligned with PaddleOCR | Apple M4 Benchmark**

English | [中文](README_CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.21-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_SME2-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

A production-ready, single-file PP-OCRv5 inference pipeline using ONNX Runtime, delivering **1.51x speedup** over PaddleOCR native inference on Apple M4 (ORT 1.24.3, 2 threads, KleidiAI SME2) with **100% text-level accuracy alignment** — verified on 228 text regions across 7 images with zero mismatch.

## Highlights

- **1.51x faster** than Paddle native inference on Apple M4 (ORT 1.24.3, threads=2)
- **100% accuracy match** — 228/228 texts identical, confidence diff < 0.00002
- **Single-file deployment** — `ppocrv5_onnx.py` (~720 lines), copy-paste into any ARM app
- **Reproducible benchmarks** — 5 configurations tested, run on your own platform in 3 commands
- **KleidiAI SME2 analysis** — investigated with ARM in [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633)

## Benchmark Results (Apple M4)

<table>
<tr>
<th>Configuration</th>
<th>Avg Latency</th>
<th>vs Paddle</th>
<th>Text Match</th>
</tr>
<tr>
<td>Paddle 3.3.0 (t=8)</td>
<td>9,567 ms</td>
<td>1.00x</td>
<td>baseline</td>
</tr>
<tr>
<td>ORT 1.21.1 (t=8, no KleidiAI)</td>
<td>6,497 ms</td>
<td>1.47x faster</td>
<td>228/228 ✓</td>
</tr>
<tr>
<td><b>ORT 1.24.3 (t=2, KleidiAI SME2)</b></td>
<td><b>6,332 ms</b></td>
<td><b>1.51x faster</b></td>
<td><b>228/228 ✓</b></td>
</tr>
<tr>
<td>ORT 1.24.3 (t=8, KleidiAI SME2)</td>
<td>7,096 ms</td>
<td>1.35x faster</td>
<td>228/228 ✓</td>
</tr>
<tr>
<td>ORT 1.24.3 (t=8, KleidiAI disabled)</td>
<td>7,155 ms</td>
<td>1.34x faster</td>
<td>228/228 ✓</td>
</tr>
</table>

> All benchmarks: 3 runs/image, 1 warmup. All backends produce 100% identical text, confidence diff < 0.00002.
> Reproduce: `python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2`

**Why does ORT 1.24.3 at t=2 beat t=8?** KleidiAI in ORT 1.24.3 uses ARM **SME2** (Scalable Matrix Extension) for both GEMM and Conv operations. On Apple M4, SME is a **shared coprocessor** (2 devices per cluster), not per-core like NEON. At threads > 2, all threads contend for these 2 devices — the Conv-heavy det model regresses from 1,328 ms (ORT 1.21.1 NEON, t=8) to 4,312 ms. At threads=2, SME contention is minimal, and rec/textline_ori benefit massively from SME2 acceleration. See [SME Thread Scaling](#sme-thread-scaling-on-apple-silicon) for the full analysis.

<details>
<summary><b>Per-model inference breakdown (ms, averaged across 7 images)</b></summary>

| Model | Paddle 3.3.0 (t=8) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) |
|-------|-------------------:|------------------:|------------------:|------------------:|
| doc_ori | 9.65 | 6.63 | **3.41** | 4.82 |
| det | 4,296.21 | **1,328.30** | 4,246.58 | 4,311.64 |
| textline_ori | 444.61 | 222.09 | **80.84** | 98.81 |
| rec | 4,728.09 | 4,850.45 | **1,931.17** | 2,579.15 |

Key insight: ORT 1.21.1 (NEON, t=8) is best for det alone, but ORT 1.24.3 (SME2, t=2) wins overall because rec and textline_ori are **2.5x** and **2.7x** faster via SME2.

</details>

<details>
<summary><b>Per-image latency breakdown</b></summary>

| Image | Texts | Paddle (t=8) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) |
|-------|:-----:|------------:|------------------:|------------------:|------------------:|
| ancient_demo.png | 12 | 2,707 ms | 2,246 ms | **1,022 ms** | 1,340 ms |
| handwrite_ch_demo.png | 10 | 1,824 ms | 1,144 ms | **673 ms** | 1,175 ms |
| handwrite_en_demo.png | 11 | 2,121 ms | 1,429 ms | **828 ms** | 2,215 ms |
| japan_demo.png | 28 | 19,706 ms | **7,971 ms** | 18,425 ms | 14,549 ms |
| magazine.png | 65 | 17,947 ms | 14,025 ms | **11,601 ms** | 16,118 ms |
| magazine_vetical.png | 65 | 17,549 ms | 14,341 ms | **9,732 ms** | 11,599 ms |
| pinyin_demo.png | 37 | 5,113 ms | 4,323 ms | **2,039 ms** | 2,676 ms |

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

ORT 1.24.x uses KleidiAI **SME2 kernels** (SGEMM, IGEMM Conv, Dynamic QGemm) that leverage ARM's Scalable Matrix Extension. On Apple M4, SME is a **shared coprocessor** (2 devices total), not per-core like NEON. This creates a thread scaling trade-off:

| Threads | det (Conv-heavy) | rec (GEMM-heavy) | Pipeline Total |
|:-------:|:-----------------:|:-----------------:|:--------------:|
| 2 | 4,247 ms | **1,931 ms** | **6,332 ms** |
| 8 | 4,312 ms | 2,579 ms | 7,096 ms |
| 8 (no KleidiAI) | 4,266 ms | 2,677 ms | 7,155 ms |
| ORT 1.21.1, 8 (NEON) | **1,328 ms** | 4,850 ms | 6,497 ms |

**Recommended: `threads=2`** for ORT >= 1.24 on Apple M4. This minimizes SME contention while benefiting from SME2 acceleration on GEMM-heavy models.

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

# ORT 1.24.3, threads=8 (shows SME contention on det)
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# ORT 1.24.3, threads=8, KleidiAI disabled (NEON fallback)
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8 --disable-kleidiai

# ORT 1.21.1, threads=8 (NEON baseline, no KleidiAI)
pip install onnxruntime==1.21.1
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# Paddle benchmark (optional)
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
├── results/                        # Reference benchmark data (Apple M4)
│   ├── paddle_3.3.0.json
│   ├── ort_1.21.1.json
│   ├── ort_1.24.3.json             # KleidiAI SME2, threads=8
│   ├── ort_1.24.3_t2.json          # KleidiAI SME2, threads=2 (recommended)
│   └── ort_1.24.3_no_kleidiai.json # KleidiAI disabled, threads=8
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
