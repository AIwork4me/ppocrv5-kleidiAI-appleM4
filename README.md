<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 on ONNX Runtime + Arm KleidiAI | 100% Accuracy Aligned with PaddleOCR | Apple M4 Benchmark**

English | [中文](README_CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.22-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_I8MM-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

A production-ready, single-file PP-OCRv5 inference pipeline using ONNX Runtime, delivering **1.72x speedup** over PaddleOCR native inference with **100% text-level accuracy alignment** — verified on 228 text regions across 7 images with zero mismatch.

## Highlights

- **1.72x faster** than Paddle native inference on Apple M4 (KleidiAI auto-enabled)
- **100% accuracy match** — 228/228 texts identical, confidence diff < 0.00002
- **Single-file deployment** — `ppocrv5_onnx.py` (~550 lines), copy-paste into any ARM app
- **Reproducible benchmarks** — run on your own platform in 3 commands
- **Zero accuracy loss** from KleidiAI acceleration (ORT 1.21 vs 1.23: 0.000000 confidence diff)

## Benchmark Results (Apple M4)

<table>
<tr>
<th>Backend</th>
<th>Avg Latency</th>
<th>vs Paddle</th>
<th>Text Match</th>
<th>Conf Diff</th>
</tr>
<tr>
<td>Paddle 3.3.0</td>
<td>9,451 ms</td>
<td>1.00x</td>
<td>baseline</td>
<td>—</td>
</tr>
<tr>
<td>ORT 1.21.1 (no KleidiAI)</td>
<td>6,407 ms</td>
<td>1.48x faster</td>
<td>228/228 (100%)</td>
<td>0.000019</td>
</tr>
<tr>
<td><b>ORT 1.23.2 (KleidiAI)</b></td>
<td><b>5,486 ms</b></td>
<td><b>1.72x faster</b></td>
<td><b>228/228 (100%)</b></td>
<td><b>0.000019</b></td>
</tr>
</table>

> Measured on Apple M4, macOS ARM64, 8 threads, 7 images, 3 runs/image.
> Reproduce: `python benchmarks/benchmark_unified.py --backend ort --num-runs 3`

<details>
<summary><b>KleidiAI per-model speedup (ORT 1.21.1 → 1.23.2)</b></summary>

| Model | ORT 1.21.1 (ms) | ORT 1.23.2 (ms) | Speedup | Role |
|-------|---------------:|---------------:|:-------:|------|
| doc_ori | 2.57 | 1.34 | **1.91x** | Document orientation (4-class) |
| textline_ori | 67.88 | 36.51 | **1.86x** | Text line orientation (2-class) |
| rec | 1,599.89 | 1,319.33 | **1.21x** | Text recognition (CTC) |
| det | 3,779.37 | 3,788.58 | 1.00x | Text detection (DB, large-kernel Conv) |

KleidiAI accelerates GEMM-dominated models (classification, recognition) via Arm I8MM instructions. Detection is dominated by large-kernel convolutions (9x9), which are not GEMM-bound.

</details>

<details>
<summary><b>Per-image latency breakdown</b></summary>

| Image | Texts | Paddle 3.3.0 | ORT 1.21.1 | ORT 1.23.2 (KleidiAI) |
|-------|:-----:|------------:|----------:|---------------------:|
| ancient_demo.png | 12 | 2,958 ms | 2,379 ms | 2,086 ms |
| handwrite_ch_demo.png | 10 | 1,834 ms | 1,230 ms | 1,064 ms |
| handwrite_en_demo.png | 11 | 2,422 ms | 1,620 ms | 1,395 ms |
| japan_demo.png | 28 | 14,017 ms | 8,606 ms | 7,313 ms |
| magazine.png | 65 | 24,095 ms | 14,625 ms | 12,519 ms |
| magazine_vertical.png | 65 | 17,279 ms | 14,803 ms | 12,803 ms |
| pinyin_demo.png | 37 | 3,553 ms | 1,585 ms | 1,220 ms |

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

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/user/ppocrv5-kleidiAI-appleM4.git
cd ppocrv5-kleidiAI-appleM4
pip install onnxruntime>=1.22.0 opencv-python-headless numpy pyclipper
```

### 2. Download models

Download from [Baidu Pan](https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw) (password: `uepw`), place under `models/`. See [models/README.md](models/README.md) for expected layout.

```bash
python scripts/download_models.py  # verify models are in place
```

### 3. Run OCR

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline("models", dict_path="data/dict/ppocrv5_dict.txt")
results = pipeline.predict("image.png")

for r in results:
    print(f"{r['text']}  ({r['confidence']:.4f})")
```

## Integration

`ppocrv5_onnx.py` is a **single-file module** (~550 lines) with minimal dependencies. Copy it directly into your project:

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=4,
)
results = pipeline.predict(bgr_image_array)  # accepts file path or BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**Dependencies**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## Reproduce Benchmarks

```bash
# ORT benchmark (recommended: ORT >= 1.22 for KleidiAI)
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

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
│   └── ort_1.23.2.json
├── data/
│   ├── dict/ppocrv5_dict.txt      # Character dictionary (18,383 chars)
│   └── images/                     # 7 test images
├── models/                         # ONNX models (download separately, ~180 MB)
├── docs/
│   ├── ACCURACY_ALIGNMENT.md       # 6-round alignment process
│   ├── BENCHMARK_RESULTS.md        # Full benchmark tables
│   └── PIPELINE_ARCHITECTURE.md    # 4-model pipeline details
├── scripts/download_models.py      # Model verification tool
└── examples/quickstart.py          # Minimal usage example
```

## Documentation

| Document | Description |
|----------|-------------|
| [Pipeline Architecture](docs/PIPELINE_ARCHITECTURE.md) | 4-model pipeline, preprocessing parameters, batch strategy |
| [Accuracy Alignment](docs/ACCURACY_ALIGNMENT.md) | 6-round debugging journey from 65.6% to 100% |
| [Benchmark Results](docs/BENCHMARK_RESULTS.md) | Full speed/accuracy tables, per-model KleidiAI analysis |

## Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python | >= 3.10 | |
| onnxruntime | >= 1.21.0 | >= 1.22.0 for KleidiAI |
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
