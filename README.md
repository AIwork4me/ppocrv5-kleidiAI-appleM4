<div align="center">

# ppocrv5-kleidiAI-appleM4

**PP-OCRv5 on ONNX Runtime + Arm KleidiAI | 100% Accuracy Aligned with PaddleOCR | Apple M4 Benchmark**

English | [中文](README_CN.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-≥1.22-purple.svg)](https://onnxruntime.ai)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v5_Server-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![KleidiAI](https://img.shields.io/badge/KleidiAI-Arm_SME+I8MM-green.svg)](https://gitlab.arm.com/kleidi/kleidiai)
[![Platform](https://img.shields.io/badge/Platform-macOS_ARM64-lightgrey.svg)]()

</div>

A production-ready, single-file PP-OCRv5 inference pipeline using ONNX Runtime, delivering **1.72x speedup** over PaddleOCR native inference (ORT 1.23.2, 8 threads) with **100% text-level accuracy alignment** — verified on 228 text regions across 7 images with zero mismatch.

## Highlights

- **1.72x faster** than Paddle native inference on Apple M4 (ORT 1.23.2, KleidiAI I8MM GEMM)
- **100% accuracy match** — 228/228 texts identical, confidence diff < 0.00002
- **Single-file deployment** — `ppocrv5_onnx.py` (~720 lines), copy-paste into any ARM app
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
<td>1.48x </td>
<td>228/228 (100%)</td>
<td>0.000019</td>
</tr>
<tr>
<td><b>ORT 1.23.2 (KleidiAI)</b></td>
<td><b>5,486 ms</b></td>
<td><b>1.72x </b></td>
<td><b>228/228 (100%)</b></td>
<td><b>0.000019</b></td>
</tr>
<tr>
<td>ORT 1.24.3 (KleidiAI + SME Conv)</td>
<td>7,842 ms</td>
<td>1.21x faster</td>
<td>228/228 (100%)</td>
<td>0.000019</td>
</tr>
</table>

> Measured on Apple M4, macOS ARM64, 8 threads, 7 images, 3 runs/image (ORT 1.24.3: 1 run, 0 warmup).
> ORT 1.24.3 is slower than 1.23.2 at 8 threads due to SME device contention — see [SME Thread Scaling](#sme-thread-scaling-on-apple-silicon) below.
> Reproduce: `python benchmarks/benchmark_unified.py --backend ort --num-runs 3`

<details>
<summary><b>KleidiAI per-model speedup (ORT 1.21.1 → 1.23.2, inference ms per image)</b></summary>

| Model | ORT 1.21.1 (ms) | ORT 1.23.2 (ms) | Speedup | Role |
|-------|---------------:|---------------:|:-------:|------|
| doc_ori | 6.16 | 3.22 | **1.91x** | Document orientation (4-class) |
| textline_ori | 219.99 | 118.23 | **1.86x** | Text line orientation (2-class) |
| rec | 4,786.54 | 3,962.88 | **1.21x** | Text recognition (CTC) |
| det | 1,305.97 | 1,311.58 | 1.00x | Text detection (DB, large-kernel Conv) |

KleidiAI accelerates GEMM-dominated models (doc_ori, textline_ori, rec) via Arm I8MM GEMM kernels. The det model shows 1.00x at threads=8 because ORT 1.23.2 does not yet have SME Conv kernels. In ORT 1.24.x, KleidiAI's SME Conv kernels are **2.8x faster at single-thread**, but cannot scale with multiple threads due to Apple M4's limited SME devices (2 total). See [docs/SME_THREAD_SCALING.md](docs/SME_THREAD_SCALING.md) for details.

</details>

<details>
<summary><b>Per-image latency breakdown</b></summary>

| Image | Texts | Paddle 3.3.0 | ORT 1.21.1 | ORT 1.23.2 (KleidiAI) |
|-------|:-----:|------------:|----------:|---------------------:|
| ancient_demo.png | 12 | 2,722 ms | 2,239 ms | 1,913 ms |
| handwrite_ch_demo.png | 10 | 1,816 ms | 1,146 ms | 985 ms |
| handwrite_en_demo.png | 11 | 2,122 ms | 1,397 ms | 1,223 ms |
| japan_demo.png | 28 | 18,994 ms | 7,916 ms | 6,933 ms |
| magazine.png | 65 | 18,494 ms | 13,843 ms | 11,661 ms |
| magazine_vetical.png | 65 | 16,924 ms | 14,073 ms | 12,138 ms |
| pinyin_demo.png | 37 | 5,088 ms | 4,235 ms | 3,552 ms |

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

ORT >= 1.24 adds KleidiAI **FP32 SME Conv kernels** that use ARM's Scalable Matrix Extension. On Apple M4, SME is a **shared coprocessor** (2 devices total), not per-core like NEON. This creates a thread scaling trade-off:

| Threads | NEON (ORT < 1.22) | SME (ORT >= 1.24) |
|:-------:|:------------------:|:------------------:|
| 1 | Baseline | **2.8x faster** |
| 2 | ~2x faster | **~2.4x faster** (optimal for SME) |
| 8 | **~4x faster** (optimal for NEON) | ~1.2x faster (contention) |

**Recommended configurations for Apple M4:**

| Goal | ORT Version | Threads | Disable KleidiAI? |
|------|:-----------:|:-------:|:------------------:|
| Best pipeline throughput | **1.23.2** | **8** | N/A (no SME Conv) |
| Lowest single-model latency | >= 1.24 | 2 | No |
| Full thread scaling on latest ORT | >= 1.24 | 8 | Yes (`mlas.disable_kleidiai=1`) |

```python
# For ORT >= 1.24: disable KleidiAI to use NEON at 8 threads
opts = ort.SessionOptions()
opts.add_session_config_entry("mlas.disable_kleidiai", "1")
opts.intra_op_num_threads = 8
```

See [docs/SME_THREAD_SCALING.md](docs/SME_THREAD_SCALING.md) for the full analysis, experimental data, and background.

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

`ppocrv5_onnx.py` is a **single-file module** (~720 lines) with minimal dependencies. Copy it directly into your project:

```python
from ppocrv5_onnx import PPOCRv5Pipeline

pipeline = PPOCRv5Pipeline(
    model_dir="path/to/onnx/models",
    dict_path="path/to/ppocrv5_dict.txt",
    threads=4,  # See docs/SME_THREAD_SCALING.md for thread tuning on Apple Silicon
)
results = pipeline.predict(bgr_image_array)  # accepts file path or BGR ndarray
# [{"text": "...", "confidence": 0.98, "bounding_box": [[x,y], ...]}, ...]
```

**Dependencies**: `onnxruntime`, `opencv-python-headless`, `numpy`, `pyclipper`

## Reproduce Benchmarks

```bash
# ORT benchmark (recommended: ORT 1.23.2 at 8 threads for best pipeline throughput)
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# ORT benchmark with KleidiAI disabled (for ORT >= 1.24, NEON fallback)
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --disable-kleidiai

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
│   ├── ort_1.23.2.json
│   └── ort_1.24.3.json
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
