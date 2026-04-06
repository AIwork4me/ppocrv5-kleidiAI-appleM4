# Benchmark Results

Reference benchmarks on Apple M4 (macOS ARM64).

## Speed Overview

| Configuration | Avg Latency | vs Paddle | Init Time | FPS |
|---|---:|:---:|---:|---:|
| Paddle 3.3.0 (t=8) | 9,567 ms | 1.00x | 2.65s | 0.1045 |
| ORT 1.21.1 (t=2) | 9,346 ms | 1.02x | 0.15s | 0.1070 |
| ORT 1.21.1 (t=8, no KleidiAI) | 6,497 ms | 1.47x faster | 0.16s | 0.1539 |
| **ORT 1.24.3 (t=2, KleidiAI SME2)** | **6,332 ms** | **1.51x faster** | **0.16s** | **0.1579** |
| ORT 1.24.3 (t=2, KleidiAI disabled) | 6,543 ms | 1.46x faster | 0.09s | 0.1528 |
| ORT 1.24.3 (t=8, KleidiAI SME2) | 7,096 ms | 1.35x faster | 0.16s | 0.1409 |
| ORT 1.24.3 (t=8, KleidiAI disabled) | 7,155 ms | 1.34x faster | 0.17s | 0.1398 |

## Per-Model Timing (avg ms per image)

> These values are from `aggregate_timing` in the result JSON files — the average across all 7 images. For textline_ori and rec, the count represents total text regions processed across all images and runs.

### ORT 1.21.1 (t=2)

| Model | Preprocess | Inference | Postprocess | Total |
|---|---:|---:|---:|---:|
| doc_ori | 0.45 | 4.94 | 0.49 | 5.88 |
| det | 3.14 | 2,571.48 | 2.02 | 2,576.64 |
| textline_ori (x684) | 1.57 | 187.45 | 0.08 | 189.10 |
| rec (x684) | 6.64 | 6,523.42 | 39.57 | 6,569.63 |
| **Total** | | | | **9,341.25** |

### ORT 1.21.1 (t=8, NEON only)

| Model | Preprocess | Inference | Postprocess | Total |
|---|---:|---:|---:|---:|
| doc_ori | 0.62 | 6.63 | 0.95 | 8.20 |
| det | 4.88 | 1,328.30 | 2.68 | 1,335.86 |
| textline_ori (x684) | 2.83 | 222.09 | 0.27 | 225.19 |
| rec (x684) | 9.53 | 4,850.45 | 59.79 | 4,919.77 |
| **Total** | | | | **6,489.02** |

### ORT 1.24.3 (t=2, KleidiAI SME2)

| Model | Preprocess | Inference | Postprocess | Total |
|---|---:|---:|---:|---:|
| doc_ori | 0.92 | 3.41 | 0.60 | 4.93 |
| det | 6.64 | 4,246.58 | 3.92 | 4,257.14 |
| textline_ori (x684) | 2.55 | 80.84 | 0.11 | 83.50 |
| rec (x684) | 9.57 | 1,931.17 | 39.61 | 1,980.35 |
| **Total** | | | | **6,325.92** |

### ORT 1.24.3 (t=8, KleidiAI SME2)

| Model | Preprocess | Inference | Postprocess | Total |
|---|---:|---:|---:|---:|
| doc_ori | 1.02 | 4.82 | 0.98 | 6.82 |
| det | 5.47 | 4,311.64 | 5.78 | 4,322.89 |
| textline_ori (x684) | 3.68 | 98.81 | 0.29 | 102.78 |
| rec (x684) | 11.57 | 2,579.15 | 64.57 | 2,655.29 |
| **Total** | | | | **7,087.78** |

## KleidiAI Acceleration by Model (ORT 1.21.1 → 1.24.3)

| Model | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) | Notes |
|---|---:|---:|---:|---|
| doc_ori | 6.63 ms | 3.41 ms (1.94x) | 4.82 ms (1.38x) | SME2 SGEMM |
| det | 1,328.30 ms | 4,246.58 ms (0.31x) | 4,311.64 ms (0.31x) | Conv kernel regression (large spatial inputs) |
| textline_ori | 222.09 ms | 80.84 ms (2.75x) | 98.81 ms (2.25x) | SME2 SGEMM + Conv |
| rec | 4,850.45 ms | 1,931.17 ms (2.51x) | 2,579.15 ms (1.88x) | SME2 SGEMM + Conv |

The det regression is resolution-dependent: images below ~500K pixels are actually faster on ORT 1.24.3. The regression only appears on high-resolution inputs (> ~1M pixels) and is NOT caused by SME contention.

**Key insight**: KleidiAI SME2 dramatically accelerates GEMM-heavy models (rec 2.5x, textline_ori 2.7x) but the Conv-heavy det model regresses due to a large-kernel Conv kernel regression in ORT 1.24.x affecting high-resolution inputs. The net effect is positive at t=2 (pipeline 1.51x vs Paddle) because rec dominates total latency.

## Per-Image Det Analysis

The det regression is strictly resolution-dependent:

| Image | Det Input | Pixels | ORT 1.21.1 (t=2) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | 1.24.3/1.21.1 (t=2) |
|---|:-:|---:|---:|---:|---:|:---:|
| ancient_demo.png | 480×672 | 322K | 889 ms | 480 ms | 369 ms | 0.41x faster |
| handwrite_ch_demo.png | 512×672 | 344K | 936 ms | 514 ms | 436 ms | 0.47x faster |
| handwrite_en_demo.png | 608×736 | 447K | 1,236 ms | 692 ms | 554 ms | 0.45x faster |
| pinyin_demo.png | 672×704 | 473K | 1,385 ms | 758 ms | 754 ms | 0.54x faster |
| magazine.png | 1472×992 | 1,460K | 4,070 ms | 2,081 ms | 6,620 ms | 1.63x slower |
| magazine_vetical.png | 992×1472 | 1,460K | 4,198 ms | 2,134 ms | 4,895 ms | 1.17x slower |
| japan_demo.png | 1216×1600 | 1,946K | 5,288 ms | 2,640 ms | 16,098 ms | 3.04x slower |

**Threshold**: ~500K pixels. Below this, ORT 1.24.3 is faster even at t=2 vs ORT 1.21.1 t=8. Above ~1M pixels, the large-kernel Conv regression dominates.

## Accuracy

All backends produce **100% identical** text output:

| Comparison | Texts | Match Rate | Avg Confidence Diff |
|:---:|:---:|:---:|:---:|
| Paddle vs ORT 1.21.1 | 228 | **100.0%** | 0.000019 |
| Paddle vs ORT 1.24.3 | 228 | **100.0%** | 0.000019 |

KleidiAI introduces **zero accuracy loss**.

## Per-Image Speed (ms)

| Image | Texts | Paddle (t=8) | ORT 1.21.1 (t=8) | ORT 1.24.3 (t=2) | ORT 1.24.3 (t=8) |
|---|:---:|---:|---:|---:|---:|
| ancient_demo.png | 12 | 2,707 | 2,246 | 1,022 | 1,340 |
| handwrite_ch_demo.png | 10 | 1,824 | 1,144 | 673 | 1,175 |
| handwrite_en_demo.png | 11 | 2,121 | 1,429 | 828 | 2,215 |
| japan_demo.png | 28 | 19,706 | 7,971 | 18,425 | 14,549 |
| magazine.png | 65 | 17,947 | 14,025 | 11,601 | 16,118 |
| magazine_vetical.png | 65 | 17,549 | 14,341 | 9,732 | 11,599 |
| pinyin_demo.png | 37 | 5,113 | 4,323 | 2,039 | 2,676 |

## Test Environment

- CPU: Apple M4
- OS: macOS (Darwin 25.3.0), ARM64
- Python: 3.11.14
- All configs: 3 runs/image, 1 warmup
- 7 test images (ancient, handwritten, Japanese, magazine, pinyin)

## Reproduce

```bash
# ORT 1.24.3, threads=2 (recommended)
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2

# ORT 1.24.3, threads=8
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# ORT 1.24.3, KleidiAI disabled
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8 --disable-kleidiai

# ORT 1.21.1 (NEON baseline)
pip install onnxruntime==1.21.1
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 8

# Paddle (optional)
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3
```
