# Benchmark Results

Reference benchmarks on Apple M4 (macOS ARM64, 8 CPU threads).

## Speed Overview

| Backend | Avg Latency | vs Paddle | Init Time | FPS |
|---------|----------:|:---------:|----------:|----:|
| Paddle 3.3.0 | 9,451 ms | 1.00x | 0.54s | 0.1058 |
| ORT 1.21.1 (no KleidiAI) | 6,407 ms | 1.48x faster | 0.16s | 0.1561 |
| **ORT 1.23.2 (KleidiAI I8MM)** | **5,486 ms** | **1.72x faster** | **0.70s** | **0.1823** |
| ORT 1.24.3 (KleidiAI SME Conv)\* | 7,842 ms | 1.21x faster | 0.08s | 0.1275 |

> \*ORT 1.24.3 was benchmarked with 1 run and 0 warmup (others used 3 runs, 1 warmup). The det model regression is caused by SME device contention at threads=8 — see [SME Thread Scaling](SME_THREAD_SCALING.md).

## Per-Model Timing (avg ms per image)

> These values are from `aggregate_timing` in the result JSON files — the average across all 7 images. For textline_ori and rec, the count represents total text regions processed across all images and runs.

### ORT 1.23.2 (KleidiAI I8MM)

| Model | Preprocess | Inference | Postprocess | Total |
|-------|----------:|---------:|------------:|------:|
| doc_ori | 0.60 | 3.22 | 0.94 | 4.76 |
| det | 4.87 | 1,311.58 | 2.64 | 1,319.09 |
| textline_ori (x228) | 2.95 | 118.23 | 0.25 | 121.43 |
| rec (x228) | 9.68 | 3,962.88 | 60.74 | 4,033.30 |
| **Total** | | | | **5,478.58** |

### ORT 1.21.1 (no KleidiAI)

| Model | Preprocess | Inference | Postprocess | Total |
|-------|----------:|---------:|------------:|------:|
| doc_ori | 0.56 | 6.16 | 0.94 | 7.66 |
| det | 4.90 | 1,305.97 | 2.83 | 1,313.70 |
| textline_ori (x228) | 2.82 | 219.99 | 0.24 | 223.05 |
| rec (x228) | 9.45 | 4,786.54 | 58.94 | 4,854.93 |
| **Total** | | | | **6,399.34** |

### ORT 1.24.3 (KleidiAI SME Conv)\*

| Model | Preprocess | Inference | Postprocess | Total |
|-------|----------:|---------:|------------:|------:|
| doc_ori | 0.61 | 3.90 | 0.73 | 5.24 |
| det | 7.41 | **5,147.35** | 10.10 | 5,164.86 |
| textline_ori (x228) | 3.53 | 92.68 | 0.28 | 96.49 |
| rec (x228) | 14.86 | 2,497.21 | 54.69 | 2,566.76 |
| **Total** | | | | **7,833.35** |

> \*ORT 1.24.3 data from 1 run, 0 warmup. Note: det inference jumped from ~1,312 ms to 5,147 ms due to SME contention. rec inference improved from 3,963 ms to 2,497 ms.

## KleidiAI Acceleration by Model

KleidiAI is automatically enabled in ORT >= 1.22 on ARM CPUs. It includes:
- **I8MM GEMM kernels** (ORT >= 1.22): Accelerates GEMM-dominated models
- **FP32 SME Conv kernels** (ORT >= 1.24): Accelerates Conv via ARM SME coprocessor

### ORT 1.21.1 → 1.23.2 (I8MM GEMM only, no SME Conv)

| Model | ORT 1.21.1 | ORT 1.23.2 | Speedup | KleidiAI Path |
|-------|----------:|----------:|:-------:|:---:|
| doc_ori | 6.16 ms | 3.22 ms | **1.91x** | I8MM GEMM |
| textline_ori | 219.99 ms | 118.23 ms | **1.86x** | I8MM GEMM |
| rec | 4,786.54 ms | 3,962.88 ms | **1.21x** | I8MM GEMM |
| det | 1,305.97 ms | 1,311.58 ms | 1.00x | Not affected (no SME Conv in 1.23.2) |

- Classification models (doc_ori, textline_ori) benefit most: ~1.9x speedup from I8MM GEMM
- Recognition model: 1.21x speedup from I8MM GEMM
- Detection model: 1.00x — ORT 1.23.2 does NOT have SME Conv kernels, so det is unchanged

### ORT 1.23.2 → 1.24.3 (adds SME Conv kernels)

| Model | ORT 1.23.2 | ORT 1.24.3\* | Change | Notes |
|-------|----------:|----------:|:------:|-------|
| doc_ori | 3.22 ms | 3.90 ms | 0.83x | Small model, negligible |
| textline_ori | 118.23 ms | 92.68 ms | **1.28x** | SME Conv helps |
| rec | 3,962.88 ms | 2,497.21 ms | **1.59x** | SME Conv + GEMM improvements |
| det | 1,311.58 ms | **5,147.35 ms** | **0.25x** | **3.9x regression — SME contention at t=8** |

> \*ORT 1.24.3 data from 1 run, 0 warmup.

**Key insight**: KleidiAI's SME Conv kernels ARE faster at low thread counts (t=1-2), but at t=8 the limited SME devices (2 on Apple M4) cause severe contention for Conv-heavy models like det. See [SME Thread Scaling](SME_THREAD_SCALING.md).

> **Important**: ORT 1.23.2 has I8MM GEMM but NOT the SME Conv kernels. This is why it shows the best pipeline results at threads=8 — no SME contention. ORT 1.24.x adds SME Conv kernels which improve single-thread performance but degrade multi-thread performance due to Apple M4's limited SME devices (2 total).

## Accuracy

All backends produce **100% identical** text output:

| Comparison | Texts | Match Rate | Avg Confidence Diff |
|:----------:|:-----:|:----------:|:-------------------:|
| Paddle vs ORT 1.21.1 | 228 | **100.0%** | 0.000019 |
| Paddle vs ORT 1.23.2 | 228 | **100.0%** | 0.000019 |
| ORT 1.21.1 vs ORT 1.23.2 | 228 | **100.0%** | 0.000000 |

KleidiAI introduces **zero accuracy loss**.

## Per-Image Speed (ms)

| Image | Texts | Paddle | ORT 1.21.1 | ORT 1.23.2 | ORT 1.24.3\* |
|-------|:-----:|-------:|----------:|----------:|----------:|
| ancient_demo.png | 12 | 2,722 | 2,239 | 1,913 | 1,505 |
| handwrite_ch_demo.png | 10 | 1,816 | 1,146 | 985 | 841 |
| handwrite_en_demo.png | 11 | 2,122 | 1,397 | 1,223 | 1,010 |
| japan_demo.png | 28 | 18,994 | 7,916 | 6,933 | 12,347 |
| magazine.png | 65 | 18,494 | 13,843 | 11,661 | 17,255 |
| magazine_vetical.png | 65 | 16,924 | 14,073 | 12,138 | 16,752 |
| pinyin_demo.png | 37 | 5,088 | 4,235 | 3,552 | 5,182 |

> \*ORT 1.24.3: 1 run, 0 warmup. Note how images with many text regions (japan, magazine) show ORT 1.24.3 regression — the det model's SME contention dominates.

## Test Environment

- CPU: Apple M4
- OS: macOS 15.x (Darwin 25.3.0), ARM64
- Python: 3.11.14
- Threads: 8 (intra_op + inter_op)
- 7 test images (ancient, handwritten, Japanese, magazine, pinyin)
- Warmup: 1 run, Benchmark: 3 runs per image (except ORT 1.24.3: 0 warmup, 1 run)

## Reproduce

```bash
# ORT benchmark (recommended: ORT 1.23.2 for best pipeline throughput at t=8)
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# ORT 1.24.3 benchmark (shows SME Conv contention)
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# ORT 1.24.3 with KleidiAI disabled (NEON fallback)
pip install onnxruntime==1.24.3
python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --disable-kleidiai

# Paddle benchmark (optional)
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3

# Compare all results
python benchmarks/compare_results.py
```
