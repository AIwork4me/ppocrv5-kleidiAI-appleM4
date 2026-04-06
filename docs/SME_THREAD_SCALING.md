# KleidiAI on Apple Silicon: Thread Scaling & Conv Kernel Regression

## Background

We filed [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) reporting a Conv performance regression in ORT 1.24.x on macOS ARM64. Investigation revealed **two distinct issues**:

1. **Large-kernel Conv regression** — ORT 1.24.x's KleidiAI IGEMM Conv kernels are slower than NEON for large kernels (≥7×7) on high-resolution feature maps, even at optimal thread counts
2. **SME thread scaling** — SME2 kernels don't scale beyond 2 threads on Apple M4 (shared coprocessor)

This document explains both issues, their combined impact on the PP-OCRv5 pipeline, and recommended configurations.

## ARM SME Architecture

**SME (Scalable Matrix Extension)** is ARM's matrix computation accelerator. Key properties:

- **Shared coprocessor**: SME is NOT per-core like NEON. Each CPU cluster shares ONE SME device.
- **Mode switching**: Requires `SMSTART`/`SMSTOP` instructions to enter/exit Streaming SVE mode.
- **Apple M4**: 2 SME devices total (1 in P-core cluster, 1 in E-core cluster).

| Cluster | CPU Cores | SME Devices | Notes |
|---------|:---------:|:-----------:|-------|
| P-core (performance) | 4 | **1** | All 4 P-cores share 1 SME device |
| E-core (efficiency) | 6 | **1** | All 6 E-cores share 1 SME device |
| **Total** | **10** | **2** | |

In contrast, **NEON** is per-core — each of the 10 cores has its own NEON unit, enabling true parallel execution.

## What KleidiAI Does

KleidiAI is ARM's open-source micro-kernel library. In ONNX Runtime:

- **ORT >= 1.22**: Added KleidiAI I8MM GEMM kernels (accelerates GEMM-dominated ops)
- **ORT >= 1.24**: Added KleidiAI **SME2-based kernels** (`convolve_kleidiai.cpp`, [PR #26402](https://github.com/microsoft/onnxruntime/pull/26402))
  - **SME2 SGEMM**: FP32 matrix multiply using Streaming SVE mode
  - **IGEMM Conv**: Indirect GEMM for Conv2D (kernel_size >= 3, batch=1, symmetric padding)
  - **Dynamic QGemm**: Quantized GEMM with dynamic quantization
  - All `intra_op_num_threads` threads dispatch to SME — **this causes contention when threads > 2**

## Issue 1: Large-Kernel Conv Regression

### Root Cause

ORT 1.24.x's KleidiAI IGEMM Conv kernels regress on **large kernels (≥7×7) with high channel counts (≥64) on large spatial feature maps (≥400×300)**. This regression:

- Was introduced between **ORT 1.23.2 and 1.24.1** (verified by version bisect)
- Is **NOT caused by SME contention** — it occurs even at t=2 (perfect SME match) and with `mlas.disable_kleidiai` enabled
- **Only affects high-resolution inputs** — small images are actually faster on 1.24.3
- Small-kernel Conv (3×3, 1×1) is 1.5-2.1x **faster** on 1.24.3

### Version Bisect (Conv 9×9 in=256, 400×304, t=8)

| ORT Version | avg (ms) | Status |
|:-----------:|--------:|:------:|
| 1.21.1 | 658 | Baseline |
| 1.22.0 | 663 | OK |
| 1.23.2 | 700 | OK |
| **1.24.1** | **1,753** | **REGRESSED** |
| **1.24.3** | **3,053** | **REGRESSED** |

### Per-Image Det Analysis (PP-OCRv5 pipeline)

The regression is strictly correlated with input spatial resolution:

| Image | Det Input | Pixels | ORT 1.21.1 t=2 | ORT 1.21.1 t=8 | ORT 1.24.3 t=2 | 1.24.3/1.21.1 (t=2) |
|-------|:---------:|-------:|------:|------:|------:|:------:|
| ancient_demo.png | 480×672 | 322K | 889 ms | 480 ms | **369 ms** | **0.41x faster** |
| handwrite_ch_demo.png | 512×672 | 344K | 936 ms | 514 ms | **436 ms** | **0.47x faster** |
| handwrite_en_demo.png | 608×736 | 447K | 1,236 ms | 692 ms | **554 ms** | **0.45x faster** |
| pinyin_demo.png | 672×704 | 473K | 1,385 ms | 758 ms | 754 ms | **0.54x faster** |
| magazine.png | 1472×992 | 1,460K | 4,070 ms | 2,081 ms | 6,620 ms | **1.63x slower** |
| magazine_vetical.png | 992×1472 | 1,460K | 4,198 ms | 2,134 ms | 4,895 ms | **1.17x slower** |
| japan_demo.png | 1216×1600 | 1,946K | 5,288 ms | 2,640 ms | 16,098 ms | **3.04x slower** |

**Below ~500K pixels**: ORT 1.24.3 at t=2 is faster than ORT 1.21.1 at t=2 (and even faster than 1.21.1 at t=8). Above ~1M pixels: 1.2-3.0x slower at the same thread count.

### Factor Decomposition

The commonly observed "3.2x det regression" (ORT 1.24.3 t=2: 4,247 ms vs ORT 1.21.1 t=8: 1,328 ms) decomposes into two independent factors:

| Factor | Contribution | Evidence |
|--------|:----:|---------|
| **Thread count (t=2 vs t=8)** | 1.94x | ORT 1.21.1: det t=2 (2,571 ms) → t=8 (1,328 ms) |
| **Conv kernel regression** | 1.65x | Same thread count: ORT 1.24.3 t=2 (4,247 ms) / ORT 1.21.1 t=2 (2,571 ms) |
| **Combined** | 3.19x | 1.94 × 1.65 = 3.20 (matches observed 4,247 / 1,328 = 3.20) |

SME device contention between t=2 and t=8 is a negligible factor for det: 4,247 ms → 4,312 ms (only +1.5%).

### Why `mlas.disable_kleidiai` Doesn't Help

The `mlas.disable_kleidiai` session option does NOT revert the Conv kernel path in ORT 1.24.3. Evidence:

| Config | det (ms) |
|--------|--------:|
| ORT 1.24.3 t=2 (KleidiAI) | 4,247 |
| ORT 1.24.3 t=2 (disable-kleidiai) | 4,442 |
| ORT 1.24.3 t=8 (KleidiAI) | 4,312 |
| ORT 1.24.3 t=8 (disable-kleidiai) | 3,871 |

The flag appears to control MLAS GEMM kernel dispatch but NOT the Conv operator registration in `platform.cpp`, which happens at init time via `MlasConvOverride`.

## Issue 2: SME Thread Scaling

### The Contention Problem

```
threads=1:  [Thread-0] → SME-P ✓                           No contention, full SME speed
threads=2:  [Thread-0] → SME-P ✓  [Thread-1] → SME-E ✓    Perfect match
threads=4:  [T0,T1,T2,T3] → SME-P (queuing!)               Severe contention
threads=8:  [T0..T7] → SME-P/E (queuing!)                   Extreme contention + high variance
```

### Isolated Conv Thread Scaling

Conv 9x9 benchmark (in=256, out=64, 400x304) on Apple M4:

| | ORT 1.21.1 (NEON) | ORT 1.24.3 (SME) |
|-|:--:|:--:|
| t=1 | 3,251 ms | **~1,180 ms (2.8x faster)** |
| t=8 | 769 ms | ~950 ms (1.2x slower) |
| Scaling t=1→t=8 | **4.2x** | ~1.2x |

SME is excellent at single-thread (2.8x faster), but barely scales due to shared coprocessor.

### Multi-Config Scaling (t=8 speedup over t=1)

| Config | ORT 1.21.1 (NEON) | ORT 1.24.3 (SME) |
|--------|:--:|:--:|
| Conv 9x9 ci=256 | **3.6x** | 0.9x (no gain!) |
| Conv 9x9 ci=64 | **4.0x** | 1.1x |
| Conv 3x3 ci=256 | **3.5x** | 1.1x |
| Conv 1x1 ci=256 | **3.8x** | 1.4x |

## Impact on PP-OCRv5 Pipeline

The PP-OCRv5 pipeline has 4 models with different characteristics:

| Model | Kernel Sizes | KleidiAI Path | Impact |
|-------|:--------:|:---:|--------|
| det (detection) | 9x9, 3x3 | SME2 IGEMM Conv | **Regressed** — large-kernel Conv on high-res inputs |
| rec (recognition) | 3x3, 1x1 | SME2 IGEMM Conv + SGEMM | **Accelerated** — SGEMM dominates, Conv are small-kernel |
| doc_ori (orientation) | 3x3, 1x1 | Dynamic QGemm | Accelerated (small model, GEMM-dominated) |
| textline_ori (line orient.) | 3x3, 1x1 | Dynamic QGemm | Accelerated (small model, GEMM-dominated) |

### Full Pipeline Comparison (7 configs, Apple M4)

| Backend | Threads | Pipeline | det | rec | textline_ori | Notes |
|---------|:-------:|--------:|----:|----:|----:|-------|
| ORT 1.21.1 (NEON) | 2 | 9,346 ms | 2,571 ms | 6,523 ms | 187 ms | NEON baseline at t=2 |
| ORT 1.21.1 (NEON) | 8 | 6,497 ms | **1,328 ms** | 4,850 ms | 222 ms | Best det latency |
| **ORT 1.24.3 (SME2)** | **2** | **6,332 ms** | 4,247 ms | **1,931 ms** | **81 ms** | **Best overall pipeline** |
| ORT 1.24.3 (no-kleidiai) | 2 | 6,543 ms | 4,442 ms | 1,943 ms | 80 ms | disable-kleidiai ≈ no effect |
| ORT 1.24.3 (SME2) | 8 | 7,096 ms | 4,312 ms | 2,579 ms | 99 ms | SME contention hurts rec |
| ORT 1.24.3 (no-kleidiai) | 8 | 6,605 ms | 3,871 ms | 2,537 ms | 97 ms | High variance on det |
| Paddle 3.3.0 | 8 | 9,567 ms | 4,296 ms | 4,728 ms | 445 ms | Baseline |

Key findings:
- **ORT 1.24.3 at t=2 is the best overall pipeline** (6,332 ms, 1.51x vs Paddle) — rec/textline_ori SME2 acceleration (3.4x, 2.3x) outweighs det regression.
- **Det regression is from Conv kernel change, NOT SME contention**: det at t=2 (4,247 ms) ≈ det at t=8 (4,312 ms). Compare with ORT 1.21.1 where NEON scales 1.94x from t=2 to t=8.
- **Det regression is resolution-dependent**: small images (< 500K pixels) are actually faster on 1.24.3. Large images (> 1M pixels) regress 1.2-3.0x at same thread count.
- **`disable-kleidiai` does NOT revert Conv path**: det latency with the flag is within noise of without it.
- **ORT 1.21.1 at t=8 still delivers the best det latency** (1,328 ms) due to NEON's per-core parallelism.
- **High run-to-run variance**: ORT 1.24.3 det shows max/min = 4.87x across 20 runs; ORT 1.21.1 shows 1.19x.

## Recommended Thread Configuration

| Goal | ORT Version | Threads | Notes |
|------|:-----------:|:-------:|-------|
| **Best pipeline throughput** | 1.24.3 | 2 | SME2 acceleration on rec/textline_ori outweighs det regression |
| **Best det latency** | 1.21.1 | 8 | NEON only, linear thread scaling |

```python
# Apple M4 + ORT >= 1.24: recommended
pipeline = PPOCRv5Pipeline(model_dir, dict_path=dict_path, threads=2)
```

### How to Disable KleidiAI (ORT >= 1.24)

> **Note**: `mlas.disable_kleidiai` does NOT fully revert the Conv kernel path in ORT 1.24.3. It may help for GEMM-dominated models (rec) but has minimal effect on Conv-dominated models (det).

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.add_session_config_entry("mlas.disable_kleidiai", "1")
opts.intra_op_num_threads = 8
sess = ort.InferenceSession("model.onnx", opts, providers=["CPUExecutionProvider"])
```

## Corrections to Our Original Analysis

In issue #27633, we made several incorrect claims that have been corrected:

| Original Claim | Correction |
|----------------|-----------|
| "Not KleidiAI-specific: KleidiAI targets quantized GEMM (I8MM)" | **Wrong.** KleidiAI in ORT 1.24.3 includes SME2 SGEMM, IGEMM Conv, and Dynamic QGemm kernels. |
| "Det regression is caused by SME device contention" | **Incomplete.** SME contention is a minor factor (+1.5% between t=2 and t=8). The primary cause is the KleidiAI IGEMM Conv kernel being slower than NEON for large kernels on high-resolution feature maps. Factor decomposition: 3.2x = 1.94x (thread count) × 1.65x (kernel regression). |
| "ORT 1.24.3 shows erratic behavior regardless of thread count" | **Partially true.** ORT 1.24.3 det shows extreme run-to-run variance (max/min = 4.87x), but at t=2 the average pipeline throughput is still the best. |
| "disable-kleidiai can revert to NEON Conv path" | **Wrong.** In ORT 1.24.3, `mlas.disable_kleidiai` does NOT revert the Conv kernel path — the Conv override is registered at init time via `MlasConvOverride` in `platform.cpp`. |

## Future Outlook

The ORT community is aware of this issue. Expected fixes:

- **Fix large-kernel Conv regression**: The IGEMM Conv kernel needs optimization for large kernels (≥7×7) with high channel counts on large spatial inputs
- **Hybrid SME/NEON dispatch**: Assign SME2 kernels to min(threads, num_sme_devices) threads, use NEON for the rest (similar to [llama.cpp PR #20070](https://github.com/ggml-org/llama.cpp/pull/20070))
- **Fix `mlas.disable_kleidiai` for Conv path**: Current flag does not revert Conv kernels registered via `MlasConvOverride`
- **`mlas.disable_kleidiai_sme`**: Fine-grained control to disable only SME2 path while keeping I8MM GEMM

## References

- [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) — Original issue report
- [onnxruntime PR #26402](https://github.com/microsoft/onnxruntime/pull/26402) — KleidiAI Conv kernel addition
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) — ARM micro-kernel library
- [llama.cpp PR #20070](https://github.com/ggml-org/llama.cpp/pull/20070) — Hybrid SME/NEON approach
