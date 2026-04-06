# SME Thread Scaling: KleidiAI on Apple Silicon

## Background

We filed [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) reporting a 2-17x Conv performance regression in ORT 1.24.x on macOS ARM64. ORT maintainer [@Colm-in-Arm](https://github.com/Colm-in-Arm) confirmed the root cause: **SME (Scalable Matrix Extension) device contention** on Apple M4.

This document explains the issue, its impact on the PP-OCRv5 pipeline, and recommended thread configurations.

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

**KleidiAI is NOT just quantized GEMM (I8MM)** — in ORT 1.24.3 it includes SME2 SGEMM, IGEMM Conv, and Dynamic QGemm kernels, all SME2-based.

## The Contention Problem

```
threads=1:  [Thread-0] → SME-P ✓                           No contention, full SME speed
threads=2:  [Thread-0] → SME-P ✓  [Thread-1] → SME-E ✓    Perfect match
threads=4:  [T0,T1,T2,T3] → SME-P (queuing!)               Severe contention
threads=8:  [T0..T7] → SME-P/E (queuing!)                   Extreme contention + high variance
```

## Experimental Results

Isolated Conv 9x9 benchmark (in=256, out=64, 400x304) on Apple M4:

### Thread Scaling

| | ORT 1.21.1 (NEON) | ORT 1.24.3 (SME) |
|-|:--:|:--:|
| t=1 | 3,251 ms | **~1,180 ms (2.8x faster)** |
| t=8 | 769 ms | ~950 ms (1.2x slower) |
| Scaling t=1→t=8 | **4.2x** | ~1.2x |

- **SME single-thread is 2.8x faster** than NEON single-thread
- **NEON scales 4.2x** from t=1 to t=8 (each core has its own NEON unit)
- **SME barely scales** (~1.2x) because all threads share 2 SME devices
- **SME is extremely sensitive to system load** — background processes amplify contention

### Multi-Config Scaling (t=8 speedup over t=1)

| Config | ORT 1.21.1 (NEON) | ORT 1.24.3 (SME) |
|--------|:--:|:--:|
| Conv 9x9 ci=256 | **3.6x** | 0.9x (no gain!) |
| Conv 9x9 ci=64 | **4.0x** | 1.1x |
| Conv 3x3 ci=256 | **3.5x** | 1.1x |
| Conv 1x1 ci=256 | **3.8x** | 1.4x |

At t=1, ORT 1.24.3 is 1.7-8.0x faster across all configs. SME kernels are excellent — the problem is purely thread scaling.

## Impact on PP-OCRv5 Pipeline

The PP-OCRv5 pipeline has 4 models with different KleidiAI behavior:

| Model | Kernel Sizes | KleidiAI Path | Impact |
|-------|:--------:|:---:|--------|
| det (detection) | 9x9, 3x3 | SME2 IGEMM Conv | **Most affected** — large-kernel Conv dominates |
| rec (recognition) | 3x3, 1x1 | SME2 IGEMM Conv + SGEMM | Mixed — SGEMM benefits, Conv contends at high threads |
| doc_ori (orientation) | 3x3, 1x1 | Dynamic QGemm | 1.9x faster (small model, GEMM-dominated) |
| textline_ori (line orient.) | 3x3, 1x1 | Dynamic QGemm | 1.9x faster (small model, GEMM-dominated) |

### Why Our Original Benchmarks Used threads=8

All original benchmarks were run with `intra_op_num_threads=8`. This was the optimal setting for ORT 1.21.1, where NEON scales linearly with per-core parallelism. Since threads=8 was the production-optimal setting on 1.21.1, that's what we used when testing newer ORT versions — which is exactly where SME contention hits hardest.

### ORT Version Comparison (pipeline, Apple M4)

| Backend | Threads | Avg Latency | det | rec | textline_ori | Notes |
|---------|:-------:|----------:|----:|----:|----:|-------|
| ORT 1.21.1 (NEON) | 8 | 6,497 ms | 1,328 ms | 4,850 ms | 222 ms | Best det latency |
| ORT 1.24.3 (SME2) | 2 | **6,332 ms** | 4,247 ms | 1,931 ms | 81 ms | **Best overall pipeline** |
| ORT 1.24.3 (SME2) | 8 | 7,096 ms | 4,312 ms | 2,579 ms | 99 ms | SME contention hurts det |
| ORT 1.24.3 no-kleidiai | 8 | 7,155 ms | 4,266 ms | 2,677 ms | 99 ms | disable-kleidiai does NOT help det |

Key findings:
- **ORT 1.24.3 at t=2 is the best overall pipeline configuration** (6,332 ms), beating even ORT 1.21.1 at t=8 (6,497 ms).
- **disable-kleidiai does NOT help det** on ORT 1.24.3: det latency is 4,266 ms vs 4,312 ms (within noise). This suggests the SME Conv path cannot be fully reverted via `mlas.disable_kleidiai` in ORT 1.24.3.
- **ORT 1.21.1 at t=8 still delivers the best det latency** (1,328 ms vs 4,247 ms) due to NEON's per-core parallelism.
- **ORT 1.24.3's SME2 kernels dramatically improve rec** (1,931 ms at t=2 vs 4,850 ms on 1.21.1 at t=8).

## Recommended Thread Configuration

| Goal | ORT Version | Threads | Notes |
|------|:-----------:|:-------:|-------|
| **Best pipeline throughput** | 1.24.3 | 2 | SME2 acceleration, minimal contention |
| **Best det latency** | 1.21.1 | 8 | NEON only, no SME contention |

### How to Disable KleidiAI (ORT >= 1.24)

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.add_session_config_entry("mlas.disable_kleidiai", "1")  # Fall back to NEON
opts.intra_op_num_threads = 8  # Safe to use 8 threads with NEON
sess = ort.InferenceSession("model.onnx", opts, providers=["CPUExecutionProvider"])
```

Or via the benchmark script:

```bash
python benchmarks/benchmark_unified.py --backend ort --threads 2                    # SME optimal
python benchmarks/benchmark_unified.py --backend ort --threads 8 --disable-kleidiai  # NEON fallback
```

## Corrections to Our Original Analysis

In issue #27633, we made several incorrect claims that have been corrected:

| Original Claim | Correction |
|----------------|-----------|
| "Not KleidiAI-specific: KleidiAI targets quantized GEMM (I8MM)" | **Wrong.** KleidiAI in ORT 1.24.3 includes SME2 SGEMM, IGEMM Conv, and Dynamic QGemm kernels. The regression IS the KleidiAI SME2 path. |
| "Detection is not GEMM-bound, KleidiAI won't help" | **Wrong at low thread counts.** KleidiAI's SME2 kernels DO accelerate rec dramatically at t=1-2. The det regression at t>=3 is SME contention. |
| "ORT 1.24.3 shows erratic behavior regardless of thread count" | **Misleading.** At t=2, ORT 1.24.3 achieves the best pipeline throughput. Contention effects only appear at t>=3. |
| "disable-kleidiai can revert to NEON Conv path" | **Incomplete.** In ORT 1.24.3, `mlas.disable_kleidiai` does NOT fully revert the det Conv path — det latency remains ~4,266 ms with the flag (vs ~4,312 ms without). |

## Future Outlook

The ORT community is aware of this issue. Expected fixes:

- **Hybrid SME/NEON dispatch**: Assign SME2 kernels to min(threads, num_sme_devices) threads, use NEON for the rest (similar to [llama.cpp PR #20070](https://github.com/ggml-org/llama.cpp/pull/20070))
- **`mlas.disable_kleidiai_sme`**: Fine-grained control to disable only SME2 path while keeping I8MM GEMM
- **Fix `mlas.disable_kleidiai` for Conv path**: Current flag does not fully revert Conv kernels to NEON in ORT 1.24.3

## References

- [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) — Original issue report
- [onnxruntime PR #26402](https://github.com/microsoft/onnxruntime/pull/26402) — KleidiAI Conv kernel addition
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) — ARM micro-kernel library
- [llama.cpp PR #20070](https://github.com/ggml-org/llama.cpp/pull/20070) — Hybrid SME/NEON approach
