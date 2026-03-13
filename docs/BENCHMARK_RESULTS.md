# Benchmark Results

Reference benchmarks on Apple M4 (macOS ARM64, 8 CPU threads).

## Speed Overview

| Backend | Avg Latency | vs Paddle | Init Time | FPS |
|---------|----------:|:---------:|----------:|----:|
| Paddle 3.3.0 | 9,451 ms | 1.00x | 5.63s | 0.1058 |
| ORT 1.21.1 (no KleidiAI) | 6,407 ms | 1.48x faster | 0.15s | 0.1561 |
| **ORT 1.23.2 (KleidiAI)** | **5,486 ms** | **1.72x faster** | **0.16s** | **0.1823** |

## Per-Model Timing (avg ms per image)

### ORT 1.23.2 (KleidiAI)

| Model | Preprocess | Inference | Postprocess | Total |
|-------|----------:|---------:|------------:|------:|
| doc_ori | 2.59 | 1.34 | 0.01 | 3.94 |
| det | 57.96 | 3,788.58 | 218.43 | 4,064.97 |
| textline_ori (x33) | 5.38 | 36.51 | 0.20 | 42.09 |
| rec (x33) | 4.38 | 1,319.33 | 0.32 | 1,324.03 |
| **Total** | | | | **5,435.03** |

### ORT 1.21.1 (no KleidiAI)

| Model | Preprocess | Inference | Postprocess | Total |
|-------|----------:|---------:|------------:|------:|
| doc_ori | 2.53 | 2.57 | 0.01 | 5.11 |
| det | 58.80 | 3,779.37 | 217.76 | 4,055.93 |
| textline_ori (x33) | 5.51 | 67.88 | 0.18 | 73.57 |
| rec (x33) | 4.42 | 1,599.89 | 0.27 | 1,604.58 |
| **Total** | | | | **5,739.19** |

## KleidiAI Acceleration by Model

KleidiAI is automatically enabled in ORT >= 1.22 on ARM CPUs with I8MM instructions (Apple M4).

| Model | ORT 1.21.1 | ORT 1.23.2 | Speedup |
|-------|----------:|----------:|:-------:|
| doc_ori | 2.57 ms | 1.34 ms | **1.91x** |
| textline_ori | 67.88 ms | 36.51 ms | **1.86x** |
| rec | 1,599.89 ms | 1,319.33 ms | **1.21x** |
| det | 3,779.37 ms | 3,788.58 ms | 1.00x |

- Classification models (doc_ori, textline_ori) benefit most: ~1.9x speedup
- Recognition model: 1.21x speedup
- Detection model: no change (large-kernel Conv, not dominated by GEMM)

## Accuracy

All backends produce **100% identical** text output:

| Comparison | Texts | Match Rate | Avg Confidence Diff |
|:----------:|:-----:|:----------:|:-------------------:|
| Paddle vs ORT 1.21.1 | 228 | **100.0%** | 0.000019 |
| Paddle vs ORT 1.23.2 | 228 | **100.0%** | 0.000019 |
| ORT 1.21.1 vs ORT 1.23.2 | 228 | **100.0%** | 0.000000 |

KleidiAI introduces **zero accuracy loss**.

## Per-Image Speed (ms)

| Image | Texts | Paddle | ORT 1.21.1 | ORT 1.23.2 |
|-------|:-----:|-------:|----------:|----------:|
| ancient_demo.png | 12 | 2,958 | 2,379 | 2,086 |
| handwrite_ch_demo.png | 10 | 1,834 | 1,230 | 1,064 |
| handwrite_en_demo.png | 11 | 2,422 | 1,620 | 1,395 |
| japan_demo.png | 28 | 14,017 | 8,606 | 7,313 |
| magazine.png | 65 | 24,095 | 14,625 | 12,519 |
| magazine_vertical.png | 65 | 17,279 | 14,803 | 12,803 |
| pinyin_demo.png | 37 | 3,553 | 1,585 | 1,220 |

## Test Environment

- CPU: Apple M4
- OS: macOS 15.x (Darwin 25.3.0), ARM64
- Python: 3.11.14
- Threads: 8 (intra_op + inter_op)
- 7 test images (ancient, handwritten, Japanese, magazine, pinyin)
- Warmup: 1 run, Benchmark: 3 runs per image

## Reproduce

```bash
# ORT benchmark
pip install onnxruntime==1.23.2
python benchmarks/benchmark_unified.py --backend ort --num-runs 3

# Paddle benchmark (optional)
pip install paddlepaddle==3.3.0
python benchmarks/benchmark_unified.py --backend paddle --num-runs 3

# Compare all results
python benchmarks/compare_results.py
```
