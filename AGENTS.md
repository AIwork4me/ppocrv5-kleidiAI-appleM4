# AGENTS.md — AI Agent Instructions

> This file makes the repository **AI-native**: any coding agent (Claude Code, Cursor,
> Copilot Workspace, Devin, OpenHands, SWE-agent, etc.) can read this file and
> immediately understand how to navigate, build, test, and contribute to the project.

## Identity

**ppocrv5-kleidiAI-appleM4** — A production-ready, single-file PP-OCRv5 inference
pipeline on ONNX Runtime, delivering 1.51x speedup over PaddleOCR native inference
on Apple M4 (ORT 1.24.3, 2 threads, KleidiAI SME2) with 100% text-level accuracy alignment (228/228 texts, 7 images).

## Quick Reference

| What | Command |
|------|---------|
| Run OCR | `python ppocrv5_onnx.py` |
| Run quickstart | `python examples/quickstart.py` |
| Run benchmark (ORT) | `python benchmarks/benchmark_unified.py --backend ort --num-runs 3 --threads 2` |
| Run benchmark (Paddle) | `python benchmarks/benchmark_unified.py --backend paddle --num-runs 3` |
| Compare results | `python benchmarks/compare_results.py` |
| Verify models | `python scripts/download_models.py` |
| Install deps | `pip install onnxruntime>=1.21.0 opencv-python-headless numpy pyclipper` |

## Architecture

### Pipeline: 4 Sequential Models

```
Image (BGR)
  │
  ▼
[doc_ori]  LCNet 224x224 → classify {0°, 90°, 180°, 270°} → rotate
  │
  ▼
[det]      PP-OCRv5 Server Det, DB algorithm → N bounding boxes (int16)
  │
  ▼  (for each box: crop → classify → rotate)
[textline_ori]  LCNet 160x80 → classify {0°, 180°} → rotate if needed
  │
  ▼  (batch_size=6, ratio-sorted, per-batch padded)
[rec]      PP-OCRv5 Server Rec, CTC decode → text + confidence
  │
  ▼
Results: [{text, confidence, bounding_box}, ...]
```

### File Map

```
ppocrv5_onnx.py                    ★ THE core file. Single-file pipeline (~720 lines).
                                      Copy-paste ready. All pre/post-processing self-contained.
                                      Class: PPOCRv5Pipeline
                                      Entry: pipeline.predict(image) -> list[dict]

benchmarks/
  benchmark_unified.py              Unified benchmark with --backend paddle|ort.
                                      Imports from ppocrv5_onnx (no code duplication).
                                      ABC: InferenceBackend → OrtBackend, PaddleBackend.
                                      Output: results/{backend}_{version}[_t{N}][_no_kleidiai].json

  compare_results.py                Auto-discovers results/*.json, prints comparison tables.

data/
  dict/ppocrv5_dict.txt             Character dictionary (18,383 chars). DO NOT MODIFY.
  images/*.png                       7 test images (~3.5 MB total). The canonical test set.

models/                              Git-ignored. User downloads ONNX models (~180 MB).
  README.md                          Download instructions (Baidu Pan).
  .gitkeep                           Placeholder.

results/                             Reference benchmark JSONs (Apple M4).
  paddle_3.3.0.json                  Baseline.
  ort_1.21.1.json                    Without KleidiAI (NEON only, t=8).
  ort_1.24.3.json                    With KleidiAI SME2 (t=8).
  ort_1.24.3_t2.json                 With KleidiAI SME2 (t=2, recommended).
  ort_1.24.3_no_kleidiai.json        KleidiAI disabled (NEON fallback, t=8).
  ort_1.21.1_t2.json                   Without KleidiAI (NEON only, t=2).
  ort_1.24.3_t2_no_kleidiai.json       KleidiAI disabled (t=2).

docs/
  ACCURACY_ALIGNMENT.md              6-round journey from 65.6% → 100%.
  BENCHMARK_RESULTS.md               Speed/accuracy tables.
  PIPELINE_ARCHITECTURE.md           4-model pipeline details, preprocessing params.
  SME_THREAD_SCALING.md              KleidiAI SME contention on Apple Silicon, thread tuning.

scripts/download_models.py           Checks 4 ONNX model files exist and are correctly sized.
examples/quickstart.py               Minimal 3-line usage example.
```

## Critical Constraints (Read Before Editing)

### 1. Accuracy Alignment is Sacred

This pipeline produces output **100% identical** to PaddleOCR/PaddleX 3.4.x. Every
preprocessing constant, rounding mode, and padding strategy is deliberate. Seemingly
"equivalent" changes can silently break accuracy:

- **`_DET_LIMIT_TYPE = "min"` / `_DET_LIMIT_SIDE_LEN = 64`**: These override what
  `inference.yml` says. PaddleX Pipeline runtime silently overrides model-level config.
  Never change these without re-running the full 228-text verification.

- **`rec batch_size=6` + ratio sorting + per-batch padding**: The recognition model is
  sensitive to padding width. The same crop padded to different widths produces
  different text. This batch scheduling is part of the accuracy contract.

- **Crop coordinates use `minAreaRect → boxPoints → float32`**, NOT int16 directly.
  A 0.5px offset flips argmax at character boundaries ("內" vs "内").

- **CTC decode uses raw logits (not softmax)** for confidence. Uses direct index
  `character[idx]`, not `idx-1`.

- **ImageNet normalization** uses `cv2.split/merge` with pre-computed `alpha/beta`,
  not vectorized numpy division.

### 2. Verification Protocol

After ANY code change to `ppocrv5_onnx.py` or `benchmarks/benchmark_unified.py`:

```bash
# Step 1: Run the pipeline directly (expect 228 total text regions)
python ppocrv5_onnx.py

# Step 2: Run benchmark and compare
python benchmarks/benchmark_unified.py --backend ort --num-runs 1
python benchmarks/compare_results.py

# Step 3: Verify 228/228 match and 0.000000 confidence diff
# The compare_results.py output MUST show:
#   228/228 texts match (100.0%)
#   avg |conf diff| = 0.000000  (between ORT versions)
```

If any text mismatches appear, the change has broken accuracy alignment. Revert.

### 3. Model Files are NOT in Git

ONNX models (~180 MB) must be downloaded separately from Baidu Pan.
The `models/` directory is git-ignored except for `README.md` and `.gitkeep`.
Run `python scripts/download_models.py` to verify models are in place.

Expected model layout:
```
models/
  PP-OCRv5_server_det_onnx/inference.onnx     (~84 MB)
  PP-OCRv5_server_rec_onnx/inference.onnx     (~81 MB)
  PP-LCNet_x1_0_doc_ori_onnx/inference.onnx   (~6.5 MB)
  PP-LCNet_x1_0_textline_ori_onnx/inference.onnx (~6.5 MB)
```

## Code Conventions

### Style

- **Python >= 3.10**, `from __future__ import annotations` in all files.
- Google Python Style Guide. Type annotations on all public functions.
- `snake_case` everywhere (no camelCase variables).
- Constants: `UPPER_CASE` for public, `_UPPER_CASE` for private.
- Private functions prefixed with `_`.

### Public API (`ppocrv5_onnx.py`)

The `__all__` exports define the public contract:

```python
# Class
PPOCRv5Pipeline           # Main entry point

# Pre/post-processing functions (used by benchmarks)
det_preprocess            # BGR image → (NCHW tensor, img_shape)
db_postprocess            # det output → (boxes int16, scores)
get_minarea_rect_crop     # image + box → cropped text region
rec_preprocess_single     # crop → normalized NCHW tensor
rec_preprocess_batch      # list[crop] → batched tensor (padded to max width)
ctc_decode                # model output → (text, confidence)
load_charset              # dict file → character list
doc_ori_preprocess        # BGR image → NCHW tensor for doc_ori model
textline_ori_preprocess   # BGR crop → NCHW tensor for textline_ori model
rotate_image              # image + angle → rotated image
sort_boxes                # boxes → reading-order sorted boxes

# Constants
DOC_ORI_LABELS            # [0, 90, 180, 270]
TEXTLINE_ORI_LABELS       # [0, 180]
REC_BATCH_SIZE            # 6
```

### Benchmark JSON Schema

Output files in `results/` follow this structure:

```json
{
  "metadata": {
    "engine": "ONNX Runtime 1.24.3",
    "avg_latency_ms": 6332.0,
    "fps": 0.158,
    "init_time_sec": 0.16,
    "cpu_threads": 2,
    "num_runs": 3,
    "total_images": 7,
    "hw_info": { "cpu": "...", "platform": "...", "memory_gb": "..." }
  },
  "aggregate_timing": {
    "doc_ori": { "preprocess_ms": 0.9, "inference_ms": 3.4, "postprocess_ms": 0.6 },
    "det": { "preprocess_ms": 6.6, "inference_ms": 4246.6, "postprocess_ms": 3.9 },
    "textline_ori": { "preprocess_ms": 2.6, "inference_ms": 80.8, "postprocess_ms": 0.1, "count": 684 },
    "rec": { "preprocess_ms": 9.6, "inference_ms": 1931.2, "postprocess_ms": 39.6, "count": 684 }
  },
  "hotspots": [
    { "model": "det", "phase": "inference", "total_ms": 29726.0, "percent": 67.3 }
  ],
  "results": [
    {
      "image_path": "magazine.png",
      "results": [{ "text": "...", "confidence": 0.9998, "bounding_box": [[x,y], ...] }],
      "avg_latency_ms": 11601.0,
      "timing": { ... }
    }
  ]
}
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `onnxruntime` | >= 1.21.0 (>= 1.24 for KleidiAI SME2) | ONNX inference engine |
| `opencv-python-headless` | >= 4.8.0 | Image I/O, resize, warp, contours |
| `numpy` | >= 1.24.0 | Array operations |
| `pyclipper` | >= 1.3.0 | Polygon expansion (DB post-processing) |
| `paddlepaddle` | >= 3.3.0 (optional) | Only for `--backend paddle` benchmark |

## Common Tasks for Agents

### "Add a new test image"

1. Place the PNG in `data/images/`.
2. Run `python ppocrv5_onnx.py` — verify it processes the new image.
3. Run `python benchmarks/benchmark_unified.py --backend ort --num-runs 1`.
4. Run `python benchmarks/compare_results.py` — verify existing images still 100% match.

### "Optimize inference speed"

- Profile with `benchmark_unified.py`. Check `hotspots` in the output JSON.
- **Thread scaling matters**: On Apple M4, KleidiAI's SME2 kernels (SGEMM) give
  2.5-2.8x speedup for rec at t=1-2. However, the det model suffers from a
  **large-kernel Conv kernel regression** in ORT 1.24.x (not SME contention).
  The regression is resolution-dependent: small images (< 500K pixels) are faster,
  large images (> 1M pixels) regress significantly vs ORT 1.21.1.
  See `docs/SME_THREAD_SCALING.md`.
- `mlas.disable_kleidiai` does NOT help det — the Conv kernel override is registered
  at init time and is not reverted by the runtime flag.
- Hotspot distribution depends on ORT version and thread count:
  - ORT 1.21.1, t=8 (NEON): `rec/inference` dominates (~75%), `det/inference` is ~20%
  - ORT 1.24.3, t=2 (SME2): `det/inference` dominates (~67%), but rec is 3.4x faster
    than ORT 1.21.1, making the overall pipeline total the lowest
- Use `--threads 2` for optimal pipeline throughput.
- `rec/inference` benefits massively from KleidiAI SME2 SGEMM (2.5x at t=2).
- After ANY optimization, re-run the 228-text verification protocol.

### "Port to a new platform"

- `ppocrv5_onnx.py` is platform-agnostic (pure Python + ONNX Runtime).
- Copy it + `data/dict/ppocrv5_dict.txt` + 4 ONNX model files. That's the full deployment.
- Run `benchmark_unified.py` on the new platform, save results to `results/`.

### "Update preprocessing parameters"

- **Don't** — unless you've verified against PaddleOCR source that the runtime actually
  uses different values. PaddleX has a 3-layer parameter override hierarchy
  (inference.yml → Pipeline.__init__ → predict() args). The values in `ppocrv5_onnx.py`
  match the **actual runtime** behavior, not the config files.

### "Add a new model variant"

- The `PPOCRv5Pipeline.__init__` accepts `det_path`, `rec_path`, `doc_ori_path`,
  `textline_ori_path` overrides. Use these for custom models.
- Preprocessing parameters are module-level constants. If a new model needs different
  params, they should be constructor arguments, not global mutations.

## Known Issues

- **Large-kernel Conv regression in ORT 1.24.x on Apple Silicon**: The det model uses
  large-kernel Conv ops that regress in ORT 1.24.x compared to 1.21.1. This is a
  Conv kernel regression, not SME contention. The regression is resolution-dependent:
  small images (< 500K pixels) are faster in 1.24.x, while large images (> 1M pixels)
  regress significantly. `mlas.disable_kleidiai=1` does NOT help because the Conv
  kernel override is registered at init time. Recommended: use ORT 1.24.3 with
  `--threads 2` for best overall pipeline throughput (rec is 3.4x faster, offsetting
  the det regression), or use ORT 1.21.1 at `threads=8` for best det-only latency. See
  [onnxruntime#27633](https://github.com/microsoft/onnxruntime/issues/27633) and
  `docs/SME_THREAD_SCALING.md`.

- `data/images/magazine_vetical.png` has a filename typo ("vetical" → "vertical").
  This is kept as-is for consistency with existing benchmark results JSONs.

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — source of PP-OCRv5 models
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — inference engine
- [KleidiAI](https://gitlab.arm.com/kleidi/kleidiai) — Arm CPU micro-kernel library
- `docs/ACCURACY_ALIGNMENT.md` — the definitive guide to why every constant matters
- `docs/PIPELINE_ARCHITECTURE.md` — model input/output specs and preprocessing details
