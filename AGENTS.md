# AGENTS.md — AI Agent Instructions

> This file makes the repository **AI-native**: any coding agent (Claude Code, Cursor,
> Copilot Workspace, Devin, OpenHands, SWE-agent, etc.) can read this file and
> immediately understand how to navigate, build, test, and contribute to the project.

## Identity

**ppocrv5-kleidiAI-appleM4** — A production-ready, single-file PP-OCRv5 inference
pipeline on ONNX Runtime, delivering 1.72x speedup over PaddleOCR native inference
on Apple M4 (ORT 1.23.2, 8 threads) with 100% text-level accuracy alignment (228/228 texts, 7 images).

## Quick Reference

| What | Command |
|------|---------|
| Run OCR | `python ppocrv5_onnx.py` |
| Run quickstart | `python examples/quickstart.py` |
| Run benchmark (ORT) | `python benchmarks/benchmark_unified.py --backend ort --num-runs 3` |
| Run benchmark (Paddle) | `python benchmarks/benchmark_unified.py --backend paddle --num-runs 3` |
| Compare results | `python benchmarks/compare_results.py` |
| Verify models | `python scripts/download_models.py` |
| Install deps | `pip install onnxruntime>=1.22.0 opencv-python-headless numpy pyclipper` |

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
                                      Output: results/{backend}_{version}.json

  compare_results.py                Auto-discovers results/*.json, prints comparison tables.

data/
  dict/ppocrv5_dict.txt             Character dictionary (18,383 chars). DO NOT MODIFY.
  images/*.png                       7 test images (~3.5 MB total). The canonical test set.

models/                              Git-ignored. User downloads ONNX models (~180 MB).
  README.md                          Download instructions (Baidu Pan).
  .gitkeep                           Placeholder.

results/                             Reference benchmark JSONs (Apple M4).
  paddle_3.3.0.json                  Baseline.
  ort_1.21.1.json                    Without KleidiAI.
  ort_1.23.2.json                    With KleidiAI I8MM GEMM (recommended at t=8).
  ort_1.24.3.json                    With KleidiAI SME Conv (det regressed at t=8).

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
    "engine": "ONNX Runtime 1.23.2",
    "avg_latency_ms": 5486.44,
    "fps": 0.1823,
    "init_time_sec": 0.70,
    "cpu_threads": 8,
    "num_runs": 3,
    "total_images": 7,
    "hw_info": { "cpu": "...", "platform": "...", "memory_gb": "..." }
  },
  "aggregate_timing": {
    "doc_ori": { "preprocess_ms": 0.5, "inference_ms": 1.3, "postprocess_ms": 0.1 },
    "det": { "preprocess_ms": 7.0, "inference_ms": 3780.0, "postprocess_ms": 8.0 },
    "textline_ori": { "preprocess_ms": 3.0, "inference_ms": 36.5, "postprocess_ms": 0.2, "count": 228 },
    "rec": { "preprocess_ms": 12.0, "inference_ms": 1319.0, "postprocess_ms": 50.0, "count": 228 }
  },
  "hotspots": [
    { "model": "det", "phase": "inference", "total_ms": 26460.0, "percent": 65.7 }
  ],
  "results": [
    {
      "image_path": "magazine.png",
      "results": [{ "text": "...", "confidence": 0.9998, "bounding_box": [[x,y], ...] }],
      "avg_latency_ms": 12519.0,
      "timing": { ... }
    }
  ]
}
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `onnxruntime` | >= 1.21.0 (>= 1.22.0 for KleidiAI) | ONNX inference engine |
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
- **Thread scaling matters**: On Apple M4, KleidiAI's SME Conv kernels are 2.8x faster at
  single-thread but barely scale beyond 2 threads (SME device contention). NEON scales ~4x
  to 8 threads. See `docs/SME_THREAD_SCALING.md`.
- Hotspot distribution depends on ORT version and thread count:
  - ORT 1.23.2, t=8: `rec/inference` dominates (~72%), `det/inference` is ~24%
  - ORT 1.24.3, t=8: `det/inference` dominates (~66%) due to SME Conv contention
- `det/inference` is large-kernel Conv — KleidiAI's SME Conv kernels DO accelerate it
  at t=1-2, but contention at t>=3 negates the benefit. Use `--threads 2` or
  `--disable-kleidiai` for optimal det performance.
- `rec/inference` benefits from KleidiAI I8MM GEMM at any thread count.
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

- **KleidiAI SME Conv contention on Apple Silicon** (ORT >= 1.24): The SME Conv kernels
  barely scale beyond 2 threads on Apple M4 (which has only 2 SME devices). At `threads=8`,
  the det model can regress 3-4x compared to ORT 1.21.1. Workarounds: use ORT 1.23.2 at
  `threads=8`, or use ORT >= 1.24 with `--threads 2` or `--disable-kleidiai`. See
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
