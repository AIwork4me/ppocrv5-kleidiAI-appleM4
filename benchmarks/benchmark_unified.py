#!/usr/bin/env python3
"""PP-OCRv5 unified benchmark: Paddle / ORT backend, per-model timing.

Supports two backends:
  - paddle:  Paddle Inference (PIR format: inference.json + .pdiparams)
  - ort:     ONNX Runtime (auto-detects version -> KleidiAI status)

Each image outputs preprocess / inference / postprocess timing for all 4 models,
plus Top-3 hotspot analysis. Output JSON compatible with compare_results.py.

Usage:
    python benchmarks/benchmark_unified.py --backend ort    [--num-runs 3] [--num-warmup 1] [--threads 8]
    python benchmarks/benchmark_unified.py --backend paddle [--num-runs 3] [--num-warmup 1] [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Add repo root to path so we can import from ppocrv5_onnx
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ppocrv5_onnx import (
    DOC_ORI_LABELS,
    REC_BATCH_SIZE,
    TEXTLINE_ORI_LABELS,
    ctc_decode,
    db_postprocess,
    det_preprocess,
    doc_ori_preprocess,
    get_minarea_rect_crop,
    load_charset,
    rec_preprocess_batch,
    rotate_image,
    sort_boxes,
    textline_ori_preprocess,
)

# ── Paths (relative to repo root) ──
IMAGES_DIR = REPO_ROOT / "data" / "images"
DICT_PATH = REPO_ROOT / "data" / "dict" / "ppocrv5_dict.txt"
RESULTS_DIR = REPO_ROOT / "results"

# Model paths
MODELS_DIR = REPO_ROOT / "models"
ONNX_DET = str(MODELS_DIR / "PP-OCRv5_server_det_onnx" / "inference.onnx")
ONNX_REC = str(MODELS_DIR / "PP-OCRv5_server_rec_onnx" / "inference.onnx")
ONNX_DOC_ORI = str(MODELS_DIR / "PP-LCNet_x1_0_doc_ori_onnx" / "inference.onnx")
ONNX_TEXTLINE_ORI = str(MODELS_DIR / "PP-LCNet_x1_0_textline_ori_onnx" / "inference.onnx")

PADDLE_DET_DIR = MODELS_DIR / "PP-OCRv5_server_det_infer"
PADDLE_REC_DIR = MODELS_DIR / "PP-OCRv5_server_rec_infer"
PADDLE_DOC_ORI_DIR = MODELS_DIR / "PP-LCNet_x1_0_doc_ori_infer"
PADDLE_TEXTLINE_ORI_DIR = MODELS_DIR / "PP-LCNet_x1_0_textline_ori_infer"


# ════════════════════════════════════════════════════════════════
#  Backend abstraction
# ════════════════════════════════════════════════════════════════

class InferenceBackend(ABC):
    """Abstract backend wrapping 4 models with raw inference."""

    @abstractmethod
    def run_doc_ori(self, tensor: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def run_textline_ori(self, tensor: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def run_det(self, tensor: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def run_rec(self, tensor: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backend_info(self) -> dict[str, Any]: ...


class OrtBackend(InferenceBackend):
    def __init__(self, threads: int = 8, disable_kleidiai: bool = False):
        import onnxruntime as ort
        self._ort = ort
        self._disable_kleidiai = disable_kleidiai
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        if disable_kleidiai:
            opts.add_session_config_entry("mlas.disable_kleidiai", "1")
        prov = ["CPUExecutionProvider"]
        self.det_sess = ort.InferenceSession(ONNX_DET, opts, providers=prov)
        self.rec_sess = ort.InferenceSession(ONNX_REC, opts, providers=prov)
        self.doc_ori_sess = ort.InferenceSession(ONNX_DOC_ORI, opts, providers=prov)
        self.textline_ori_sess = ort.InferenceSession(ONNX_TEXTLINE_ORI, opts, providers=prov)

    def run_doc_ori(self, tensor: np.ndarray) -> np.ndarray:
        return self.doc_ori_sess.run(None, {"x": tensor})[0]

    def run_textline_ori(self, tensor: np.ndarray) -> np.ndarray:
        return self.textline_ori_sess.run(None, {"x": tensor})[0]

    def run_det(self, tensor: np.ndarray) -> np.ndarray:
        return self.det_sess.run(None, {"x": tensor})[0]

    def run_rec(self, tensor: np.ndarray) -> np.ndarray:
        return self.rec_sess.run(None, {"x": tensor})[0]

    def backend_info(self) -> dict[str, Any]:
        ver = self._ort.__version__
        parts = ver.split(".")
        major, minor = int(parts[0]), int(parts[1])
        kleidi_available = major > 1 or (major == 1 and minor >= 22)
        kleidi_enabled = kleidi_available and not self._disable_kleidiai
        has_sme_conv = major > 1 or (major == 1 and minor >= 24)
        info: dict[str, Any] = {
            "engine": f"ONNX Runtime {ver}",
            "ort_version": ver,
            "kleidi_ai": kleidi_enabled,
            "kleidi_ai_disabled_by_user": self._disable_kleidiai,
            "provider": "CPUExecutionProvider",
            "model_format": "ONNX",
        }
        if has_sme_conv and not self._disable_kleidiai:
            info["sme_note"] = (
                "ORT >= 1.24 uses KleidiAI SME Conv kernels. On Apple M4 (2 SME devices), "
                "threads > 2 may cause SME contention. Use --disable-kleidiai for NEON fallback. "
                "See docs/SME_THREAD_SCALING.md"
            )
        return info


class PaddleBackend(InferenceBackend):
    def __init__(self, threads: int = 8):
        import paddle
        from paddle.inference import Config, create_predictor

        self._paddle = paddle
        self._predictors: dict[str, Any] = {}
        self._input_names: dict[str, list[str]] = {}
        self._output_names: dict[str, list[str]] = {}

        model_dirs = {
            "doc_ori": str(PADDLE_DOC_ORI_DIR),
            "textline_ori": str(PADDLE_TEXTLINE_ORI_DIR),
            "det": str(PADDLE_DET_DIR),
            "rec": str(PADDLE_REC_DIR),
        }
        for name, model_dir in model_dirs.items():
            json_path = str(Path(model_dir) / "inference.json")
            params_path = str(Path(model_dir) / "inference.pdiparams")
            config = Config(json_path, params_path)
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(threads)
            config.switch_ir_optim(True)
            config.disable_mkldnn()
            predictor = create_predictor(config)
            self._predictors[name] = predictor
            self._input_names[name] = predictor.get_input_names()
            self._output_names[name] = predictor.get_output_names()

    def _run(self, name: str, tensor: np.ndarray) -> np.ndarray:
        predictor = self._predictors[name]
        input_handle = predictor.get_input_handle(self._input_names[name][0])
        input_handle.reshape(tensor.shape)
        input_handle.copy_from_cpu(tensor)
        predictor.run()
        return predictor.get_output_handle(self._output_names[name][0]).copy_to_cpu()

    def run_doc_ori(self, tensor: np.ndarray) -> np.ndarray:
        return self._run("doc_ori", tensor)

    def run_textline_ori(self, tensor: np.ndarray) -> np.ndarray:
        return self._run("textline_ori", tensor)

    def run_det(self, tensor: np.ndarray) -> np.ndarray:
        return self._run("det", tensor)

    def run_rec(self, tensor: np.ndarray) -> np.ndarray:
        return self._run("rec", tensor)

    def backend_info(self) -> dict[str, Any]:
        ver = self._paddle.__version__
        return {
            "engine": f"Paddle Inference {ver}",
            "paddle_version": ver,
            "model_format": "Paddle PIR",
        }


# ════════════════════════════════════════════════════════════════
#  Timed OCR pipeline
# ════════════════════════════════════════════════════════════════

def _elapsed_ms(start: float, end: float) -> float:
    """Convert a perf_counter interval to milliseconds."""
    return (end - start) * 1000.0


def timed_predict(
    backend: InferenceBackend,
    img_bgr: np.ndarray,
    character: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Run the full OCR pipeline with per-model timing breakdown.

    Args:
        backend: Inference backend (ORT or Paddle).
        img_bgr: BGR input image.
        character: Character set from ``load_charset()``.

    Returns:
        A tuple of (results, timing) where results is the OCR output list
        and timing is a nested dict of {model: {phase: ms}}.
    """
    timing: dict[str, dict[str, float]] = {
        "doc_ori": {"preprocess_ms": 0, "inference_ms": 0, "postprocess_ms": 0},
        "det": {"preprocess_ms": 0, "inference_ms": 0, "postprocess_ms": 0},
        "textline_ori": {"preprocess_ms": 0, "inference_ms": 0, "postprocess_ms": 0, "count": 0},
        "rec": {"preprocess_ms": 0, "inference_ms": 0, "postprocess_ms": 0, "count": 0},
    }

    # 1) doc_ori
    t0 = time.perf_counter()
    doc_tensor = doc_ori_preprocess(img_bgr)
    t1 = time.perf_counter()
    doc_logits = backend.run_doc_ori(doc_tensor)
    t2 = time.perf_counter()
    angle = DOC_ORI_LABELS[int(np.argmax(doc_logits[0]))]
    if angle != 0:
        img_bgr = rotate_image(img_bgr, angle)
    t3 = time.perf_counter()

    timing["doc_ori"]["preprocess_ms"] = _elapsed_ms(t0, t1)
    timing["doc_ori"]["inference_ms"] = _elapsed_ms(t1, t2)
    timing["doc_ori"]["postprocess_ms"] = _elapsed_ms(t2, t3)

    # 2) det
    t0 = time.perf_counter()
    det_input, img_shape = det_preprocess(img_bgr)
    t1 = time.perf_counter()
    det_out = backend.run_det(det_input)
    t2 = time.perf_counter()
    boxes, det_scores = db_postprocess(det_out, img_shape)
    t3 = time.perf_counter()

    timing["det"]["preprocess_ms"] = _elapsed_ms(t0, t1)
    timing["det"]["inference_ms"] = _elapsed_ms(t1, t2)
    timing["det"]["postprocess_ms"] = _elapsed_ms(t2, t3)

    if len(boxes) == 0:
        return [], timing

    sorted_boxes_list = sort_boxes(boxes)

    # 3) textline_ori
    crops = []
    for box in sorted_boxes_list:
        crop = get_minarea_rect_crop(img_bgr, box)

        t0 = time.perf_counter()
        tl_tensor = textline_ori_preprocess(crop)
        t1 = time.perf_counter()
        tl_logits = backend.run_textline_ori(tl_tensor)
        t2 = time.perf_counter()
        line_angle = TEXTLINE_ORI_LABELS[int(np.argmax(tl_logits[0]))]
        if line_angle == 180:
            crop = rotate_image(crop, 180)
        t3 = time.perf_counter()

        timing["textline_ori"]["preprocess_ms"] += _elapsed_ms(t0, t1)
        timing["textline_ori"]["inference_ms"] += _elapsed_ms(t1, t2)
        timing["textline_ori"]["postprocess_ms"] += _elapsed_ms(t2, t3)
        timing["textline_ori"]["count"] += 1

        crops.append(crop)

    # 4) rec (batched)
    crop_indices = list(range(len(crops)))
    crop_indices.sort(key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))

    rec_texts: list[str | None] = [None] * len(crops)
    rec_confs: list[float | None] = [None] * len(crops)

    for batch_start in range(0, len(crop_indices), REC_BATCH_SIZE):
        batch_idx = crop_indices[batch_start:batch_start + REC_BATCH_SIZE]
        batch_crops = [crops[i] for i in batch_idx]

        t0 = time.perf_counter()
        batch_tensor = rec_preprocess_batch(batch_crops)
        t1 = time.perf_counter()
        timing["rec"]["preprocess_ms"] += _elapsed_ms(t0, t1)

        for j, idx in enumerate(batch_idx):
            single = batch_tensor[j:j + 1]

            t0 = time.perf_counter()
            rec_out = backend.run_rec(single)
            t1 = time.perf_counter()

            text, conf = ctc_decode(rec_out, character)
            t2 = time.perf_counter()

            timing["rec"]["inference_ms"] += _elapsed_ms(t0, t1)
            timing["rec"]["postprocess_ms"] += _elapsed_ms(t1, t2)
            timing["rec"]["count"] += 1

            rec_texts[idx] = text
            rec_confs[idx] = conf

    results = []
    for i, box in enumerate(sorted_boxes_list):
        results.append({
            "bounding_box": box.tolist(),
            "text": rec_texts[i],
            "confidence": round(rec_confs[i], 6),
        })

    return results, timing


# ════════════════════════════════════════════════════════════════
#  Analysis helpers
# ════════════════════════════════════════════════════════════════

def find_hotspots(
    all_timings: list[dict[str, dict[str, float]]], top_n: int = 3
) -> list[dict[str, Any]]:
    """Rank (model, phase) pairs by total time across all images."""
    totals: dict[tuple[str, str], float] = {}
    for timing in all_timings:
        for model, phases in timing.items():
            for phase in ("preprocess_ms", "inference_ms", "postprocess_ms"):
                key = (model, phase.replace("_ms", ""))
                totals[key] = totals.get(key, 0.0) + phases.get(phase, 0.0)

    grand_total = sum(totals.values())
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    hotspots = []
    for (model, phase), ms in ranked[:top_n]:
        pct = (ms / grand_total * 100) if grand_total > 0 else 0
        hotspots.append({
            "model": model, "phase": phase,
            "total_ms": round(ms, 2), "percent": round(pct, 1),
        })
    return hotspots


def aggregate_timing(
    all_timings: list[dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    """Average timing per model across all images."""
    n = len(all_timings)
    if n == 0:
        return {}

    agg: dict[str, dict[str, float]] = {}
    for timing in all_timings:
        for model, phases in timing.items():
            if model not in agg:
                agg[model] = {}
            for k, v in phases.items():
                agg[model][k] = agg[model].get(k, 0.0) + v

    for model in agg:
        for k in agg[model]:
            if k == "count":
                pass  # keep total count
            else:
                agg[model][k] = round(agg[model][k] / n, 2)
    for model in agg:
        if "count" in agg[model]:
            agg[model]["count"] = round(agg[model]["count"], 1)
    return agg


def collect_hw_sw_info() -> dict[str, str]:
    """Collect hardware/software info for the result JSON."""
    info: dict[str, str] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            info["cpu"] = chip
        except (subprocess.SubprocessError, OSError):
            info["cpu"] = platform.processor()
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip())
            info["memory_gb"] = f"{mem_bytes / (1024**3):.0f}"
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
    else:
        info["cpu"] = platform.processor()
    return info


# ════════════════════════════════════════════════════════════════
#  Main benchmark
# ════════════════════════════════════════════════════════════════

def get_image_files() -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in extensions)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    num_runs = args.num_runs
    num_warmup = args.num_warmup
    threads = args.threads

    hw_info = collect_hw_sw_info()

    print("=" * 70)
    print(f"PP-OCRv5 Unified Benchmark — {args.backend} backend")
    print("=" * 70)

    print(f"\n[1] Initializing {args.backend} backend (4 models)...")
    init_start = time.perf_counter()

    if args.backend == "ort":
        backend = OrtBackend(threads=threads,
                             disable_kleidiai=getattr(args, 'disable_kleidiai', False))
    else:
        backend = PaddleBackend(threads=threads)

    init_time = time.perf_counter() - init_start
    binfo = backend.backend_info()
    print(f"    Engine: {binfo['engine']}")
    print(f"    Init time: {init_time:.2f}s")
    print(f"    Threads: {threads}")
    if "kleidi_ai" in binfo:
        status = "enabled" if binfo["kleidi_ai"] else "disabled"
        if binfo.get("kleidi_ai_disabled_by_user"):
            status += " (user-disabled via --disable-kleidiai)"
        print(f"    KleidiAI: {status}")
        if "sme_note" in binfo:
            print(f"    Note: {binfo['sme_note']}")

    character = load_charset(str(DICT_PATH))

    image_files = get_image_files()
    print(f"\n[2] Found {len(image_files)} test images")

    # Warmup
    print(f"\n[3] Warmup ({num_warmup} runs)...")
    if image_files:
        img = cv2.imread(str(image_files[0]))
        for _ in range(num_warmup):
            timed_predict(backend, img, character)
        print("    Done")

    # Benchmark
    print(f"\n[4] Benchmarking ({num_runs} runs per image)...")
    all_results = []
    all_timings: list[dict[str, dict[str, float]]] = []
    all_latencies = []

    for i, image_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}] {image_path.name}")
        img = cv2.imread(str(image_path))
        if img is None:
            print("      *** Cannot read image, skipping ***")
            continue

        run_latencies = []
        run_result = None
        run_timings_list: list[dict[str, dict[str, float]]] = []

        for run in range(num_runs):
            start = time.perf_counter()
            result, timing = timed_predict(backend, img, character)
            latency = _elapsed_ms(start, time.perf_counter())
            run_latencies.append(latency)
            run_timings_list.append(timing)
            if run == 0:
                run_result = result

        avg_latency = sum(run_latencies) / len(run_latencies)
        min_latency = min(run_latencies)
        max_latency = max(run_latencies)
        all_latencies.append(avg_latency)

        avg_timing = aggregate_timing(run_timings_list)
        all_timings.append(avg_timing)

        formatted = {
            "image_path": image_path.name,
            "results": run_result or [],
            "latency_ms": round(avg_latency, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "timing": avg_timing,
        }
        all_results.append(formatted)

        text_count = len(run_result) if run_result else 0
        print(f"      avg: {avg_latency:.2f} ms | min: {min_latency:.2f} ms | texts: {text_count}")

    # Summary
    total_time = sum(all_latencies)
    avg_latency = total_time / len(all_latencies) if all_latencies else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0

    agg_timing = aggregate_timing(all_timings)
    hotspots = find_hotspots(all_timings, top_n=3)

    print("\n" + "=" * 70)
    print(f"Benchmark complete — {binfo['engine']}")
    print("=" * 70)

    # Per-model breakdown
    print(f"\n{'Model':<16} {'Preprocess':>12} {'Inference':>12} {'Postprocess':>12} {'Total':>12}")
    print("-" * 68)
    grand_total_ms = 0.0
    for model in ("doc_ori", "det", "textline_ori", "rec"):
        t = agg_timing.get(model, {})
        pre = t.get("preprocess_ms", 0)
        inf = t.get("inference_ms", 0)
        post = t.get("postprocess_ms", 0)
        total = pre + inf + post
        grand_total_ms += total
        count_str = ""
        if "count" in t:
            count_str = f"  (x{t['count']:.0f})"
        print(f"{model + count_str:<16} {pre:>12.2f} {inf:>12.2f} {post:>12.2f} {total:>12.2f}")
    print("-" * 68)
    print(f"{'Total':<16} {'':>12} {'':>12} {'':>12} {grand_total_ms:>12.2f}")

    print(f"\nTop {len(hotspots)} hotspots:")
    for i, h in enumerate(hotspots, 1):
        print(f"  {i}. {h['model']} / {h['phase']:<14} {h['total_ms']:>8.1f} ms  ({h['percent']:>5.1f}%)")

    print(f"\nImages: {len(image_files)} | Runs/image: {num_runs}")
    print(f"Total: {total_time:.2f} ms | Avg latency: {avg_latency:.2f} ms | FPS: {fps:.4f}")
    print(f"Init: {init_time:.2f}s")

    # Output filename
    if args.backend == "ort":
        ver = binfo.get("ort_version", "unknown")
        output_file = RESULTS_DIR / f"ort_{ver}.json"
    else:
        ver = binfo.get("paddle_version", "unknown")
        output_file = RESULTS_DIR / f"paddle_{ver}.json"

    output = {
        "metadata": {
            **binfo,
            "hw_info": hw_info,
            "model_type": "PP-OCRv5 Server",
            "cpu_threads": threads,
            "num_runs": num_runs,
            "num_warmup": num_warmup,
            "total_images": len(image_files),
            "total_time_ms": round(total_time, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 4),
            "init_time_sec": round(init_time, 2),
        },
        "aggregate_timing": agg_timing,
        "hotspots": hotspots,
        "results": all_results,
        "_output_file": str(output_file),
    }

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="PP-OCRv5 unified benchmark (per-model timing)")
    parser.add_argument("--backend", required=True, choices=["paddle", "ort"], help="Inference backend")
    parser.add_argument("--num-runs", type=int, default=3, help="Runs per image (default: 3)")
    parser.add_argument("--num-warmup", type=int, default=1, help="Warmup runs (default: 1)")
    parser.add_argument("--threads", type=int, default=8,
                        help="CPU threads (default: 8). For KleidiAI SME on Apple M4, "
                             "threads=2 avoids SME contention. See docs/SME_THREAD_SCALING.md")
    parser.add_argument("--disable-kleidiai", action="store_true",
                        help="Disable KleidiAI (fall back to NEON). "
                             "Useful for ORT >= 1.24 at threads > 2 on Apple Silicon.")
    args = parser.parse_args()

    if not IMAGES_DIR.exists():
        print(f"Error: images directory not found: {IMAGES_DIR}")
        sys.exit(1)
    if not DICT_PATH.exists():
        print(f"Error: dictionary not found: {DICT_PATH}")
        sys.exit(1)

    if args.backend == "ort":
        models = [
            ("det", ONNX_DET), ("rec", ONNX_REC),
            ("doc_ori", ONNX_DOC_ORI), ("textline_ori", ONNX_TEXTLINE_ORI),
        ]
    else:
        models = [
            ("det", str(PADDLE_DET_DIR / "inference.json")),
            ("rec", str(PADDLE_REC_DIR / "inference.json")),
            ("doc_ori", str(PADDLE_DOC_ORI_DIR / "inference.json")),
            ("textline_ori", str(PADDLE_TEXTLINE_ORI_DIR / "inference.json")),
        ]
    for label, path in models:
        if not Path(path).exists():
            print(f"Error: {label} model not found: {path}")
            sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_data = run_benchmark(args)
    output_file = benchmark_data.pop("_output_file")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
