#!/usr/bin/env python3
"""
ONNX Runtime KleidiAI IGEMM Conv Regression Reproducer
=======================================================

Reproduces the large-kernel Conv performance regression introduced in ORT 1.24.x
on ARM64 (Apple Silicon).

Related issue: https://github.com/microsoft/onnxruntime/issues/27633

Key finding: isolated single Conv nodes are 3-5x FASTER on ORT 1.24.3 vs 1.21.1,
but the full PP-OCRv5 det model (142 Conv nodes) is 1.65x SLOWER. This script
reproduces both the single-node throughput cliff AND the full-model regression.

4 test sections:
  1. IGEMM Throughput Cliff — single Conv 9x9, throughput drops >40% at high-res
  2. Det Model Regression   — real PP-OCRv5 det model, low-res vs high-res
  3. Memory Explosion        — RSS delta +2.6 GB for a single Conv node
  4. Variance Analysis       — run-to-run max/min ratio on det model

Requirements:
    pip install onnx onnxruntime numpy

Usage:
    # With det model (full reproduction):
    python repro_igemm_regression.py --model-path PP-OCRv5_server_det_onnx/inference.onnx

    # Without det model (synthetic tests only, Sections 2 & 4 skipped):
    python repro_igemm_regression.py

    # JSON output:
    python repro_igemm_regression.py --model-path /path/to/inference.onnx --json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import tempfile
import time

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    sys.exit("pip install onnx")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime")


# ── Model builders ───────────────────────────────────────────────────────────


def make_conv_model(
    ci: int, co: int, ks: int, h: int, w: int, path: str
) -> str:
    """Single Conv node, same-padding, batch=1."""
    pad = ks // 2
    W = numpy_helper.from_array(
        np.random.randn(co, ci, ks, ks).astype(np.float32) * 0.01, name="W"
    )
    B = numpy_helper.from_array(np.zeros(co, dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv", ["X", "W", "B"], ["Y"],
        kernel_shape=[ks, ks], pads=[pad] * 4, strides=[1, 1],
    )
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, ci, h, w])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    graph = helper.make_graph([node], "conv", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


# ── Benchmark helpers ────────────────────────────────────────────────────────


def _make_session(model_path: str, threads: int) -> ort.InferenceSession:
    """Create ORT session with fixed config for reproducible benchmarks."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3  # suppress warnings
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def bench(
    model_path: str,
    feeds: dict[str, np.ndarray],
    threads: int,
    warmup: int = 2,
    runs: int = 5,
) -> dict:
    """Benchmark model, return timing stats in ms."""
    sess = _make_session(model_path, threads)

    for _ in range(warmup):
        sess.run(None, feeds)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return {
        "avg_ms": round(avg, 1),
        "min_ms": round(mn, 1),
        "max_ms": round(mx, 1),
        "std_ms": round(std, 1),
        "max_min_ratio": round(mx / mn, 2) if mn > 0 else 0,
        "times_ms": [round(t, 1) for t in times],
    }


def bench_det(
    model_path: str,
    h: int, w: int,
    threads: int,
    warmup: int = 2,
    runs: int = 3,
) -> dict:
    """Benchmark det model at a specific resolution with random input."""
    sess = _make_session(model_path, threads)
    input_name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, h, w).astype(np.float32)

    for _ in range(warmup):
        sess.run(None, {input_name: x})

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return {
        "avg_ms": round(avg, 1),
        "min_ms": round(mn, 1),
        "max_ms": round(mx, 1),
        "std_ms": round(std, 1),
        "max_min_ratio": round(mx / mn, 2) if mn > 0 else 0,
        "times_ms": [round(t, 1) for t in times],
    }


# ── Test sections ────────────────────────────────────────────────────────────


def section_resolution_scaling(tmpdir: str, threads: int) -> dict:
    """
    Section 1: IGEMM Throughput Cliff.

    Creates single Conv 9×9 (ci=co=64) at increasing resolutions.
    On healthy ORT builds, throughput (pixels/ms) stays roughly constant (~280).
    On 1.24.x, throughput drops >40% above ~226K pixels due to IGEMM
    indirection table exceeding cache hierarchy.

    NOTE: Even at the worst resolution, 1.24.3 is still FASTER than 1.21.1
    for single Conv nodes (489 vs 298 pix/ms). The regression only manifests
    in the full model context (Section 2).
    """
    ci, co, ks = 64, 64, 9
    resolutions = [
        (56, 56), (224, 160), (400, 304), (544, 416), (672, 512), (800, 608),
    ]

    results = []
    for h, w in resolutions:
        path = os.path.join(tmpdir, f"conv_{h}_{w}.onnx")
        make_conv_model(ci, co, ks, h, w, path)
        x = np.random.randn(1, ci, h, w).astype(np.float32)
        r = bench(path, {"X": x}, threads=threads, warmup=2, runs=5)
        pixels = h * w
        pix_ms = round(pixels / r["avg_ms"], 1) if r["avg_ms"] > 0 else 0
        results.append({"h": h, "w": w, "pixels": pixels, "pix_ms": pix_ms, **r})
        os.remove(path)

    first_pix_ms = results[0]["pix_ms"]
    last_pix_ms = results[-1]["pix_ms"]
    drop_pct = round((1 - last_pix_ms / first_pix_ms) * 100, 1) if first_pix_ms > 0 else 0
    regression = drop_pct > 40

    return {
        "test": "resolution_scaling",
        "config": f"Conv {ks}x{ks} ci=co={ci}, threads={threads}",
        "results": results,
        "throughput_drop_pct": drop_pct,
        "regression": regression,
        "threshold": "throughput drop > 40%",
    }


_DET_DOWNLOAD_MSG = """\
  Section 2 & 4 require the PP-OCRv5 server det model (84 MB ONNX).

  Download options:
    1. PaddleOCR official (Paddle format, then convert with paddle2onnx):
       https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/PP-OCRv5_server_det_infer.tar
    2. Pre-converted ONNX (see models/README.md):
       https://github.com/AIwork4me/ppocrv5-kleidiAI-appleM4

  Place at: ./PP-OCRv5_server_det_onnx/inference.onnx
  Or specify: --model-path /path/to/inference.onnx
"""


def section_det_model(model_path: str | None, threads: int) -> dict:
    """
    Section 2: Det Model Regression — the smoking gun.

    Runs the real PP-OCRv5 det model at low-res and high-res with random input.
    On healthy builds, latency scales linearly with pixel count (ratio ≈ 6.0x).
    On 1.24.x, latency scaling is super-linear (ratio > 9x) due to multi-factor
    regression: IGEMM throughput cliff + non-Conv op regression + memory pressure.

    Key data from profiling:
      ORT 1.21.1 t=2: det low-res  889ms, det high-res  5,288ms → ratio 5.9x
      ORT 1.24.3 t=2: det low-res  369ms, det high-res 16,098ms → ratio 43.6x
    """
    if model_path is None or not os.path.isfile(model_path):
        return {
            "test": "det_model",
            "skipped": True,
            "reason": "model not found",
            "regression": False,
        }

    # Low-res: 672×480 (322K pixels) — matches ancient_demo.png det input
    # High-res: 1600×1216 (1.95M pixels) — matches japan_demo.png det input
    # Both are multiples of 32 as required by the det model.
    configs = [
        (672, 480, "low-res"),
        (1600, 1216, "high-res"),
    ]

    results = []
    for h, w, label in configs:
        r = bench_det(model_path, h, w, threads=threads, warmup=2, runs=3)
        pixels = h * w
        results.append({"label": label, "h": h, "w": w, "pixels": pixels, **r})

    low = results[0]["avg_ms"]
    high = results[1]["avg_ms"]
    pixel_ratio = round(results[1]["pixels"] / results[0]["pixels"], 1)
    latency_ratio = round(high / low, 1) if low > 0 else 0
    excess = round(latency_ratio / pixel_ratio, 2) if pixel_ratio > 0 else 0

    # On healthy builds: excess ≈ 1.0 (linear scaling)
    # On 1.24.x: excess > 1.5 (super-linear, regression)
    regression = excess > 1.5

    return {
        "test": "det_model",
        "skipped": False,
        "model_path": model_path,
        "results": results,
        "pixel_ratio": pixel_ratio,
        "latency_ratio": latency_ratio,
        "excess_ratio": excess,
        "regression": regression,
        "threshold": "latency/pixel excess > 1.5x",
    }


def section_memory(threads: int) -> dict:
    """
    Section 3: Memory Explosion.

    ORT 1.24.x allocates massive internal buffers for IGEMM Conv path.
    Measured in a subprocess for clean RSS baseline.

    Expected: ORT 1.21.1 → +96 MB, ORT 1.24.3 → +2,640 MB
    """
    import subprocess

    measure_script = f'''
import gc, resource, sys, os, tempfile, numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

def get_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024

ci, co, ks, h, w = 64, 64, 9, 400, 304
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "conv.onnx")

pad = ks // 2
W = numpy_helper.from_array(np.random.randn(co, ci, ks, ks).astype(np.float32) * 0.01, name="W")
B = numpy_helper.from_array(np.zeros(co, dtype=np.float32), name="B")
node = helper.make_node("Conv", ["X", "W", "B"], ["Y"], kernel_shape=[ks, ks], pads=[pad]*4, strides=[1, 1])
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, ci, h, w])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
graph = helper.make_graph([node], "conv", [X], [Y], initializer=[W, B])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 9
onnx.save(model, path)

gc.collect()
rss_before = get_rss_mb()

opts = ort.SessionOptions()
opts.intra_op_num_threads = {threads}
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
opts.log_severity_level = 3

sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
x = np.random.randn(1, ci, h, w).astype(np.float32)
sess.run(None, {{"X": x}})

rss_after = get_rss_mb()
delta = rss_after - rss_before
print(f"{{rss_before:.1f}} {{rss_after:.1f}} {{delta:.1f}}")

os.remove(path)
os.rmdir(tmpdir)
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", measure_script],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "test": "memory",
            "config": "Conv 9x9 ci=co=64 400x304 (subprocess)",
            "rss_before_mb": -1, "rss_after_mb": -1, "rss_delta_mb": -1,
            "regression": False, "threshold": "RSS delta > 500 MB",
            "error": "timeout",
        }

    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split()
        rss_before = float(parts[0])
        rss_after = float(parts[1])
        delta_mb = float(parts[2])
    else:
        rss_before = rss_after = delta_mb = -1.0

    regression = delta_mb > 500

    return {
        "test": "memory",
        "config": "Conv 9x9 ci=co=64 400x304 (subprocess)",
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_delta_mb": round(delta_mb, 1),
        "regression": regression,
        "threshold": "RSS delta > 500 MB",
    }


def section_variance(
    model_path: str | None, tmpdir: str, threads: int,
) -> dict:
    """
    Section 4: Variance Analysis.

    ORT 1.24.x shows extreme latency variance on the det model at high-res.
    Profiling data: ORT 1.24.3 det max/min = 4.87x; ORT 1.21.1 = 1.19x.

    Uses real det model if available; falls back to synthetic Conv 9×9.
    """
    use_det = model_path is not None and os.path.isfile(model_path)

    if use_det:
        # High-res det model, 10 runs
        r = bench_det(model_path, 1600, 1216, threads=threads, warmup=2, runs=10)
        config = "det model [1,3,1600,1216], 10 runs"
    else:
        # Fallback: synthetic Conv above throughput cliff
        ci, co, ks, h, w = 64, 64, 9, 544, 416
        path = os.path.join(tmpdir, "var_conv.onnx")
        make_conv_model(ci, co, ks, h, w, path)
        x = np.random.randn(1, ci, h, w).astype(np.float32)
        r = bench(path, {"X": x}, threads=threads, warmup=3, runs=20)
        os.remove(path)
        config = f"Conv {ks}x{ks} ci=co={ci} {h}x{w}, 20 runs (synthetic fallback)"

    regression = r["max_min_ratio"] > 2.0

    return {
        "test": "variance",
        "config": config,
        "uses_det_model": use_det,
        **r,
        "regression": regression,
        "threshold": "max/min ratio > 2.0x",
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ORT KleidiAI IGEMM Conv regression reproducer"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to PP-OCRv5 det ONNX model (inference.onnx). "
             "If not provided, searches default locations.",
    )
    parser.add_argument("--threads", type=int, default=2, help="intra_op_num_threads")
    parser.add_argument("--json", action="store_true", help="JSON output to stdout")
    args = parser.parse_args()

    # Auto-detect model path if not specified
    model_path = args.model_path
    if model_path is None:
        for candidate in [
            "PP-OCRv5_server_det_onnx/inference.onnx",
            "../PP-OCRv5_server_det_onnx/inference.onnx",
            "models/PP-OCRv5_server_det_onnx/inference.onnx",
        ]:
            if os.path.isfile(candidate):
                model_path = candidate
                break

    env = {
        "ort_version": ort.__version__,
        "providers": ort.get_available_providers(),
        "python": sys.version.split()[0],
        "platform": f"{sys.platform} / {platform.machine()}",
        "threads": args.threads,
        "det_model": model_path if model_path and os.path.isfile(model_path) else None,
    }
    try:
        env["macos"] = platform.mac_ver()[0]
    except Exception:
        pass

    if not args.json:
        print("=" * 70)
        print("  ONNX Runtime KleidiAI IGEMM Conv Regression Reproducer")
        print("=" * 70)
        for k, v in env.items():
            print(f"  {k:<14s}: {v}")
        print()

    tmpdir = tempfile.mkdtemp()
    all_results = {"environment": env, "sections": []}
    skipped_count = 0

    try:
        # ── Section 1: Resolution Scaling ────────────────────────────────
        if not args.json:
            print("Section 1: IGEMM Throughput Cliff (Conv 9x9 ci=co=64)")
            print("-" * 70)
        s1 = section_resolution_scaling(tmpdir, args.threads)
        all_results["sections"].append(s1)
        if not args.json:
            print(f"  {'H×W':>10s}  {'pixels':>8s}  {'avg_ms':>8s}  {'pix/ms':>8s}")
            for r in s1["results"]:
                mark = "  ← cliff" if r["pix_ms"] < s1["results"][0]["pix_ms"] * 0.6 else ""
                print(f"  {r['h']:>4d}×{r['w']:<4d}  {r['pixels']:>7d}  {r['avg_ms']:>7.1f}ms  {r['pix_ms']:>7.1f}{mark}")
            status = "REGRESSION" if s1["regression"] else "OK"
            print(f"  Throughput drop: {s1['throughput_drop_pct']}%  [{status}]")
            print()

        # ── Section 2: Det Model Regression ──────────────────────────────
        if not args.json:
            print("Section 2: Det Model Regression (PP-OCRv5 server det)")
            print("-" * 70)
        s2 = section_det_model(model_path, args.threads)
        all_results["sections"].append(s2)
        if not args.json:
            if s2.get("skipped"):
                print("  [SKIPPED] Det model not found.")
                print(_DET_DOWNLOAD_MSG)
                skipped_count += 1
            else:
                for r in s2["results"]:
                    pixels_k = r["pixels"] // 1000
                    print(f"  {r['label']:<10s} [{r['h']}×{r['w']}]:  {r['avg_ms']:>9.1f}ms  ({pixels_k}K pixels)")
                print(f"  Pixel ratio: {s2['pixel_ratio']}x  Latency ratio: {s2['latency_ratio']}x  Excess: {s2['excess_ratio']}x")
                status = "REGRESSION" if s2["regression"] else "OK"
                print(f"  [{status}]")
            print()

        # ── Section 3: Memory ────────────────────────────────────────────
        if not args.json:
            print("Section 3: Memory Explosion (RSS delta, Conv 9x9 ci=co=64)")
            print("-" * 70)
        s3 = section_memory(args.threads)
        all_results["sections"].append(s3)
        if not args.json:
            if s3["rss_delta_mb"] >= 0:
                print(f"  Before: {s3['rss_before_mb']:.0f} MB → After: {s3['rss_after_mb']:.0f} MB")
                print(f"  Delta: +{s3['rss_delta_mb']:.0f} MB", end="")
                status = "REGRESSION" if s3["regression"] else "OK"
                print(f"  [{status}]")
            else:
                print(f"  [ERROR] Subprocess measurement failed")
            print()

        # ── Section 4: Variance ──────────────────────────────────────────
        if not args.json:
            print("Section 4: Variance Analysis")
            print("-" * 70)
        s4 = section_variance(model_path, tmpdir, args.threads)
        all_results["sections"].append(s4)
        if not args.json:
            if not s4.get("uses_det_model") and model_path is None:
                print(f"  (det model not available, using synthetic fallback)")
            print(f"  Config: {s4['config']}")
            print(f"  avg: {s4['avg_ms']:.1f}ms  std: {s4['std_ms']:.1f}ms  max/min: {s4['max_min_ratio']}x")
            status = "REGRESSION" if s4["regression"] else "OK"
            print(f"  [{status}]")
            print()

    finally:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    # Summary
    active_sections = [s for s in all_results["sections"] if not s.get("skipped")]
    regression_count = sum(1 for s in active_sections if s["regression"])
    total = len(active_sections)
    all_results["summary"] = {
        "regressions": regression_count,
        "total_tests": total,
        "skipped": skipped_count,
        "verdict": "REGRESSION DETECTED" if regression_count > 0 else "ALL PASS",
    }

    if args.json:
        json.dump(all_results, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print("=" * 70)
        if regression_count > 0:
            print(f"  RESULT: {regression_count}/{total} sections show regression", end="")
            if skipped_count:
                print(f" ({skipped_count} skipped)")
            else:
                print()
            print(f"  This ORT build ({ort.__version__}) exhibits the IGEMM Conv regression.")
            print(f"  See: https://github.com/microsoft/onnxruntime/issues/27633")
        else:
            print(f"  RESULT: ALL {total} sections PASS — no regression detected", end="")
            if skipped_count:
                print(f" ({skipped_count} skipped)")
            else:
                print()
        print("=" * 70)


if __name__ == "__main__":
    main()
