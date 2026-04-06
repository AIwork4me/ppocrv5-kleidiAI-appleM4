#!/usr/bin/env python3
"""
ONNX Runtime KleidiAI IGEMM Conv Regression Reproducer
=======================================================

Reproduces the large-kernel Conv performance regression introduced in ORT 1.24.x
on ARM64 (Apple Silicon). This script creates synthetic ONNX models — no external
models or data required.

Related issue: https://github.com/microsoft/onnxruntime/issues/27633

Key findings this script demonstrates:
  1. Resolution-dependent throughput cliff for IGEMM Conv on large feature maps
  2. Non-Conv ops (Resize, Concat) also regress on large feature maps
  3. Massive memory allocation (+2.6 GB RSS for a single Conv 9×9 node)
  4. High run-to-run variance (max/min > 4x) on ORT 1.24.x

Requirements:
    pip install onnx onnxruntime numpy

Usage:
    pip install onnxruntime==1.24.3
    python repro_igemm_regression.py              # expect regressions
    python repro_igemm_regression.py --threads 1  # single-thread mode

    pip install onnxruntime==1.21.1
    python repro_igemm_regression.py              # expect all PASS

    python repro_igemm_regression.py --json       # machine-readable output
"""
from __future__ import annotations

import argparse
import gc
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


def make_mixed_det_model(h: int, w: int, path: str) -> str:
    """
    Simulates PP-OCRv5 det model head: large-kernel Conv + Resize + Concat.

    Architecture:
        X (1, 3, H, W)
        → Conv 3×3 ci=3→64 stride=2  (downsample to H/2 × W/2)
        → Conv 3×3 ci=64→128 stride=2 (downsample to H/4 × W/4)
        → Conv 9×9 ci=128→64          (large kernel, same-padding)
        → Conv 9×9 ci=64→64           (large kernel, same-padding)
        → Resize (upsample 2x to H/2 × W/2)
        → Conv 3×3 ci=64→32           (refine)
        → Y
    """
    nodes = []
    inits = []

    def add_conv(name, prev, ci, co, ks, stride=1):
        pad = ks // 2
        w_data = np.random.randn(co, ci, ks, ks).astype(np.float32) * 0.01
        b_data = np.zeros(co, dtype=np.float32)
        inits.append(numpy_helper.from_array(w_data, name=f"{name}_W"))
        inits.append(numpy_helper.from_array(b_data, name=f"{name}_B"))
        out = f"{name}_out"
        nodes.append(helper.make_node(
            "Conv", [prev, f"{name}_W", f"{name}_B"], [out],
            kernel_shape=[ks, ks], pads=[pad] * 4, strides=[stride, stride],
        ))
        relu = f"{name}_relu"
        nodes.append(helper.make_node("Relu", [out], [relu]))
        return relu

    # Downsample path
    x = add_conv("down1", "X", 3, 64, 3, stride=2)
    x = add_conv("down2", x, 64, 128, 3, stride=2)

    # Large-kernel head (this is where regression manifests)
    x = add_conv("head1", x, 128, 64, 9)
    x = add_conv("head2", x, 64, 64, 9)

    # Upsample via Resize (this also regresses on 1.24.x)
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="scales"
    )
    inits.extend([roi, scales])
    resize_out = "resize_out"
    nodes.append(helper.make_node(
        "Resize", [x, "roi", "scales"], [resize_out], mode="nearest",
    ))

    # Refine
    final = add_conv("refine", resize_out, 64, 32, 3)

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, h, w])
    Y = helper.make_tensor_value_info(final, TensorProto.FLOAT, None)
    graph = helper.make_graph(nodes, "det_like", [X], [Y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def make_resize_model(ci: int, h: int, w: int, scale: float, path: str) -> str:
    """Pure Resize model (no Conv) to isolate Resize regression."""
    roi = numpy_helper.from_array(np.array([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, scale, scale], dtype=np.float32), name="scales"
    )
    node = helper.make_node("Resize", ["X", "roi", "scales"], ["Y"], mode="nearest")
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, ci, h, w])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    graph = helper.make_graph([node], "resize", [X], [Y], initializer=[roi, scales])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def make_concat_model(ci: int, h: int, w: int, n: int, path: str) -> str:
    """Pure Concat model: concatenate n identity copies along channel axis."""
    inputs = [f"X{i}" for i in range(n)]
    node = helper.make_node("Concat", inputs, ["Y"], axis=1)
    input_vis = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, ci, h, w])
        for name in inputs
    ]
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)
    graph = helper.make_graph([node], "concat", input_vis, [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


# ── Benchmark helpers ────────────────────────────────────────────────────────


def bench(
    model_path: str,
    feeds: dict[str, np.ndarray],
    threads: int,
    warmup: int = 2,
    runs: int = 5,
) -> dict:
    """Benchmark model, return timing stats in ms."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.log_severity_level = 3  # suppress warnings

    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

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


def get_rss_mb() -> float:
    """Peak RSS in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


# ── Test sections ────────────────────────────────────────────────────────────


def section_resolution_scaling(tmpdir: str, threads: int) -> dict:
    """
    Section 1: Resolution Scaling — throughput cliff detection.

    Creates single Conv 9×9 (ci=co=64) at increasing resolutions.
    On healthy ORT builds, throughput (pixels/ms) stays roughly constant.
    On 1.24.x, throughput drops >40% above ~226K pixels due to IGEMM
    indirection table exceeding cache hierarchy.
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

    # Regression detection: compare first and last throughput
    first_pix_ms = results[0]["pix_ms"]
    last_pix_ms = results[-1]["pix_ms"]
    drop_pct = round((1 - last_pix_ms / first_pix_ms) * 100, 1) if first_pix_ms > 0 else 0
    regression = drop_pct > 40

    return {
        "test": "resolution_scaling",
        "config": f"Conv {ks}x{ks} ci=co={ci}",
        "results": results,
        "throughput_drop_pct": drop_pct,
        "regression": regression,
        "threshold": "throughput drop > 40%",
    }


def section_mixed_model(tmpdir: str, threads: int) -> dict:
    """
    Section 2: Mixed Det-like Model — full model amplification.

    On healthy builds, high-res / low-res latency ratio should roughly match
    the pixel count ratio. On 1.24.x, it exceeds this due to compounding
    regressions in Conv + Resize + other ops.
    """
    configs = [
        (224, 160, "low-res"),
        (800, 608, "high-res"),
    ]

    results = []
    for h, w, label in configs:
        path = os.path.join(tmpdir, f"det_{h}_{w}.onnx")
        make_mixed_det_model(h, w, path)
        x = np.random.randn(1, 3, h, w).astype(np.float32)
        r = bench(path, {"X": x}, threads=threads, warmup=2, runs=3)
        results.append({"label": label, "h": h, "w": w, **r})
        os.remove(path)

    # Expected pixel ratio: (800*608)/(224*160) = 13.6
    # Expected latency ratio for linear scaling: ~13.6
    # With regression, ratio will be much higher
    low = results[0]["avg_ms"]
    high = results[1]["avg_ms"]
    pixel_ratio = (800 * 608) / (224 * 160)
    latency_ratio = round(high / low, 1) if low > 0 else 0
    excess = round(latency_ratio / pixel_ratio, 2)
    regression = excess > 1.5  # latency grows much faster than pixels

    return {
        "test": "mixed_det_model",
        "results": results,
        "pixel_ratio": round(pixel_ratio, 1),
        "latency_ratio": latency_ratio,
        "excess_ratio": excess,
        "regression": regression,
        "threshold": "latency/pixel excess > 1.5x",
    }


def section_non_conv_ops(tmpdir: str, threads: int) -> dict:
    """
    Section 3: Non-Conv Op Regression — Resize and Concat.

    Even without Conv nodes, Resize and Concat regress on large feature maps
    in ORT 1.24.x. This measures absolute throughput (output MB/s) for Resize
    to detect abnormal slowdowns independent of input size.

    The regression observed in profiling: Resize 97ms→1044ms (10.7x) on
    japan_demo.png (1600×1216) det model. We test with 256-channel feature maps
    at sizes matching the det model's internal feature maps.
    """
    results = {}

    # Resize: 256-channel feature maps at det-model-like sizes
    # In the det model, Resize upsamples ~400×300 → ~800×600 at ci=256
    for label, ci, h, w, scale in [
        ("resize_small", 256, 56, 56, 2.0),     # small: 56→112 (baseline)
        ("resize_large", 256, 200, 152, 2.0),    # large: 200→400 (det-like)
    ]:
        path = os.path.join(tmpdir, f"{label}.onnx")
        make_resize_model(ci, h, w, scale, path)
        x = np.random.randn(1, ci, h, w).astype(np.float32)
        r = bench(path, {"X": x}, threads=threads, warmup=3, runs=5)
        # Compute output throughput: output_bytes / time
        oh, ow = int(h * scale), int(w * scale)
        output_mb = ci * oh * ow * 4 / 1024 / 1024  # float32
        throughput_mb_s = round(output_mb / (r["avg_ms"] / 1000), 1) if r["avg_ms"] > 0 else 0
        results[label] = {
            "ci": ci, "h": h, "w": w, "scale": scale,
            "output_size": f"{oh}×{ow}", "output_mb": round(output_mb, 1),
            "throughput_mb_s": throughput_mb_s, **r,
        }
        os.remove(path)

    # Concat: small vs large feature map
    for label, ci, h, w, n in [
        ("concat_small", 64, 56, 56, 4),
        ("concat_large", 64, 400, 304, 4),
    ]:
        path = os.path.join(tmpdir, f"{label}.onnx")
        make_concat_model(ci, h, w, n, path)
        feeds = {f"X{i}": np.random.randn(1, ci, h, w).astype(np.float32) for i in range(n)}
        r = bench(path, feeds, threads=threads, warmup=3, runs=5)
        results[label] = {"ci": ci, "h": h, "w": w, "n_inputs": n, **r}
        os.remove(path)

    # Regression detection: Resize throughput should be consistent (memory-bound).
    # On healthy builds, throughput stays ~10-15 GB/s regardless of size.
    # On 1.24.x with large feature maps, throughput can drop to <2 GB/s.
    small_tp = results["resize_small"]["throughput_mb_s"]
    large_tp = results["resize_large"]["throughput_mb_s"]
    tp_drop_pct = round((1 - large_tp / small_tp) * 100, 1) if small_tp > 0 else 0
    regression = tp_drop_pct > 50  # >50% throughput drop is abnormal for memory-bound op

    return {
        "test": "non_conv_ops",
        "results": results,
        "resize_throughput_drop_pct": tp_drop_pct,
        "regression": regression,
        "threshold": "Resize throughput drop > 50% (small→large)",
    }


def section_memory(tmpdir: str, threads: int) -> dict:
    """
    Section 4: Memory Measurement — RSS delta after inference.

    ORT 1.24.x allocates massive internal buffers for IGEMM Conv path.
    We measure RSS in a subprocess to get a clean baseline.
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

    result = subprocess.run(
        [sys.executable, "-c", measure_script],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode == 0:
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


def section_variance(tmpdir: str, threads: int) -> dict:
    """
    Section 5: Variance Analysis — run-to-run stability.

    ORT 1.24.x shows extreme latency variance for large-kernel Conv on
    large feature maps. Healthy builds: max/min < 1.3x. Regression: > 2x.
    """
    ci, co, ks, h, w = 64, 64, 9, 544, 416  # above throughput cliff
    path = os.path.join(tmpdir, "var_conv.onnx")
    make_conv_model(ci, co, ks, h, w, path)
    x = np.random.randn(1, ci, h, w).astype(np.float32)

    r = bench(path, {"X": x}, threads=threads, warmup=3, runs=20)
    os.remove(path)

    regression = r["max_min_ratio"] > 2.0

    return {
        "test": "variance",
        "config": f"Conv {ks}x{ks} ci=co={ci} {h}x{w}, 20 runs",
        **r,
        "regression": regression,
        "threshold": "max/min ratio > 2.0x",
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="ORT KleidiAI IGEMM Conv regression reproducer"
    )
    parser.add_argument("--threads", type=int, default=2, help="intra_op_num_threads")
    parser.add_argument("--json", action="store_true", help="JSON output to stdout")
    args = parser.parse_args()

    env = {
        "ort_version": ort.__version__,
        "providers": ort.get_available_providers(),
        "python": sys.version.split()[0],
        "platform": f"{sys.platform} / {platform.machine()}",
        "threads": args.threads,
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

    try:
        # Section 1: Resolution Scaling
        if not args.json:
            print("Section 1: Resolution Scaling (Conv 9x9 ci=co=64)")
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

        # Section 2: Mixed Det-like Model
        if not args.json:
            print("Section 2: Mixed Det-like Model (Conv 9x9 + Resize + ...)")
            print("-" * 70)
        s2 = section_mixed_model(tmpdir, args.threads)
        all_results["sections"].append(s2)
        if not args.json:
            for r in s2["results"]:
                print(f"  {r['label']:<10s} ({r['h']}×{r['w']}):  {r['avg_ms']:>8.1f}ms")
            print(f"  Pixel ratio: {s2['pixel_ratio']}x, Latency ratio: {s2['latency_ratio']}x, Excess: {s2['excess_ratio']}x")
            status = "REGRESSION" if s2["regression"] else "OK"
            print(f"  [{status}]")
            print()

        # Section 3: Non-Conv Ops
        if not args.json:
            print("Section 3: Non-Conv Op Regression (Resize, Concat)")
            print("-" * 70)
        s3 = section_non_conv_ops(tmpdir, args.threads)
        all_results["sections"].append(s3)
        if not args.json:
            for label, r in s3["results"].items():
                size = f"{r['h']}×{r['w']}"
                tp = f"  {r['throughput_mb_s']} MB/s" if "throughput_mb_s" in r else ""
                print(f"  {label:<16s}  ({size:>9s}):  {r['avg_ms']:>8.1f}ms{tp}")
            print(f"  Resize throughput drop: {s3['resize_throughput_drop_pct']}%")
            status = "REGRESSION" if s3["regression"] else "OK"
            print(f"  [{status}]")
            print()

        # Section 4: Memory
        if not args.json:
            print("Section 4: Memory (RSS delta for 1 Conv 9x9 inference)")
            print("-" * 70)
        s4 = section_memory(tmpdir, args.threads)
        all_results["sections"].append(s4)
        if not args.json:
            print(f"  Before: {s4['rss_before_mb']:.0f} MB → After: {s4['rss_after_mb']:.0f} MB")
            print(f"  Delta: +{s4['rss_delta_mb']:.0f} MB", end="")
            status = "REGRESSION" if s4["regression"] else "OK"
            print(f"  [{status}]")
            print()

        # Section 5: Variance
        if not args.json:
            print("Section 5: Variance (20 runs, Conv 9x9 ci=co=64 544x416)")
            print("-" * 70)
        s5 = section_variance(tmpdir, args.threads)
        all_results["sections"].append(s5)
        if not args.json:
            print(f"  avg: {s5['avg_ms']:.1f}ms  std: {s5['std_ms']:.1f}ms  max/min: {s5['max_min_ratio']}x")
            status = "REGRESSION" if s5["regression"] else "OK"
            print(f"  [{status}]")
            print()

    finally:
        for f in os.listdir(tmpdir):
            os.remove(os.path.join(tmpdir, f))
        os.rmdir(tmpdir)

    # Summary
    regression_count = sum(1 for s in all_results["sections"] if s["regression"])
    total = len(all_results["sections"])
    all_results["summary"] = {
        "regressions": regression_count,
        "total_tests": total,
        "verdict": "REGRESSION DETECTED" if regression_count > 0 else "ALL PASS",
    }

    if args.json:
        json.dump(all_results, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        print("=" * 70)
        if regression_count > 0:
            print(f"  RESULT: {regression_count}/{total} sections show regression")
            print(f"  This ORT build ({ort.__version__}) exhibits the IGEMM Conv regression.")
            print(f"  See: https://github.com/microsoft/onnxruntime/issues/27633")
        else:
            print(f"  RESULT: ALL {total} sections PASS — no regression detected")
        print("=" * 70)


if __name__ == "__main__":
    main()
