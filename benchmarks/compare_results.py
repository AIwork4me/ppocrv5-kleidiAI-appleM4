#!/usr/bin/env python3
"""Compare benchmark results across backends.

Auto-discovers all JSON files in results/ and generates a comparison report
including speed, accuracy, per-model timing, and KleidiAI acceleration analysis.

Usage:
    python benchmarks/compare_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def discover_results() -> list[tuple[str, dict]]:
    """Auto-discover and load all result JSON files."""
    results = []
    if not RESULTS_DIR.exists():
        return results

    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            label = f.stem  # e.g. "paddle_3.3.0", "ort_1.21.1"
            results.append((label, data))
            print(f"  Loaded: {f.name}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skip: {f.name} ({e})")

    return results


def compare_speed(results: list[tuple[str, dict]]) -> None:
    """Compare inference speed across backends."""
    print("\n" + "=" * 85)
    print("Speed Comparison")
    print("=" * 85)

    header = f"  {'Backend':<28} {'Avg (ms)':>10} {'FPS':>10} {'Init (s)':>10}"
    print(f"\n{header}")
    print("  " + "-" * 60)

    latencies = []
    for label, data in results:
        metadata = data.get("metadata", {})
        avg = metadata.get("avg_latency_ms", 0)
        fps = metadata.get("fps", 0)
        init = metadata.get("init_time_sec", 0)
        latencies.append((label, avg))
        print(f"  {label:<28} {avg:>10.2f} {fps:>10.4f} {init:>10.2f}")

    if len(latencies) >= 2:
        latencies.sort(key=lambda x: x[1])
        fastest = latencies[0]
        print(f"\n  Fastest: {fastest[0]} ({fastest[1]:.2f} ms)")
        for label, avg in latencies[1:]:
            ratio = avg / fastest[1] if fastest[1] > 0 else 0
            print(f"    vs {label}: {ratio:.2f}x slower")


def compare_per_model_timing(results: list[tuple[str, dict]]) -> None:
    """Compare per-model timing across backends."""
    print("\n" + "=" * 85)
    print("Per-Model Timing (avg ms per image)")
    print("=" * 85)

    agg_data = []
    labels = []
    for label, data in results:
        agg = data.get("aggregate_timing", {})
        if agg:
            agg_data.append(agg)
            labels.append(label)

    if not agg_data:
        print("  No per-model timing data available")
        return

    models = ["doc_ori", "det", "textline_ori", "rec"]
    header = f"  {'Model':<16}"
    for lbl in labels:
        header += f" {lbl:>18}"
    if len(labels) >= 2:
        header += f" {'Speedup':>10}"
    print(f"\n{header}")
    print("  " + "-" * (16 + 19 * len(labels) + (11 if len(labels) >= 2 else 0)))

    for model in models:
        row = f"  {model:<16}"
        totals = []
        for agg in agg_data:
            phases = agg.get(model, {})
            total = (
                phases.get("preprocess_ms", 0)
                + phases.get("inference_ms", 0)
                + phases.get("postprocess_ms", 0)
            )
            totals.append(total)
            row += f" {total:>18.2f}"
        if len(totals) >= 2 and totals[-1] > 0:
            speedup = totals[0] / totals[-1]
            row += f" {speedup:>9.2f}x"
        print(row)


def compare_accuracy(results: list[tuple[str, dict]]) -> None:
    """Compare text recognition accuracy across backends."""
    print("\n" + "=" * 85)
    print("Accuracy Comparison")
    print("=" * 85)

    if len(results) < 2:
        print("  Need at least 2 results for comparison")
        return

    # Use first result as baseline
    base_label, base_data = results[0]
    base_texts = {}
    for item in base_data.get("results", []):
        img = item.get("image_path", "")
        texts = [r.get("text", "") for r in item.get("results", [])]
        base_texts[img] = texts

    print(f"\n  Baseline: {base_label}")

    for comp_label, comp_data in results[1:]:
        comp_texts = {}
        for item in comp_data.get("results", []):
            img = item.get("image_path", "")
            texts = [r.get("text", "") for r in item.get("results", [])]
            comp_texts[img] = texts

        all_images = sorted(base_texts.keys() | comp_texts.keys())
        total_base = 0
        total_match = 0

        for img in all_images:
            base = base_texts.get(img, [])
            comp = comp_texts.get(img, [])
            total_base += len(base)
            total_match += sum(1 for t in base if t in comp)

        rate = total_match / total_base * 100 if total_base > 0 else 0
        print(f"  {base_label} vs {comp_label}: {total_match}/{total_base} texts match ({rate:.1f}%)")

    # Confidence comparison
    print("\n  Confidence differences:")
    for comp_label, comp_data in results[1:]:
        base_confs = {}
        for item in base_data.get("results", []):
            for r in item.get("results", []):
                base_confs[r.get("text", "")] = r.get("confidence", 0)

        comp_confs = {}
        for item in comp_data.get("results", []):
            for r in item.get("results", []):
                comp_confs[r.get("text", "")] = r.get("confidence", 0)

        diffs = []
        for text, conf in base_confs.items():
            if text in comp_confs:
                diffs.append(abs(conf - comp_confs[text]))

        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            print(f"    {base_label} vs {comp_label}: avg |conf diff| = {avg_diff:.6f}")


def compare_per_image_speed(results: list[tuple[str, dict]]) -> None:
    """Compare per-image speed across backends."""
    print("\n" + "=" * 85)
    print("Per-Image Speed (avg ms)")
    print("=" * 85)

    labels = [label for label, _ in results]
    sources = []
    for _, data in results:
        by_img = {r["image_path"]: r for r in data.get("results", [])}
        sources.append(by_img)

    all_images = sorted(set().union(*(s.keys() for s in sources)))

    header = f"  {'Image':<28}"
    for lbl in labels:
        header += f" {lbl:>14}"
    print(f"\n{header}")
    print("  " + "-" * (28 + 15 * len(labels)))

    for img in all_images:
        row = f"  {img:<28}"
        for src in sources:
            lat = src.get(img, {}).get("avg_latency_ms", 0)
            row += f" {lat:>14.2f}"
        print(row)


def main() -> None:
    print("=" * 85)
    print("PP-OCRv5 Benchmark Comparison")
    print("=" * 85)
    print("\nLoading results:")

    results = discover_results()

    if not results:
        print("\nNo result files found in results/")
        print("Run benchmarks first: python benchmarks/benchmark_unified.py --backend ort")
        sys.exit(1)

    if len(results) == 1:
        print("\nOnly 1 result found. Run more backends for comparison.")

    compare_speed(results)
    compare_per_model_timing(results)
    compare_accuracy(results)
    compare_per_image_speed(results)

    print()


if __name__ == "__main__":
    main()
