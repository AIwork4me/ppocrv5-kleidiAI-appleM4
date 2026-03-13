#!/usr/bin/env python3
"""Verify that ONNX models are downloaded and placed correctly."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

REQUIRED_ONNX = [
    ("PP-OCRv5_server_det_onnx/inference.onnx", "Text detection", 80),
    ("PP-OCRv5_server_rec_onnx/inference.onnx", "Text recognition", 75),
    ("PP-LCNet_x1_0_doc_ori_onnx/inference.onnx", "Doc orientation", 5),
    ("PP-LCNet_x1_0_textline_ori_onnx/inference.onnx", "Textline orientation", 5),
]

OPTIONAL_PADDLE = [
    "PP-OCRv5_server_det_infer/inference.json",
    "PP-OCRv5_server_rec_infer/inference.json",
    "PP-LCNet_x1_0_doc_ori_infer/inference.json",
    "PP-LCNet_x1_0_textline_ori_infer/inference.json",
]


def main() -> int:
    print("PP-OCRv5 Model Verification")
    print("=" * 60)
    print(f"Models directory: {MODELS_DIR}\n")

    all_ok = True

    print("Required ONNX models:")
    for rel_path, desc, min_mb in REQUIRED_ONNX:
        full_path = MODELS_DIR / rel_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            if size_mb >= min_mb:
                print(f"  [OK]   {desc:<22} {rel_path} ({size_mb:.1f} MB)")
            else:
                print(f"  [WARN] {desc:<22} {rel_path} ({size_mb:.1f} MB, expected >= {min_mb} MB)")
        else:
            print(f"  [MISS] {desc:<22} {rel_path}")
            all_ok = False

    print("\nOptional Paddle models (for benchmark --backend paddle):")
    paddle_count = 0
    for rel_path in OPTIONAL_PADDLE:
        full_path = MODELS_DIR / rel_path
        if full_path.exists():
            print(f"  [OK]   {rel_path}")
            paddle_count += 1
        else:
            print(f"  [    ] {rel_path}")

    print(f"\n{'=' * 60}")
    if all_ok:
        print("All required ONNX models are present. Ready to run!")
        print(f"Optional Paddle models: {paddle_count}/{len(OPTIONAL_PADDLE)}")
        return 0
    else:
        print("Some required models are missing.")
        print("\nDownload from Baidu Pan:")
        print("  Link: https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw")
        print("  Password: uepw")
        print(f"\nPlace model directories under: {MODELS_DIR}/")
        print("See models/README.md for the expected directory layout.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
