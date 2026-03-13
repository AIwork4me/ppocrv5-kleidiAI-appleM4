#!/usr/bin/env python3
"""Quick start: PP-OCRv5 OCR with ONNX Runtime."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for development usage.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from ppocrv5_onnx import PPOCRv5Pipeline


def main() -> None:
    model_dir = _REPO_ROOT / "models"
    dict_path = _REPO_ROOT / "data" / "dict" / "ppocrv5_dict.txt"
    images_dir = _REPO_ROOT / "data" / "images"

    pipeline = PPOCRv5Pipeline(model_dir, dict_path=dict_path, threads=4)

    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        print("No images found. Place PNG files in data/images/")
        sys.exit(1)

    results = pipeline.predict(image_files[0])

    print(f"Image: {image_files[0].name}")
    print(f"Detected {len(results)} text regions:\n")
    for r in results:
        print(f"  [{r['confidence']:.4f}] {r['text']}")


if __name__ == "__main__":
    main()
