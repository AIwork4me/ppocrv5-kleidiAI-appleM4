#!/usr/bin/env python3
"""PP-OCRv5 ONNX Runtime inference pipeline — single-file, copy-paste ready.

A complete 4-model OCR pipeline (doc_ori + det + textline_ori + rec) that
produces results **100% identical** to PaddleOCR / PaddleX 3.4.x native
inference. Verified on 7 images, 228 text regions, with zero text mismatch
and < 0.00002 average confidence difference.

Requirements:
    pip install onnxruntime opencv-python-headless numpy pyclipper

Quick start:
    from ppocrv5_onnx import PPOCRv5Pipeline

    pipeline = PPOCRv5Pipeline("models")
    results = pipeline.predict("image.png")
    for r in results:
        print(f"{r['text']}  ({r['confidence']:.4f})")

Architecture:
    image -> doc_ori (orientation) -> det (DB text detection)
          -> textline_ori (line orientation) -> rec (CTC recognition)

Accuracy alignment details: see docs/ACCURACY_ALIGNMENT.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pyclipper

__all__ = [
    "PPOCRv5Pipeline",
    # Pre/post-processing functions (used by benchmarks and integrations)
    "det_preprocess",
    "db_postprocess",
    "get_minarea_rect_crop",
    "rec_preprocess_single",
    "rec_preprocess_batch",
    "ctc_decode",
    "load_charset",
    "doc_ori_preprocess",
    "textline_ori_preprocess",
    "rotate_image",
    "sort_boxes",
    # Constants (used by benchmarks)
    "DOC_ORI_LABELS",
    "TEXTLINE_ORI_LABELS",
    "REC_BATCH_SIZE",
]

# ---------------------------------------------------------------------------
#  Constants aligned with PaddleOCR/PaddleX 3.4.x runtime parameters
# ---------------------------------------------------------------------------

# Detection — Pipeline runtime overrides inference.yml defaults:
#   limit_type="min", limit_side_len=64 (NOT resize_long=960)
_DET_LIMIT_SIDE_LEN = 64
_DET_LIMIT_TYPE = "min"
_DET_MAX_SIDE_LIMIT = 4000
_DET_STRIDE = 32
_DET_THRESH = 0.3
_DET_BOX_THRESH = 0.6
_DET_UNCLIP_RATIO = 1.5
_DET_MIN_SIZE = 3
_DET_MAX_CANDIDATES = 1000

# Recognition
_REC_IMG_H = 48
_REC_IMG_W_BASE = 320
_REC_MAX_IMG_W = 3200
_REC_BATCH_SIZE = 6
REC_BATCH_SIZE = _REC_BATCH_SIZE

# ImageNet normalization (shared by det + classification models)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DOC_ORI_LABELS = [0, 90, 180, 270]
TEXTLINE_ORI_LABELS = [0, 180]


# ═══════════════════════════════════════════════════════════════════════════
#  Detection pre/post-processing
# ═══════════════════════════════════════════════════════════════════════════

def det_preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize and normalize an image for the DB text detection model.

    Args:
        img: BGR input image (H, W, C).

    Returns:
        A tuple of (tensor, img_shape) where tensor is NCHW float32 and
        img_shape is [src_h, src_w, ratio_h, ratio_w].
    """
    src_h, src_w = img.shape[:2]

    if _DET_LIMIT_TYPE == "max":
        if max(src_h, src_w) > _DET_LIMIT_SIDE_LEN:
            ratio = float(_DET_LIMIT_SIDE_LEN) / (src_h if src_h > src_w else src_w)
        else:
            ratio = 1.0
    elif _DET_LIMIT_TYPE == "min":
        if min(src_h, src_w) < _DET_LIMIT_SIDE_LEN:
            ratio = float(_DET_LIMIT_SIDE_LEN) / (src_h if src_h < src_w else src_w)
        else:
            ratio = 1.0
    else:
        ratio = float(_DET_LIMIT_SIDE_LEN) / max(src_h, src_w)

    resize_h = int(src_h * ratio)
    resize_w = int(src_w * ratio)

    if max(resize_h, resize_w) > _DET_MAX_SIDE_LIMIT:
        r = float(_DET_MAX_SIDE_LIMIT) / max(resize_h, resize_w)
        resize_h = int(resize_h * r)
        resize_w = int(resize_w * r)

    resize_h = max(int(round(resize_h / _DET_STRIDE) * _DET_STRIDE), _DET_STRIDE)
    resize_w = max(int(round(resize_w / _DET_STRIDE) * _DET_STRIDE), _DET_STRIDE)

    if resize_h == src_h and resize_w == src_w:
        resized = img
    else:
        resized = cv2.resize(img, (resize_w, resize_h))

    ratio_h = resize_h / float(src_h)
    ratio_w = resize_w / float(src_w)

    # NormalizeImage: (pixel * scale - mean) / std via pre-computed alpha/beta
    tensor = resized.astype(np.float32)
    for c in range(3):
        alpha = (1.0 / 255.0) / _IMAGENET_STD[c]
        beta = -_IMAGENET_MEAN[c] / _IMAGENET_STD[c]
        tensor[:, :, c] = tensor[:, :, c] * alpha + beta

    tensor = tensor.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
    img_shape = np.array([src_h, src_w, ratio_h, ratio_w])
    return tensor, img_shape


def _get_mini_boxes(contour: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Return sorted box points and minimum side length from MinAreaRect."""
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def _box_score_fast(bitmap: np.ndarray, box_coords: np.ndarray) -> float:
    """Compute mean score inside a polygon using math.floor/ceil boundaries."""
    h, w = bitmap.shape[:2]
    box = box_coords.copy()
    xmin = max(0, min(math.floor(box[:, 0].min()), w - 1))
    xmax = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    ymin = max(0, min(math.floor(box[:, 1].min()), h - 1))
    ymax = max(0, min(math.ceil(box[:, 1].max()), h - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _unclip_box(box: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Expand polygon by offset proportional to area/perimeter ratio."""
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    distance = area * unclip_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded = np.array(offset.Execute(distance))
    except ValueError:
        expanded = np.array(offset.Execute(distance)[0])
    return expanded


def db_postprocess(
    pred: np.ndarray, img_shape: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """DB post-processing: threshold, find contours, unclip, and rescale.

    Args:
        pred: Raw detection output (1, 1, H, W).
        img_shape: Array of [src_h, src_w, ratio_h, ratio_w].

    Returns:
        A tuple of (boxes, scores) where boxes is int16 array of shape
        (N, 4, 2) and scores is a list of N float values.
    """
    src_h, src_w, ratio_h, ratio_w = img_shape
    prob = pred[0, 0]
    bitmap = prob > _DET_THRESH
    height, width = bitmap.shape
    dest_width = src_w
    dest_height = src_h
    width_scale = dest_width / width
    height_scale = dest_height / height

    outs = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = outs[0] if len(outs) == 2 else outs[1]
    num_contours = min(len(contours), _DET_MAX_CANDIDATES)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = _get_mini_boxes(contour)
        if sside < _DET_MIN_SIZE:
            continue
        points = np.array(points)
        score = _box_score_fast(prob, points.reshape(-1, 2))
        if _DET_BOX_THRESH > score:
            continue
        expanded = _unclip_box(points, _DET_UNCLIP_RATIO).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(expanded)
        if sside < _DET_MIN_SIZE + 2:
            continue
        box = np.array(box)
        for i in range(box.shape[0]):
            box[i, 0] = max(0, min(round(box[i, 0] * width_scale), dest_width))
            box[i, 1] = max(0, min(round(box[i, 1] * height_scale), dest_height))
        boxes.append(box.astype(np.int16))
        scores.append(score)

    if boxes:
        return np.array(boxes, dtype=np.int16), scores
    return np.zeros((0, 4, 2), dtype=np.int16), scores


# ═══════════════════════════════════════════════════════════════════════════
#  Text region cropping
# ═══════════════════════════════════════════════════════════════════════════

def get_minarea_rect_crop(src: np.ndarray, box_int: np.ndarray) -> np.ndarray:
    """Crop a text region using minAreaRect with float32 coordinate precision.

    This matches Paddle's CropByPolys.get_minarea_rect_crop exactly:
    int16 box -> minAreaRect(int32) -> boxPoints(float32) -> sort -> crop.
    Using int16 coordinates directly would cause 0.5-1.0px offset, leading
    to argmax flips in the recognition model.
    """
    bounding_box = cv2.minAreaRect(box_int.astype(np.int32))
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1, index_4 = 0, 1
    else:
        index_1, index_4 = 1, 0
    if points[3][1] > points[2][1]:
        index_2, index_3 = 2, 3
    else:
        index_2, index_3 = 3, 2

    float_box = np.array(
        [points[index_1], points[index_2], points[index_3], points[index_4]],
        dtype=np.float32,
    )
    return _get_rotate_crop_image(src, float_box)


def _get_rotate_crop_image(src: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Perspective-transform crop of a text region."""
    pts = points.astype(np.float32)
    img_crop_width = int(
        max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    )
    img_crop_height = int(
        max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))
    )
    if img_crop_width <= 0 or img_crop_height <= 0:
        return np.zeros((_REC_IMG_H, _REC_IMG_W_BASE, 3), dtype=np.uint8)

    pts_std = np.float32([
        [0, 0], [img_crop_width, 0],
        [img_crop_width, img_crop_height], [0, img_crop_height],
    ])
    transform_matrix = cv2.getPerspectiveTransform(pts, pts_std)
    dst_img = cv2.warpPerspective(
        src, transform_matrix, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    dst_h, dst_w = dst_img.shape[0:2]
    if dst_h * 1.0 / dst_w >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


# ═══════════════════════════════════════════════════════════════════════════
#  Recognition pre-processing
# ═══════════════════════════════════════════════════════════════════════════

def rec_preprocess_single(crop: np.ndarray) -> np.ndarray:
    """Resize and normalize a single crop for the recognition model."""
    img_c, img_h, img_w_base = 3, _REC_IMG_H, _REC_IMG_W_BASE
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((img_c, img_h, img_w_base), dtype=np.float32)

    max_wh_ratio = max(img_w_base / img_h, w * 1.0 / h)
    img_w = int(img_h * max_wh_ratio)
    if img_w > _REC_MAX_IMG_W:
        resized = cv2.resize(crop, (_REC_MAX_IMG_W, img_h))
        resized_w = _REC_MAX_IMG_W
        img_w = _REC_MAX_IMG_W
    else:
        ratio = w / float(h)
        if math.ceil(img_h * ratio) > img_w:
            resized_w = img_w
        else:
            resized_w = int(math.ceil(img_h * ratio))
        resized = cv2.resize(crop, (resized_w, img_h))

    resized = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
    resized -= 0.5
    resized /= 0.5

    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized
    return padding_im


def rec_preprocess_batch(crops: list[np.ndarray]) -> np.ndarray:
    """Batch-preprocess crops for recognition, padding to the batch max width.

    This matches Paddle's ToBatch.__pad_imgs: each crop is resized
    independently, then all are padded to the batch's maximum width.
    The padding width affects recognition output.
    """
    if not crops:
        return np.zeros((0, 3, _REC_IMG_H, _REC_IMG_W_BASE), dtype=np.float32)

    imgs = [rec_preprocess_single(crop) for crop in crops]
    max_width = max(img.shape[2] for img in imgs)
    padded = []
    for img in imgs:
        _, h, w = img.shape
        if w < max_width:
            img = np.pad(
                img, ((0, 0), (0, 0), (0, max_width - w)),
                mode="constant", constant_values=0,
            )
        padded.append(img)
    return np.stack(padded, axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  CTC decoding
# ═══════════════════════════════════════════════════════════════════════════

def load_charset(dict_path: str | Path) -> list[str]:
    """Load character set from a dictionary file.

    The returned list follows PaddleOCR's convention:
    ``["blank"] + dict_chars + [" "]``.

    Args:
        dict_path: Path to the ppocrv5_dict.txt file.

    Returns:
        Ordered list of characters including blank token at index 0.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    lines.append(" ")  # use_space_char=True
    character = ["blank"] + lines  # CTCLabelDecode.add_special_char
    return character


def ctc_decode(output: np.ndarray, character: list[str]) -> tuple[str, float]:
    """CTC greedy decode using raw logits (not softmax) for confidence.

    This matches Paddle exactly: argmax selects characters, raw logit max
    values are averaged for confidence scoring.

    Args:
        output: Model output of shape (1, seq_len, num_classes).
        character: Character set from ``load_charset()``.

    Returns:
        A tuple of (text, confidence).
    """
    preds = np.array(output[0])  # [seq_len, num_classes]
    preds_idx = preds.argmax(axis=-1)
    preds_prob = preds.max(axis=-1)

    selection = np.ones(len(preds_idx), dtype=bool)
    selection[1:] = preds_idx[1:] != preds_idx[:-1]  # deduplicate
    selection &= preds_idx != 0  # remove blank

    char_list = [character[int(idx)] for idx in preds_idx[selection]]
    conf_list = preds_prob[selection]

    text = "".join(char_list)
    avg_conf = float(np.mean(conf_list)) if len(conf_list) > 0 else 0.0
    return text, avg_conf


# ═══════════════════════════════════════════════════════════════════════════
#  Classification model pre-processing
# ═══════════════════════════════════════════════════════════════════════════

def _cls_normalize_rgb(img_rgb: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization to an RGB image using cv2.split/merge."""
    scale = 1.0 / 255.0
    alpha = [scale / _IMAGENET_STD[c] for c in range(3)]
    beta = [-_IMAGENET_MEAN[c] / _IMAGENET_STD[c] for c in range(3)]
    split_im = list(cv2.split(img_rgb))
    for c in range(3):
        split_im[c] = split_im[c].astype(np.float32)
        split_im[c] *= alpha[c]
        split_im[c] += beta[c]
    return cv2.merge(split_im)


def doc_ori_preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Preprocess image for the document orientation classifier.

    Pipeline: BGR->RGB, ResizeByShort(256), CenterCrop(224), Normalize, NCHW.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = 256.0 / min(h, w)
    h_resize = round(h * scale)
    w_resize = round(w * scale)
    if h_resize != h or w_resize != w:
        img = cv2.resize(img, (w_resize, h_resize), interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    cw, ch = 224, 224
    x1 = max(0, (w - cw) // 2)
    y1 = max(0, (h - ch) // 2)
    img = img[y1:y1 + ch, x1:x1 + cw]
    img = _cls_normalize_rgb(img)
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def textline_ori_preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a crop for the text line orientation classifier.

    Pipeline: BGR->RGB, Resize(160x80), Normalize, NCHW.
    """
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 80), interpolation=cv2.INTER_LINEAR)
    img = _cls_normalize_rgb(img)
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  Rotation / sorting utilities
# ═══════════════════════════════════════════════════════════════════════════

def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by the given angle using warpAffine with INTER_CUBIC."""
    if angle < 1e-7:
        return image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(mat[0, 0])
    sin = np.abs(mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    mat[0, 2] += (new_w - w) / 2
    mat[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, mat, (new_w, new_h), flags=cv2.INTER_CUBIC)


def sort_boxes(dt_polys: np.ndarray) -> list[np.ndarray]:
    """Sort text boxes in reading order (top-to-bottom, left-to-right)."""
    if len(dt_polys) == 0:
        return []
    dt_boxes = np.array(dt_polys)
    sorted_boxes = sorted(list(dt_boxes), key=lambda x: (x[0][1], x[0][0]))
    num_boxes = len(sorted_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if (abs(sorted_boxes[j + 1][0][1] - sorted_boxes[j][0][1]) < 10
                    and sorted_boxes[j + 1][0][0] < sorted_boxes[j][0][0]):
                sorted_boxes[j], sorted_boxes[j + 1] = (
                    sorted_boxes[j + 1], sorted_boxes[j]
                )
            else:
                break
    return sorted_boxes


# ═══════════════════════════════════════════════════════════════════════════
#  PPOCRv5Pipeline — main entry point
# ═══════════════════════════════════════════════════════════════════════════

class PPOCRv5Pipeline:
    """PP-OCRv5 inference pipeline using ONNX Runtime.

    A complete 4-model pipeline that produces results 100% identical to
    PaddleOCR/PaddleX 3.4.x native inference:

        doc_ori -> det -> textline_ori -> rec

    Args:
        model_dir: Directory containing the 4 ONNX model sub-directories.
            Expected layout::

                model_dir/
                    PP-OCRv5_server_det_onnx/inference.onnx
                    PP-OCRv5_server_rec_onnx/inference.onnx
                    PP-LCNet_x1_0_doc_ori_onnx/inference.onnx
                    PP-LCNet_x1_0_textline_ori_onnx/inference.onnx

        dict_path: Path to ppocrv5_dict.txt. Defaults to
            ``model_dir/../data/dict/ppocrv5_dict.txt`` if not specified.
        threads: Number of CPU threads for ORT (default: 4).
        det_path: Override path to detection model ONNX file.
        rec_path: Override path to recognition model ONNX file.
        doc_ori_path: Override path to doc orientation model ONNX file.
        textline_ori_path: Override path to textline orientation model ONNX file.

    Example:
        >>> pipeline = PPOCRv5Pipeline("models")
        >>> results = pipeline.predict("image.png")
        >>> for r in results:
        ...     print(f"{r['text']}  ({r['confidence']:.4f})")
    """

    def __init__(
        self,
        model_dir: str | Path,
        dict_path: str | Path | None = None,
        threads: int = 4,
        *,
        det_path: str | Path | None = None,
        rec_path: str | Path | None = None,
        doc_ori_path: str | Path | None = None,
        textline_ori_path: str | Path | None = None,
    ):
        model_dir = Path(model_dir)

        det = str(det_path or model_dir / "PP-OCRv5_server_det_onnx" / "inference.onnx")
        rec = str(rec_path or model_dir / "PP-OCRv5_server_rec_onnx" / "inference.onnx")
        doc_ori = str(doc_ori_path or model_dir / "PP-LCNet_x1_0_doc_ori_onnx" / "inference.onnx")
        textline_ori = str(textline_ori_path or model_dir / "PP-LCNet_x1_0_textline_ori_onnx" / "inference.onnx")

        if dict_path is None:
            # Try common locations
            candidates = [
                model_dir.parent / "data" / "dict" / "ppocrv5_dict.txt",
                model_dir / "ppocrv5_dict.txt",
                model_dir.parent / "dict" / "ppocrv5_dict.txt",
            ]
            for c in candidates:
                if c.exists():
                    dict_path = c
                    break
            if dict_path is None:
                raise FileNotFoundError(
                    f"Cannot find ppocrv5_dict.txt. Searched: {[str(c) for c in candidates]}. "
                    "Please provide dict_path explicitly."
                )

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = threads
        opts.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]

        self.det_sess = ort.InferenceSession(det, opts, providers=prov)
        self.rec_sess = ort.InferenceSession(rec, opts, providers=prov)
        self.doc_ori_sess = ort.InferenceSession(doc_ori, opts, providers=prov)
        self.textline_ori_sess = ort.InferenceSession(textline_ori, opts, providers=prov)
        self.character = load_charset(str(dict_path))

    def classify_doc_orientation(self, img_bgr: np.ndarray) -> int:
        """Classify document orientation. Returns angle in {0, 90, 180, 270}."""
        tensor = doc_ori_preprocess(img_bgr)
        logits = self.doc_ori_sess.run(None, {"x": tensor})[0]
        return DOC_ORI_LABELS[int(np.argmax(logits[0]))]

    def classify_textline_orientation(self, crop_bgr: np.ndarray) -> int:
        """Classify text line orientation. Returns angle in {0, 180}."""
        tensor = textline_ori_preprocess(crop_bgr)
        logits = self.textline_ori_sess.run(None, {"x": tensor})[0]
        return TEXTLINE_ORI_LABELS[int(np.argmax(logits[0]))]

    def predict(self, image: str | Path | np.ndarray) -> list[dict[str, Any]]:
        """Run the full OCR pipeline on a single image.

        Args:
            image: A file path (str or Path) or a BGR numpy array (HxWxC uint8).

        Returns:
            List of dicts, each with keys:
                - ``text`` (str): Recognized text.
                - ``confidence`` (float): Recognition confidence (raw logit mean).
                - ``bounding_box`` (list): 4-point polygon [[x,y], ...].
        """
        if isinstance(image, (str, Path)):
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img_bgr = image

        # 1) Document orientation classification + correction
        angle = self.classify_doc_orientation(img_bgr)
        if angle != 0:
            img_bgr = rotate_image(img_bgr, angle)

        # 2) Text detection (DB)
        det_input, img_shape = det_preprocess(img_bgr)
        det_out = self.det_sess.run(None, {"x": det_input})[0]
        boxes, det_scores = db_postprocess(det_out, img_shape)

        if len(boxes) == 0:
            return []

        # 3) Sort boxes in reading order
        sorted_boxes_list = sort_boxes(boxes)

        # 4) Crop + text line orientation correction
        crops = []
        for box in sorted_boxes_list:
            crop = get_minarea_rect_crop(img_bgr, box)
            line_angle = self.classify_textline_orientation(crop)
            if line_angle == 180:
                crop = rotate_image(crop, 180)
            crops.append(crop)

        # 5) Recognition (batched, matching Paddle's batch_size=6 + ratio sort)
        crop_indices = list(range(len(crops)))
        crop_indices.sort(key=lambda i: crops[i].shape[1] / float(crops[i].shape[0]))

        rec_texts: list[str | None] = [None] * len(crops)
        rec_confs: list[float | None] = [None] * len(crops)

        for batch_start in range(0, len(crop_indices), _REC_BATCH_SIZE):
            batch_idx = crop_indices[batch_start:batch_start + _REC_BATCH_SIZE]
            batch_crops = [crops[i] for i in batch_idx]
            batch_tensor = rec_preprocess_batch(batch_crops)

            for j, idx in enumerate(batch_idx):
                single = batch_tensor[j:j + 1]
                rec_out = self.rec_sess.run(None, {"x": single})[0]
                text, conf = ctc_decode(rec_out, self.character)
                rec_texts[idx] = text
                rec_confs[idx] = conf

        results = []
        for i, box in enumerate(sorted_boxes_list):
            results.append({
                "bounding_box": box.tolist(),
                "text": rec_texts[i],
                "confidence": round(rec_confs[i], 6),
            })

        return results


# ═══════════════════════════════════════════════════════════════════════════
#  Built-in demo
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "models"
    dict_path = repo_root / "data" / "dict" / "ppocrv5_dict.txt"
    images_dir = repo_root / "data" / "images"

    det_model = model_dir / "PP-OCRv5_server_det_onnx" / "inference.onnx"
    if not det_model.exists():
        print(f"Models not found in {model_dir}/")
        print("Download models first: python scripts/download_models.py")
        sys.exit(1)

    if not dict_path.exists():
        print(f"Dictionary not found: {dict_path}")
        sys.exit(1)

    pipeline = PPOCRv5Pipeline(model_dir, dict_path=dict_path, threads=4)
    print(f"Pipeline initialized (ORT {ort.__version__})")

    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ) if images_dir.exists() else []

    if not image_files:
        print(f"No images found in {images_dir}/")
        sys.exit(1)

    total_texts = 0
    for img_path in image_files:
        results = pipeline.predict(img_path)
        total_texts += len(results)
        print(f"\n{img_path.name}: {len(results)} text regions")
        for r in results:
            print(f"  [{r['confidence']:.4f}] {r['text']}")

    print(f"\nTotal: {len(image_files)} images, {total_texts} text regions")
