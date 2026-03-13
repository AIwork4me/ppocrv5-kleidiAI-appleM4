# Pipeline Architecture

## 4-Model OCR Pipeline

PP-OCRv5 uses a 4-model pipeline for full-page OCR:

```
                    ┌─────────────┐
                    │  Input Image │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   doc_ori    │  Document orientation classification
                    │  (LCNet)    │  → 0° / 90° / 180° / 270°
                    └──────┬──────┘
                           │ rotate if needed
                    ┌──────▼──────┐
                    │     det      │  Text detection (DB)
                    │  (PP-OCRv5)  │  → N bounding boxes
                    └──────┬──────┘
                           │ crop each box
                    ┌──────▼──────┐
                    │textline_ori  │  Text line orientation (per crop)
                    │  (LCNet)    │  → 0° / 180°
                    └──────┬──────┘
                           │ rotate if needed
                    ┌──────▼──────┐
                    │     rec      │  Text recognition (CTC)
                    │  (PP-OCRv5)  │  → text + confidence
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Results    │
                    │ [{text, conf,│
                    │  bbox}, ...] │
                    └─────────────┘
```

## Model Details

### 1. doc_ori — Document Orientation Classification

| Property | Value |
|----------|-------|
| Model | PP-LCNet_x1_0 |
| ONNX file | `PP-LCNet_x1_0_doc_ori_onnx/inference.onnx` |
| Input | 1x3x224x224 (RGB, ImageNet normalized) |
| Output | 4 classes: [0°, 90°, 180°, 270°] |
| Preprocessing | BGR→RGB → ResizeByShort(256, `round()`) → CenterCrop(224) → ImageNet normalize |

### 2. det — Text Detection (DB)

| Property | Value |
|----------|-------|
| Model | PP-OCRv5 Server Det |
| ONNX file | `PP-OCRv5_server_det_onnx/inference.onnx` |
| Input | 1x3xHxW (BGR, ImageNet normalized, stride=32 aligned) |
| Output | Probability map [1,1,H,W] |
| Preprocessing | Resize(`limit_type=min, limit_side_len=64`, `stride=32`) → ImageNet normalize |
| Postprocessing | DB: threshold=0.3, box_threshold=0.6, unclip_ratio=1.5 |

**Important**: The Pipeline runtime overrides `inference.yml` defaults. The actual resize parameters are `limit_type="min", limit_side_len=64` (not `resize_long=960`). This means large images are processed at original resolution.

### 3. textline_ori — Text Line Orientation Classification

| Property | Value |
|----------|-------|
| Model | PP-LCNet_x1_0 |
| ONNX file | `PP-LCNet_x1_0_textline_ori_onnx/inference.onnx` |
| Input | 1x3x80x160 (RGB, ImageNet normalized) |
| Output | 2 classes: [0°, 180°] |
| Preprocessing | BGR→RGB → Resize(160x80, no aspect ratio) → ImageNet normalize |

### 4. rec — Text Recognition (CTC)

| Property | Value |
|----------|-------|
| Model | PP-OCRv5 Server Rec |
| ONNX file | `PP-OCRv5_server_rec_onnx/inference.onnx` |
| Input | 1x3x48xW (BGR, (pixel/255 - 0.5) / 0.5 normalized) |
| Output | Logits [1, seq_len, 18385] |
| Dictionary | 18383 characters + blank + space = 18385 entries |
| Preprocessing | Resize to height=48, pad to batch max width |
| Postprocessing | CTC greedy decode, raw logit max for confidence |

**Critical**: Recognition is batched with `batch_size=6`. Crops are sorted by width/height ratio, split into batches, and each batch is padded to the batch-internal max width. This padding width affects model output — it is part of the accuracy contract.

## Preprocessing Parameters

### ImageNet Normalization (det, doc_ori, textline_ori)

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
scale = 1.0 / 255.0
# Applied as: pixel * (scale / std) + (-mean / std)
```

### Recognition Normalization

```python
# Simpler: (pixel / 255 - 0.5) / 0.5 → range [-1, 1]
```

## Text Region Cropping

The cropping pipeline preserves sub-pixel precision:

```
int16 bounding box
    → cv2.minAreaRect(int32)
    → cv2.boxPoints(float32)     ← sub-pixel coordinates
    → sort by x → assign TL/TR/BR/BL
    → cv2.getPerspectiveTransform
    → cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE)
```

Using int16 coordinates directly would cause 0.5-1.0px offset, leading to character recognition errors at region boundaries.

## Box Sorting

Text boxes are sorted in reading order:
1. Primary: top-to-bottom (sort by top-left Y coordinate)
2. Secondary: left-to-right (if Y difference < 10px, sort by X)

## Character Dictionary

The character set is constructed as:
```python
character = ["blank"] + dict_chars + [" "]
# "blank" at index 0 (CTC blank token)
# 18383 characters from ppocrv5_dict.txt
# space " " at the end (use_space_char=True)
# Total: 18385 entries
```

CTC decode uses **direct indexing** (`character[idx]`), not `idx-1`.
