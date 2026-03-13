# Accuracy Alignment: Paddle to ONNX Runtime

## Summary

Through 6 iterations of debugging and alignment, we achieved **100% text match** between PaddleOCR/PaddleX 3.4.x native inference and our ONNX Runtime pipeline across all 4 models:

| Comparison | Text Regions | Text Match | Avg Confidence Diff |
|:----------:|:-----------:|:----------:|:-------------------:|
| Paddle vs ORT | 228 vs 228 | **100.0%** | 0.000019 |

Tested on 7 images including rotated documents, handwritten Chinese/English, Japanese, and pinyin.

## The Journey: 65.6% → 100%

### Round 1: Initial Alignment (65.6%)

Starting from a naive ONNX implementation, 159 vs 163 detected boxes, only 65.6% text match.

Issues found and fixed:
1. Classification models used BGR instead of **RGB** input
2. `box_score_fast` used `int()` truncation instead of `math.floor/ceil`
3. `unclip` used shapely instead of `cv2.contourArea/arcLength`
4. `get_mini_boxes` used sum/diff sorting instead of x-sort + y-assign
5. Coordinate mapping used float ratios instead of `round()` + int16 clamp
6. CTC decode used softmax instead of **raw logits max** for confidence
7. CTC decode used `idx-1` offset instead of **direct indexing** `character[idx]`
8. Crop rotation used `cv2.rotate` instead of `np.rot90` with `>= 1.5` threshold
9. Normalize used vectorized division instead of **alpha/beta pre-computation**

### Round 2: CTC Fix (71.8%)

After fixing CTC decoding (raw logits + direct indexing), confidence difference dropped from 0.976 to 0.011.

### Round 3: Critical Breakthrough — det Resize Parameters (90.8%)

**Root cause**: PaddleOCR Pipeline runtime overrides `inference.yml` defaults:

| Layer | Source | det Resize Parameter | Active? |
|:-----:|--------|---------------------|:-------:|
| 1 | `inference.yml` | `resize_long=960` (type2, stride=128) | Overridden |
| 2 | `OCRPipeline.__init__` | **`limit_type="min", limit_side_len=64`** (type0, stride=32) | **Active** |
| 3 | `ocr.predict()` args | User can override | Not used |

**Lesson**: Never trust config files at face value. Use `pp.get_text_det_params()` to verify actual runtime parameters. PaddleX has a 3-layer parameter override hierarchy, and the Pipeline silently overrides model-level defaults.

### Round 4: Verification (90.8%)

Isolated testing: given identical crop images, Paddle rec and ONNX rec produce identical argmax outputs. This proved **rec models themselves are equivalent** — all remaining differences come from upstream processing.

### Round 5: Crop Coordinate Precision (93.3%)

**Problem**: Paddle uses `minAreaRect → boxPoints → float32` coordinates for cropping, while we used int16 directly.

```
DBPostProcess outputs int16 box (identical both sides)
         │
    ┌────┴────┐
    ▼         ▼
  Paddle    ONNX (before fix)
    │         │
    ▼         ▼
  cv2.minAreaRect(int32)   Direct int16 → float32
    │
    ▼
  cv2.boxPoints → float32 (sub-pixel precision)
```

The 0.5-1.0px difference causes crop pixel differences up to 253, which flips argmax at boundary characters (e.g., "內" vs "内").

**Fix**: Replicate `get_minarea_rect_crop` exactly: int16 → minAreaRect(int32) → boxPoints(float32) → sort → crop.

### Round 6: Final Breakthrough — rec Batch Padding (100.0%)

**The most subtle issue**: Paddle's rec model is **sensitive to padding width**. The same crop padded to different widths produces different text:

```
Same crop (48×120):
  pad to imgW=320  → "口"  (batched with 5 other crops)
  pad to imgW=3200 → "Q"   (processed independently)
```

**Paddle's rec pipeline**:
1. Sort all crops by width/height ratio
2. Split into batches of 6
3. Each batch: pad to batch-internal max width
4. Infer per-sample within batch

**Fix**: Replicate the exact same batch_size=6 sorting + per-batch padding logic.

**Result**: 93.3% → **100.0%**

## Key Insights

### 1. Config File ≠ Runtime Parameters

PaddleOCR has a 3-layer override hierarchy. The `inference.yml` values are silently overridden by Pipeline defaults. Always verify runtime parameters programmatically.

### 2. Methodology: Layer-by-Layer Isolation

```
image → [det preprocess] → [det model] → [det postprocess] → boxes
                                                                ↓
                                                    [crop] → [rec preprocess] → [rec model] → [CTC decode] → text
```

At each layer, feed identical input to both pipelines and compare output. This systematically narrows down the source of differences.

### 3. Zero-Padding is NOT Lossless

Padding width affects the model's global context (attention, conv boundaries), causing argmax flips at boundary characters. **Batch scheduling strategy is part of the pipeline's accuracy contract.**

### 4. Precision Alignment = Exact Translation

Accuracy alignment is not about matching "the big picture" — it's about matching **every floating-point operation order and boundary condition**. Any "equivalent" reimplementation may introduce differences. The safest approach is **line-by-line source code translation**.

### 5. All Fixes (Ranked by Impact)

| Fix | Impact | Description |
|-----|:------:|-------------|
| det resize params | 65.6% → 90.8% | Pipeline runtime uses `limit_type=min, limit_side_len=64` |
| rec batch padding | 93.3% → 100% | Align batch_size=6 sort + per-batch padding width |
| crop coordinates | 90.8% → 93.3% | minAreaRect → boxPoints float32 precision |
| CTC decode | text + confidence | Direct index `character[idx]`, raw logits max |
| DB post-processing | box detection | floor/ceil, cv2 contourArea, x-sort y-assign |
| doc_ori preprocess | orientation | `round()` not `int()`, cv2.split/merge normalize |
| Rotation logic | orientation | warpAffine(INTER_CUBIC) not cv2.rotate |
