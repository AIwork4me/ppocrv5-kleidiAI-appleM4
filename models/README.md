# Models

ONNX models are **not included** in this repository due to their size (~180 MB total).

## Download

Download from Baidu Pan:

- **Link**: https://pan.baidu.com/s/1-t7U07_kDgEcy7HdJe9-VQ?pwd=uepw
- **Password**: uepw

## Required Models (ONNX)

After downloading, place the ONNX model directories here:

```
models/
├── PP-OCRv5_server_det_onnx/
│   └── inference.onnx          (~84 MB)
├── PP-OCRv5_server_rec_onnx/
│   └── inference.onnx          (~81 MB)
├── PP-LCNet_x1_0_doc_ori_onnx/
│   └── inference.onnx          (~6.5 MB)
└── PP-LCNet_x1_0_textline_ori_onnx/
    └── inference.onnx          (~6.5 MB)
```

## Optional Models (Paddle, for benchmark only)

If you want to run `benchmarks/benchmark_unified.py --backend paddle`, also place:

```
models/
├── PP-OCRv5_server_det_infer/
│   ├── inference.json
│   └── inference.pdiparams
├── PP-OCRv5_server_rec_infer/
│   ├── inference.json
│   └── inference.pdiparams
├── PP-LCNet_x1_0_doc_ori_infer/
│   ├── inference.json
│   └── inference.pdiparams
└── PP-LCNet_x1_0_textline_ori_infer/
    ├── inference.json
    └── inference.pdiparams
```

## Verify

```bash
python scripts/download_models.py
```
