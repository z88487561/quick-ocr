# 第三方元件

- **PP-OCRv5 mobile 偵測／辨識模型與字典**（PaddlePaddle / PaddleOCR，Apache License 2.0），ONNX 轉檔版本取自 npm 套件 `pdfmarkdown-ppocrv5-models`（Apache-2.0，未修改）。
- **ch_ppocr_mobile_v2.0_cls_infer.onnx**（文字方向分類，PaddleOCR，Apache License 2.0），取自 `rapidocr_onnxruntime` 套件（未修改）。
- **ONNX Runtime Web 1.31.1**（`ort.wasm.min.mjs`、`ort.webgpu.min.mjs`、`ort-wasm-simd-threaded.*`（含 asyncify，供 WebGPU 選項使用），Microsoft，MIT License）。
- `pp-ocr.mjs`、`worker.mjs` 為本專案自行撰寫（偵測後處理、文字行裁切、CTC 解碼）。
