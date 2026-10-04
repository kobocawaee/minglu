"""
compare_int8_fp32.py — 健全性檢查：比較視覺編碼器 fp32 vs 量化後的輸出

同一張圖分別餵進 fp32 和量化模型，量測 image_features 有多接近
INT8 和 BF16 都能比（第 2 個參數傳要比較的模型路徑）

標準：
  cosine > 0.99  = 非常忠實（可以繼續上 NPU）
  cosine 0.95-99 = 可以接受（在完整流程裡再看文字輸出）
  cosine < 0.95  = 令人擔心（量化讓輸出走樣）

Usage (env ryzen-ai-1.7.1):
    python code/compare_int8_fp32.py [image_path] [quant_model_path]
    預設圖片 = data/sample_indoor.jpg（不在校準集中）
    default quant = vision_encoder_xint8.onnx
比較 bf16 的範例：
    python code/compare_int8_fp32.py data/sample_indoor.jpg models/smolvlm256m_onnx/onnx/vision_encoder_bf16.onnx
"""
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
FP32_PATH = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"
DEFAULT_QUANT = "models/smolvlm256m_onnx/onnx/vision_encoder_xint8.onnx"


def build_inputs(image_path):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="np")
    pv = inputs["pixel_values"].astype(np.float32)
    if "pixel_attention_mask" in inputs:
        pam = inputs["pixel_attention_mask"].astype(bool)
    else:
        b, n, h, w = pv.shape[0], pv.shape[1], pv.shape[-2], pv.shape[-1]
        pam = np.ones((b, n, h, w), dtype=bool)
    return {"pixel_values": pv, "pixel_attention_mask": pam}


def run_model(path, feed):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t0 = time.time()
    out = sess.run(None, feed)[0]
    return out, time.time() - t0


def main(image_path, quant_path):
    qname = os.path.basename(quant_path)
    print(f"=== FP32 vs {qname} sanity check ===\nimage: {image_path} (不在校準集中)\n")
    feed = build_inputs(image_path)
    print(f"input pixel_values: {feed['pixel_values'].shape}\n")

    print("running fp32 model...")
    f32, t32 = run_model(FP32_PATH, feed)
    print(f"running quant model ({qname})...")
    q, tq = run_model(quant_path, feed)

    print(f"\nfp32  output: shape={f32.shape}  time={t32:.2f}s")
    print(f"quant output: shape={q.shape}  time={tq:.2f}s")

    if f32.shape != q.shape:
        print("\n[!] shape 不一致 — 無法比較")
        return

    print(f"\nquant NaN? {np.isnan(q).any()}   Inf? {np.isinf(q).any()}")
    print(f"\nvalue range  fp32 : [{f32.min():.2f}, {f32.max():.2f}] mean={f32.mean():.3f}")
    print(f"value range  quant: [{q.min():.2f}, {q.max():.2f}] mean={q.mean():.3f}")

    a, b = f32.reshape(-1).astype(np.float64), q.reshape(-1).astype(np.float64)
    max_abs = np.abs(a - b).max()
    mean_abs = np.abs(a - b).mean()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    rel_l2 = float(np.linalg.norm(a - b) / np.linalg.norm(a))

    print(f"\n--- DIFFERENCE METRICS ---")
    print(f"  max abs diff   : {max_abs:.4f}")
    print(f"  mean abs diff  : {mean_abs:.4f}")
    print(f"  cosine sim     : {cos:.5f}")
    print(f"  relative L2 err: {rel_l2:.4f}")

    print(f"\n--- VERDICT ---")
    if np.isnan(q).any() or np.isinf(q).any():
        print("  [FAIL] 量化結果有 NaN/Inf — 量化壞掉了")
    elif cos > 0.99:
        print("  [PASS] cosine > 0.99 — 量化非常忠實，可以直接上 NPU")
    elif cos > 0.95:
        print("  [OK]   cosine 0.95-0.99 — 可以接受，但應在實際流程中檢查文字輸出")
    else:
        print("  [WARN] cosine < 0.95 — 偏差很大，量化設定還不合適")
    print("\n注意：這裡比較的是 embedding。最終證明是比較 decoder 產生的「文字」")
    print("（量化版 vs fp32，在下一階段的流程中進行）")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_indoor.jpg"
    quant_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUANT
    main(image_path, quant_path)
