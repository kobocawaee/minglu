"""
compare_int8_fp32.py — Sanity check: เทียบ output ของ vision encoder fp32 vs quantized

ป้อนรูปเดียวกันเข้าทั้ง fp32 และ quantized model แล้ววัดว่า image_features ใกล้กันไหม
ใช้เทียบได้ทั้ง INT8 และ BF16 (ส่ง path ของ model ที่จะเทียบเป็น arg ที่ 2)

เกณฑ์:
  cosine > 0.99  = ซื่อสัตย์มาก (ไปต่อ NPU ได้)
  cosine 0.95-99 = พอรับได้ (ดู text ตอน pipeline อีกที)
  cosine < 0.95  = น่ากังวล (quantization ทำ output เพี้ยน)

Usage (env ryzen-ai-1.7.1):
    python code/compare_int8_fp32.py [image_path] [quant_model_path]
    default image = data/sample_indoor.jpg (นอก calibration set)
    default quant = vision_encoder_xint8.onnx
ตัวอย่างเทียบ bf16:
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
    print(f"=== FP32 vs {qname} sanity check ===\nimage: {image_path} (นอก calibration set)\n")
    feed = build_inputs(image_path)
    print(f"input pixel_values: {feed['pixel_values'].shape}\n")

    print("running fp32 model...")
    f32, t32 = run_model(FP32_PATH, feed)
    print(f"running quant model ({qname})...")
    q, tq = run_model(quant_path, feed)

    print(f"\nfp32  output: shape={f32.shape}  time={t32:.2f}s")
    print(f"quant output: shape={q.shape}  time={tq:.2f}s")

    if f32.shape != q.shape:
        print("\n[!] shape ไม่ตรงกัน — เทียบไม่ได้")
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
        print("  [FAIL] quant มี NaN/Inf — quantization พัง")
    elif cos > 0.99:
        print("  [PASS] cosine > 0.99 — quant ซื่อสัตย์มาก ไปต่อ NPU ได้เลย")
    elif cos > 0.95:
        print("  [OK]   cosine 0.95-0.99 — พอรับได้ แต่ควรเช็ก text ตอน pipeline จริง")
    else:
        print("  [WARN] cosine < 0.95 — เพี้ยนเยอะ quantization config ยังไม่เหมาะ")
    print("\nหมายเหตุ: นี่เทียบ embedding. การพิสูจน์สุดท้ายคือเทียบ 'ข้อความ' ที่ decoder")
    print("ผลิตจาก quant vs fp32 (ทำตอน pipeline ขั้นถัดไป)")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_indoor.jpg"
    quant_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUANT
    main(image_path, quant_path)
