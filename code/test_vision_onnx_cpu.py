"""
test_vision_onnx_cpu.py — ทดสอบ vision_encoder.onnx (SmolVLM-256M) บน CPU

เป้าหมาย:
  1. ยืนยันว่า ONNX vision encoder ที่โหลดมา (onnx-community) รันได้จริง
  2. ดู shape ของ pixel_values ที่ processor สร้าง (ต้องตรงกับ ONNX input 512x512)
  3. ได้โค้ด preprocessing (รูป -> pixel_values) ที่จะ reuse เป็น calibration data ตอน Quark quantize

Usage:
    python code/test_vision_onnx_cpu.py [image_path]
    (default image = data/sample_indoor.jpg)

Env: vlm_research (transformers + onnxruntime CPU)
"""
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
ONNX_PATH = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"


def build_inputs(image_path):
    """ใช้ AutoProcessor ของ SmolVLM แปลงรูป -> pixel_values + pixel_attention_mask"""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")

    # สร้าง chat message มาตรฐานของ SmolVLM (มี <image> placeholder)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image."},
        ],
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="np")

    pv = inputs["pixel_values"].astype(np.float32)
    # pixel_attention_mask อาจไม่มี ถ้าไม่มี padding -> สร้าง all-ones
    if "pixel_attention_mask" in inputs:
        pam = inputs["pixel_attention_mask"].astype(bool)
    else:
        b, n = pv.shape[0], pv.shape[1]
        h, w = pv.shape[-2], pv.shape[-1]
        pam = np.ones((b, n, h, w), dtype=bool)

    return pv, pam


def main(image_path):
    print(f"=== Vision encoder ONNX test (CPU) ===")
    print(f"image: {image_path}\n")

    pv, pam = build_inputs(image_path)
    print(f"pixel_values        shape={pv.shape} dtype={pv.dtype}")
    print(f"pixel_attention_mask shape={pam.shape} dtype={pam.dtype}")
    print(f"  -> num_images (tiles) = {pv.shape[1]}, tile size = {pv.shape[-2]}x{pv.shape[-1]}\n")

    print("loading ONNX session (CPUExecutionProvider)...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    feed = {"pixel_values": pv, "pixel_attention_mask": pam}
    t0 = time.time()
    out = sess.run(None, feed)[0]
    dt = time.time() - t0

    print(f"\nimage_features  shape={out.shape} dtype={out.dtype}")
    print(f"  stats: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
    print(f"  has NaN? {np.isnan(out).any()}  has Inf? {np.isinf(out).any()}")
    print(f"\ninference time (CPU): {dt:.2f}s")
    print("\nOK - ONNX vision encoder works on CPU." if not np.isnan(out).any()
          else "\nWARNING - output has NaN!")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_indoor.jpg"
    main(image_path)
