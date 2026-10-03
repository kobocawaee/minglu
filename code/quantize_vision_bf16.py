"""
quantize_vision_bf16.py — Quark BF16 quantize ของ SmolVLM vision encoder

ทำไม BF16: XINT8 (per-tensor INT8 power-of-2) พังกับ SigLIP (cosine -0.52) เพราะ activation
range กว้าง + outlier. BF16 เป็น float 16-bit (เก็บ exponent เต็ม) → fidelity สูงกว่ามาก
ควรได้ cosine ~1.0. และ BF16 = cast ตรงๆ ไม่ต้อง MinMSE calibration search → เร็วกว่า INT8 มาก

VitisAI EP รับทั้ง INT8 และ BF16 (BF16 ต้อง pre-compile ตอน deploy บน NPU)

Usage (env ryzen-ai-1.7.1):
    python code/quantize_vision_bf16.py
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
from transformers import AutoProcessor
from quark.onnx import ModelQuantizer, BF16_QCONFIG, CalibrationDataReader

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ONNX_IN = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"
ONNX_OUT = "models/smolvlm256m_onnx/onnx/vision_encoder_bf16.onnx"

# BF16 ใช้ MinMax (เร็ว) + ไม่พึ่ง range เป๊ะ → ใช้แค่ 3 รูปพอ
CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
]


class VisionCalibReader(CalibrationDataReader):
    """ป้อนทีละ 1 tile กัน OOM (เหมือน xint8) — vision encoder ไม่มี cross-tile attention"""

    def __init__(self, image_paths):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.samples = []
        for p in image_paths:
            if not os.path.exists(p):
                continue
            pv, pam = self._prep(p)
            for i in range(pv.shape[1]):
                self.samples.append({
                    "pixel_values": pv[:, i:i + 1],
                    "pixel_attention_mask": pam[:, i:i + 1],
                })
            print(f"  [calib] {os.path.basename(p)} -> {pv.shape[1]} tiles")
        self.idx = 0

    def _prep(self, path):
        img = Image.open(path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[img], return_tensors="np")
        pv = inputs["pixel_values"].astype(np.float32)
        if "pixel_attention_mask" in inputs:
            pam = inputs["pixel_attention_mask"].astype(bool)
        else:
            b, n, h, w = pv.shape[0], pv.shape[1], pv.shape[-2], pv.shape[-1]
            pam = np.ones((b, n, h, w), dtype=bool)
        return pv, pam

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        d = self.samples[self.idx]
        self.idx += 1
        return d

    def rewind(self):
        self.idx = 0


def main():
    print("=== Quark BF16 quantization: SmolVLM vision encoder ===\n")
    print("preparing calibration data (MinMax, เร็ว)...")
    reader = VisionCalibReader(CALIB_IMAGES)
    print(f"calibration samples: {len(reader.samples)}\n")

    print("quantizing (BF16)...")
    t0 = time.time()
    quantizer = ModelQuantizer(BF16_QCONFIG)
    quantizer.quantize_model(ONNX_IN, ONNX_OUT, reader)
    dt = time.time() - t0

    size_in = os.path.getsize(ONNX_IN) / 1e6
    size_out = os.path.getsize(ONNX_OUT) / 1e6 if os.path.exists(ONNX_OUT) else 0
    print(f"\nDONE in {dt:.1f}s")
    print(f"  fp32 : {size_in:.1f} MB -> {ONNX_IN}")
    print(f"  bf16 : {size_out:.1f} MB -> {ONNX_OUT}")


if __name__ == "__main__":
    main()
