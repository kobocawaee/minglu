"""
quantize_vision_xint8.py — Quark INT8 (XINT8) quantize ของ SmolVLM vision encoder

XINT8 = symmetric INT8 + power-of-two scales, โหมดที่ AMD ทำมาเพื่อ Ryzen AI NPU โดยเฉพาะ
(VitisAI EP รับ INT8 model นี้ไปวางบน NPU)

flow:
  vision_encoder.onnx (fp32, 374MB)
    --[Quark XINT8 + calibration รูปจริง]-->
  vision_encoder_xint8.onnx (INT8, เล็กลง ~4x)

Usage (ต้องอยู่ใน env ryzen-ai-1.7.1):
    python code/quantize_vision_xint8.py

Env: ryzen-ai-1.7.1 (quark 0.11rc1)
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

from quark.onnx import ModelQuantizer, XINT8_QCONFIG, CalibrationDataReader

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ONNX_IN = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"
ONNX_OUT = "models/smolvlm256m_onnx/onnx/vision_encoder_xint8.onnx"

# calibration: เลือก 5 รูปคลุม 3 use case (street / indoor / outdoor walkway)
CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/crosswalk_greenlight.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
    "data/test_images/istockphoto-157374644-612x612.jpg",
]


class VisionCalibReader(CalibrationDataReader):
    """ป้อน pixel_values + pixel_attention_mask ทีละ 1 tile (กัน OOM)

    vision encoder เข้ารหัสแต่ละ tile อิสระกัน (ไม่มี cross-tile attention) ป้อนทีละ tile
    ให้ค่า calibration เหมือนป้อนทั้ง batch พร้อมกัน แต่หน่วยความจำต่อ sample ลดลง ~13x.
    (batch=13 ทำให้ softmax attention tensor = 13*12*1024*1024*4B = 654MB/layer สะสมจน OOM)
    """

    def __init__(self, image_paths):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.samples = []
        for p in image_paths:
            if not os.path.exists(p):
                print(f"  [skip] not found: {p}")
                continue
            pv, pam = self._prep(p)
            n_tiles = pv.shape[1]
            for i in range(n_tiles):
                self.samples.append({
                    "pixel_values": pv[:, i:i + 1],
                    "pixel_attention_mask": pam[:, i:i + 1],
                })
            print(f"  [calib] {os.path.basename(p)} -> {n_tiles} tiles (flattened)")
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
    print("=== Quark XINT8 quantization: SmolVLM vision encoder ===\n")
    print("preparing calibration data...")
    reader = VisionCalibReader(CALIB_IMAGES)
    print(f"\ncalibration samples: {len(reader.samples)}\n")

    print("quantizing (XINT8)... [จะรันโมเดลหลายรอบเพื่อ calibrate อาจใช้เวลาหลายนาที]")
    t0 = time.time()
    quantizer = ModelQuantizer(XINT8_QCONFIG)
    quantizer.quantize_model(ONNX_IN, ONNX_OUT, reader)
    dt = time.time() - t0

    size_in = os.path.getsize(ONNX_IN) / 1e6
    size_out = os.path.getsize(ONNX_OUT) / 1e6 if os.path.exists(ONNX_OUT) else 0
    print(f"\nDONE in {dt:.1f}s")
    print(f"  fp32 : {size_in:.1f} MB -> {ONNX_IN}")
    print(f"  int8 : {size_out:.1f} MB -> {ONNX_OUT}")


if __name__ == "__main__":
    main()
