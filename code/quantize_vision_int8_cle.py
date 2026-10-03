"""
quantize_vision_int8_cle.py - INT8 re-quantize แบบแก้ถูก (รอบ 2)

แก้ 2 จุดที่ XINT8_QCONFIG รอบแรกตั้งผิด (ทำให้ cosine -0.52):
  1. enable_npu_transformer=True  (รอบแรกใช้ enable_npu_cnn=True ผิดโหมด - model เป็น transformer)
  2. include_cle=True             (รอบแรก False - CLE แก้ outlier/activation range กว้าง)
+ calibrate_method = PowerOfTwoMethod.NonOverflow (power-of-2 scale สำหรับ NPU + เร็ว ~10 นาที
  แทน MinMSE 3.5 ชม. - เอาไว้ทดสอบสมมติฐานก่อน ถ้า cosine ดีค่อย refine ด้วย MinMSE)

INT8 = standard QDQ ops -> โหลดได้บน ORT ธรรมดา (ไม่ต้อง custom_ops.dll แบบ BF16) + เล็ก 4x

Usage (env ryzen-ai-1.7.1):
    python code/quantize_vision_int8_cle.py
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
from quark.onnx import quantize_static, CalibrationDataReader, PowerOfTwoMethod, QuantType

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
ONNX_IN = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"
ONNX_OUT = "models/smolvlm256m_onnx/onnx/vision_encoder_int8v3.onnx"

CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
]


class VisionCalibReader(CalibrationDataReader):
    """ป้อนทีละ 1 tile กัน OOM (vision encoder ไม่มี cross-tile attention)"""

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
    print("=== INT8 re-quantize v2 (transformer mode + CLE + NonOverflow) ===\n")
    reader = VisionCalibReader(CALIB_IMAGES)
    print(f"calibration samples: {len(reader.samples)}\n")

    print("quantizing INT8 (NonOverflow, เร็ว)...")
    t0 = time.time()
    quantize_static(
        ONNX_IN, ONNX_OUT, reader,
        calibrate_method=PowerOfTwoMethod.NonOverflow,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        enable_npu_transformer=True,
        include_cle=True,
        per_channel=True,   # v3: แต่ละ channel มี scale ของตัวเอง แก้ outlier channels
    )
    dt = time.time() - t0

    size_in = os.path.getsize(ONNX_IN) / 1e6
    size_out = os.path.getsize(ONNX_OUT) / 1e6 if os.path.exists(ONNX_OUT) else 0
    print(f"\nDONE in {dt:.1f}s")
    print(f"  fp32   : {size_in:.1f} MB")
    print(f"  int8v2 : {size_out:.1f} MB -> {ONNX_OUT}")


if __name__ == "__main__":
    main()
