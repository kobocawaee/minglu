"""
quantize_vision_bf16.py — 用 Quark 把 SmolVLM 視覺編碼器量化成 BF16

為什麼用 BF16：XINT8（逐張量、2 的次方縮放的 INT8）用在 SigLIP 會壞掉（cosine -0.52），因為 activation
範圍很廣又有離群值。BF16 是 16 位元浮點數（保留完整指數）→ 保真度高很多
cosine 應該接近 1.0。而且 BF16 = 直接轉型，不需要 MinMSE 校準搜尋 → 比 INT8 快很多

VitisAI EP 同時支援 INT8 和 BF16（BF16 要在部署到 NPU 時預先編譯）

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

# BF16 用 MinMax（快）＋不依賴精確範圍 → 3 張圖就夠
CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
]


class VisionCalibReader(CalibrationDataReader):
    """一次餵一個 tile 避免記憶體不足（和 xint8 一樣）— 視覺編碼器沒有跨 tile 的 attention"""

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
    print("preparing calibration data (MinMax, 快速)...")
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
