"""
quantize_vision_xint8.py — 用 Quark 把 SmolVLM 視覺編碼器量化成 INT8（XINT8）

XINT8 = 對稱 INT8 + 2 的次方縮放，是 AMD 專為 Ryzen AI NPU 設計的模式
（VitisAI EP 會把這個 INT8 模型放到 NPU 上執行）

flow:
  vision_encoder.onnx (fp32, 374MB)
    --[Quark XINT8 + 用真實圖片校準]-->
  vision_encoder_xint8.onnx（INT8，約小 4 倍）

Usage（必須在 ryzen-ai-1.7.1 環境中）：
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

# 校準：選 5 張圖涵蓋 3 種使用情境（街道／室內／戶外步道）
CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/crosswalk_greenlight.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
    "data/test_images/istockphoto-157374644-612x612.jpg",
]


class VisionCalibReader(CalibrationDataReader):
    """一次餵一個 tile 的 pixel_values + pixel_attention_mask（避免記憶體不足）

    視覺編碼器各個 tile 獨立編碼（沒有跨 tile 的 attention），一次餵一個 tile
    得到的校準值和整批一起餵相同，但每個樣本的記憶體用量少約 13 倍。
    （batch=13 會讓 softmax attention 張量 = 13*12*1024*1024*4B = 每層 654MB，累積到記憶體不足）
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

    print("quantizing (XINT8)... [要跑模型好幾次來校準，可能需要好幾分鐘]")
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
