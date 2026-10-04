"""
quantize_vision_int8_cle.py - 修正設定後重新量化成 INT8（第 2 輪）

修正第一輪 XINT8_QCONFIG 設錯的兩個地方（導致 cosine -0.52）：
  1. enable_npu_transformer=True  （第一輪用了 enable_npu_cnn=True，模式錯了 - 模型是 transformer）
  2. include_cle=True             （第一輪是 False - CLE 能處理離群值／activation 範圍過廣）
+ calibrate_method = PowerOfTwoMethod.NonOverflow（給 NPU 的 2 的次方縮放＋快，約 10 分鐘
  取代 MinMSE 的 3.5 小時 - 先驗證假設，cosine 好的話再用 MinMSE 精修）

INT8 = 標準 QDQ 運算 -> 一般的 ORT 就能載入（不需要像 BF16 那樣的 custom_ops.dll）＋檔案小 4 倍

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
    """一次餵一個 tile 避免記憶體不足（視覺編碼器沒有跨 tile 的 attention）"""

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

    print("quantizing INT8 (NonOverflow, 快速)...")
    t0 = time.time()
    quantize_static(
        ONNX_IN, ONNX_OUT, reader,
        calibrate_method=PowerOfTwoMethod.NonOverflow,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        enable_npu_transformer=True,
        include_cle=True,
        per_channel=True,   # v3：每個 channel 有自己的縮放，處理離群的 channel
    )
    dt = time.time() - t0

    size_in = os.path.getsize(ONNX_IN) / 1e6
    size_out = os.path.getsize(ONNX_OUT) / 1e6 if os.path.exists(ONNX_OUT) else 0
    print(f"\nDONE in {dt:.1f}s")
    print(f"  fp32   : {size_in:.1f} MB")
    print(f"  int8v2 : {size_out:.1f} MB -> {ONNX_OUT}")


if __name__ == "__main__":
    main()
