"""
quantize_vision_static_int8.py - 把 vision_static[_bN].onnx 量化成 INT8

支援 batch：
    python code/quantize_vision_static_int8.py        # batch 1  -> vision_static_int8.onnx
    python code/quantize_vision_static_int8.py 13      # batch 13 -> vision_static_b13_int8.onnx

設定：enable_npu_transformer + include_cle + NonOverflow（實驗中效果最好）
Env: ryzen-ai-1.7.1
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
# batch 13 用 2 張圖就夠（避免記憶體不足），batch 1 用 3 張圖
CALIB_IMAGES = [
    "data/test_images/crosswalk_car.jpg",
    "data/test_images/Indoor_livingroom.jpg",
    "data/test_images/pngtree-park-corridor-ground-walk-way-photo-image_36618132.jpg",
]


class TileCalibReader(CalibrationDataReader):
    """輸入 [batch,3,512,512]。batch=1 -> 一次一個 tile／batch>1 -> 整疊（補齊或截斷成 batch 大小）"""

    def __init__(self, image_paths, batch):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.batch = batch
        self.samples = []
        imgs = image_paths if batch == 1 else image_paths[:2]  # batch 大時少用幾張圖，避免記憶體不足
        for p in imgs:
            if not os.path.exists(p):
                continue
            pv = self._tiles(p)  # [num_tiles, 3, 512, 512]
            if batch == 1:
                for i in range(pv.shape[0]):
                    self.samples.append({"pixel_values": pv[i:i + 1]})
            else:
                n = pv.shape[0]
                if n >= batch:
                    stack = pv[:batch]
                else:  # 用重複的 tile 補齊 batch
                    pad = np.repeat(pv[-1:], batch - n, axis=0)
                    stack = np.concatenate([pv, pad], axis=0)
                self.samples.append({"pixel_values": stack.astype(np.float32)})
            print(f"  [calib] {os.path.basename(p)} -> {pv.shape[0]} tiles")
        self.idx = 0

    def _tiles(self, path):
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        prompt = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        pv = self.processor(text=prompt, images=[img], return_tensors="np")["pixel_values"].astype(np.float32)
        return pv[0]

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        d = self.samples[self.idx]
        self.idx += 1
        return d

    def rewind(self):
        self.idx = 0


def main(batch):
    onnx_in = "models/smolvlm256m_onnx/onnx/vision_static.onnx" if batch == 1 \
        else f"models/smolvlm256m_onnx/onnx/vision_static_b{batch}.onnx"
    onnx_out = onnx_in.replace(".onnx", "_int8.onnx")
    print(f"=== quantize {os.path.basename(onnx_in)} -> INT8 (batch={batch}) ===\n")
    reader = TileCalibReader(CALIB_IMAGES, batch)
    print(f"calibration samples: {len(reader.samples)}\n")

    t0 = time.time()
    quantize_static(
        onnx_in, onnx_out, reader,
        calibrate_method=PowerOfTwoMethod.NonOverflow,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        enable_npu_transformer=True, include_cle=True,
    )
    size = os.path.getsize(onnx_out) / 1e6 if os.path.exists(onnx_out) else 0
    print(f"\nDONE in {time.time()-t0:.1f}s -> {onnx_out} ({size:.1f} MB)")


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(batch)
