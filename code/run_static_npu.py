"""
run_static_npu.py - Stage 4c (retry): run STATIC vision encoder INT8 on NPU (VitisAI EP)

static 模型已經沒有 NonZero -> 看看 VitisAI 編譯器這次會不會當掉（和上一輪不同）
只測一個 tile [1,3,512,512] 就夠了（只看能否編譯＋準確度）

Usage (env ryzen-ai-1.7.1):
    python code/run_static_npu.py [image_path]
"""
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("RYZEN_AI_INSTALLATION_PATH", r"C:\Program Files\RyzenAI\1.7.1")

import numpy as np
from PIL import Image
import onnxruntime as ort
from transformers import AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
FP32_PATH = "models/smolvlm256m_onnx/onnx/vision_static.onnx"
INT8_PATH = "models/smolvlm256m_onnx/onnx/vision_static_int8.onnx"


def first_tile(image_path):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="np")
    pv = inputs["pixel_values"].astype(np.float32)  # [1, num_tiles, 3, 512, 512]
    return pv[0, 0:1]  # 第一個 tile [1,3,512,512]


def main(image_path):
    print("=== Stage 4c retry: STATIC vision encoder INT8 on NPU ===")
    print(f"RYZEN_AI_INSTALLATION_PATH = {os.environ.get('RYZEN_AI_INSTALLATION_PATH')}\n")
    feed = {"pixel_values": first_tile(image_path)}
    print(f"input: {feed['pixel_values'].shape}\n")

    print("running fp32 static (CPU)...")
    f32 = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"]).run(None, feed)[0]

    print("creating VitisAI EP session for INT8 static (NPU)... [compile may take minutes]")
    so = ort.SessionOptions()
    so.log_severity_level = 1
    t0 = time.time()
    npu = ort.InferenceSession(INT8_PATH, sess_options=so,
                               providers=["VitisAIExecutionProvider", "CPUExecutionProvider"])
    print(f"*** SESSION CREATED in {time.time()-t0:.1f}s (= 沒有當掉!) ***")
    print(f"providers ACTIVE: {npu.get_providers()}")

    npu.run(None, feed)  # warm-up
    t1 = time.time()
    q = npu.run(None, feed)[0]
    npu_time = time.time() - t1

    a, b = f32.reshape(-1).astype(np.float64), q.reshape(-1).astype(np.float64)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"\noutput shape: {q.shape}  npu_time={npu_time:.3f}s  NaN={np.isnan(q).any()}")
    print(f"cosine vs fp32: {c:.5f}")
    print("\n--- VERDICT ---")
    if "VitisAIExecutionProvider" in npu.get_providers():
        print("  [WIN] static INT8 視覺編碼器可以在 VitisAI/NPU 上載入並執行（沒有當掉）!")
        print(f"        accuracy cosine={c:.3f}")
    else:
        print("  [!] VitisAI 沒有啟用 - 退回 CPU")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_indoor.jpg"
    main(image_path)
