"""
bench_b13_npu.py - 比較 NPU 延遲：batch-13（呼叫 1 次）vs 逐 tile（呼叫 13 次）

原本逐 tile = 每張圖 13.48 秒（每次呼叫的額外開銷 x13）。batch-13 = 一次呼叫處理全部 13 個 tile
看額外開銷能省掉多少

Usage (env ryzen-ai-1.7.1):
    python code/bench_b13_npu.py [image_path]
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
B1 = "models/smolvlm256m_onnx/onnx/vision_static_int8.onnx"
B13 = "models/smolvlm256m_onnx/onnx/vision_static_b13_int8.onnx"
EPS = ["VitisAIExecutionProvider", "CPUExecutionProvider"]


def tiles(image_path):
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    return proc(text=prompt, images=[img], return_tensors="np")["pixel_values"].astype(np.float32)[0]


def main(image_path):
    print("=== NPU latency: batch-13 (1 call) vs per-tile (13 calls) ===\n")
    tl = tiles(image_path)
    n = tl.shape[0]
    print(f"tiles: {n}\n")

    # per-tile (batch 1) x13
    print("per-tile NPU session (compile/cache)...")
    s1 = ort.InferenceSession(B1, providers=EPS)
    s1.run(None, {"pixel_values": tl[0:1]})  # warm
    t = time.time()
    for i in range(n):
        s1.run(None, {"pixel_values": tl[i:i + 1]})
    t_pertile = time.time() - t
    print(f"  per-tile: {t_pertile:.2f}s ({n} calls)")

    # batch-13 (1 call)
    print("\nbatch-13 NPU session (compile/cache)...")
    s13 = ort.InferenceSession(B13, providers=EPS)
    stack = tl[:13] if n >= 13 else np.concatenate([tl, np.repeat(tl[-1:], 13 - n, 0)], 0)
    stack = stack.astype(np.float32)
    s13.run(None, {"pixel_values": stack})  # warm
    t = time.time()
    s13.run(None, {"pixel_values": stack})
    t_batch = time.time() - t
    print(f"  batch-13: {t_batch:.2f}s (1 call)")

    print(f"\n--- RESULT ---")
    print(f"  per-tile (13 calls): {t_pertile:.2f}s")
    print(f"  batch-13 (1 call)  : {t_batch:.2f}s")
    if t_batch > 0:
        print(f"  batching 加速  : {t_pertile/t_batch:.1f}x")
    print(f"\n  對照原本 iGPU（每張預填約 0.75 秒）| 原本 CPU（約 1.3 秒）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/test_images/crosswalk_car.jpg")
