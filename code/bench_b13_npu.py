"""
bench_b13_npu.py - เทียบ latency NPU: batch-13 (1 call) vs per-tile (13 calls)

per-tile เดิม = 13.48s/รูป (overhead ต่อ call x13). batch-13 = 1 call ทั้ง 13 tiles
ดูว่า overhead หายไปแค่ไหน

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
        print(f"  batching เร็วขึ้น  : {t_pertile/t_batch:.1f}x")
    print(f"\n  เทียบ iGPU เดิม (~0.75s prefill/รูป) | CPU เดิม (~1.3s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/test_images/crosswalk_car.jpg")
