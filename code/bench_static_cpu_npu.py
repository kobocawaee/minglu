"""
bench_static_cpu_npu.py - วัด latency static INT8 vision encoder: CPU vs NPU (ทุก tile ของ 1 รูป)

Usage (env ryzen-ai-1.7.1):
    python code/bench_static_cpu_npu.py [image_path]
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
INT8 = "models/smolvlm256m_onnx/onnx/vision_static_int8.onnx"


def tiles(image_path):
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    pv = proc(text=prompt, images=[img], return_tensors="np")["pixel_values"].astype(np.float32)
    return pv[0]  # [num_tiles, 3, 512, 512]


def bench(sess, tlist, label):
    sess.run(None, {"pixel_values": tlist[0:1]})  # warm-up
    t0 = time.time()
    for i in range(tlist.shape[0]):
        sess.run(None, {"pixel_values": tlist[i:i + 1]})
    dt = time.time() - t0
    print(f"  {label:8s}: {dt:.2f}s / {tlist.shape[0]} tiles = {dt/tlist.shape[0]*1000:.0f} ms/tile")
    return dt


def main(image_path):
    print(f"=== static INT8 vision encoder: CPU vs NPU latency ===\nimage: {image_path}\n")
    tl = tiles(image_path)
    print(f"tiles: {tl.shape[0]} x [1,3,512,512]\n")

    print("CPU session...")
    cpu = ort.InferenceSession(INT8, providers=["CPUExecutionProvider"])
    t_cpu = bench(cpu, tl, "CPU")

    print("\nNPU session (VitisAI, compile/cache)...")
    npu = ort.InferenceSession(INT8, providers=["VitisAIExecutionProvider", "CPUExecutionProvider"])
    t_npu = bench(npu, tl, "NPU")

    print(f"\n--- RESULT (1 image, {tl.shape[0]} tiles) ---")
    print(f"  CPU total: {t_cpu:.2f}s")
    print(f"  NPU total: {t_npu:.2f}s")
    print(f"  speedup (CPU/NPU): {t_cpu/t_npu:.2f}x  {'(NPU เร็วกว่า)' if t_npu < t_cpu else '(CPU เร็วกว่า!)'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/test_images/crosswalk_car.jpg")
