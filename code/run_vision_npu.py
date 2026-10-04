"""
run_vision_npu.py - Stage 4c: run quantized vision encoder on AMD NPU (VitisAI EP)

回答主要問題：VitisAI EP 能不能接受這個模型＋運算是否真的落在 NPU（＋和 fp32 比較準確度）
INT8（標準 QDQ）可以在 VitisAI 上載入（不像 BF16 卡在自訂運算的 dll）

Usage (env ryzen-ai-1.7.1):
    python code/run_vision_npu.py [image_path] [quant_model_path]
    default quant = vision_encoder_int8v3.onnx
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
FP32_PATH = "models/smolvlm256m_onnx/onnx/vision_encoder.onnx"
DEFAULT_QUANT = "models/smolvlm256m_onnx/onnx/vision_encoder_int8v3.onnx"


def build_inputs(image_path):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="np")
    pv = inputs["pixel_values"].astype(np.float32)
    if "pixel_attention_mask" in inputs:
        pam = inputs["pixel_attention_mask"].astype(bool)
    else:
        b, n, h, w = pv.shape[0], pv.shape[1], pv.shape[-2], pv.shape[-1]
        pam = np.ones((b, n, h, w), dtype=bool)
    return {"pixel_values": pv, "pixel_attention_mask": pam}


def main(image_path, quant_path):
    qname = os.path.basename(quant_path)
    print("=== Stage 4c: quantized vision encoder on NPU (VitisAI EP) ===")
    print(f"model: {qname}")
    print(f"RYZEN_AI_INSTALLATION_PATH = {os.environ.get('RYZEN_AI_INSTALLATION_PATH')}")
    print(f"image: {image_path}\n")

    feed = build_inputs(image_path)
    print(f"input pixel_values: {feed['pixel_values'].shape}\n")

    print("running fp32 reference (CPU)...")
    cpu_sess = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
    f32 = cpu_sess.run(None, feed)[0]

    print("creating VitisAI EP session (NPU)... [first compile may take minutes]")
    so = ort.SessionOptions()
    so.log_severity_level = 1  # INFO -> 看得到切分到 NPU 的紀錄
    t_create = time.time()
    npu_sess = ort.InferenceSession(
        quant_path, sess_options=so,
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"session created in {time.time()-t_create:.1f}s")
    print(f"providers ACTIVE: {npu_sess.get_providers()}")

    npu_sess.run(None, feed)  # warm-up
    t0 = time.time()
    q = npu_sess.run(None, feed)[0]
    npu_time = time.time() - t0

    print(f"\nfp32  output: shape={f32.shape}")
    print(f"quant output: shape={q.shape}  npu_run_time={npu_time:.2f}s")

    if f32.shape != q.shape:
        print("\n[!] shape mismatch - cannot compare")
        return

    print(f"\nquant(NPU) NaN? {np.isnan(q).any()}  Inf? {np.isinf(q).any()}")
    a, b = f32.reshape(-1).astype(np.float64), q.reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"cosine vs fp32 : {cos:.5f}")

    print("\n--- VERDICT ---")
    if "VitisAIExecutionProvider" not in npu_sess.get_providers():
        print("  [!] VitisAI EP not active - ran on CPU fallback only")
    else:
        print("  [OK] VitisAI EP active - model loaded on NPU stack")
        print(f"       accuracy vs fp32: cosine={cos:.3f} (refine later)")
        print("  看上方紀錄找 'VitisAI'／分配到的節點數，就知道有多少比例的運算落在 NPU")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_indoor.jpg"
    quant_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUANT
    main(image_path, quant_path)
