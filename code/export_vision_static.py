"""
export_vision_static.py - 把 SmolVLM 視覺編碼器匯出成 STATIC shape 的 ONNX

卡關點：Idefics3VisionEmbeddings 依 mask 做 scatter -> NonZero/ScatterND -> NPU 當掉
解法：monkey-patch embeddings 成 static（position_ids = arange）-> 沒有 NonZero

支援 batch 大小（給 NPU batching 最佳化用）：
    python code/export_vision_static.py        # batch 1（一次一個 tile）-> vision_static.onnx
    python code/export_vision_static.py 13      # batch 13（一次呼叫全部）-> vision_static_b13.onnx

Env: vlm_research
"""
import sys
import types
import warnings

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
OPSET = 17


def static_emb_forward(self, pixel_values, patch_attention_mask=None):
    """static：完整 tile -> position_ids = arange(num_patches)（沒有 scatter/mask），支援任何 batch"""
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(2).transpose(1, 2)  # [B, 1024, 768]
    n = embeddings.shape[1]
    position_ids = torch.arange(n, device=embeddings.device).unsqueeze(0).expand(embeddings.shape[0], -1)
    return embeddings + self.position_embedding(position_ids)


class VisionConnector(torch.nn.Module):
    def __init__(self, vision_model, connector):
        super().__init__()
        self.vision_model = vision_model
        self.connector = connector

    def forward(self, pixel_values):
        h = self.vision_model(pixel_values=pixel_values).last_hidden_state
        return self.connector(h)


def cos(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(batch):
    out_path = "models/smolvlm256m_onnx/onnx/vision_static.onnx" if batch == 1 \
        else f"models/smolvlm256m_onnx/onnx/vision_static_b{batch}.onnx"
    print(f"loading SmolVLM (PyTorch)... batch={batch}")
    m = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
    m.eval()
    wrapper = VisionConnector(m.model.vision_model, m.model.connector).eval()

    pv = torch.randn(batch, 3, 512, 512)
    with torch.no_grad():
        ref_orig = wrapper(pv).numpy()

    emb = wrapper.vision_model.embeddings
    emb.forward = types.MethodType(static_emb_forward, emb)
    with torch.no_grad():
        ref_static = wrapper(pv).numpy()
    c = cos(ref_orig, ref_static)
    print(f"patched-static vs original: cosine={c:.6f}")
    if c < 0.99999:
        print("  [STOP] static forward 和原本的結果不一致")
        return
    print("  [OK] static = original\n")

    print(f"exporting ONNX (static, opset {OPSET}) -> {out_path}")
    torch.onnx.export(
        wrapper, (pv,), out_path,
        input_names=["pixel_values"], output_names=["image_features"],
        opset_version=OPSET, dynamic_axes=None, do_constant_folding=True, dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"pixel_values": pv.numpy()})[0]
    print(f"  onnx shape: {onnx_out.shape}  cosine vs PyTorch: {cos(ref_static, onnx_out):.6f}")

    import onnx
    ops = set(n.op_type for n in onnx.load(out_path).graph.node)
    bad = {"NonZero", "GatherND", "ScatterND", "Bucketize"} & ops
    print(f"  data-dependent ops: {bad if bad else 'NONE (static!) ✓'}")


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(batch)
