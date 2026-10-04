"""
fig6: NPU op-placement comparison — SmolVLM (ViT) vs Gemma-3-4b
顯示運算量大的運算（MatMul/Softmax/LayerNorm/GELU/Conv）實際落在哪個裝置
數字來自節點配置紀錄（Mhiu, 2026-06-25）— docs/npu_setup_progress.md、STATUS.md

執行方式：
    python code/plot_npu_op_placement.py
輸出：results/fig6_npu_op_placement.png
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# SmolVLM-256M vision encoder (static INT8, VitisAI EP): node counts from node-placement log
smolvlm_npu_nodes = 50 + 123  # QuantizeLinear + DequantizeLinear (boundary only, no compute)
smolvlm_cpu_nodes = 13 + 12 + 12 + 24 + 12 + 1  # MatMul + FusedMatMul + Softmax + LayerNorm + GELU + Conv

models = ["SmolVLM-256M\n(VitisAI EP)", "Gemma-3-4b\n(OGA / RyzenAI)"]
compute_on_npu_pct = [0, 100]  # SmolVLM: 0% of compute ops on NPU; Gemma: matmul kernel runs every layer on NPU
compute_on_cpu_pct = [100, 0]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
y = range(len(models))
ax.barh(y, compute_on_npu_pct, color="#2e7d32", label="Compute ops on NPU")
ax.barh(y, compute_on_cpu_pct, left=compute_on_npu_pct, color="#c62828", label="Compute ops on CPU (fallback)")

ax.set_yticks(list(y))
ax.set_yticklabels(models)
ax.set_xlabel("% of compute-heavy ops (MatMul / Softmax / LayerNorm / GELU / Conv)")
ax.set_xlim(0, 100)
ax.set_title("NPU acceleration depends on model architecture, not just size")
ax.legend(loc="lower right")

ax.text(2, 0, f"NPU gets only Quantize/Dequantize\n({smolvlm_npu_nodes} boundary nodes), no compute",
        va="center", fontsize=8, color="white")
ax.text(98, 1, "AMDMatMulNBitsKernel\nruns every layer", va="center", ha="right", fontsize=8, color="white")

fig.tight_layout()
out_path = OUT / "fig6_npu_op_placement.png"
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
