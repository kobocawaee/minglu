"""
inspect_onnx.py — 檢視 ONNX 檔的結構（輸入／輸出／運算類型／opset）

用來看模型圖裡有哪些運算，以評估 VitisAI EP（NPU）
能支援幾成，再決定要不要真的投入量化

Usage:
    python code/inspect_onnx.py models/smolvlm256m_onnx/onnx/vision_encoder.onnx
"""
import sys
from collections import Counter

# Windows 主控台 = cp1252 -> 避免 print 時出錯
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import onnx


def tensor_shape(t):
    """把 shape 取出成好讀的 list（動態維度會是名稱字串）"""
    dims = []
    for d in t.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)        # 動態，例如 'batch'、'seq'
        else:
            dims.append(d.dim_value)        # 靜態，例如 3、512
    dtype = onnx.TensorProto.DataType.Name(t.type.tensor_type.elem_type)
    return dims, dtype


def main(path):
    print(f"=== Inspecting: {path} ===\n")
    model = onnx.load(path)

    # opset
    opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
    print(f"Opset: {opsets}")
    print(f"IR version: {model.ir_version}\n")

    g = model.graph

    # inputs / outputs
    print(f"--- INPUTS ({len(g.input)}) ---")
    for inp in g.input:
        dims, dtype = tensor_shape(inp)
        print(f"  {inp.name:30s} {dtype:10s} {dims}")

    print(f"\n--- OUTPUTS ({len(g.output)}) ---")
    for out in g.output:
        dims, dtype = tensor_shape(out)
        print(f"  {out.name:30s} {dtype:10s} {dims}")

    # 運算類型統計 — 重點：看有哪些運算
    ops = Counter(node.op_type for node in g.node)
    print(f"\n--- OP TYPES ({len(g.node)} nodes, {len(ops)} unique) ---")
    for op, cnt in ops.most_common():
        print(f"  {op:25s} {cnt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/inspect_onnx.py <model.onnx>")
        sys.exit(1)
    main(sys.argv[1])
