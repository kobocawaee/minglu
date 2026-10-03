"""
inspect_onnx.py — ส่องโครงสร้างไฟล์ ONNX (inputs/outputs/op-types/opset)

ใช้ดูว่า model graph มี operator อะไรบ้าง เพื่อประเมินว่า VitisAI EP (NPU)
จะรองรับได้กี่ % ก่อนจะลงทุน quantize จริง

Usage:
    python code/inspect_onnx.py models/smolvlm256m_onnx/onnx/vision_encoder.onnx
"""
import sys
from collections import Counter

# Windows console = cp1252 -> กัน error เวลา print
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import onnx


def tensor_shape(t):
    """ดึง shape ออกมาเป็น list อ่านง่าย (dim ที่เป็น dynamic จะเป็นชื่อ string)"""
    dims = []
    for d in t.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)        # dynamic เช่น 'batch', 'seq'
        else:
            dims.append(d.dim_value)        # static เช่น 3, 512
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

    # op-type histogram — หัวใจ: ดูว่ามี operator อะไรบ้าง
    ops = Counter(node.op_type for node in g.node)
    print(f"\n--- OP TYPES ({len(g.node)} nodes, {len(ops)} unique) ---")
    for op, cnt in ops.most_common():
        print(f"  {op:25s} {cnt}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code/inspect_onnx.py <model.onnx>")
        sys.exit(1)
    main(sys.argv[1])
