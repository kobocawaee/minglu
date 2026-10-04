"""
eval_lytnet.py — 在和 VLM 測試相同的 PTL 子集上驗證 LYTNetV2（預訓練權重）
==================================================================================
目標：同條件比較的表格 — VLM（分辨能力 ≈ 0，§5.5.1）vs 專用 CNN
用同一批圖 → 補上混合式架構論點的最後一塊證據

需要知道的設定：
  - 模型：LYTNetV2 + ImVisible repo（MIT）的權重 → clone 到 external/ImVisible
  - ⚠️ 輸入：repo 程式寫 768x576，但 V2 實際有 AvgPool2d(12,9)，要求特徵圖 ≥12x12
    → 輸入必須 ≥768x768。我們用 1024x768（保持原圖 4032x3024 的 4:3 比例）
  - 像素 = 原始 0-255 浮點數，不做正規化（依 repo 的 dataset.py — 正規化那行被註解掉）
  - 類別：red / green / none / countdown_blank / countdown_green（索引依 testing.py）

執行：  python code/eval_lytnet.py [--limit-per-class 30] [--out results/lytnet_ptl.csv]
"""

import sys, os, csv, time, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "external/ImVisible/Model"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import numpy as np
from PIL import Image
from LYTNetV2 import LYTNetV2

CLASSES = ["red", "green", "none", "countdown_blank", "countdown_green"]
WEIGHTS = REPO / "external/ImVisible/Model/LytNetV2_weights"
IMGDIR = REPO / "data/ptl/images/PTL_Dataset_876x657"
SUBSET = REPO / "data/ptl/ptl_subset150.csv"


def load_model():
    net = LYTNetV2()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()
    return net


def predict(net, path):
    im = Image.open(path).convert("RGB").resize((1024, 768))
    x = torch.from_numpy(np.transpose(np.asarray(im, dtype=np.float32), (2, 0, 1))).unsqueeze(0)
    with torch.no_grad():
        cls, _direction = net(x)
    return CLASSES[int(cls.argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-class", type=int, default=30)
    ap.add_argument("--out", default="results/lytnet_ptl.csv")
    args = ap.parse_args()

    net = load_model()
    rows = list(csv.DictReader(open(SUBSET, encoding="utf-8-sig")))
    per_class = {}
    picked = []
    for r in rows:
        c = r["ptl_class"]
        if c not in ("red", "green"):
            continue
        if per_class.get(c, 0) >= args.limit_per_class:
            continue
        per_class[c] = per_class.get(c, 0) + 1
        picked.append(r)

    out_rows, t0 = [], time.perf_counter()
    stats = {"red": {"n": 0, "ok": 0, "as_green": 0}, "green": {"n": 0, "ok": 0, "as_red": 0}}
    for r in picked:
        p = IMGDIR / r["filename"]
        if not p.exists():
            print("missing:", r["filename"]); continue
        pred = predict(net, p)
        truth = r["ptl_class"]
        s = stats[truth]; s["n"] += 1
        if pred == truth: s["ok"] += 1
        if truth == "red" and pred == "green": s["as_green"] += 1
        if truth == "green" and pred == "red": s["as_red"] += 1
        out_rows.append({"filename": r["filename"], "truth": truth, "pred": pred,
                         "correct": pred == truth})
    n = sum(s["n"] for s in stats.values())
    avg_ms = (time.perf_counter() - t0) / max(n, 1) * 1000

    os.makedirs(Path(args.out).parent, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "truth", "pred", "correct"])
        w.writeheader(); w.writerows(out_rows)

    print(f"n={n}  avg {avg_ms:.0f}ms/img (CPU)")
    print(f"RED   : {stats['red']['ok']}/{stats['red']['n']} correct  | red→green (dangerous false-clear): {stats['red']['as_green']}")
    print(f"GREEN : {stats['green']['ok']}/{stats['green']['n']} correct | green→red (false alarm): {stats['green']['as_red']}")
    # 和 §5.5.1 相同的分辨能力：P(說綠|綠) − P(說綠|紅)
    p_g_g = stats["green"]["ok"] / max(stats["green"]["n"], 1)
    p_g_r = stats["red"]["as_green"] / max(stats["red"]["n"], 1)
    print(f"discrimination (say-green on green − say-green on red): {100*(p_g_g - p_g_r):.0f} pp  (VLMs: ~0 pp)")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
