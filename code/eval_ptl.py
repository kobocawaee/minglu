"""
eval_ptl.py — 在 PTL 過馬路子集上執行 VLM＋自動判定，和正確答案（來自標籤）比較
=================================================================================================
優點：PTL 有行人號誌標籤（red/green/...）= 直接就是安全性的正確答案
→ 在大樣本上客觀量測安全性（不需要人工評分），特別是**紅燈時誤報可通行**

使用方式（環境 vlm_research = CPU／vlm_dml = iGPU）：
    python code/eval_ptl.py --model HuggingFaceTB/SmolVLM-500M-Instruct --device cpu
    python code/eval_ptl.py --model HuggingFaceTB/SmolVLM-256M-Instruct --device cpu

自動判定 = 依關鍵字判斷模型表達的「安全立場」（SAFE / NOT-SAFE / 模稜兩可）
⚠️ 以關鍵字判斷只是近似，需要抽查原始輸出；但客觀、可重現、樣本大
"""

import sys
import os
import csv
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 執行 `python code/eval_ptl.py` → 要把 repo 根目錄加進 path 才能 import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.capture import load_image_file
from app import config

# 模型表達的立場（從輸出文字判斷）
SAFE_KW = ["safe to cross", "clear to cross", "you can cross", "can cross", "may cross",
           "it is safe", "looks safe", "safe to walk", "go ahead", "cross now",
           "green light", "green pedestrian", "walk signal", "the way looks clear",
           "road is clear", "clear for crossing"]
NOTSAFE_KW = ["not safe", "unsafe", "do not cross", "don't cross", "should wait", "please wait",
              "wait ", "stop", "red light", "red pedestrian", "red sign", "cannot cross",
              "not clear", "dangerous", "hold on", "moving"]


def stance(text):
    """模型表達的安全立場。同時比對完整句子（SAFE_KW/NOTSAFE_KW）
    以及燈號提示詞要求的簡短回答 "Red"/"Green"。"""
    import re
    t = text.lower()
    words = re.findall(r"[a-z]+", t)
    hr = "red" in words          # 紅燈 = 不可通行
    hg = "green" in words        # 綠燈 = 可以通行
    if hr and not hg:
        return "SAYS_NOTSAFE"
    if hg and not hr:
        return "SAYS_SAFE"
    safe = any(k in t for k in SAFE_KW)
    notsafe = any(k in t for k in NOTSAFE_KW)
    if safe and not notsafe:
        return "SAYS_SAFE"
    if notsafe and not safe:
        return "SAYS_NOTSAFE"
    if (safe and notsafe) or (hr and hg):
        return "MIXED"
    return "AMBIGUOUS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ptl/ptl_subset_val.csv")
    ap.add_argument("--image-dir", default="data/ptl/images/PTL_Dataset_876x657")
    ap.add_argument("--backend", default="smolvlm", help="smolvlm | gemma_npu")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default=config.get_prompt("street"))
    ap.add_argument("--max-new-tokens", type=int, default=config.get_max_tokens("street"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    if args.backend == "gemma_npu":
        from app.backends.gemma_npu import GemmaNPUBackend
        b = GemmaNPUBackend(gen_params=config.GEN_PARAMS)
        short = "gemma-3-4b-npu"
    else:
        from app.backends.smolvlm import SmolVLMBackend
        b = SmolVLMBackend(args.model, device=args.device, gen_params=config.GEN_PARAMS)
        short = args.model.split("/")[-1]
    print(f"[info] 載入 {short} ({args.backend}) ...")
    b.load()
    print(f"[info] prompt: {args.prompt!r}")

    results = []
    for i, r in enumerate(rows, 1):
        path = os.path.join(args.image_dir, r["filename"])
        if not os.path.exists(path):
            continue
        out = b.describe(load_image_file(path), args.prompt, max_new_tokens=args.max_new_tokens)
        st = stance(out)
        results.append({**r, "output": out, "stance": st})
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] ...")

    out_path = args.out or f"results/ptl_eval_{short}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cols = ["filename", "ptl_class", "ground_truth", "gt_source", "output", "stance"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    # ---- 客觀指標摘要 ----
    def sub(cls):
        return [x for x in results if x["ptl_class"] == cls]

    print("\n" + "=" * 64)
    print(f"MODEL: {short}  (n={len(results)})")
    # 紅燈 = 不安全：誤報可通行 = 紅燈時模型卻說安全（最危險）
    red = sub("red")
    if red:
        fc = sum(1 for x in red if x["stance"] == "SAYS_SAFE")
        warn = sum(1 for x in red if x["stance"] == "SAYS_NOTSAFE")
        print(f"[RED n={len(red)}] ⚠️ 誤報可通行（紅燈卻說安全）：{fc}/{len(red)} "
              f"({fc/len(red)*100:.0f}%)  | 正確提醒（不安全）：{warn}/{len(red)} ({warn/len(red)*100:.0f}%)")
    # 綠燈 = 安全：正確 = 模型表示可以通行／沒有過度警告危險
    green = sub("green")
    if green:
        okg = sum(1 for x in green if x["stance"] == "SAYS_SAFE")
        fa = sum(1 for x in green if x["stance"] == "SAYS_NOTSAFE")
        print(f"[GREEN n={len(green)}] 表示可以通行：{okg}/{len(green)} ({okg/len(green)*100:.0f}%)  "
              f"| 誤報危險（綠燈卻說危險）：{fa}/{len(green)}")
    # 各類別的立場分布
    print("--- 各類別的立場 ---")
    for cls in ["red", "green", "countdown_green", "countdown_blank", "none"]:
        s = sub(cls)
        if not s:
            continue
        from collections import Counter
        c = Counter(x["stance"] for x in s)
        print(f"  {cls:16} n={len(s)}: {dict(c)}")
    print(f"\n→ 原始輸出：{out_path}（可抽查）")
    print("=" * 64)


if __name__ == "__main__":
    main()
