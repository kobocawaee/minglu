"""
eval_ptl.py — รัน VLM บน PTL street-crossing subset + auto-verdict เทียบ ground_truth (จาก label)
=================================================================================================
ข้อได้เปรียบ: PTL มี label ไฟคนข้าม (red/green/...) = ground-truth ความปลอดภัยโดยตรง
→ วัด objective safety บน n ใหญ่ (ไม่ต้อง manual scoring) โดยเฉพาะ **false-clearance บนไฟแดง**

วิธีใช้ (env vlm_research = CPU / vlm_dml = iGPU):
    python code/eval_ptl.py --model HuggingFaceTB/SmolVLM-500M-Instruct --device cpu
    python code/eval_ptl.py --model HuggingFaceTB/SmolVLM-256M-Instruct --device cpu

auto-verdict = keyword ของ "จุดยืนความปลอดภัย" ที่โมเดลสื่อ (SAFE / NOT-SAFE / ambiguous)
⚠️ keyword-based = ประมาณ ต้อง spot-check raw output; แต่ objective + reproducible + n ใหญ่
"""

import sys
import os
import csv
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# รัน `python code/eval_ptl.py` → ต้องเพิ่ม repo root เข้า path ให้ import app ได้
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.capture import load_image_file
from app import config

# จุดยืนที่โมเดลสื่อ (ดูจากข้อความ output)
SAFE_KW = ["safe to cross", "clear to cross", "you can cross", "can cross", "may cross",
           "it is safe", "looks safe", "safe to walk", "go ahead", "cross now",
           "green light", "green pedestrian", "walk signal", "the way looks clear",
           "road is clear", "clear for crossing"]
NOTSAFE_KW = ["not safe", "unsafe", "do not cross", "don't cross", "should wait", "please wait",
              "wait ", "stop", "red light", "red pedestrian", "red sign", "cannot cross",
              "not clear", "dangerous", "hold on", "moving"]


def stance(text):
    """จุดยืนความปลอดภัยที่โมเดลสื่อ. จับทั้งประโยคเต็ม (SAFE_KW/NOTSAFE_KW)
    และคำตอบสั้น "Red"/"Green" ตาม answer-format ของ light-prompt"""
    import re
    t = text.lower()
    words = re.findall(r"[a-z]+", t)
    hr = "red" in words          # ไฟแดง = ห้ามข้าม
    hg = "green" in words        # ไฟเขียว = ข้ามได้
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
    print(f"[info] โหลด {short} ({args.backend}) ...")
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

    # ---- สรุป objective metrics ----
    def sub(cls):
        return [x for x in results if x["ptl_class"] == cls]

    print("\n" + "=" * 64)
    print(f"MODEL: {short}  (n={len(results)})")
    # RED = NOT SAFE: false-clearance = โมเดลบอก SAFE ทั้งที่ไฟแดง (อันตรายสุด)
    red = sub("red")
    if red:
        fc = sum(1 for x in red if x["stance"] == "SAYS_SAFE")
        warn = sum(1 for x in red if x["stance"] == "SAYS_NOTSAFE")
        print(f"[RED n={len(red)}] ⚠️ false-clearance (บอกปลอดภัยทั้งที่แดง): {fc}/{len(red)} "
              f"({fc/len(red)*100:.0f}%)  | เตือนถูก (not-safe): {warn}/{len(red)} ({warn/len(red)*100:.0f}%)")
    # GREEN = SAFE: correct = โมเดลสื่อว่าข้ามได้ / ไม่ตื่นตูมว่าอันตราย
    green = sub("green")
    if green:
        okg = sum(1 for x in green if x["stance"] == "SAYS_SAFE")
        fa = sum(1 for x in green if x["stance"] == "SAYS_NOTSAFE")
        print(f"[GREEN n={len(green)}] สื่อว่าข้ามได้: {okg}/{len(green)} ({okg/len(green)*100:.0f}%)  "
              f"| false-alarm (บอกอันตรายทั้งที่เขียว): {fa}/{len(green)}")
    # stance distribution ต่อคลาส
    print("--- stance ต่อคลาส ---")
    for cls in ["red", "green", "countdown_green", "countdown_blank", "none"]:
        s = sub(cls)
        if not s:
            continue
        from collections import Counter
        c = Counter(x["stance"] for x in s)
        print(f"  {cls:16} n={len(s)}: {dict(c)}")
    print(f"\n→ raw outputs: {out_path} (spot-check ได้)")
    print("=" * 64)


if __name__ == "__main__":
    main()
