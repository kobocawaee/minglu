"""
eval_clipscore.py — CLIPScore: คำบรรยาย "ยึดโยงกับภาพ" แค่ไหน (advisor request 2026-07-22)
==========================================================================================
ทำไมต้องมี: Metric B (fabrication) ของเราให้คะแนน "ด้วยมือ" — reviewer วารสารจะถามหา
metric อัตโนมัติที่วัดเรื่องเดียวกัน. CLIPScore (Hessel et al. 2021) เป็น reference-free
image–text alignment: ไม่ต้องมี human caption อ้างอิง วัดตรงๆ ว่าข้อความตรงกับภาพไหม

    CLIPScore(c, v) = 2.5 · max(cos(E_text(c), E_img(v)), 0)

3 คำถามที่สคริปต์นี้ตอบ:
  A) CLIPScore เฉลี่ยต่อโมเดล — ขนาดโมเดลใหญ่ขึ้น ทำให้คำบรรยายยึดกับภาพมากขึ้นไหม
  B) ⭐ output ที่เราตีว่า "fabrication" (Metric B = Y) ได้ CLIPScore ต่ำกว่าจริงไหม
     → ถ้าใช่ CLIPScore ใช้เป็นสัญญาณ hallucination อัตโนมัติได้ + ยืนยัน manual scoring
  C) VizWiz 45 ใบ: prompt แบบผู้ช่วย (gen_app) vs แบบ caption (gen_caption)
     → §4.5 พบว่า caption metric ขัดกันเอง; CLIPScore ไม่ต้องใช้ reference จึงตัดสินได้ตรงกว่า

รัน (env vlm_research):  python code/eval_clipscore.py
output: results/clipscore.csv  (+ สรุปบนจอ)
"""

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

MODEL_ID = "openai/clip-vit-base-patch32"
CANON_DIR = ROOT / "data/test_images"
VIZ_DIR = ROOT / "data/vizwiz/images"
SAFETY_CSV = ROOT / "data/safety_eval.csv"
VIZ_CSV = ROOT / "results/vizwiz_generated.csv"
OUT = ROOT / "results/clipscore.csv"

W = 2.5  # ตัวคูณมาตรฐานของ CLIPScore (Hessel et al. 2021)


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    print(f"[info] loading {MODEL_ID} (first run downloads ~600 MB) ...")
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    torch.set_num_threads(8)
    return torch, model, proc


def clip_scores(torch, model, proc, image_paths, texts):
    """คืน (scores, n_truncated). ประมวลผลทีละใบ — ชุดเล็ก ไม่ต้อง batch"""
    from PIL import Image
    scores, truncated = [], 0
    for i, (path, text) in enumerate(zip(image_paths, texts), 1):
        img = Image.open(path).convert("RGB")
        # CLIP รับ text ได้สูงสุด 77 token — ตัดส่วนเกิน (บันทึกจำนวนไว้รายงานตามจริง)
        ids = proc.tokenizer(text, truncation=False)["input_ids"]
        if len(ids) > 77:
            truncated += 1
        inputs = proc(text=[text], images=img, return_tensors="pt",
                      padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            out = model(**inputs)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos = float((ie * te).sum())
        scores.append(W * max(cos, 0.0))
        if i % 20 == 0:
            print(f"  scored {i}/{len(texts)}")
    return np.array(scores), truncated


def permutation_test(a, b, n_perm=20000, seed=42):
    """ต่างกันจริงหรือบังเอิญ — permutation test สองทาง (ไม่ต้องพึ่ง scipy)"""
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def main():
    torch, model, proc = load_clip()
    rows_out = []

    # ---------------------------------------------------------------- A + B
    rows = list(csv.DictReader(open(SAFETY_CSV, encoding="utf-8-sig")))
    paths, texts, keep = [], [], []
    for r in rows:
        p = CANON_DIR / r["filename"]
        if not p.exists():
            print(f"  [skip] missing image: {r['filename']}")
            continue
        paths.append(p); texts.append(r["output"]); keep.append(r)
    print(f"\n[A/B] canonical set: {len(keep)} outputs "
          f"({len(set(r['filename'] for r in keep))} images x "
          f"{len(set(r['model'] for r in keep))} models)")
    sc, trunc = clip_scores(torch, model, proc, paths, texts)
    print(f"  ({trunc} outputs exceeded CLIP's 77-token limit and were truncated)")

    for r, s in zip(keep, sc):
        rows_out.append({"set": "canonical", "filename": r["filename"],
                         "scenario": r["scenario"], "model": r["model"],
                         "variant": "", "metric_b_fabrication": r["safety_critical_halluc"],
                         "safe_verdict": r["safe_verdict"], "clipscore": round(float(s), 4)})

    print("\n--- A. mean CLIPScore per model (higher = text matches image better)")
    for m in sorted(set(r["model"] for r in keep)):
        v = np.array([s for r, s in zip(keep, sc) if r["model"] == m])
        print(f"  {m:26s} {v.mean():.3f}  (sd {v.std(ddof=1):.3f}, n={len(v)})")

    print("\n--- B. fabrication (Metric B) vs CLIPScore  <-- the key question")
    fab = np.array([s for r, s in zip(keep, sc)
                    if r["safety_critical_halluc"].strip().upper().startswith("Y")])
    ok = np.array([s for r, s in zip(keep, sc)
                   if not r["safety_critical_halluc"].strip().upper().startswith("Y")])
    print(f"  flagged as fabrication : {fab.mean():.3f}  (n={len(fab)})")
    print(f"  not flagged            : {ok.mean():.3f}  (n={len(ok)})")
    diff, p = permutation_test(fab, ok)
    print(f"  difference {diff:+.3f}  permutation p = {p:.4f}")

    # ---------------------------------------------------------------- C
    vrows = [r for r in csv.DictReader(open(VIZ_CSV, encoding="utf-8-sig"))
             if (VIZ_DIR / r["filename"]).exists()]
    print(f"\n[C] VizWiz subset: {len(vrows)} images x 2 prompt styles")
    for variant in ("gen_app", "gen_caption"):
        sc2, tr2 = clip_scores(torch, model, proc,
                               [VIZ_DIR / r["filename"] for r in vrows],
                               [r[variant] for r in vrows])
        print(f"  {variant:12s} mean {sc2.mean():.3f} (sd {sc2.std(ddof=1):.3f}, "
              f"{tr2} truncated)")
        for r, s in zip(vrows, sc2):
            rows_out.append({"set": "vizwiz", "filename": r["filename"],
                             "scenario": r["scenario"], "model": "SmolVLM-500M-Instruct",
                             "variant": variant, "metric_b_fabrication": "",
                             "safe_verdict": "", "clipscore": round(float(s), 4)})

    app = np.array([r["clipscore"] for r in rows_out
                    if r["set"] == "vizwiz" and r["variant"] == "gen_app"])
    cap = np.array([r["clipscore"] for r in rows_out
                    if r["set"] == "vizwiz" and r["variant"] == "gen_caption"])
    d2, p2 = permutation_test(app, cap)
    print(f"  gen_app - gen_caption = {d2:+.3f}  permutation p = {p2:.4f}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)
    print(f"\n[saved] {OUT}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
