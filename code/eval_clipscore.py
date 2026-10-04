"""
eval_clipscore.py — CLIPScore：描述和畫面的「對應程度」（指導教授 2026-07-22 的要求）
==========================================================================================
為什麼需要：我們的指標 B（捏造）是「人工」評分 — 期刊審稿人會要求
衡量同一件事的自動指標。CLIPScore（Hessel et al. 2021）是不需要參考答案的
圖文對應指標：不需要人工描述當參考，直接衡量文字和畫面是否相符

    CLIPScore(c, v) = 2.5 · max(cos(E_text(c), E_img(v)), 0)

這支腳本回答 3 個問題：
  A) 各模型的平均 CLIPScore — 模型變大，描述是否更貼近畫面
  B) ⭐ 我們判定為「捏造」（指標 B = Y）的輸出，CLIPScore 是否真的比較低
     → 如果是，CLIPScore 可以當作自動的幻覺訊號＋佐證人工評分
  C) VizWiz 45 張：助理式提示詞（gen_app）vs 描述式提示詞（gen_caption）
     → §4.5 發現描述類指標彼此矛盾；CLIPScore 不需要參考答案，判斷更直接

執行（vlm_research 環境）：  python code/eval_clipscore.py
輸出：results/clipscore.csv（＋螢幕上的摘要）
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

W = 2.5  # CLIPScore 的標準倍數（Hessel et al. 2021）


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    print(f"[info] loading {MODEL_ID} (first run downloads ~600 MB) ...")
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    torch.set_num_threads(8)
    return torch, model, proc


def clip_scores(torch, model, proc, image_paths, texts):
    """回傳 (scores, n_truncated)。一張一張處理 — 資料量小，不需要 batch"""
    from PIL import Image
    scores, truncated = [], 0
    for i, (path, text) in enumerate(zip(image_paths, texts), 1):
        img = Image.open(path).convert("RGB")
        # CLIP 的文字最多 77 個 token — 截掉多的部分（記錄數量，如實報告）
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
    """差異是真的還是巧合 — 雙尾排列檢定（不依賴 scipy）"""
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
