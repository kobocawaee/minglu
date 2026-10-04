"""
分析自動選模式的結果 — 混淆矩陣＋各模式的精確率／召回率
+ ⭐ 過馬路召回率（依風險框架最重要：把過馬路場景選錯模式 = 繼承 L4 風險）

輸入：Mhiu 產生的 CSV（在約 60 張有標註的圖上執行 app/pipeline.route）— 欄位：
    filename, true_mode, routed_mode   （欄位名稱可用 --true-col/--pred-col 調整）

使用方式：
    python code/eval_router.py results/router_output.csv
    python code/eval_router.py results/router_output.csv --true-col label --pred-col routed
結果 → results/router_eval.md（表格＋過馬路召回率＋選錯的清單）＋ print

⚠️ 過馬路召回率 = 所有*應該是 street* 的圖中，系統正確選到 street 的比例（street 類別的召回率）
   street → 其他模式的誤判 = 最危險（漏掉 L4 等級的危險）→ 列出來檢查
"""
import argparse
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent


def load(path, true_col, pred_col, file_col):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r.get(file_col, "").strip(),
                         r.get(true_col, "").strip().lower(),
                         r.get(pred_col, "").strip().lower()))
    return [r for r in rows if r[1] and r[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="router output CSV (filename,true_mode,routed_mode)")
    ap.add_argument("--file-col", default="filename")
    ap.add_argument("--true-col", default="true_mode")
    ap.add_argument("--pred-col", default="routed_mode")
    ap.add_argument("--out", default=str(ROOT / "results" / "router_eval.md"))
    args = ap.parse_args()

    rows = load(args.csv_path, args.true_col, args.pred_col, args.file_col)
    n = len(rows)
    modes = sorted(set([r[1] for r in rows]) | set([r[2] for r in rows]))

    # confusion[true][pred] = count
    conf = defaultdict(Counter)
    correct = 0
    for _, t, p in rows:
        conf[t][p] += 1
        if t == p:
            correct += 1
    overall_acc = correct / n if n else 0.0

    # per-mode precision/recall
    per = {}
    for m in modes:
        tp = conf[m][m]
        support = sum(conf[m].values())                     # 實際是 m
        pred_pos = sum(conf[t][m] for t in modes)           # 預測成 m
        recall = tp / support if support else None
        prec = tp / pred_pos if pred_pos else None
        per[m] = (tp, support, pred_pos, prec, recall)

    # 選錯的情況，特別是 street → 其他
    street_miss = [(fn, p) for fn, t, p in rows if t == "street" and p != "street"]
    all_miss = [(fn, t, p) for fn, t, p in rows if t != p]

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, float) else "N/A"

    L = []
    L.append("# Auto-router evaluation — routing accuracy + street-recall (Aomam)\n")
    L.append(f"n = {n} images · modes = {', '.join(modes)} · source: `{Path(args.csv_path).name}`\n")
    L.append(f"**Overall routing accuracy: {correct}/{n} = {pct(overall_acc)}**\n")

    L.append("\n## Per-mode precision / recall\n")
    L.append("| Mode | support | routed→correct | Precision | Recall |")
    L.append("|---|---|---|---|---|")
    for m in modes:
        tp, sup, pp, pr, rc = per[m]
        star = " ⭐" if m == "street" else ""
        L.append(f"| {m}{star} | {sup} | {tp} | {pct(pr) if pr is not None else 'N/A'} | "
                 f"{pct(rc) if rc is not None else 'N/A'} |")

    L.append("\n## Confusion matrix (rows = true, cols = routed)\n")
    L.append("| true \\ routed | " + " | ".join(modes) + " |")
    L.append("|" + "---|" * (len(modes) + 1))
    for t in modes:
        L.append(f"| **{t}** | " + " | ".join(str(conf[t][p]) for p in modes) + " |")

    st = per.get("street")
    L.append("\n## ⭐ Street-recall (safety-critical — misroute street inherits L4)\n")
    if st and st[1]:
        L.append(f"- **street-recall = {st[0]}/{st[1]} = {pct(st[4])}** "
                 f"（所有應該是 street 的圖中，正確選到 street 的比例）")
        if street_miss:
            L.append(f"- ⚠️ **street 選錯 {len(street_miss)} 例**（應該是 street 卻選到其他模式 = 漏掉危險）：")
            for fn, p in street_miss:
                L.append(f"  - `{fn}` → routed **{p}**")
        else:
            L.append("- ✅ 沒有 street 選錯（過馬路召回率 100%）")
    else:
        L.append("- （資料中沒有 street 的樣本 — 請先補上再下結論）")

    if all_miss:
        L.append("\n## 所有選錯的情況（正確 → 選到）\n")
        for fn, t, p in all_miss:
            L.append(f"- `{fn}`: {t} → **{p}**")

    L.append("\n---\n*reproduce:* `python code/eval_router.py <csv>` · "
             "risk framing: `docs/risk_analysis.md` (auto = risk amplifier)")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
