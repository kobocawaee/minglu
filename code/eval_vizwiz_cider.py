"""
在 VizWiz 子集上執行 pycocoevalcap（主要看 CIDEr-D，另有 BLEU/METEOR/ROUGE/SPICE）
比較兩個系統：gen_app（程式周遭模式的提示詞）vs gen_caption（一般中性的描述提示詞）
對照人工參考答案（ref_1..5，已過濾 rejected/precanned）

報告整體結果，以及各情境（street/indoor/surrounding）分開的結果
使用方式：  python code/eval_vizwiz_cider.py   （base anaconda 環境；METEOR/SPICE 需要 Java）
結果 → results/vizwiz_cider.md
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


gen_rows = load_csv(ROOT / "results" / "vizwiz_generated.csv")
ref_rows = load_csv(ROOT / "results" / "vizwiz_references.csv")

refs = {}
for r in ref_rows:
    fn = r["filename"]
    rs = [r.get(f"ref_{i}", "").strip() for i in range(1, 6)]
    refs[fn] = [x for x in rs if x]

gen = {}
for r in gen_rows:
    gen[r["filename"]] = {"scenario": r["scenario"].strip(),
                          "gen_app": r["gen_app"].strip(),
                          "gen_caption": r["gen_caption"].strip()}

# 只取同時有模型輸出和參考答案的圖
fns = [fn for fn in gen if fn in refs and refs[fn]]
tok = PTBTokenizer()


def score_subset(field, subset_fns):
    gts_raw = {fn: [{"caption": c} for c in refs[fn]] for fn in subset_fns}
    res_raw = {fn: [{"caption": gen[fn][field]}] for fn in subset_fns}
    gts = tok.tokenize(gts_raw)
    res = tok.tokenize(res_raw)
    out = {}
    for scorer, names in [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr-D"]),
    ]:
        try:
            sc, _ = scorer.compute_score(gts, res)
            if isinstance(sc, list):
                for n, s in zip(names, sc):
                    out[n] = s
            else:
                out[names[0]] = sc
        except Exception as e:
            for n in names:
                out[n] = None
            print(f"  [warn] {names} failed: {e}")
    return out


# SPICE 分開跑（需要 Stanford CoreNLP — 電腦上可能沒有 → 略過）
def try_spice(subset_fns):
    try:
        from pycocoevalcap.spice.spice import Spice
        gts = tok.tokenize({fn: [{"caption": c} for c in refs[fn]] for fn in subset_fns})
        res = tok.tokenize({fn: [{"caption": gen[fn]["gen_app"]}] for fn in subset_fns})
        s, _ = Spice().compute_score(gts, res)
        return s
    except Exception as e:
        print(f"  [warn] SPICE skipped: {e}")
        return None


scenarios = ["street", "indoor", "surrounding"]
groups = {"ALL": fns}
for sc in scenarios:
    groups[sc] = [fn for fn in fns if gen[fn]["scenario"] == sc]

metric_order = ["CIDEr-D", "BLEU-4", "METEOR", "ROUGE-L", "BLEU-1"]
results = {}  # (system, group) -> metric dict
for system in ["gen_app", "gen_caption"]:
    for gname, gfns in groups.items():
        if gfns:
            print(f"scoring {system} / {gname} (n={len(gfns)})")
            results[(system, gname)] = score_subset(system, gfns)

spice_all = try_spice(fns)


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


lines = []
lines.append("# VizWiz-Captions — CIDEr-D & caption metrics (Aomam)\n")
lines.append(f"Ran on {len(fns)} VizWiz val images with ≥1 non-rejected reference "
             f"(refs already filtered is_rejected/is_precanned). pycocoevalcap "
             f"(CIDEr-D primary per `docs/evaluation_metrics.md`).\n")
lines.append("**Two systems compared vs the same human references:** `gen_app` = the app's "
             "surrounding-mode prompt (assistant-style, e.g. \"in front of you\"); "
             "`gen_caption` = a neutral captioning prompt (\"Describe this image in one sentence\").\n")

for gname in ["ALL", "street", "indoor", "surrounding"]:
    n = len(groups[gname])
    if not n:
        continue
    lines.append(f"\n## {gname} (n={n})\n")
    lines.append("| System | CIDEr-D | BLEU-4 | METEOR | ROUGE-L | BLEU-1 |")
    lines.append("|---|---|---|---|---|---|")
    for system in ["gen_app", "gen_caption"]:
        m = results.get((system, gname), {})
        lines.append("| " + system + " | " + " | ".join(fmt(m.get(k)) for k in metric_order) + " |")

lines.append(f"\n**SPICE (gen_app, ALL):** {fmt(spice_all)}"
             + ("" if spice_all is not None else "  (skipped — needs Stanford CoreNLP)"))

out_md = ROOT / "results" / "vizwiz_cider.md"
# 接在後面要補的文字段落（由程式產生＋人工補寫）
(out_md.with_suffix(".auto.md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved metric tables -> {out_md.with_suffix('.auto.md')}")
