# -*- coding: utf-8 -*-
"""Object mode: does the assistant name the thing the photographer was holding?

Section 3.7 promised a per-mode metric for every mode, "object, top-1
identification", and Section 4 never reported one. This closes that gap.

Scoring, stated precisely because "top-1 identification" is vaguer than it
sounds. Each VizWiz image carries up to five independent human captions. A
content word that two or more of them use is treated as agreed: it is something
the picture demonstrably contains, rather than one annotator's choice of word.
The assistant's answer counts as correct when it uses at least one agreed word.
This is an agreement measure against human descriptions, not a classification
accuracy against a label set, and Section 4.5 says so; VizWiz carries no object
labels, so a stricter reading is not available.

Images whose captions cannot support the rule are reported separately rather than
scored: VizWiz photographs are taken by blind users, and some are of a wall or a
worktop with no object in them at all.

    python code/eval_object_mode.py

Writes results/object_mode_eval.csv.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from PIL import Image

from app import config
from app.backends import get_backend
from eval_robustness import content_tokens

ROUTER = ROOT / "results/router_eval_v2.csv"
REFS = ROOT / "results/vizwiz_references.csv"
IMAGES = ROOT / "data/vizwiz/images"
OUT = ROOT / "results/object_mode_eval.csv"
MIN_AGREE = 2       # how many captions must share a word before it counts as agreed


def agreed_words(refs):
    """Content words that at least MIN_AGREE captions use."""
    counts = Counter()
    used = 0
    for r in refs:
        if not r.strip():
            continue
        used += 1
        counts.update(content_tokens(r))
    if used < MIN_AGREE:
        return None, used          # cannot apply the rule
    return {w for w, n in counts.items() if n >= MIN_AGREE}, used


def main():
    router = list(csv.DictReader(open(ROUTER, encoding="utf-8-sig")))
    refs = {r["filename"]: [r[f"ref_{i}"] for i in range(1, 6)]
            for r in csv.DictReader(open(REFS, encoding="utf-8-sig"))}
    targets = [r["filename"] for r in router if r["true_mode"] == "object"]

    backend = get_backend("smolvlm", model_id=config.SMOLVLM_MODEL,
                          device="cpu", gen_params=config.GEN_PARAMS)
    print(f"object-mode images: {len(targets)}")
    print("loading SmolVLM-500M ...")
    backend.load(); backend.warmup()

    prompt = config.get_prompt("object")
    mx = config.get_max_tokens("object")

    rows, scored, correct, skipped = [], 0, 0, 0
    for name in targets:
        path = IMAGES / name
        if not path.exists():
            continue
        answer = backend.describe(Image.open(path).convert("RGB"), prompt,
                                  max_new_tokens=mx)
        agreed, n_refs = agreed_words(refs.get(name, []))
        if agreed is None:
            verdict = f"not scored ({n_refs} caption(s))"
            skipped += 1
            hit = ""
        else:
            hit_words = content_tokens(answer) & agreed
            ok = bool(hit_words)
            scored += 1
            correct += ok
            verdict = "correct" if ok else "wrong"
            hit = " ".join(sorted(hit_words))
        rows.append({"filename": name, "answer": answer.strip(),
                     "n_refs": n_refs,
                     "agreed_words": " ".join(sorted(agreed)) if agreed else "",
                     "matched": hit, "verdict": verdict})
        print(f"  [{verdict:22}] {name}  ->  {answer.strip()[:60]!r}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"\nscored {scored} images, {skipped} not scorable "
          f"(fewer than {MIN_AGREE} usable captions)")
    if scored:
        print(f"top-1 agreement: {correct}/{scored} ({100*correct/scored:.0f}%)")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
