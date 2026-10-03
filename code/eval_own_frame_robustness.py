# -*- coding: utf-8 -*-
"""Run the six robustness variants on a frame from our own field footage.

Figure 5 used to reprint a stock photograph that data/dataset_manifest.csv flags
as carrying a visible Dreamstime watermark, while Section 3.5 states the
canonical images are "referenced, not reprinted". A frame we filmed has neither
problem. This checks that it still shows the finding, and writes the outputs so
the figure can be rebuilt without loading the model again.

The input is the *blurred* frame, the same file the report prints. Publishing one
image while reporting outputs measured on another would not be reproducible, and
the blur covers distant heads far from the signal, so being strict costs nothing.

    python code/eval_own_frame_robustness.py            # uses the default frame
    python code/eval_own_frame_robustness.py other.png

Writes results/own_frame_variants.csv.
"""
import csv
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from PIL import Image

from app import config, quality
from app.backends import get_backend
from eval_robustness import (make_variants, safety_stance, has_person,
                             mean_pairwise_jaccard)
from eval_ptl import stance as colour_aware_stance

DEFAULT_FRAME = ROOT / "data/field_frames/crossing1_t24_red_blur.png"
OUT = ROOT / "results/own_frame_variants.csv"
TRUTH = "red"      # confirmed by magnifying the frame; results/own_frame_robustness.md


def main():
    frame = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAME
    base = Image.open(frame).convert("RGB")
    print(f"frame: {frame.name}  {base.size[0]}x{base.size[1]}  ground truth: {TRUTH}")

    variants = make_variants(base)
    gate = {k: quality.assess(v, config.QUALITY)[0] for k, v in variants.items()}
    print(f"quality gate: {sum(gate.values())}/{len(gate)}"
          + ("" if all(gate.values()) else f"  FAILED {[k for k, v in gate.items() if not v]}"))

    backend = get_backend("smolvlm", model_id=config.SMOLVLM_MODEL,
                          device="cpu", gen_params=config.GEN_PARAMS)
    print("loading SmolVLM-500M ...")
    backend.load(); backend.warmup()

    prompt = config.get_prompt("street")
    mx = config.get_max_tokens("street")

    rows, texts = [], []
    for name, img in variants.items():
        text = backend.describe(img, prompt, max_new_tokens=mx)
        texts.append(text)
        low = text.lower()
        said = "green" if "green" in low else "red" if "red" in low else ""
        rows.append({"variant": name, "text": text, "said": said,
                     "agrees_with_truth": said == TRUTH,
                     "phrase_stance": safety_stance(text),
                     "colour_stance": colour_aware_stance(text),
                     "person": has_person(text)})
        flag = "" if said == TRUTH else "   <-- disagrees with the light"
        print(f"  [{name:8}] {text!r}{flag}")

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    wrong = [r["variant"] for r in rows if not r["agrees_with_truth"]]
    print(f"\nvariants disagreeing with the light: {wrong or 'none'}")
    print(f"phrase-based flip : {len({r['phrase_stance'] for r in rows}) > 1}"
          "   (blind to a bare colour word)")
    print(f"colour-aware flip : {len({r['colour_stance'] for r in rows}) > 1}")
    print(f"word-Jaccard      : {mean_pairwise_jaccard(texts):.3f}"
          "   (17-image average 0.39)")
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
