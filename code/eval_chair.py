# -*- coding: utf-8 -*-
"""Object hallucination at a larger n, and whether a detector can catch it.

Section 4.3 counts safety-critical fabrication on 17 hand-scored images: 4 of 17,
the same for every model size. Twelve labelled fabrications is a thin base for the
follow-up question of Section 4.5, whether the fabrications are catchable, so this
measures a *different* quantity on more images and reports it separately.

The method is CHAIR (Rohrbach et al. 2018): fix a vocabulary of concrete objects,
and count an object as fabricated when the model names it and no human caption of
that image does. VizWiz gives five independent captions per image, and 55 images
are held locally, so the vocabulary is the only thing left to choose. It is the 80
COCO classes, taken from the detector itself, plus the synonym map below. That
choice is deliberate: every object in the vocabulary is one YOLOv8n can be asked
about, so the same run answers the cross-check question without a second labelling
pass.

Two numbers come out, in CHAIR's own terms:

    CHAIR_i   fabricated mentions / all object mentions
    CHAIR_s   outputs containing at least one fabrication / all outputs

and then, over every object mention, whether YOLOv8n supports it:

    caught        fabricated mentions the detector also fails to see
    false alarm   grounded mentions the detector fails to see

What this is not. It is not Metric B and cannot replace 4/17. Metric B asks
whether an output would mislead a blind user about their safety, judged by a
human; this asks whether a noun appears in five other people's sentences. A
caption is not an inventory, so an object really in the picture that all five
annotators left out is scored as fabricated here. That biases CHAIR_i upward and
is the standard caveat on caption-only CHAIR. VizWiz images are also held objects
and indoor scenes, so nothing here transfers to the street results of Section 4.4.

    python code/eval_chair.py            # both model sizes, 55 images
    python code/eval_chair.py --models 500m

Writes results/chair_eval.csv and results/chair_mentions.csv.
"""
import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

from app import config, detect
from app.backends import get_backend

IMAGES = ROOT / "data/vizwiz/images"
ANNOTATIONS = ROOT / "data/vizwiz/annotations/val.json"
OUT = ROOT / "results/chair_eval.csv"
OUT_MENTIONS = ROOT / "results/chair_mentions.csv"

MODELS = {
    "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
}

# Words that mean a COCO class without being its name. Kept deliberately short:
# every entry widens what counts as "the human said it too", so a missing synonym
# inflates the fabrication count and a wrong one hides a real fabrication.
SYNONYMS = {
    "person": ["man", "woman", "men", "women", "boy", "girl", "child", "children",
               "kid", "people", "guy", "lady", "pedestrian", "someone", "player",
               "human", "worker"],
    "car": ["taxi", "cab", "suv", "sedan", "van", "jeep", "automobile"],
    "truck": ["pickup", "lorry"],
    "motorcycle": ["motorbike", "moped", "scooter"],
    "bicycle": ["bike"],
    "airplane": ["plane", "aircraft", "jet"],
    "couch": ["sofa"],
    "tv": ["television", "monitor"],       # not "screen": a phone has one too
    "cell phone": ["phone", "smartphone", "iphone", "mobile phone"],
    "dining table": ["table", "desk"],
    "potted plant": ["plant", "houseplant", "flowerpot"],
    "cup": ["mug"],
    "wine glass": ["wineglass"],
    "donut": ["doughnut"],
    "laptop": ["computer"],
    "remote": ["remote control"],
    "refrigerator": ["fridge"],
    "handbag": ["purse"],
    "backpack": ["rucksack"],
    "teddy bear": ["teddy"],
    "traffic light": ["stoplight", "traffic signal"],
    "sink": ["basin", "washbasin"],
    "toilet": ["lavatory"],
    "dog": ["puppy"],
    "cat": ["kitten"],
    "bed": ["mattress"],
    "sports ball": ["ball"],
}


# One COCO class is also a colour, and both models list colours. "orange" counts
# as the fruit only in the plural or after a determiner with no noun following it,
# so "an orange." is a mention and "orange, purple, pink" and "orange juice" are not.
CUSTOM = {
    "orange": re.compile(r"\boranges\b|\b(?:an|the|one|this|that)\s+orange\b(?!\s+[a-z])",
                         re.I),
}


# CHAIR counts any occurrence of an object word as a claim to have seen it. Our
# prompt invites the opposite ("If unsure, say you are not sure"), so a model that
# obeys it writes "cannot tell if someone else is nearby or not" and CHAIR scores
# that as inventing a person. Sentences carrying one of these markers are reported
# separately rather than silently dropped, because deciding they do not count is a
# judgement about our own prompt, not part of the published metric.
HEDGE = re.compile(r"\b(?:not sure|unsure|cannot|can not|can't|unclear|not clear|"
                   r"n't clear|hard to tell|difficult to tell|don't see|do not see|"
                   r"no one|nobody|or not)\b", re.I)


def hedged(text, obj, vocab):
    """True when every sentence naming this object also hedges or denies it."""
    hits = [s for s in re.split(r"(?<=[.!?])\s+", text) if vocab[obj].search(s)]
    return bool(hits) and all(HEDGE.search(s) for s in hits)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(0.0, c - m), 100 * min(1.0, c + m)


def build_vocabulary():
    """COCO class names, from the detector, each with its surface forms."""
    from ultralytics import YOLO
    names = list(YOLO("yolov8n.pt").names.values())
    vocab = {}
    for cls in names:
        forms = [cls] + SYNONYMS.get(cls, [])
        pats = []
        for f in forms:
            # allow the regular plurals; "bus" needs "buses", "person" needs "people"
            pats.append(re.escape(f) + r"(?:e?s)?")
        vocab[cls] = CUSTOM.get(cls, re.compile(r"\b(?:" + "|".join(pats) + r")\b", re.I))
    return vocab


def mentioned(text, vocab):
    """Which vocabulary objects this sentence names."""
    return {cls for cls, pat in vocab.items() if pat.search(text)}


def load_references():
    d = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    by_id = {i["id"]: i["file_name"] for i in d["images"]}
    refs = {}
    for a in d["annotations"]:
        refs.setdefault(by_id[a["image_id"]], []).append(a["caption"])
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS),
                    help="which sizes to run (default: both)")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the summary from the saved CSVs, without the models")
    args = ap.parse_args()
    sizes = [s for s in MODELS if s.lower() in {m.lower() for m in args.models}]

    if args.report_only:
        rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
        mentions = list(csv.DictReader(open(OUT_MENTIONS, encoding="utf-8-sig")))
        for r in rows:
            r["n_mentions"] = int(r["n_mentions"])
            r["n_fabricated"] = int(r["n_fabricated"])
        for m in mentions:
            m["grounded"] = m["grounded"] == "True"
            m["detector_sees"] = m["detector_sees"] == "True"
        report(rows, mentions, [s for s in sizes if any(r["model"] == s for r in rows)])
        return

    refs = load_references()
    files = sorted(p.name for p in IMAGES.glob("*.jpg") if p.name in refs)
    print(f"images with five human captions: {len(files)}")

    vocab = build_vocabulary()
    print(f"vocabulary: {len(vocab)} COCO classes "
          f"({sum(len(v) for v in SYNONYMS.values())} synonyms added)")

    # --- what the detector sees, once per image (the cross-check's evidence)
    print("\nrunning YOLOv8n over the set ...")
    seen = {}
    for name in files:
        dets, _ = detect.detect(Image.open(IMAGES / name).convert("RGB"),
                                conf=0.35, classes="all")
        seen[name] = {d["cls"] for d in dets}
    empty = sum(1 for v in seen.values() if not v)
    print(f"  detector finds nothing at all in {empty}/{len(files)} images")

    # --- what the humans said
    human = {name: set().union(*(mentioned(c, vocab) for c in refs[name]))
             for name in files}

    prompt = config.get_prompt("surrounding")
    mx = config.get_max_tokens("surrounding")
    rows, mentions = [], []

    for size in sizes:
        print(f"\nloading SmolVLM-{size} ...")
        backend = get_backend("smolvlm", model_id=MODELS[size], device="cpu",
                              gen_params=config.GEN_PARAMS)
        backend.load(); backend.warmup()
        t0 = time.time()
        for i, name in enumerate(files, 1):
            out = backend.describe(Image.open(IMAGES / name).convert("RGB"),
                                   prompt, max_new_tokens=mx).strip()
            said = mentioned(out, vocab)
            fabricated = said - human[name]
            rows.append({
                "model": size, "filename": name,
                "n_mentions": len(said), "n_fabricated": len(fabricated),
                "fabricated": " ".join(sorted(fabricated)),
                "said": " ".join(sorted(said)),
                "human_objects": " ".join(sorted(human[name])),
                "yolo_sees": " ".join(sorted(seen[name])),
                "output": out,
            })
            for obj in sorted(said):
                mentions.append({
                    "model": size, "filename": name, "object": obj,
                    "grounded": obj in human[name],
                    "detector_sees": obj in seen[name],
                })
            if i % 10 == 0:
                print(f"  {i}/{len(files)}  ({(time.time()-t0)/i:.1f}s/image)")
        del backend

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT_MENTIONS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mentions[0])); w.writeheader(); w.writerows(mentions)

    report(rows, mentions, sizes)
    print(f"\nsaved {OUT.relative_to(ROOT)} and {OUT_MENTIONS.relative_to(ROOT)}")


def report(rows, mentions, sizes):
    vocab = build_vocabulary()
    print("\n" + "=" * 74)
    print("object hallucination (CHAIR), five human captions as the reference")
    print("=" * 74)
    for size in sizes:
        r = [x for x in rows if x["model"] == size]
        m = sum(x["n_mentions"] for x in r)
        fab = sum(x["n_fabricated"] for x in r)
        imgs = sum(1 for x in r if x["n_fabricated"])
        lo, hi = wilson(fab, m)
        slo, shi = wilson(imgs, len(r))
        print(f"\n  SmolVLM-{size}   ({m} object mentions over {len(r)} images)")
        print(f"    CHAIR_i  {fab:3d}/{m:<4d} = {100*fab/m:.0f}%  95% CI [{lo:.0f}, {hi:.0f}]")
        print(f"    CHAIR_s  {imgs:3d}/{len(r):<4d} = {100*imgs/len(r):.0f}%  "
              f"95% CI [{slo:.0f}, {shi:.0f}]")
        # counted from the per-mention rows: half the COCO names are two words,
        # so splitting the joined column on spaces invents classes ("traffic", "light")
        top = Counter(x["object"] for x in mentions
                      if x["model"] == size and not x["grounded"])
        print("    most fabricated: "
              + ", ".join(f"{o} x{n}" for o, n in top.most_common(5)))
        fab_of = {}
        for x in mentions:
            if x["model"] == size and not x["grounded"]:
                fab_of.setdefault(x["filename"], []).append(x["object"])
        hedge = [(x["filename"], o) for x in r
                 for o in fab_of.get(x["filename"], [])
                 if hedged(x["output"], o, vocab)]
        if hedge:
            print(f"    of these, {len(hedge)} sit in a hedged or negated sentence "
                  f"and are arguably not claims: "
                  f"{', '.join(o for _, o in hedge[:4])}")
            print(f"    excluding them: CHAIR_i {fab-len(hedge)}/{m-len(hedge)} = "
                  f"{100*(fab-len(hedge))/(m-len(hedge)):.0f}%")

    print("\n" + "=" * 74)
    print("can YOLOv8n catch them? a mention is flagged when the detector cannot see it")
    print("=" * 74)
    blind = {x["filename"] for x in rows if not x["yolo_sees"].strip()}
    print(f"\n  the detector reports nothing at all in {len(blind)} of "
          f"{len({x['filename'] for x in rows})} images")
    for size in sizes:
        m = [x for x in mentions if x["model"] == size]
        fab = [x for x in m if not x["grounded"]]
        good = [x for x in m if x["grounded"]]
        caught = [x for x in fab if not x["detector_sees"]]
        alarm = [x for x in good if not x["detector_sees"]]
        lo, hi = wilson(len(caught), len(fab))
        alo, ahi = wilson(len(alarm), len(good))
        print(f"\n  SmolVLM-{size}")
        print(f"    caught       {len(caught):3d}/{len(fab):<4d} = "
              f"{100*len(caught)/len(fab):.0f}%  95% CI [{lo:.0f}, {hi:.0f}]")
        print(f"    false alarm  {len(alarm):3d}/{len(good):<4d} = "
              f"{100*len(alarm)/len(good):.0f}%  95% CI [{alo:.0f}, {ahi:.0f}]")
        for label, sub in (("in the images it sees nothing in", True),
                           ("in the images it detects something in", False)):
            g = [x for x in good if (x["filename"] in blind) == sub]
            a = [x for x in g if not x["detector_sees"]]
            if g:
                print(f"      {label:38} {len(a):3d}/{len(g):<3d} = "
                      f"{100*len(a)/len(g):.0f}%")


if __name__ == "__main__":
    main()
