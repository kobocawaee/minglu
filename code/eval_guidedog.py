# -*- coding: utf-8 -*-
"""Our models on GuideDogQA: a comparison against published work, same data, same metric.

The advisor asked for a real comparison with other research rather than a table of
features. Kim et al. (GuideDog, ACL 2026), already cited in this report, release
both the benchmark and the numbers eight systems score on it, so our models can be
run on their data and the result dropped into their table.

GuideDogQA protocol, from their paper:

  object   435 four-way questions. `choices` already carries the letters
           ("A. Curb", "B. Fire hydrant", ...) and `answer` is the letter, not
           the label. Chance 25%, and the gold letter is evenly spread
           (105/107/118/105), so there is no position to exploit.
  depth    383 items. `choices` is two instance names ("Motorcycle_2",
           "Person_0"); the model must identify which is closer AND which is
           farther, and the item counts only if both are right. That is why their
           table lists chance at 25% for a binary task: two binary answers.

Two properties of the data that the paper's summary does not make obvious, and
that change what can be claimed here:

  The label vocabulary is not COCO. It is 79 BLV-guidance labels, of which only
  ten are COCO classes (Person, Car, Bus, Bicycle, Motorcycle, Bench, Chair, Dog,
  Fire hydrant, Stop sign). Curb, Sidewalk, Lamp Post, Barrier Post, Road
  Shoulder and Closed Sidewalk are the substance of the benchmark and no
  general-purpose detector names them. So the detector row is not the easy win it
  looked like: it can even attempt only the 292 of 435 questions that mention at
  least one COCO-nameable object, and the correct answer is a COCO class in only
  237.

  A third of the depth pairs, 127 of 383, name two instances of the same class.
  Nothing in the label alone separates "Motorcycle_2" from "Motorcycle_0", so
  those items are close to unanswerable from names. The overall figure is
  reported as the paper defines it, and the same-class subset is reported beside
  it, because a reader should be able to see whether that third is carrying the
  result.

Their published baselines (Table 4), depth / object:

  Random chance 25.0 / 25.0 · Cambrian-1 24.3 / 82.3 · Molmo 28.5 / 34.0
  LLaVA-1.6 30.0 / 80.9 · LLaVA-OneVision 32.4 / 87.4 · Qwen2.5-VL 22.2 / 85.7
  Qwen2.5-VL fine-tuned 41.5 / 83.9 · Gemini 2.0 Flash 53.0 / 65.7
  GPT-4o 67.1 / 74.7

They do not publish the prompt text, so ours is defined here and printed with the
results. Systems evaluated:

  256M, 500M   the deployed VLMs, prompted plainly, which is what their baselines
               are. This is the apples-to-apples row.
  detector     YOLOv8n plus the rule the assistant already uses, on the object
               subset only. Our central claim is that a small dedicated detector
               beats a tiny VLM on exactly this kind of question, and this tests
               it on someone else's data.

  ⚠ The detector row carries a caveat, and it is the opposite of the one first
  expected: far from being handed a COCO benchmark, YOLOv8n cannot name 69 of the
  79 labels at all. Its coverage is reported with its score, and it abstains
  rather than guessing when no choice maps to a class it knows. On depth the
  assistant has no component at all, so it is not run: an absent capability
  should be reported as absent rather than as a score near chance.

The dataset is gated (CC BY-NC 4.0): accept the terms while logged in at
huggingface.co/datasets/kjunh/GuideDog, then `huggingface-cli login`.

    python code/eval_guidedog.py --peek          # schema and two examples, no models
    python code/eval_guidedog.py --subset object
    python code/eval_guidedog.py                 # both subsets, both models

Writes results/guidedog_eval.csv and results/guidedog_items.csv.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import config, detect
from app.backends import get_backend

OUT = ROOT / "results/guidedog_eval.csv"
ITEMS = ROOT / "results/guidedog_items.csv"
REPO = "kjunh/GuideDog"

MODELS = {
    "256M": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "500M": "HuggingFaceTB/SmolVLM-500M-Instruct",
}

# depth / object, as printed in their Table 4
PUBLISHED = [
    ("Random chance", 25.0, 25.0), ("Cambrian-1", 24.3, 82.3),
    ("Molmo", 28.5, 34.0), ("LLaVA-1.6", 30.0, 80.9),
    ("LLaVA-OneVision", 32.4, 87.4), ("Qwen2.5-VL", 22.2, 85.7),
    ("Qwen2.5-VL (fine-tuned)", 41.5, 83.9), ("Gemini 2.0 Flash", 53.0, 65.7),
    ("GPT-4o", 67.1, 74.7),
]

LETTERS = "ABCDEFGH"
COCO_NAMES = set()          # filled from YOLOv8n the first time the detector runs
RAW = []                    # every raw model answer, in the order asked

OBJECT_PROMPT = ("I am visually impaired. Exactly one of these objects is in the "
                 "image. Which one? Answer with the letter only.\n{options}\nAnswer:")
DEPTH_PROMPT = ("I am visually impaired. Looking at the image, which of these is "
                "{which} to the camera? Answer with the letter only.\n{options}\nAnswer:")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * max(0.0, c - m), 100 * min(1.0, c + m)


def load(subset):
    from datasets import load_dataset
    ds = load_dataset(REPO, subset, split="train")
    return ds


def options_block(choices):
    """The object subset already ships lettered choices; the depth subset does
    not, and its instance names carry an underscore index that has to survive,
    since a third of the pairs are two instances of one class."""
    if choices and re.match(r"^[A-H]\.\s", choices[0]):
        return "\n".join(choices)
    return "\n".join(f"{LETTERS[i]}. {c.replace('_', ' ')}" for i, c in enumerate(choices))


def label_of(choice):
    """"B. Fire hydrant" -> "Fire hydrant";  "Motorcycle_2" -> "Motorcycle 2"."""
    return re.sub(r"^[A-H]\.\s*", "", choice).replace("_", " ").strip()


def pick(answer, choices):
    """Which choice the model went for: a bare letter first, then the choice text.

    Lenient on purpose. A tiny model that answers "a dog" instead of "B" has
    identified the object, and scoring that wrong would measure instruction
    following rather than perception. Every parse outcome is written to the item
    CSV so the leniency can be audited.
    """
    txt = (answer or "").strip()

    # Tried in this order. The first version of this anchored at the start of the
    # string, so "Answer: D" could not be read at all: the regex reached the "A"
    # of "Answer" and stopped. A fifth of the smoke test was scored wrong for
    # answers that were in fact correct, which is why raw text is now in the CSV.
    for pattern, how in (
            (r"(?:answer|option|choice)\s*(?:is)?\s*[:\-]?\s*([A-H])\b", "labelled"),
            (r"\b([A-H])\s*[\.\):]", "letter"),
            (r"^\s*([A-H])\s*$", "bare letter"),
    ):
        m = re.search(pattern, txt, re.I)
        if m:
            i = LETTERS.index(m.group(1).upper())
            if i < len(choices):
                return i, how

    low = txt.lower()
    hits = [i for i, c in enumerate(choices)
            if re.search(rf"\b{re.escape(label_of(c).lower())}\b", low)]
    if len(hits) == 1:
        return hits[0], "text"
    if len(hits) > 1:
        return None, f"ambiguous:{len(hits)}"
    return None, "unparsed"


def detector_pick(image, choices):
    """The assistant's own machinery: whichever named class YOLOv8n sees best.

    Abstains when no choice names a class YOLOv8n knows, which is 143 of the 435
    object questions. An abstention is scored wrong, and counted separately, so
    the row cannot be read as if the detector had answered."""
    known = {i: label_of(c).lower() for i, c in enumerate(choices)}
    known = {i: n for i, n in known.items() if n in COCO_NAMES}
    if not known:
        return None, "out of vocabulary"
    dets, _ = detect.detect(image, conf=0.25, classes="all")
    best, conf = None, 0.0
    for i, name in known.items():
        for d in dets:
            if d["cls"].lower() == name and d["conf"] > conf:
                best, conf = i, d["conf"]
    return (best, "detected" if best is not None else "nothing detected")


def run_object(ds, ask, tag, rows):
    correct = 0
    for r in ds:
        choices = list(r["choices"])
        gold = LETTERS.index(r["answer"]) if r["answer"] in LETTERS else None
        got, how = ask(r["image"], choices, None)
        ok = gold is not None and got == gold
        correct += ok
        rows.append({"system": tag, "subset": "object", "id": r.get("id", ""),
                     "gold": choices[gold] if gold is not None else r["answer"],
                     "picked": choices[got] if got is not None else "",
                     "parse": how, "correct": ok, "note": "",
                     "q_right": int(ok), "same_pick": "",
                     "raw": RAW.pop() if RAW else ""})
    return correct, len(ds)


def run_depth(ds, ask, tag, rows):
    """Both halves must be right, which is their rule and their chance level."""
    correct = 0
    for r in ds:
        choices = list(r["choices"])
        same = len({re.sub(r"_\d+$", "", c) for c in choices}) == 1
        both, picks, raw_pair, got_both, right = True, [], [], [], 0
        for which, gold_name in (("closer", r["closer"]), ("farther", r["farther"])):
            gold = choices.index(gold_name) if gold_name in choices else None
            got, how = ask(r["image"], choices, which)
            got_both.append(got)
            right += int(gold is not None and got == gold)
            picks.append(f"{which}={label_of(choices[got]) if got is not None else '?'}({how})")
            raw_pair.append(RAW.pop() if RAW else "")
            both = both and gold is not None and got == gold
        correct += both
        rows.append({"system": tag, "subset": "depth", "id": r.get("id", ""),
                     "gold": f"closer={r['closer']} farther={r['farther']}",
                     "picked": " ".join(picks), "parse": "", "correct": both,
                     "note": "same-class pair" if same else "",
                     "q_right": right, "same_pick": int(len(set(got_both)) == 1),
                     "raw": " | ".join(raw_pair)})
    return correct, len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peek", action="store_true",
                    help="print the schema and two rows, then stop")
    ap.add_argument("--subset", choices=["object", "depth"], action="append")
    ap.add_argument("--models", nargs="+", default=list(MODELS) + ["detector"])
    ap.add_argument("--limit", type=int, help="first N items only, for a smoke test")
    args = ap.parse_args()
    subsets = args.subset or ["object", "depth"]

    data = {}
    for s in subsets:
        data[s] = load(s)
        print(f"{s}: {len(data[s])} items")

    if args.peek:
        for s, ds in data.items():
            print(f"\n=== {s} ===")
            print("features:", {k: str(v) for k, v in ds.features.items()})
            for r in ds.select(range(min(2, len(ds)))):
                for k, v in r.items():
                    if k == "image":
                        print(f"  image: {v.size} {v.mode}")
                    else:
                        print(f"  {k}: {str(v)[:110]}")
                print("  ---")
        print("\nprompts that would be used:")
        print(" ", OBJECT_PROMPT.replace("\n", " | "))
        print(" ", DEPTH_PROMPT.replace("\n", " | "))
        return

    if args.limit:
        data = {s: ds.select(range(min(args.limit, len(ds)))) for s, ds in data.items()}

    rows, summary = [], []

    for name in args.models:
        if name == "detector":
            if "object" not in data:
                continue
            print("\nrunning YOLOv8n + rule on the object subset ...")
            from ultralytics import YOLO
            COCO_NAMES.update(n.lower() for n in YOLO("yolov8n.pt").names.values())
            attempt = sum(1 for r in data["object"]
                          if any(label_of(c).lower() in COCO_NAMES for c in r["choices"]))
            print(f"  questions with a choice YOLOv8n can name: "
                  f"{attempt}/{len(data['object'])}")

            def ask(image, choices, which):
                return detector_pick(image.convert("RGB"), choices)

            k, n = run_object(data["object"], ask, "detector", rows)
            summary.append(("ours: YOLOv8n + rule", None, (k, n)))
            print(f"  object {k}/{n}")
            continue

        print(f"\nloading SmolVLM-{name} ...")
        backend = get_backend("smolvlm", model_id=MODELS[name], device="cpu",
                              gen_params=config.GEN_PARAMS)
        backend.load(); backend.warmup()

        def ask(image, choices, which, backend=backend):
            tmpl = OBJECT_PROMPT if which is None else DEPTH_PROMPT
            prompt = tmpl.format(options=options_block(choices), which=which or "")
            out = backend.describe(image.convert("RGB"), prompt, max_new_tokens=12)
            i, how = pick(out, choices)
            RAW.append(out.strip())
            return i, how

        got = {}
        for s in subsets:
            runner = run_object if s == "object" else run_depth
            k, n = runner(data[s], ask, name, rows)
            got[s] = (k, n)
            print(f"  {s} {k}/{n} = {100*k/n:.1f}%")
        summary.append((f"ours: SmolVLM-{name}", got.get("depth"), got.get("object")))
        del backend

    with open(ITEMS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 78)
    print("GuideDogQA accuracy (%), our rows measured here, the rest as published")
    print("=" * 78)
    print(f"  {'system':26} {'depth':>18}  {'object':>18}")
    for name, d, o in PUBLISHED:
        print(f"  {name:26} {d:>17.1f}  {o:>17.1f}")
    print("  " + "-" * 66)
    out_rows = []
    for name, d, o in summary:
        cells = []
        for pair in (d, o):
            if pair is None:
                cells.append(f"{'not applicable':>18}")
            else:
                k, n = pair
                lo, hi = wilson(k, n)
                cells.append(f"{100*k/n:>8.1f} [{lo:.0f},{hi:.0f}]")
        print(f"  {name:26} {cells[0]}  {cells[1]}")
        out_rows.append({"system": name,
                         "depth": "" if d is None else f"{100*d[0]/d[1]:.1f}",
                         "depth_n": "" if d is None else d[1],
                         "object": "" if o is None else f"{100*o[0]/o[1]:.1f}",
                         "object_n": "" if o is None else o[1]})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)

    # a third of the depth pairs name two instances of one class; show whether
    # they are carrying the number rather than leaving a reader to wonder
    for size in {r["system"] for r in rows if r["subset"] == "depth"}:
        sub = [r for r in rows if r["system"] == size and r["subset"] == "depth"]
        same = [r for r in sub if r["note"] == "same-class pair"]
        diff = [r for r in sub if r["note"] != "same-class pair"]
        if same and diff:
            print(f"\n  depth split for {size}: "
                  f"same-class pairs {sum(r['correct'] for r in same)}/{len(same)}"
                  f" = {100*sum(r['correct'] for r in same)/len(same):.0f}%, "
                  f"different classes {sum(r['correct'] for r in diff)}/{len(diff)}"
                  f" = {100*sum(r['correct'] for r in diff)/len(diff):.0f}%")

    for size in {r["system"] for r in rows if r["subset"] == "depth"}:
        sub = [r for r in rows if r["system"] == size and r["subset"] == "depth"]
        q = sum(r["q_right"] for r in sub)
        const = sum(r["same_pick"] for r in sub)
        lo, hi = wilson(q, 2 * len(sub))
        print(f"  per-question for {size}, chance 50%: {q}/{2*len(sub)} = "
              f"{100*q/(2*len(sub)):.0f}% [{lo:.0f},{hi:.0f}]  ·  gave the same answer "
              f"to closer and farther in {const}/{len(sub)} items "
              f"({100*const/len(sub):.0f}%), which the paired rule scores as wrong")

    unparsed = sum(1 for r in rows if r["parse"].startswith(("unparsed", "ambiguous")))
    oov = sum(1 for r in rows if r["parse"] == "out of vocabulary")
    print(f"\nunparsable or ambiguous answers: {unparsed}/{len(rows)} "
          f"(scored wrong; see {ITEMS.name} to audit the leniency)")
    if oov:
        print(f"detector abstentions, no choice in its vocabulary: {oov}")
    print(f"saved {OUT.relative_to(ROOT)} and {ITEMS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
