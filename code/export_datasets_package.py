# -*- coding: utf-8 -*-
"""Build the dataset package the advisor asked for: a description plus the data.

His request was "I need all the datasets you used. Can be accessed from the
internet or GitHub repository." Most of what we used is public, so the answer is
mostly links; what is ours is the ground truth we wrote and the outputs the models
produced, and those travel as CSV with no images in them, which is what makes the
package shareable at all.

Two things deliberately not in the zip:

  The 17 canonical images. Section 3.3 says they are referenced rather than
  reprinted, and the manifest rates eight of them high-risk (AP, iStock, PNGTree,
  Dreamstime, commercial catalogue). What goes instead is dataset_manifest.csv,
  which records each image's source and our hand-written ground truth, and
  safety_eval.csv, which holds every model output. Every number in Section 4.3 can
  be recomputed from those two without the pixels.

  The field recordings. They show identifiable pedestrians. The scored transcripts
  go in, the video does not.

Writes docs/thesis/DATASETS.md and docs/thesis/datasets_package.zip.

    python code/export_datasets_package.py
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs/thesis/DATASETS.md"
ZIP = ROOT / "docs/thesis/datasets_package.zip"

# (folder in the zip, path in the repo, what it is)
FILES = [
    ("ground_truth", "data/dataset_manifest.csv",
     "The 17 canonical images: filename, scenario, our hand-written description, "
     "key objects, ground truth, source type and licence risk."),
    ("ground_truth", "results/vizwiz_references.csv",
     "45 VizWiz images with their five human reference captions each."),
    ("ground_truth", "data/ptl/ptl_subset_val.csv",
     "The 30 red + 30 green PTL subset used for the VLM versus LYTNet comparison."),
    ("ground_truth", "data/ptl/ptl_subset150.csv",
     "The 150-image PTL subset used for the larger light-reading run."),
    ("ground_truth", "data/ptl/validation_file.csv",
     "PTL validation split (n = 645 after filtering) used to calibrate the "
     "light-announcement confidences. Original labels from ImVisible."),
    ("ground_truth", "results/router_eval_verified.csv",
     "The 60-image routing set with manually verified scenario labels. The "
     "keyword-based labels they replaced were 69% wrong."),

    ("model_outputs", "data/safety_eval.csv",
     "Every model output on the 17 canonical images, with the safety verdict and "
     "fabrication flag. Source for Metric A and Metric B, Section 4.3."),
    ("model_outputs", "results/ptl_light_256M.csv",
     "SmolVLM-256M on the PTL subset: what it said about each light."),
    ("model_outputs", "results/ptl_light_500M.csv", "SmolVLM-500M, same images."),
    ("model_outputs", "results/ptl_light_gemma.csv", "Gemma-3-4b, same images."),
    ("model_outputs", "results/lytnet_ptl.csv",
     "LYTNetV2 on the same images, which is what makes the comparison "
     "apples-to-apples."),
    ("model_outputs", "results/vizwiz_generated.csv",
     "Generated captions for the 45 VizWiz images under two prompts."),
    ("model_outputs", "results/guidedog_items.csv",
     "Every one of the 2,071 answers our systems gave on GuideDogQA, with the raw "
     "text and how it was parsed."),

    ("results", "data/benchmark.csv",
     "Platform benchmark: latency and peak memory per model per device. "
     "Section 4.1."),
    ("results", "results/gemma_npu_17.csv",
     "Gemma-3-4b on the NPU: time to first token and total latency. Section 4.2."),
    ("results", "results/robustness/robustness_raw.csv",
     "Every output for the six renderings of each of the 17 images."),
    ("results", "results/robustness/robustness_per_image.csv",
     "Per-image flip rate and word-level consistency. Section 4.3."),
    ("results", "results/perturbation_magnitude.csv",
     "SSIM, LPIPS and CLIP-cosine for each perturbation, showing how little the "
     "image changed when the output flipped."),
    ("results", "results/field_T90.csv",
     "Field clips scored at the deployed night guard, 170 frames with "
     "second-by-second ground truth. Section 4.4."),
    ("results", "results/field_T80.csv", "The same clips at the previous threshold."),
    ("results", "results/field_noguard.csv", "The same clips with no night guard."),
    ("results", "results/field_alwaysbox.csv",
     "The same clips requiring a detected light box in every frame, the "
     "no-threshold ablation of Table 4."),
    ("results", "results/clipscore.csv",
     "CLIPScore per output, and whether we had flagged it as a fabrication. "
     "Section 4.5."),
    ("results", "results/chair_eval.csv",
     "CHAIR object-hallucination labelling on 55 VizWiz images."),
    ("results", "results/chair_mentions.csv",
     "One row per object mentioned: grounded or fabricated, and whether the "
     "detector could see it."),
    ("results", "results/detector_crosscheck.csv",
     "Whether YOLOv8n and OCR could catch each fabrication on the canonical set."),
    ("results", "results/object_mode_eval.csv", "Object-mode agreement scoring."),
    ("results", "results/router_eval_v2.csv",
     "Router accuracy on verified labels, before and after the street bias."),
    ("results", "results/guidedog_eval.csv",
     "GuideDogQA summary: our three systems beside the eight published baselines."),
]

DOCTEXT = """# Datasets used in this report

Prepared for Assoc. Prof. Huang-Chia Shih, {date}.

Everything below is either public and linked, or attached as CSV in
`datasets_package.zip`. Two categories of image cannot be redistributed and are
explained in section 3; in both cases the ground truth and the model outputs are
attached instead, so every number in the report can still be recomputed and
checked.

## 1. Public datasets

| Dataset | Where to get it | Licence | What we used |
|---|---|---|---|
| **ImVisible / PTL** | <https://github.com/samuelyu2002/ImVisible> (images via the Google Drive links in the README) | MIT | 30 red + 30 green for the VLM versus LYTNet comparison; 150-image subset; validation split (n = 645) for threshold calibration |
| **VizWiz-Captions** | <https://vizwiz.org/tasks-and-datasets/image-captioning/> | CC BY 4.0 | 45 images for caption metrics; 55 images with five human captions each for the CHAIR measurement |
| **GuideDogQA** | <https://huggingface.co/datasets/kjunh/GuideDog> (`object` and `depth` configs) | CC BY-NC 4.0 | All 818 questions, evaluation only, not redistributed |

The GuideDog dataset is gated: the terms have to be accepted while signed in to
Hugging Face before the files can be downloaded.

## 2. Models, for completeness

All public and used as published, with no fine-tuning anywhere in this work.

| Model | Where |
|---|---|
| SmolVLM-256M / 500M Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct`, `HuggingFaceTB/SmolVLM-500M-Instruct` |
| Gemma-3-4b (NPU build) | AMD Ryzen AI / onnxruntime-genai |
| LYTNetV2 | <https://github.com/samuelyu2002/ImVisible> |
| YOLOv8n | Ultralytics |
| RapidOCR | <https://github.com/RapidAI/RapidOCR> |

## 3. What cannot be redistributed, and what is attached instead

**The 17 canonical images.** They were collected from the open web and their
licences are mixed: the manifest records one Associated Press photograph, one
iStock image, one PNGTree image, one with a visible Dreamstime watermark and two
commercial catalogue images. Section 3.3 of the report therefore states that these
images are referenced rather than reprinted. Attached instead:

- `ground_truth/dataset_manifest.csv`, which names every image, its source, its
  licence risk and the ground truth we wrote for it
- `model_outputs/safety_eval.csv`, which holds every output from all three models

Metric A, Metric B and the read-mode comparison can all be recomputed from those
two files. The manifest also identifies each image precisely enough to find it.

**The field recordings.** Six clips filmed at real crossings in Taiwan. They show
identifiable pedestrians, so the video is not distributed and only face-blurred
frames appear in the report. Attached instead are the scored transcripts,
`results/field_*.csv`, which carry the second-by-second ground truth we read off
the footage, what the assistant said in each frame, and whether it was correct.

## 4. Attached files

`datasets_package.zip` contains {n} CSV files in three folders. No images.

{index}

## 5. Where each result in the report comes from

| Report section | File |
|---|---|
| 4.1 Platform benchmark | `results/benchmark.csv` |
| 4.2 NPU deployment gap | `results/gemma_npu_17.csv` |
| 4.3 Metric A and Metric B | `model_outputs/safety_eval.csv` + `ground_truth/dataset_manifest.csv` |
| 4.3 Light discrimination (PTL) | `model_outputs/ptl_light_*.csv` + `model_outputs/lytnet_ptl.csv` |
| 4.3 Robustness, flip rate | `results/robustness_raw.csv`, `results/robustness_per_image.csv` |
| 4.3 Perturbation magnitude | `results/perturbation_magnitude.csv` |
| 4.4 Field validation, night guard | `results/field_T90.csv` and the three ablations |
| 4.5 Caption metrics | `model_outputs/vizwiz_generated.csv` + `ground_truth/vizwiz_references.csv` |
| 4.5 CLIPScore | `results/clipscore.csv` |
| 4.5 CHAIR | `results/chair_eval.csv`, `results/chair_mentions.csv` |
| 4.5 Detector cross-check | `results/detector_crosscheck.csv` |
| 4.5 Object mode | `results/object_mode_eval.csv` |
| 4.5 Mode routing | `results/router_eval_v2.csv` + `ground_truth/router_eval_verified.csv` |
| 4.5 GuideDogQA | `results/guidedog_eval.csv`, `model_outputs/guidedog_items.csv` |

## 6. Code

The evaluation scripts that produced every file above are available on request.
The repository is private because it still holds the canonical images and the
unblurred field footage described in section 3.
"""


def main():
    import datetime
    rows, missing, n = [], [], 0
    for folder, rel, desc in FILES:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        n += 1
        with open(src, encoding="utf-8-sig") as f:
            count = sum(1 for _ in f) - 1
        rows.append(f"- **`{folder}/{src.name}`** ({count} rows) {desc}")

    index = "\n".join(rows)
    DOC.write_text(DOCTEXT.format(date=datetime.date.today().strftime("%d %B %Y"),
                                  n=n, index=index), encoding="utf-8")

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(DOC, "DATASETS.md")
        for folder, rel, _ in FILES:
            src = ROOT / rel
            if src.exists():
                z.write(src, f"{folder}/{src.name}")

    size = ZIP.stat().st_size / 1024
    print(f"  {DOC.relative_to(ROOT)}")
    print(f"  {ZIP.relative_to(ROOT)}  ({n} CSV files, {size:.0f} KB)")
    if missing:
        print("  MISSING, not included:", *missing, sep="\n    ")

    # nothing in the package should be an image or a video
    with zipfile.ZipFile(ZIP) as z:
        bad = [i for i in z.namelist()
               if not i.lower().endswith((".csv", ".md"))]
    print("  contents check:", "only CSV and markdown" if not bad else f"UNEXPECTED {bad}")


if __name__ == "__main__":
    main()
