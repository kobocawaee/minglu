# Evaluation code

Companion to the report and to `datasets_package.zip`. Every number in Section 4
was produced by a script in here.

This is a snapshot of the working code, not a packaged library: the scripts were
written to be run from the project root, and they read and write the CSV files in
the data package. Paths inside them are relative to that root.

## Layout

Two folders, matching the repository. The scripts add the project root to
`sys.path` and import one another, so the tree has to stay as it is; run them from
the root, as `python code/eval_ptl.py`.

**`app/`** (18 files) is the deployed assistant of Section 3.6. `pipeline.py`
turns one frame into speech and is Algorithm 1; the router at the top of the same
file is Algorithm 3; `street_mode.py` with `light_classifier.py` is Algorithm 2.
`server.py` is the offline HTTPS server that lets a phone act as camera and
speaker, `quality.py` the frame-quality gate of Equations 1 to 3.

**`code/`** (41 files), grouped by what they do:

- *Measurement* (30): `eval_ptl.py` light discrimination,
  `eval_robustness.py` the perturbation study, `eval_chair.py` object
  hallucination, `eval_detector_crosscheck.py` whether a detector can catch a
  fabrication, `eval_guidedog.py` the external comparison, `eval_object_mode.py`,
  `eval_router.py` and `run_router_eval.py` routing, `eval_clipscore.py` and
  `eval_perturbation_magnitude.py` the metric studies of Section 4.5,
  `score_field_clips.py` and `eval_night_threshold.py` the field clips and the
  night guard.
- *Platform and NPU* : `test_smolvlm.py` and `test_directml.py` the CPU and iGPU
  benchmarks, `bench_*.py` the NPU timing, `quantize_*.py` and
  `export_vision_static.py` the ONNX work behind Section 4.2,
  `inspect_onnx.py` the node-placement analysis in Table 3.
- *Figures* (6): `plot_benchmark.py`, `plot_analysis.py` and
  `make_thesis_figures.py`, with `figstyle.py` holding the shared palette.
- *Utilities* (5): subset selection, caption generation, `blur_faces.py`
  (applied to the field frames before any model saw them), and the builder for
  the data package.

## Environment

Python 3.10. The main environment uses PyTorch 2.12 (CPU) with Transformers 5.12;
the DirectML measurements need a separate environment with PyTorch 2.4.1 and
Transformers 4.49, and the NPU work uses AMD Ryzen AI 1.7.1 with
onnxruntime-genai. Those three cannot share one environment, which is itself part
of the deployment-gap finding.

Model weights are downloaded from Hugging Face on first run. Nothing here is
fine-tuned.

## What is not included

- **Report-production scripts.** Roughly half the repository is python-docx code
  that edits the manuscript: alt text, equation typesetting, the acmart export,
  the bibliography. Available if useful, but it is not research code.
- **Images and video.** The 17 canonical images have mixed licences and the field
  recordings show identifiable pedestrians. See `DATASETS.md` in the data package.
- **LYTNetV2 itself.** `app/light_classifier.py` and `code/eval_lytnet.py` import
  the class and the weights from the ImVisible repository
  (<https://github.com/samuelyu2002/ImVisible>, MIT). Clone it into `external/`
  and those two will find it; we did not vendor someone else's model.
- **University paperwork scripts**, which contain personal administrative detail.

## Running order, if you want to reproduce a result

1. Fetch the public datasets listed in `DATASETS.md`.
2. `python code/eval_ptl.py` for the light benchmark, and so on. Each script
   prints its summary and writes a CSV whose name matches the one in the data
   package, so its output can be compared directly against ours.
