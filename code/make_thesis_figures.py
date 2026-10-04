"""
make_thesis_figures.py — 論文主要發現的圖（fig7-9）
====================================================================
fig7_ptl_lytnet.png       : 長條圖 — VLM 讀不出號誌（分辨能力≈0）vs LYTNet 90 個百分點
fig8_robustness_grid.png  : 同一張圖的 6 種變化 — 只有 2/6 出現提醒（發現 4）
fig9_hybrid_architecture.png : 程式的混合式架構（方法章節）

採用的原則（資料視覺化方法）：依資料用途選圖形、依意義選顏色（紅 = 誤報可通行，危險），
每根長條直接標數值（解決對比度警告）、不用雙 Y 軸、配色已驗證

執行（vlm_research 環境）：python code/make_thesis_figures.py
"""

import sys, csv, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "code"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 配色（取自簡報 — 已用資料視覺化驗證工具檢查；灰色只用於註記）
NAVY, TEAL, CYAN, GREEN = "#0E2438", "#12A594", "#2AA6CE", "#1AA35A"
DANGER, MUTE, LIGHT = "#C2410C", "#51617A", "#F1F4F8"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150,
                     "axes.edgecolor": "#DCE3EC", "axes.linewidth": 0.8})


# ---------------------------------------------------------------- fig 7
def fig7_ptl():
    # The VLM bars are the says-safe rate from eval_ptl.stance(), not a test for
    # the word "green": an output can convey safety without naming a colour, and
    # one naming both is MIXED. The LYTNetV2 bars are its predicted class, which
    # the footnote states. Section 3.8 defines the metric the same way.
    models = ["SmolVLM-256M", "SmolVLM-500M", "Gemma-3-4b", "LYTNetV2\n(dedicated CNN)"]
    on_green = [57, 73, 20, 93.3]   # conveys "safe" when the light IS green (want high)
    on_red = [67, 77, 20, 3.3]      # conveys "safe" when the light IS red (want low)
    disc = [-0.10, -0.04, 0.00, +0.90]        # = P(safe | green) − P(safe | red)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(models)); w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], on_green, w, color=TEAL,
                label="conveys it is safe when the light is GREEN (want high)")
    b2 = ax.bar([i + w / 2 for i in x], on_red, w, color=DANGER,
                label="conveys it is safe when the light is RED (dangerous false clearance)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                    f"{b.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=9, color=NAVY)
    # 區分 VLM 和專用 CNN 的區塊
    ax.axvline(2.5, color=MUTE, lw=0.8, ls=(0, (4, 4)))
    ax.text(1.0, 112, "general VLMs: discrimination ≈ 0", ha="center",
            fontsize=9, color=MUTE, style="italic")
    ax.text(3.0, 112, "ours (hybrid)", ha="center", fontsize=9, color=TEAL, style="italic")

    # 分辨能力 = 兩根長條之間的差距 → 直接在圖上畫成「標示差距的箭頭」
    # （rev. 07-22，依指導教授意見：軸下方的文字超出邊界，而且看不懂）
    for i, d in enumerate(disc[:3]):                       # 3 個 VLM：幾乎沒有差距
        top = max(on_green[i], on_red[i])
        ax.annotate("", xy=(i - w / 2, top + 9), xytext=(i + w / 2, top + 9),
                    arrowprops=dict(arrowstyle="<->", color=MUTE, lw=1.1))
        ax.text(i, top + 11, f"gap {int(round(d * 100)):+d} pp".replace("-", "−"),
                ha="center", fontsize=9, color=MUTE)

    ax.annotate("", xy=(3.46, on_green[3]), xytext=(3.46, on_red[3]),   # LYTNet：有明顯差距
                arrowprops=dict(arrowstyle="<->", color=GREEN, lw=2.4))
    ax.text(3.56, (on_green[3] + on_red[3]) / 2, "+90 pp", ha="left", va="center",
            fontsize=12, fontweight="bold", color=GREEN)
    ax.text(3.56, (on_green[3] + on_red[3]) / 2 - 9, "discrimination", ha="left",
            va="center", fontsize=8.5, color=GREEN)

    ax.set_xticks(list(x)); ax.set_xticklabels(models, fontsize=9.5)
    ax.set_xlim(-0.6, 4.25)
    ax.set_ylim(0, 108); ax.set_ylabel("% of images", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#EDF1F6"); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, bbox_to_anchor=(0.01, 1.0))
    ax.set_title("Reading the pedestrian light: every VLM fails, a 14 MB dedicated CNN does not",
                 fontsize=11.5, color=NAVY, pad=26, loc="left")
    fig.text(0.01, 0.012, "Same PTL test set (30 red + 30 green; Gemma 20+20) · "
             "discrimination = P(safe | green) − P(safe | red) · "
             "for LYTNetV2 the bars are its predicted class",
             fontsize=7.8, color=MUTE)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "results/fig7_ptl_lytnet.png", bbox_inches="tight")
    plt.close(fig); print("fig7 done")


# ---------------------------------------------------------------- fig 8
def fig8_robustness():
    """Six variants of one frame from our own footage.

    Rewritten 2026-07-27. This used to reprint crosswalk_redlight.webp, which
    data/dataset_manifest.csv marks as carrying a visible Dreamstime watermark,
    against Section 3.5's statement that the canonical images are referenced
    rather than reprinted. The frame here is ours, filmed at a Taipei crossing,
    with bystanders' faces blurred by code/blur_faces.py, and it is the same file
    the model was run on (code/eval_own_frame_robustness.py).

    It is also a plainer illustration. The model answers this scene in one word,
    so the failure is legible at a glance: five variants say Red, and an 8% centre
    crop says Green on a red light.
    """
    from PIL import Image
    from eval_robustness import make_variants

    frame = ROOT / "data/field_frames/crossing1_t24_red_blur.png"
    base = Image.open(frame).convert("RGB")
    variants = make_variants(base)
    rows = {r["variant"]: r for r in csv.DictReader(
        open(ROOT / "results/own_frame_variants.csv", encoding="utf-8-sig"))}

    pretty = {"orig": "original", "rot+4": "rotate +4°", "rot-4": "rotate −4°",
              "bright+": "brightness +15%", "bright-": "brightness −15%",
              "crop92": "centre crop 92%"}
    fig, axes = plt.subplots(1, 6, figsize=(13.5, 5.0))
    for ax, name in zip(axes.flat, variants):
        r = rows[name]
        wrong = r["agrees_with_truth"].lower() != "true"
        ax.imshow(variants[name]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(DANGER if wrong else "#B9C4D0")
            s.set_linewidth(3 if wrong else 1.0)
        ax.set_title(pretty[name], fontsize=9, color=MUTE, pad=6)
        ax.set_xlabel(f"“{r['text']}”" + ("\nwrong colour" if wrong else ""),
                      fontsize=11 if wrong else 10.5,
                      color=DANGER if wrong else NAVY,
                      fontweight="bold" if wrong else "normal", labelpad=8)

    fig.suptitle("One red light, six variants that all pass the quality gate: "
                 "an 8% centre crop turns it green", fontsize=12, color=NAVY, y=0.99)
    fig.text(0.01, 0.015, "SmolVLM-500M, greedy decoding · our own footage, faces "
             "blurred · source: results/own_frame_variants.csv",
             fontsize=7.8, color=MUTE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(ROOT / "results/fig8_robustness_grid.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig); print("fig8 done")


# ---------------------------------------------------------------- fig 9
def _box(ax, x, y, w, h, text, fc, tc="white", fs=9, lw=0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                fc=fc, ec=NAVY if lw else fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, label=None, color=MUTE, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4))
    if label:
        ax.text(lx if lx is not None else (x1 + x2) / 2,
                ly if ly is not None else (y1 + y2) / 2 + 0.018,
                label, ha="center", fontsize=7.5, color=color)


def _split(ax, source, bus_x, targets, color=MUTE):
    """One source to several targets through a shared vertical bus.

    Diagonals across a block diagram look sketched; right angles look drawn. The
    source runs right to bus_x, the bus spans the targets, and each target gets
    its own horizontal arrow off the bus."""
    ys = [y for _, y in targets] + [source[1]]
    ax.plot([source[0], bus_x], [source[1], source[1]], color=color, lw=1.4, zorder=1)
    ax.plot([bus_x, bus_x], [min(ys), max(ys)], color=color, lw=1.4, zorder=1)
    for x, y in targets:
        ax.add_patch(FancyArrowPatch((bus_x, y), (x, y), arrowstyle="-|>",
                                     mutation_scale=13, color=color, lw=1.4))


def _merge(ax, sources, bus_x, target, color=MUTE):
    """Several sources into one target through a shared vertical bus."""
    ys = [y for _, y in sources] + [target[1]]
    for x, y in sources:
        ax.plot([x, bus_x], [y, y], color=color, lw=1.4, zorder=1)
    ax.plot([bus_x, bus_x], [min(ys), max(ys)], color=color, lw=1.4, zorder=1)
    ax.add_patch(FancyArrowPatch((bus_x, target[1]), target, arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4))


def _elbow(ax, x1, y1, x2, y2, label=None, color=MUTE):
    """Right-angled connector: straight down from the source, then across to the
    target. Used for the read channel, which leaves the quality gate and has to
    reach its box without crossing the router it never passes through."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=1.4,
                                 connectionstyle="angle,angleA=-90,angleB=180,rad=8"))
    if label:
        ax.text(x1 + 0.012, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=7.5, color=color)


# Every box is drawn with this much padding around its nominal rectangle, so an
# arrow that should touch an edge has to allow for it. Ignoring this is why the
# old diagram had arrows starting inside boxes and ending part-way through them.
PAD = 0.012


def fig9_architecture():
    """The deployed pipeline, drawn to match app/pipeline.py.

    Rewritten 2026-07-26. The previous version had an arrow from the auto router
    to the read channel, which the code never takes: route() returns only street,
    object, indoor or surrounding, because wanting to read is an intention that
    cannot be inferred from a frame. Read is now drawn leaving the quality gate
    directly, which is what happens when the user selects it.

    Also removed, on the advisor's note that em dashes read as machine-written:
    the two dashes, the sentence-long embedded title and the footer line. The
    figure caption in the report already carries that text, so repeating it here
    was redundant as well as untidy. Arrow endpoints are now computed from the
    box rectangles plus PAD instead of being typed in by hand.

    Revised 2026-07-30, on the advisor's note that the person re-check should be
    drawn as a branch rather than hidden in a line of box text. Their description
    of it ran the other way, "if you miss a person you check again and you go to
    the VLM", so the branch is drawn as the code actually runs it
    (app/pipeline.py:99): the VLM answers once, its text is searched for a word
    meaning person, and only if none is found is a second detector pass made, at
    a higher confidence and with a size floor. The VLM is never re-invoked, which
    is the point of the arrangement, since Section 5.1 argues a model that has
    just missed something is the wrong thing to ask twice.
    """
    fig, ax = plt.subplots(figsize=(11.6, 5.8))
    ax.set_xlim(0, 1.01); ax.set_ylim(0, 1); ax.axis("off")

    # (x, y, w, h) for every box, so the arrows below can be derived from them
    cam = (0.015, 0.460, 0.115, 0.150)
    gate = (0.185, 0.460, 0.130, 0.150)
    stop = (0.185, 0.775, 0.130, 0.120)
    router = (0.400, 0.460, 0.125, 0.150)
    # the right-hand column now carries four boxes, so it is respaced to leave a
    # gap the branch label can sit inside rather than across a box edge
    street = (0.600, 0.760, 0.255, 0.180)
    scene = (0.600, 0.520, 0.255, 0.180)
    recheck = (0.600, 0.280, 0.255, 0.130)
    read = (0.600, 0.055, 0.255, 0.120)
    post = (0.910, 0.460, 0.080, 0.150)

    def right(b):  return b[0] + b[2] + PAD
    def left(b):   return b[0] - PAD
    def top(b):    return b[1] + b[3] + PAD
    def bottom(b): return b[1] - PAD
    def cx(b):     return b[0] + b[2] / 2
    def cy(b):     return b[1] + b[3] / 2

    _box(ax, *cam, "Phone camera\n(offline hotspot)", NAVY)
    _box(ax, *gate, "Quality gate\nblur + brightness\n~1 ms", TEAL)
    _box(ax, *stop, "Refuse and ask for\na better frame", LIGHT, NAVY, 8.2, 1)
    _box(ax, *router, "Auto router\nYOLO cues\n~50 ms", TEAL)
    _box(ax, *street, "STREET (no VLM, ~0.15 s)\nLYTNetV2 light (14 MB, ~70 ms)\n"
                      "+ YOLO vehicles + position rule", NAVY, fs=8.8)
    _box(ax, *scene, "SURROUNDING / INDOOR / OBJECT\nSmolVLM-500M on iGPU (~2 to 4 s)\n"
                     "greedy decoding, one pass", NAVY, fs=8.8)
    _box(ax, *recheck, "Person re-check (no second VLM pass)\n"
                       "YOLOv8n conf ≥ 0.5, box > 4% of frame\n"
                       "appends “A person is in front of you.”", TEAL, fs=8.2)
    _box(ax, *read, "READ\nRapidOCR (offline, ~0.8 s)", NAVY, fs=8.8)
    _box(ax, *post, "Post-process\nand speak", TEAL)

    _arrow(ax, right(cam), cy(cam), left(gate), cy(gate))
    # the reject label sits beside its arrow, not on it
    _arrow(ax, cx(gate), top(gate), cx(stop), bottom(stop), "reject", DANGER,
           lx=cx(gate) + 0.040, ly=(top(gate) + bottom(stop)) / 2)
    _arrow(ax, right(gate), cy(gate), left(router), cy(router), "pass")
    # no label on the router's outputs: the box is already named "Auto router",
    # so "mode = auto" only added a word for the arrow to run through
    _split(ax, (right(router), cy(router)), (right(router) + left(street)) / 2,
           [(left(street), cy(street)), (left(scene), cy(scene))])
    _elbow(ax, cx(gate), bottom(gate), left(read), cy(read), "mode = read\n(user choice)")

    # the conditional branch: taken only when the VLM's own text names nobody.
    # the arrow sits left of centre so its label has the rest of the column width
    branch_x = left(scene) + 0.102
    _arrow(ax, branch_x, bottom(scene), branch_x, top(recheck),
           "no person in the text", DANGER,
           lx=branch_x + 0.100, ly=(bottom(scene) + top(recheck)) / 2)

    # every path, including the branch, rejoins the same bus into post-process
    _merge(ax, [(right(b), cy(b)) for b in (street, scene, recheck, read)],
           (right(scene) + left(post)) / 2, (left(post), cy(post)))

    fig.savefig(ROOT / "results/fig9_hybrid_architecture.png",
                bbox_inches="tight", facecolor="white")
    plt.close(fig); print("fig9 done")


if __name__ == "__main__":
    fig7_ptl()
    fig8_robustness()
    fig9_architecture()
