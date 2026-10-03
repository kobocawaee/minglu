"""
สร้างกราฟจาก data/benchmark.csv สำหรับใส่สไลด์/รายงาน
- fig1: เทียบ SmolVLM-256M vs 500M (latency, out_tokens, prefill/decode) @ best config (le=384)
- fig2: latency vs resolution ของ 256M (384 vs 512) — โชว์ว่าแบนเพราะ vision_tokens คงที่
- fig3: เทียบ CPU vs iGPU (DirectML) × 256M/500M — platform 1 vs platform 2

วิธีรัน:
    python code/plot_benchmark.py
ผลลัพธ์เซฟลง results/fig1..fig3 .png
"""

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

figstyle.apply()

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "benchmark.csv"
OUT = ROOT / "results"

# ----- โหลดข้อมูล -----
df = pd.read_csv(CSV)
df["model_short"] = df["model"].str.extract(r"(SmolVLM-\d+M)")  # ชื่อสั้นไว้ใส่ legend

def short_label(name):
    """ตัดชื่อไฟล์ให้สั้น + ตัดอักขระที่ไม่ใช่ ascii ออก (กัน font ภาษาไทยเป็นกล่อง)"""
    stem = re.sub(r"[^\x00-\x7F]", "", Path(name).stem)
    return stem[:14]

# ----- แยก batch -----
# concise config (prompt "two short sentences", le=384) → ใช้เทียบ 256M vs 500M
# จำกัด device เป็น CPU เดิม (vlm_research) กัน row ของ CPU-t2.4/iGPU มาปน image ซ้ำ
cmp = df[
    df["prompt"].str.contains("two short sentences")
    & (df["longest_edge"] == 384)
    & (df["device"] == "AMD-Ryzen-CPU")
].copy()
m256 = cmp[cmp["model_short"] == "SmolVLM-256M"].set_index("image")
m500 = cmp[cmp["model_short"] == "SmolVLM-500M"].set_index("image")
images = [img for img in m256.index if img in m500.index]  # เฉพาะรูปที่มีทั้ง 2 model
labels = [short_label(i) for i in images]

# long-prompt sweep ของ 256M (มี 384 + 512) → ใช้ทำ latency vs resolution
sweep = df[df["prompt"].str.contains("Briefly") & (df["model_short"] == "SmolVLM-256M")].copy()

# =====================================================================
# FIG 1 — เทียบ 256M vs 500M @ best config (le=384)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("SmolVLM-256M vs 500M @ best config (longest_edge=384, max_new_tokens=50)\n"
             "AMD Ryzen AI 7 350 (CPU), 17 real test images", fontsize=14, fontweight="bold")

x = range(len(images))
w = 0.4
C256, C500 = "#4C9BE8", "#E8744C"  # ฟ้า=256M ส้ม=500M

# (a) latency รายรูป
ax = axes[0, 0]
ax.bar([i - w/2 for i in x], m256.loc[images, "latency_s"], w, label="256M", color=C256)
ax.bar([i + w/2 for i in x], m500.loc[images, "latency_s"], w, label="500M", color=C500)
ax.set_title("(a) Latency per image")
ax.set_ylabel("latency (s)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# (b) latency เฉลี่ย
ax = axes[0, 1]
means = [m256.loc[images, "latency_s"].mean(), m500.loc[images, "latency_s"].mean()]
bars = ax.bar(["256M", "500M"], means, color=[C256, C500], width=0.5)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}s", ha="center", va="bottom", fontweight="bold")
ax.set_title("(b) Average latency")
ax.set_ylabel("latency (s)")
ax.grid(axis="y", alpha=0.3)

# (c) out_tokens รายรูป — โชว์ว่า 500M หยุดเองตาม "2 sentences" ส่วน 256M ชน cap 50
ax = axes[1, 0]
ax.bar([i - w/2 for i in x], m256.loc[images, "out_tokens"], w, label="256M", color=C256)
ax.bar([i + w/2 for i in x], m500.loc[images, "out_tokens"], w, label="500M", color=C500)
ax.axhline(50, color="red", ls="--", lw=1, label="max_new_tokens cap")
ax.set_title("(c) Output tokens (256M hits cap, 500M stops earlier)")
ax.set_ylabel("out tokens")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# (d) prefill vs decode (เฉลี่ย, stacked) — prefill เกือบคงที่ ความต่างมาจาก decode
ax = axes[1, 1]
pf = [m256.loc[images, "prefill_s"].mean(), m500.loc[images, "prefill_s"].mean()]
dec = [means[0] - pf[0], means[1] - pf[1]]
ax.bar(["256M", "500M"], pf, 0.5, label="prefill (image)", color="#9B9B9B")
ax.bar(["256M", "500M"], dec, 0.5, bottom=pf, label="decode (text)", color="#7BC47F")
for i, (p, d) in enumerate(zip(pf, dec)):
    ax.text(i, p/2, f"{p:.2f}s", ha="center", va="center", fontsize=9)
    ax.text(i, p + d/2, f"{d:.2f}s", ha="center", va="center", fontsize=9)
ax.set_title("(d) Avg prefill vs decode breakdown")
ax.set_ylabel("seconds")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
f1 = OUT / "fig1_model_comparison.png"
plt.savefig(f1, dpi=150)
print(f"เซฟ {f1}")

# =====================================================================
# FIG 2 — latency vs resolution ของ 256M (384 vs 512)
# =====================================================================
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("SmolVLM-256M: latency vs resolution (longest_edge)\n"
              "latency is flat because vision_tokens stay constant at the sweet spot",
              fontsize=13, fontweight="bold")

piv_lat = sweep.pivot_table(index="image", columns="longest_edge", values="latency_s")
piv_tok = sweep.pivot_table(index="image", columns="longest_edge", values="vision_tokens")
res_cols = sorted(piv_lat.columns)

# ซ้าย: latency เฉลี่ย ต่อ resolution (มี error bar = ช่วงรายรูป)
lat_mean = [piv_lat[c].mean() for c in res_cols]
lat_std = [piv_lat[c].std() for c in res_cols]
axL.bar([str(c) for c in res_cols], lat_mean, yerr=lat_std, capsize=6,
        color="#4C9BE8", width=0.5)
for i, v in enumerate(lat_mean):
    axL.text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontweight="bold")
axL.set_title("Avg latency per resolution")
axL.set_xlabel("longest_edge (px)")
axL.set_ylabel("latency (s)")
axL.grid(axis="y", alpha=0.3)

# ขวา: vision_tokens ต่อ resolution — อธิบายว่าทำไม latency ถึงแบน
tok_mean = [piv_tok[c].mean() for c in res_cols]
bars = axR.bar([str(c) for c in res_cols], tok_mean, color="#E8744C", width=0.5)
for b, v in zip(bars, tok_mean):
    axR.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
axR.set_title("Vision tokens per resolution (constant -> flat latency)")
axR.set_xlabel("longest_edge (px)")
axR.set_ylabel("vision tokens")
axR.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
f2 = OUT / "fig2_latency_vs_resolution.png"
plt.savefig(f2, dpi=150)
print(f"เซฟ {f2}")

# ----- สรุปตัวเลขลง console -----
print("\n=== สรุป (เฉลี่ย 17 รูป, le=384, concise config) ===")
print(f"256M: latency {means[0]:.2f}s | prefill {pf[0]:.2f}s | decode {dec[0]:.2f}s | "
      f"out {m256.loc[images,'out_tokens'].mean():.0f} tok | RAM {m256.loc[images,'peak_ram_mb'].mean():.0f} MB")
print(f"500M: latency {means[1]:.2f}s | prefill {pf[1]:.2f}s | decode {dec[1]:.2f}s | "
      f"out {m500.loc[images,'out_tokens'].mean():.0f} tok | RAM {m500.loc[images,'peak_ram_mb'].mean():.0f} MB")

# =====================================================================
# FIG 3 — CPU vs iGPU (DirectML), เทียบ 256M และ 500M (platform 1 vs 2)
# ใช้ device CPU-t2.4 vs iGPU (env vlm_dml เดียวกัน → torch version เท่ากัน เทียบแฟร์)
# =====================================================================
hw = df[df["device"].isin(["AMD-Ryzen-CPU-t2.4", "AMD-Radeon-iGPU"])].copy()
if not hw.empty:
    models = ["SmolVLM-256M", "SmolVLM-500M"]
    devs = ["AMD-Ryzen-CPU-t2.4", "AMD-Radeon-iGPU"]
    dev_name = {"AMD-Ryzen-CPU-t2.4": "CPU", "AMD-Radeon-iGPU": "iGPU (DirectML)"}
    # palette เดียวกับรูปอื่นทั้งเล่ม (code/figstyle.py)
    CCPU, CGPU = figstyle.NAVY, figstyle.TEAL

    def avg(metric, model, dev):
        s = hw[(hw["model_short"] == model) & (hw["device"] == dev)][metric]
        return s.mean() if len(s) else 0

    fig3, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6))
    figstyle.fig_title(fig3, "CPU against iGPU (DirectML), SmolVLM at le=384, "
                             "17 images")
    xm = range(len(models))
    wb = 0.36

    # (a) latency เฉลี่ย
    cpu_lat = [avg("latency_s", m, devs[0]) for m in models]
    gpu_lat = [avg("latency_s", m, devs[1]) for m in models]
    a1.bar([i - wb/2 for i in xm], cpu_lat, wb, label="CPU", color=CCPU)
    a1.bar([i + wb/2 for i in xm], gpu_lat, wb, label="iGPU (DirectML)", color=CGPU)
    for i, (c, g) in enumerate(zip(cpu_lat, gpu_lat)):
        a1.text(i - wb/2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
        a1.text(i + wb/2, g, f"{g:.2f}", ha="center", va="bottom", fontsize=8)
    figstyle.panel_title(a1, "(a) Average latency")
    a1.set_ylabel("latency (s)")
    a1.set_xticks(list(xm)); a1.set_xticklabels(models)
    a1.legend(fontsize=8.5); figstyle.tidy(a1)

    # (b) prefill เฉลี่ย — โชว์ว่า iGPU prefill เกือบคงที่/เร็วกว่ามาก
    cpu_pf = [avg("prefill_s", m, devs[0]) for m in models]
    gpu_pf = [avg("prefill_s", m, devs[1]) for m in models]
    a2.bar([i - wb/2 for i in xm], cpu_pf, wb, label="CPU", color=CCPU)
    a2.bar([i + wb/2 for i in xm], gpu_pf, wb, label="iGPU (DirectML)", color=CGPU)
    for i, (c, g) in enumerate(zip(cpu_pf, gpu_pf)):
        a2.text(i - wb/2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=8)
        a2.text(i + wb/2, g, f"{g:.2f}", ha="center", va="bottom", fontsize=8)
    figstyle.panel_title(a2, "(b) Average prefill (image processing)")
    a2.set_ylabel("prefill (s)")
    a2.set_xticks(list(xm)); a2.set_xticklabels(models)
    a2.legend(fontsize=8.5); figstyle.tidy(a2)

    # (c) speedup iGPU/CPU ต่อ model
    speedups = [cpu_lat[i] / gpu_lat[i] if gpu_lat[i] else 0 for i in range(len(models))]
    bars = a3.bar(models, speedups, color=CGPU, width=0.5)
    a3.axhline(1.0, color=figstyle.DANGER, ls="--", lw=1.1, label="break-even (1.0x)")
    for b, v in zip(bars, speedups):
        a3.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}x", ha="center", va="bottom", fontweight="bold")
    figstyle.panel_title(a3, "(c) iGPU speed-up over CPU")
    a3.set_ylabel("speedup (x)")
    a3.legend(fontsize=8.5); figstyle.tidy(a3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    f3 = OUT / "fig3_cpu_vs_igpu.png"
    plt.savefig(f3, dpi=150)
    print(f"เซฟ {f3}")

    print("\n=== CPU vs iGPU (เฉลี่ย 17 รูป, le=384) ===")
    for m in models:
        print(f"{m}: CPU lat {avg('latency_s', m, devs[0]):.2f}s (pf {avg('prefill_s', m, devs[0]):.2f}) | "
              f"iGPU lat {avg('latency_s', m, devs[1]):.2f}s (pf {avg('prefill_s', m, devs[1]):.2f}) | "
              f"speedup {avg('latency_s', m, devs[0]) / avg('latency_s', m, devs[1]):.2f}x")
