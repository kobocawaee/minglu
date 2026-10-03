"""
กราฟวิเคราะห์เพิ่มเติม (Aomam) — ต่อยอดจาก results/analysis_cpu_vs_igpu.md
แยกจาก plot_benchmark.py ของ Mhiu เพื่อไม่ให้ merge ชนกัน

- fig4: RAM trade-off — iGPU เร็วแต่กิน RAM (500M@iGPU เกินงบเครื่อง 8GB)
        (a) RAM CPU vs iGPU + เส้นอ้างอิง 8GB   (b) latency-vs-RAM trade-off
- fig5: prompt effect — concise vs verbose (latency + out_tokens) บน CPU 256M

วิธีรัน:
    python code/plot_analysis.py
ผลลัพธ์เซฟลง results/fig4_ram_tradeoff.png, results/fig5_prompt_effect.png

หมายเหตุความแฟร์:
- เทียบ CPU vs iGPU ใช้ device CPU-t2.4 คู่กับ iGPU (env DirectML เดียวกัน = torch เท่ากัน)
  ตามเหตุผลเดียวกับ fig3 ของ Mhiu
- ทุกค่าใช้ concise prompt + le=384 ยกเว้น fig5 ที่ตั้งใจเทียบ prompt
"""

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

df = pd.read_csv(CSV)
df["model_short"] = df["model"].str.extract(r"(SmolVLM-\d+M)")
df["is_concise"] = df["prompt"].str.contains("two short sentences")

# palette เดียวกับรูปอื่นทั้งเล่ม (code/figstyle.py) — เดิมเป็นม่วง/เขียว default
CCPU, CGPU = figstyle.NAVY, figstyle.TEAL
C256, C500 = figstyle.CYAN, figstyle.DANGER
RAM_LIMIT_GB = 8.0                  # เครื่อง consumer ทั่วไป (เช่นเครื่อง Aomam)
USABLE_GB = 6.0                     # งบ RAM ที่ใช้ได้จริงหลังหัก OS/แอป (~2GB)

models = ["SmolVLM-256M", "SmolVLM-500M"]
devs = ["AMD-Ryzen-CPU-t2.4", "AMD-Radeon-iGPU"]
hw = df[df["device"].isin(devs) & df["is_concise"] & (df["longest_edge"] == 384)].copy()


def avg(metric, model, dev):
    s = hw[(hw["model_short"] == model) & (hw["device"] == dev)][metric]
    return s.mean() if len(s) else 0


# =====================================================================
# FIG 4 — RAM trade-off
# =====================================================================
fig4, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
figstyle.fig_title(fig4, "Peak memory cost of offloading to the iGPU "
                         "(SmolVLM at le=384, 17 images)")

xm = range(len(models))
wb = 0.36

# (a) RAM CPU vs iGPU + เส้น 8GB / usable
cpu_ram = [avg("peak_ram_mb", m, devs[0]) / 1024 for m in models]   # -> GB
gpu_ram = [avg("peak_ram_mb", m, devs[1]) / 1024 for m in models]
a1.bar([i - wb/2 for i in xm], cpu_ram, wb, label="CPU", color=CCPU)
a1.bar([i + wb/2 for i in xm], gpu_ram, wb, label="iGPU (DirectML)", color=CGPU)
for i, (c, g) in enumerate(zip(cpu_ram, gpu_ram)):
    a1.text(i - wb/2, c, f"{c:.1f} GB", ha="center", va="bottom", fontsize=9)
    a1.text(i + wb/2, g, f"{g:.1f} GB", ha="center", va="bottom", fontsize=9, fontweight="bold")
a1.axhline(RAM_LIMIT_GB, color=figstyle.DANGER, ls="--", lw=1.3, label=f"{RAM_LIMIT_GB:.0f} GB device")
a1.axhline(USABLE_GB, color=figstyle.MUTE, ls=":", lw=1.3, label=f"~{USABLE_GB:.0f} GB usable (after OS)")
figstyle.panel_title(a1, "(a) Peak RAM: the 500M model on the iGPU exceeds the usable budget of an 8 GB device")
a1.set_ylabel("peak RAM (GB)")
a1.set_xticks(list(xm)); a1.set_xticklabels(models)
a1.set_ylim(0, 9.6)
a1.legend(fontsize=8.5, loc="upper left"); figstyle.tidy(a1)

# (b) latency-vs-RAM trade-off scatter
for m, mk in zip(models, ["o", "s"]):
    a2.scatter(avg("peak_ram_mb", m, devs[0]) / 1024, avg("latency_s", m, devs[0]),
               s=180, color=CCPU, marker=mk, edgecolor="white", lw=1.2, zorder=3,
               label=f"CPU {m[-4:]}")
    a2.scatter(avg("peak_ram_mb", m, devs[1]) / 1024, avg("latency_s", m, devs[1]),
               s=180, color=CGPU, marker=mk, edgecolor="white", lw=1.2, zorder=3,
               label=f"iGPU {m[-4:]}")
a2.axvline(USABLE_GB, color=figstyle.MUTE, ls=":", lw=1.3)
a2.text(USABLE_GB - 0.06, 0.04, "usable limit", rotation=90, va="bottom",
        ha="right", color=figstyle.MUTE, fontsize=8,
        transform=a2.get_xaxis_transform())
figstyle.panel_title(a2, "(b) Latency against RAM: CPU is the low-memory corner, iGPU the low-latency one")
a2.set_xlabel("peak RAM (GB)")
a2.set_ylabel("latency (s)")
a2.legend(fontsize=8); figstyle.tidy(a2, grid_axis="both")

plt.tight_layout(rect=[0, 0, 1, 0.94])
f4 = OUT / "fig4_ram_tradeoff.png"
plt.savefig(f4, dpi=150)
print(f"เซฟ {f4}")

# =====================================================================
# FIG 5 — prompt effect (concise vs verbose) บน CPU 256M le=384
# =====================================================================
cpu256 = df[(df["device"] == "AMD-Ryzen-CPU") & (df["model_short"] == "SmolVLM-256M")
            & (df["longest_edge"] == 384)].copy()
verb = cpu256[~cpu256["is_concise"]]
conc = cpu256[cpu256["is_concise"]]

fig5, (b1, b2) = plt.subplots(1, 2, figsize=(13, 5))
fig5.suptitle("Prompt effect: concise + anti-guess is faster AND safer\n"
              "SmolVLM-256M, AMD Ryzen CPU, le=384, 17 images",
              fontsize=13, fontweight="bold")

cats = ["verbose\n(\"Briefly describe...\")", "concise\n(\"two short sentences,\nDo not guess\")"]
CV, CC = "#C0392B", "#27AE60"   # แดง=verbose เขียว=concise

# (a) latency
lat = [verb["latency_s"].mean(), conc["latency_s"].mean()]
bars = b1.bar(cats, lat, color=[CV, CC], width=0.55)
for bar, v in zip(bars, lat):
    b1.text(bar.get_x() + bar.get_width()/2, v, f"{v:.2f}s", ha="center", va="bottom", fontweight="bold")
drop = (lat[0] - lat[1]) / lat[0] * 100
b1.set_title(f"(a) Avg latency  (concise -{drop:.0f}%)")
b1.set_ylabel("latency (s)")
b1.grid(axis="y", alpha=0.3)

# (b) out tokens
tok = [verb["out_tokens"].mean(), conc["out_tokens"].mean()]
bars = b2.bar(cats, tok, color=[CV, CC], width=0.55)
for bar, v in zip(bars, tok):
    b2.text(bar.get_x() + bar.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom", fontweight="bold")
b2.set_title("(b) Output tokens (shorter = faster + better for TTS)")
b2.set_ylabel("out tokens")
b2.grid(axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.9])
f5 = OUT / "fig5_prompt_effect.png"
plt.savefig(f5, dpi=150)
print(f"เซฟ {f5}")

print("\n=== สรุปตัวเลขที่ plot ===")
for m in models:
    print(f"{m}: CPU {avg('latency_s', m, devs[0]):.2f}s/{avg('peak_ram_mb', m, devs[0])/1024:.1f}GB | "
          f"iGPU {avg('latency_s', m, devs[1]):.2f}s/{avg('peak_ram_mb', m, devs[1])/1024:.1f}GB")
print(f"prompt 256M: verbose {lat[0]:.2f}s/{tok[0]:.0f}tok -> concise {lat[1]:.2f}s/{tok[1]:.0f}tok (-{drop:.0f}% latency)")
