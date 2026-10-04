"""
用 data/benchmark.csv 畫圖，放進簡報／報告
- fig1：比較 SmolVLM-256M vs 500M（延遲、輸出 token、預填／解碼）@ 最佳設定（le=384）
- fig2：256M 的延遲 vs 解析度（384 vs 512）— 顯示曲線平坦是因為 vision_tokens 固定
- fig3：比較 CPU vs iGPU（DirectML）× 256M/500M — 平台 1 vs 平台 2

執行方式：
    python code/plot_benchmark.py
結果存到 results/fig1..fig3 .png
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

# ----- 載入資料 -----
df = pd.read_csv(CSV)
df["model_short"] = df["model"].str.extract(r"(SmolVLM-\d+M)")  # 圖例用的短名稱

def short_label(name):
    """把檔名縮短＋去掉非 ASCII 字元（避免非英文字型顯示成方框）"""
    stem = re.sub(r"[^\x00-\x7F]", "", Path(name).stem)
    return stem[:14]

# ----- 分組 -----
# 簡潔設定（提示詞 "two short sentences"、le=384）→ 用來比較 256M vs 500M
# 限定原本的 CPU（vlm_research），避免 CPU-t2.4／iGPU 的資料混進來造成同一張圖重複
cmp = df[
    df["prompt"].str.contains("two short sentences")
    & (df["longest_edge"] == 384)
    & (df["device"] == "AMD-Ryzen-CPU")
].copy()
m256 = cmp[cmp["model_short"] == "SmolVLM-256M"].set_index("image")
m500 = cmp[cmp["model_short"] == "SmolVLM-500M"].set_index("image")
images = [img for img in m256.index if img in m500.index]  # 只取兩個模型都有的圖
labels = [short_label(i) for i in images]

# 256M 的長提示詞掃描（有 384 + 512）→ 用來畫延遲 vs 解析度
sweep = df[df["prompt"].str.contains("Briefly") & (df["model_short"] == "SmolVLM-256M")].copy()

# =====================================================================
# FIG 1 — 比較 256M vs 500M @ 最佳設定（le=384）
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("SmolVLM-256M vs 500M @ best config (longest_edge=384, max_new_tokens=50)\n"
             "AMD Ryzen AI 7 350 (CPU), 17 real test images", fontsize=14, fontweight="bold")

x = range(len(images))
w = 0.4
C256, C500 = "#4C9BE8", "#E8744C"  # 藍 = 256M，橘 = 500M

# (a) 每張圖的延遲
ax = axes[0, 0]
ax.bar([i - w/2 for i in x], m256.loc[images, "latency_s"], w, label="256M", color=C256)
ax.bar([i + w/2 for i in x], m500.loc[images, "latency_s"], w, label="500M", color=C500)
ax.set_title("(a) Latency per image")
ax.set_ylabel("latency (s)")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# (b) 平均延遲
ax = axes[0, 1]
means = [m256.loc[images, "latency_s"].mean(), m500.loc[images, "latency_s"].mean()]
bars = ax.bar(["256M", "500M"], means, color=[C256, C500], width=0.5)
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.2f}s", ha="center", va="bottom", fontweight="bold")
ax.set_title("(b) Average latency")
ax.set_ylabel("latency (s)")
ax.grid(axis="y", alpha=0.3)

# (c) 每張圖的輸出 token — 顯示 500M 會依「2 sentences」自己停，256M 則碰到上限 50
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

# (d) 預填 vs 解碼（平均、堆疊）— 預填幾乎固定，差異來自解碼
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
print(f"已存 {f1}")

# =====================================================================
# FIG 2 — 256M 的延遲 vs 解析度（384 vs 512）
# =====================================================================
fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("SmolVLM-256M: latency vs resolution (longest_edge)\n"
              "latency is flat because vision_tokens stay constant at the sweet spot",
              fontsize=13, fontweight="bold")

piv_lat = sweep.pivot_table(index="image", columns="longest_edge", values="latency_s")
piv_tok = sweep.pivot_table(index="image", columns="longest_edge", values="vision_tokens")
res_cols = sorted(piv_lat.columns)

# 左：各解析度的平均延遲（誤差線 = 各圖的範圍）
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

# 右：各解析度的 vision_tokens — 說明延遲為什麼是平的
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
print(f"已存 {f2}")

# ----- 在主控台印出數字摘要 -----
print("\n=== 摘要（17 張圖平均，le=384，簡潔設定）===")
print(f"256M: latency {means[0]:.2f}s | prefill {pf[0]:.2f}s | decode {dec[0]:.2f}s | "
      f"out {m256.loc[images,'out_tokens'].mean():.0f} tok | RAM {m256.loc[images,'peak_ram_mb'].mean():.0f} MB")
print(f"500M: latency {means[1]:.2f}s | prefill {pf[1]:.2f}s | decode {dec[1]:.2f}s | "
      f"out {m500.loc[images,'out_tokens'].mean():.0f} tok | RAM {m500.loc[images,'peak_ram_mb'].mean():.0f} MB")

# =====================================================================
# FIG 3 — CPU vs iGPU（DirectML），比較 256M 和 500M（平台 1 vs 2）
# 用 CPU-t2.4 vs iGPU（同一個 vlm_dml 環境 → torch 版本相同，比較公平）
# =====================================================================
hw = df[df["device"].isin(["AMD-Ryzen-CPU-t2.4", "AMD-Radeon-iGPU"])].copy()
if not hw.empty:
    models = ["SmolVLM-256M", "SmolVLM-500M"]
    devs = ["AMD-Ryzen-CPU-t2.4", "AMD-Radeon-iGPU"]
    dev_name = {"AMD-Ryzen-CPU-t2.4": "CPU", "AMD-Radeon-iGPU": "iGPU (DirectML)"}
    # 和論文其他圖相同的配色（code/figstyle.py）
    CCPU, CGPU = figstyle.NAVY, figstyle.TEAL

    def avg(metric, model, dev):
        s = hw[(hw["model_short"] == model) & (hw["device"] == dev)][metric]
        return s.mean() if len(s) else 0

    fig3, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6))
    figstyle.fig_title(fig3, "CPU against iGPU (DirectML), SmolVLM at le=384, "
                             "17 images")
    xm = range(len(models))
    wb = 0.36

    # (a) 平均延遲
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

    # (b) 平均預填 — 顯示 iGPU 的預填幾乎固定／快很多
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

    # (c) 各模型的 iGPU/CPU 加速倍數
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
    print(f"已存 {f3}")

    print("\n=== CPU vs iGPU（17 張圖平均，le=384）===")
    for m in models:
        print(f"{m}: CPU lat {avg('latency_s', m, devs[0]):.2f}s (pf {avg('prefill_s', m, devs[0]):.2f}) | "
              f"iGPU lat {avg('latency_s', m, devs[1]):.2f}s (pf {avg('prefill_s', m, devs[1]):.2f}) | "
              f"speedup {avg('latency_s', m, devs[0]) / avg('latency_s', m, devs[1]):.2f}x")
