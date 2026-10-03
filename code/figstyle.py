# -*- coding: utf-8 -*-
"""One visual language for every figure in the report.

The figures were written at different times by different people, so Figures 2 and
3 came out in matplotlib's default purple-and-green with black panel frames and
bold black two-line titles, while Figures 1, 4 and 5 used the project's navy and
teal. Side by side in one document that reads as assembled rather than designed,
which is exactly the impression the advisor asked us to avoid.

This module holds the palette and the rcParams so a figure cannot drift again.
It also carries the two rules that were being applied inconsistently: a title is
left-aligned, unbold and navy because the caption below already carries the
sentence, and a panel keeps only the axes it needs.
"""
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# Validated against the colour-vision checks used for the slide deck. Grey is for
# annotation only, never for a data series.
NAVY, TEAL, CYAN, GREEN = "#0E2438", "#12A594", "#2AA6CE", "#1AA35A"
DANGER, MUTE, LIGHT = "#C2410C", "#51617A", "#F1F4F8"
GRID = "#EDF1F6"


def apply():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.facecolor": "white",
        "axes.edgecolor": "#DCE3EC",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "axes.titlecolor": NAVY,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "xtick.color": MUTE,
        "ytick.color": MUTE,
        "legend.frameon": False,
    })


def tidy(ax, grid_axis="y"):
    """Drop the frame the data does not need and put the grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.9)
        ax.set_axisbelow(True)


def panel_title(ax, text):
    ax.set_title(text, fontsize=9.5, color=NAVY, loc="left", pad=8)


def fig_title(fig, text, y=0.98):
    """A short label, not a sentence: the report caption states the finding."""
    fig.suptitle(text, fontsize=11.5, color=NAVY, x=0.01, ha="left", y=y)
