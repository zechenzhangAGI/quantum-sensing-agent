"""Build the README hero figure from the 14 May 2025 session data in this folder.

Panels (a) and (b) are the lab toolkit's own PNG renderings, because the 2D arrays in the
JSON files were truncated by numpy's repr when saved. Panel (c) is re-plotted from the
complete 1D arrays in esr_data_20250514_114852.json. No fit is applied.

Usage (from the repository root):
    python examples/2025-05-14-nv-odmr/make_hero_figure.py
Requires numpy and matplotlib.
"""
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "..", "docs", "figures", "hero_2025-05-14.png")

INK, MUTED, GRID, ACCENT = "#1f2933", "#7b8794", "#e4e7eb", "#2563a8"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})


def arr(s):
    """Parse a numpy-repr string saved by the toolkit into a 1D array."""
    s = s if isinstance(s, str) else str(s)
    assert "..." not in s, "array was truncated"
    return np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)])


esr = json.load(open(os.path.join(DATA, "esr_data_20250514_114852.json")))
f = arr(esr["frequency"]) / 1e9
y = arr(esr["data"])
e = arr(esr["data_err"])
assert len(f) == len(y) == len(e) == 40

fig = plt.figure(figsize=(13, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.2, 1], wspace=0.08)
titles = [
    "1  galvo scan, 08:31\nwhole field of view (126 × 126 pts, ±4 V)",
    "2  find_nv on candidate 3, 11:23\nzoomed scan + 2D Gaussian peak",
    "3  CW ESR (ODMR) on candidate 3, 11:48\n2.85–2.89 GHz, 0 dBm, 9.6 kcps",
]
for i, fn in enumerate(["GalvoScan_plot_20250514_083103.png", "FindNV_plot_20250514_112326.png"]):
    ax = fig.add_subplot(gs[i])
    ax.imshow(mpimg.imread(os.path.join(DATA, fn)))
    ax.set_axis_off()
    ax.set_title(titles[i], fontsize=10, color=INK, loc="left")

ax = fig.add_subplot(gs[2])
ax.errorbar(f, y, yerr=e, fmt="o-", ms=4, lw=1.2, elinewidth=0.8, capsize=0,
            color=ACCENT, mfc="white", mec=ACCENT)
ax.axhline(1.0, color=MUTED, lw=0.6)
ax.axvline(2.870, color=MUTED, lw=0.8, ls="--")
ax.text(2.8705, 1.036, "D = 2.870 GHz", fontsize=9, color=MUTED)
ax.set_ylim(0.955, 1.045)
ax.set_xlabel("microwave frequency (GHz)")
ax.set_ylabel("normalized fluorescence")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title(titles[2], fontsize=10, color=INK, loc="left")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("wrote", os.path.normpath(OUT))
