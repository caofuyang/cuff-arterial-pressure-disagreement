# -*- coding: utf-8 -*-
"""
Generate Figures 1-3 of the manuscript.

Output
------
  Figure_1.png   agreement between cuff and arterial pressure (three databases)
  Figure_2.png   discordance by MAP threshold (two panels)
  Figure_3.png   acute kidney injury by hypotension group

Figure 3 is written at 1000 dpi and 6.6 inches wide, giving a raster of roughly
6600 x 3100 pixels, which exceeds the journal's minimum width of 3543 pixels for
bitmapped line art. Figures 1 and 2 are written at 600 dpi (about 4300 pixels
wide) and also meet that requirement.

Requires matplotlib, pandas, numpy.

Run:  python make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ------------------------------------------------------------------ paths
DERIVED = r"E:\occult_htn\derived"      # derived parquet files
FIGDIR  = r"E:\occult_htn\figures"
SINGLE  = os.path.join(FIGDIR, "single_panel")
for d in (FIGDIR, SINGLE):
    os.makedirs(d, exist_ok=True)

DPI = 600
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

DB = [
    ("MOVER (OR)",     "mover_pairs_t5.parquet",  "#C1272D"),
    ("MIMIC-IV (ICU)", "mimic_pairs_t5.parquet",  "#0072B2"),
    ("eICU-CRD (ICU)", "eicu_pairs_t5.parquet",   "#009E73"),
]

THRESHOLDS = [65, 60, 55]
GROUPS = ["neither", "cuff_only", "art_only", "both"]
GROUP_LABELS = ["Neither", "Cuff only", "Arterial only", "Both"]

AKI_SETS = [
    ("MIMIC-IV (ICU)", "mimic_aki.parquet", "mimic_pairs_t5.parquet",
     "stay_id", "#0072B2"),
    ("eICU-CRD (ICU)", "eicu_aki.parquet", "eicu_pairs_t5.parquet",
     "patientunitstayid", "#009E73"),
]


def load_pairs(fname):
    """Load paired readings, restricted to same-minute pairs."""
    p = pd.read_parquet(os.path.join(DERIVED, fname))
    return p[p["gap"] == 0].copy()


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion, returned as percentages."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


# ==================================================================
# load
# ==================================================================
print("loading...")
DATA = {}
for label, fname, col in DB:
    path = os.path.join(DERIVED, fname)
    if os.path.exists(path):
        DATA[label] = (load_pairs(fname), col)
        print("  %-16s %s rows" % (label, format(len(DATA[label][0]), ",")))
    else:
        print("  missing:", path)


# ==================================================================
# Figure 1 - agreement (Bland-Altman style, hexbin density)
# ==================================================================
print("Figure 1 ...")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
summary = []
for ax, (label, (p, col)) in zip(axes, DATA.items()):
    mean_ = (p["map_art"] + p["map_nibp"]) / 2.0
    diff = p["map_nibp"] - p["map_art"]
    bias, sd = diff.mean(), diff.std()
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    summary.append((label, bias, sd, lo, hi))

    m = (mean_ >= 20) & (mean_ <= 160) & (diff >= -60) & (diff <= 60)
    ax.hexbin(mean_[m], diff[m], gridsize=70, cmap="Blues", mincnt=1,
              bins="log", linewidths=0)
    ax.axhline(bias, color=col, lw=1.3)
    ax.axhline(lo, color=col, lw=0.8, ls="--")
    ax.axhline(hi, color=col, lw=0.8, ls="--")
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_title(label, fontsize=8.5)
    ax.set_xlabel("Mean of cuff and arterial MAP (mmHg)")
    ax.set_xlim(20, 160)
    ax.set_ylim(-60, 60)
    ax.text(0.03, 0.96, "bias %+.1f\nLoA %+.0f to %+.0f" % (bias, lo, hi),
            transform=ax.transAxes, va="top", fontsize=7)
axes[0].set_ylabel("Cuff minus arterial MAP (mmHg)")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "Figure_1.png"), dpi=DPI)
plt.close(fig)

# single-panel alternates, in case composite figures are queried by the journal
for label, (p, col) in DATA.items():
    f2, a2 = plt.subplots(figsize=(3.4, 2.6))
    mean_ = (p["map_art"] + p["map_nibp"]) / 2.0
    diff = p["map_nibp"] - p["map_art"]
    bias, sd = diff.mean(), diff.std()
    m = (mean_ >= 20) & (mean_ <= 160) & (diff >= -60) & (diff <= 60)
    a2.hexbin(mean_[m], diff[m], gridsize=70, cmap="Blues", mincnt=1,
              bins="log", linewidths=0)
    a2.axhline(bias, color=col, lw=1.3)
    a2.axhline(bias - 1.96 * sd, color=col, lw=0.8, ls="--")
    a2.axhline(bias + 1.96 * sd, color=col, lw=0.8, ls="--")
    a2.axhline(0, color="0.4", lw=0.6)
    a2.set_xlabel("Mean of cuff and arterial MAP (mmHg)")
    a2.set_ylabel("Cuff minus arterial MAP (mmHg)")
    a2.set_title(label, fontsize=8.5)
    a2.set_xlim(20, 160); a2.set_ylim(-60, 60)
    f2.tight_layout()
    f2.savefig(os.path.join(SINGLE, "Figure_1_%s.png" % label.split()[0]), dpi=DPI)
    plt.close(f2)


# ==================================================================
# Figure 2 - discordance by threshold
# ==================================================================
print("Figure 2 ...")
rows = []
for label, (p, col) in DATA.items():
    for T in THRESHOLDS:
        al = int((p["map_art"] < T).sum())
        nl = int((p["map_nibp"] < T).sum())
        ms = int(((p["map_nibp"] >= T) & (p["map_art"] < T)).sum())
        uc = int(((p["map_nibp"] < T) & (p["map_art"] >= T)).sum())
        rows.append(dict(db=label, thr=T,
                         missed=100.0 * ms / al if al else np.nan,
                         unconf=100.0 * uc / nl if nl else np.nan))
R = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
w = 0.26
xs = np.arange(len(THRESHOLDS))
for ax, field, title in (
        (axes[0], "missed", "Arterial hypotension not shown by the cuff"),
        (axes[1], "unconf", "Cuff hypotension not confirmed by the arterial line")):
    for i, (label, (p, col)) in enumerate(DATA.items()):
        vals = [R[(R.db == label) & (R.thr == T)][field].values[0]
                for T in THRESHOLDS]
        ax.bar(xs + (i - 1) * w, vals, w, label=label, color=col,
               edgecolor="white", linewidth=0.4)
        for x, v in zip(xs + (i - 1) * w, vals):
            ax.text(x, v + 1.5, "%.0f" % v, ha="center", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["<%d" % T for T in THRESHOLDS])
    ax.set_xlabel("MAP threshold (mmHg)")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("Proportion (%)")
axes[1].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "Figure_2.png"), dpi=DPI)
plt.close(fig)


# ==================================================================
# Figure 3 - AKI by hypotension group
#
# ------------------------------------------------------------------
# NOTE: this is the version used in the manuscript. It is written at 1000 dpi
# and 6.6 inches wide (about 6600 x 3100 pixels), rather than at the 600 dpi
# used for Figures 1 and 2, because the group labels and the sample-size
# annotations beneath the axis required a wider canvas.
# ------------------------------------------------------------------
# ==================================================================
print("Figure 3 ...")
fig, ax = plt.subplots(figsize=(6.6, 3.1))
w = 0.36
xs = np.arange(len(GROUPS))

for i, (label, akif, pairf, key, col) in enumerate(AKI_SETS):
    akp, pk = os.path.join(DERIVED, akif), os.path.join(DERIVED, pairf)
    if not (os.path.exists(akp) and os.path.exists(pk)):
        continue
    p = load_pairs(pairf)
    g = p.groupby(key).agg(n_pairs=("map_art", "size"),
                           min_art=("map_art", "min"),
                           min_nib=("map_nibp", "min")).reset_index()
    g = g[g["n_pairs"] >= 5]
    d = g.merge(pd.read_parquet(akp), on=key, how="inner")
    d["grp"] = np.where((d.min_art < 65) & (d.min_nib < 65), "both",
               np.where(d.min_art < 65, "art_only",
               np.where(d.min_nib < 65, "cuff_only", "neither")))

    vals, los, his, ns = [], [], [], []
    for gname in GROUPS:
        s = d[d["grp"] == gname]
        k, n = int(s["aki"].sum()), len(s)
        vals.append(100.0 * k / n if n else np.nan)
        lo_, hi_ = wilson(k, n)
        los.append(lo_); his.append(hi_); ns.append(n)
    vals = np.array(vals); los = np.array(los); his = np.array(his)

    ax.bar(xs + (i - 0.5) * w, vals, w, color=col, label=label,
           edgecolor="white", linewidth=0.4)
    ax.errorbar(xs + (i - 0.5) * w, vals, yerr=[vals - los, his - vals],
                fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2)
    for x, v, n in zip(xs + (i - 0.5) * w, vals, ns):
        ax.text(x, v + 2.5, "%.1f" % v, ha="center", fontsize=6.5)
        ax.text(x, -8.5, "n=%s" % format(n, ","), ha="center",
                fontsize=6, color="0.35")

ax.set_xticks(xs)
ax.set_xticklabels(GROUP_LABELS, fontsize=8)
ax.tick_params(axis="x", pad=14)
ax.set_xlabel("Group, by lowest MAP below 65 mmHg", labelpad=16)
ax.set_ylabel("Acute kidney injury (%)")
ax.set_ylim(-12, 66)
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.spines["bottom"].set_position(("data", 0))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "Figure_3.png"), dpi=1000)
plt.close(fig)

print("")
print("Bias summary:")
for label, bias, sd, lo, hi in summary:
    print("  %-16s bias %+.2f  SD %.2f  LoA %+.2f to %+.2f"
          % (label, bias, sd, lo, hi))
print("")
print("saved to:", FIGDIR)
