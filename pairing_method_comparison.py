# -*- coding: utf-8 -*-
"""
Step 6 - MIMIC-IV: make the pairing method defensible.

Compares three pairing strategies and quantifies discordance in BOTH
directions (the reviewer's first question is "how do you know the cuff
reading is contemporaneous with the arterial reading?").

  M1  nearest arterial within +/-5 min          (previous step)
  M2  arterial value in the SAME minute (gap=0) (strictest)
  M3  mean arterial MAP over NIBP time +/-2 min (smoothed, PRIMARY)

Run:  python E:\\claude-tools\\step6_methods.py
"""
import os
import time
import traceback
import numpy as np
import pandas as pd

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step6_methods_report.txt")
BPFILE  = os.path.join(DERIVED, "mimic_bp.parquet")
OUTPRIM = os.path.join(DERIVED, "mimic_pairs_primary.parquet")

os.makedirs(AUDIT, exist_ok=True)
lines = []


def W(t=""):
    s = str(t)
    print(s, flush=True)
    lines.append(s)


def Save():
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print("save failed: %r" % e)


t0 = time.time()
W("=" * 66)
W(" STEP 6 - pairing method comparison (MIMIC-IV)")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

# ------------------------------------------------------------------
bp = pd.read_parquet(BPFILE)
bp = bp[(bp["valuenum"] >= 20) & (bp["valuenum"] <= 200)]
art = (bp.loc[bp.itemid == 220052, ["stay_id", "tmin", "valuenum"]]
         .rename(columns={"valuenum": "map_art"}).reset_index(drop=True))
nib = (bp.loc[bp.itemid == 220181, ["stay_id", "tmin", "valuenum"]]
         .rename(columns={"valuenum": "map_nibp"}).reset_index(drop=True))
del bp
nib = nib.reset_index(drop=True)
nib["nid"] = np.arange(len(nib), dtype="int64")
W("invasive %s | non-invasive %s" % (format(len(art), ","), format(len(nib), ",")))
W("")
Save()


def analyse(pairs, label, save_as=None):
    """pairs must have: stay_id, map_nibp, map_art, gap"""
    W("=" * 66)
    W(" %s" % label)
    W("=" * 66)
    n = len(pairs)
    if n == 0:
        W("  no pairs")
        W("")
        return
    W("  paired readings : %s" % format(n, ","))
    W("  ICU stays       : %s" % format(pairs["stay_id"].nunique(), ","))
    if "gap" in pairs.columns:
        W("  gap (min) median: %.1f   max %.1f"
          % (pairs["gap"].median(), pairs["gap"].max()))
    W("")

    d = (pairs["map_nibp"] - pairs["map_art"]).astype(float)
    bias, sd = float(d.mean()), float(d.std())
    W("  --- agreement (NIBP - invasive) ---")
    W("    bias      : %+.2f mmHg" % bias)
    W("    SD        : %.2f mmHg" % sd)
    W("    95%% limits: %+.2f to %+.2f" % (bias - 1.96 * sd, bias + 1.96 * sd))
    W("    NIBP>inv  : %.1f %%" % (100.0 * float((d > 0).mean())))
    W("    |diff|>10 : %.1f %%" % (100.0 * float((d.abs() > 10).mean())))
    W("")

    W("  --- discordance, both directions ---")
    for THR in (65, 60, 55):
        low_art = int((pairs["map_art"] < THR).sum())
        low_nib = int((pairs["map_nibp"] < THR).sum())
        con = int(((pairs["map_nibp"] < THR) & (pairs["map_art"] < THR)).sum())
        occ = int(((pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)).sum())
        fal = int(((pairs["map_nibp"] < THR) & (pairs["map_art"] >= THR)).sum())
        bn = int(((pairs["map_nibp"] >= THR) & (pairs["map_art"] >= THR)).sum())
        W("    threshold %d mmHg" % THR)
        W("      invasive-low readings     : %s" % format(low_art, ","))
        W("      cuff-low readings         : %s" % format(low_nib, ","))
        W("      MISSED  (art low, cuff ok): %s  (%.1f%% of invasive-low)"
          % (format(occ, ","), (100.0 * occ / low_art) if low_art else 0.0))
        W("      FALSE ALARM (cuff low, art ok): %s  (%.1f%% of cuff-low)"
          % (format(fal, ","), (100.0 * fal / low_nib) if low_nib else 0.0))
        W("      concordant low            : %s" % format(con, ","))
        W("      concordant normal         : %s" % format(bn, ","))
        agree = (con + bn) / n * 100.0
        W("      overall agreement         : %.1f %%" % agree)
        W("")

    THR = 65
    pairs = pairs.assign(art_low=pairs["map_art"] < THR,
                         occult=(pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR),
                         falarm=(pairs["map_nibp"] < THR) & (pairs["map_art"] >= THR))
    per = pairs.groupby("stay_id").agg(n=("occult", "size"),
                                       low=("art_low", "sum"),
                                       occ=("occult", "sum"),
                                       fal=("falarm", "sum"))
    wl = per[per["low"] > 0]
    W("  --- per ICU stay, threshold 65 ---")
    W("    stays with >=1 pair        : %s" % format(len(per), ","))
    W("    ... with >=1 invasive-low  : %s" % format(len(wl), ","))
    if len(wl):
        fr = wl["occ"] / wl["low"]
        W("    ALL hypotensive missed     : %s" % format(int((fr >= 1).sum()), ","))
        W("    NONE missed                : %s" % format(int((fr <= 0).sum()), ","))
        W("    median missed share        : %.0f %%" % (100.0 * float(fr.median())))
        W("    IQR of missed share        : %.0f - %.0f %%"
          % (100.0 * float(fr.quantile(.25)), 100.0 * float(fr.quantile(.75))))
    W("")
    Save()

    if save_as:
        keep = pairs[["stay_id", "tmin", "map_nibp", "map_art", "gap"]].copy()
        keep.to_parquet(save_as, index=False)
        W("  saved -> %s" % save_as)
        W("")
        Save()


# ==================================================================
# M1  nearest within +/-5
# ==================================================================
try:
    art5 = art.rename(columns={"tmin": "art_t"})
    matched = np.zeros(len(nib), dtype=bool)
    frames = []
    for off in sorted(range(-5, 6), key=abs):
        sub = nib.loc[~matched]
        left = sub[["nid", "stay_id", "tmin", "map_nibp"]].copy()
        left["art_t"] = left["tmin"] + off
        m = left.merge(art5, on=["stay_id", "art_t"], how="inner")
        if len(m):
            m["gap"] = abs(off)
            frames.append(m)
            matched[m["nid"].values] = True
    m1 = pd.concat(frames, ignore_index=True)
    del frames
    analyse(m1, "M1  nearest arterial within +/-5 min")
except Exception:
    W("M1 FAILED"); W(traceback.format_exc()); Save()

# ==================================================================
# M2  same minute only
# ==================================================================
try:
    left = nib[["nid", "stay_id", "tmin", "map_nibp"]].copy()
    left["art_t"] = left["tmin"]
    m2 = left.merge(art5, on=["stay_id", "art_t"], how="inner")
    m2["gap"] = 0
    analyse(m2, "M2  arterial value in the SAME minute (gap = 0)")
except Exception:
    W("M2 FAILED"); W(traceback.format_exc()); Save()

# ==================================================================
# M3  mean arterial MAP over +/-2 min  (PRIMARY)
# ==================================================================
try:
    frames = []
    for off in range(-2, 3):
        left = nib[["nid", "stay_id", "tmin", "map_nibp"]].copy()
        left["art_t"] = left["tmin"] + off
        m = left.merge(art5, on=["stay_id", "art_t"], how="inner")
        if len(m):
            frames.append(m)
    allm = pd.concat(frames, ignore_index=True)
    del frames
    W("  M3 raw matches: %s" % format(len(allm), ","))
    g = allm.groupby("nid").agg(n_art=("map_art", "size"),
                                sum_art=("map_art", "sum"),
                                map_art_mean=("map_art", "mean"))
    g = g.reset_index()
    m3 = nib.merge(g, on="nid", how="inner")
    m3["map_art"] = m3["map_art_mean"]
    m3["gap"] = 2.0
    m3 = m3[m3["n_art"] >= 1]
    analyse(m3, "M3  mean arterial MAP over NIBP time +/-2 min  (PRIMARY)",
            save_as=OUTPRIM)

    # stricter variant: require >=2 arterial readings in the window
    m3b = m3[m3["n_art"] >= 2]
    analyse(m3b, "M3b same, but requiring >=2 arterial readings in window")
except Exception:
    W("M3 FAILED"); W(traceback.format_exc()); Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
