# -*- coding: utf-8 -*-
"""
Step 13 - MOVER (intraoperative): label semantics, time granularity, pairing.

Open questions from earlier:
  * is "MAP (mmHg)" invasive or non-invasive?
  * is recorded_time only hourly?

Answers first, pairing second. No outcomes here.

Run:  python E:\\claude-tools\\step13_mover.py
"""
import os
import sys
import glob
import time
import traceback
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step13_mover_report.txt")
BPFILE  = os.path.join(DERIVED, "mover_bp.parquet")
PAIRF   = os.path.join(DERIVED, "mover_pairs_t5.parquet")
THR     = 65

os.makedirs(AUDIT, exist_ok=True)
lines = []


def W(t=""):
    s = str(t)
    print(s, flush=True)
    lines.append(s)


def Save():
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _hook(t, v, tb):
    lines.append("")
    lines.append("!" * 66)
    lines.append(" SCRIPT FAILED")
    lines.append("!" * 66)
    lines.append("".join(traceback.format_exception(t, v, tb)))
    Save()


sys.excepthook = _hook

t0 = time.time()
W("=" * 66)
W(" STEP 13 - MOVER label semantics + granularity + pairing")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

parts = sorted(glob.glob(r"E:\MOVER\derived\epic_map_part*.parquet"))
W("  parts: %d" % len(parts))
W("")

INV = lambda s: s.str.upper().str.contains("ART", na=False)


def classify(name_series):
    up = name_series.astype(str).str.upper()
    inv = up.str.contains("ART", na=False)
    nibp = up.str.contains("NIBP", na=False)
    generic = (~inv) & (~nibp) & up.str.contains("MAP", na=False)
    return inv, nibp, generic


# ==================================================================
# A. label x FLO_NAME, per-LOG_ID combinations, granularity
# ==================================================================
W("=" * 66)
W(" A. label semantics")
W("=" * 66)
Save()

lab_flo = {}
has = {}          # LOG_ID -> set of classes present
minute_hist = {}
hour_hist = {}
n_total = 0

for p in parts:
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=3_000_000,
                                 columns=["LOG_ID", "FLO_DISPLAY_NAME",
                                          "FLO_NAME", "recorded_time"]):
        df = batch.to_pandas()
        n_total += len(df)
        inv, nibp, gen = classify(df["FLO_DISPLAY_NAME"])
        for nm, mask, tag in (("FLO_DISPLAY_NAME", inv, "inv"),
                              ("FLO_DISPLAY_NAME", nibp, "nibp"),
                              ("FLO_DISPLAY_NAME", gen, "generic")):
            pass
        lab_flo_pairs = df.loc[inv | nibp | gen,
                               ["FLO_NAME", "FLO_DISPLAY_NAME"]].value_counts()
        for (f, d), c in lab_flo_pairs.items():
            lab_flo[(str(f), str(d))] = lab_flo.get((str(f), str(d)), 0) + int(c)

        for nm, mask in (("inv", inv), ("nibp", nibp), ("generic", gen)):
            for lid in df.loc[mask, "LOG_ID"].unique():
                has.setdefault(lid, set()).add(nm)

        ts = pd.to_datetime(df["recorded_time"])
        mh = ts.dt.minute.value_counts()
        for k, v in mh.items():
            minute_hist[int(k)] = minute_hist.get(int(k), 0) + int(v)
    W("    processed %s | total %s rows" % (os.path.basename(p), format(n_total, ",")))
    Save()

W("")
W("  --- FLO_NAME x label ---")
for (f, d), c in sorted(lab_flo.items(), key=lambda x: -x[1])[:25]:
    W("    %-28s | %-26s %s" % (f[:28], d[:26], format(c, ",")))
W("")
Save()

W("  --- how many LOG_IDs have which combination ---")
combo = {}
for lid, s in has.items():
    key = "+".join(sorted(s))
    combo[key] = combo.get(key, 0) + 1
for k, v in sorted(combo.items(), key=lambda x: -x[1]):
    W("    %-28s %s" % (k, format(v, ",")))
W("")
Save()

W("  --- time granularity: distribution of minute-of-hour ---")
tot = sum(minute_hist.values())
top = sorted(minute_hist.items(), key=lambda x: -x[1])[:10]
for m, c in top:
    W("    minute %2d : %12s  (%.1f%%)" % (m, format(c, ","), 100.0 * c / tot))
W("    minute==0 share : %.1f %%"
  % (100.0 * minute_hist.get(0, 0) / tot))
W("    distinct minutes present: %d / 60" % len(minute_hist))
W("")
Save()

# ==================================================================
# B. build the BP subset for the pairing cohort
# ==================================================================
W("=" * 66)
W(" B. build BP subset")
W("=" * 66)
Save()
# a LOG_ID qualifies if it has an invasive label AND a NIBP label
qual = [lid for lid, s in has.items() if "inv" in s and "nibp" in s]
qual_set = set(qual)
W("  LOG_IDs with BOTH invasive and NIBP labels: %s" % format(len(qual), ","))
W("")
Save()

if os.path.exists(BPFILE):
    bp = pd.read_parquet(BPFILE)
    W("  cached: %s rows" % format(len(bp), ","))
else:
    frames = []
    for p in parts:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=3_000_000,
                                     columns=["LOG_ID", "FLO_DISPLAY_NAME",
                                              "recorded_time", "map_value"]):
            df = batch.to_pandas()
            df = df[df["LOG_ID"].isin(qual_set)]
            if not len(df):
                continue
            inv, nibp, gen = classify(df["FLO_DISPLAY_NAME"])
            df = df[inv | nibp]
            if not len(df):
                continue
            up = df["FLO_DISPLAY_NAME"].astype(str).str.upper()
            df["cls"] = np.where(up.str.contains("ART", na=False), "art",
                                 np.where(up.str.contains("NIBP", na=False), "nibp", "other"))
            frames.append(df[["LOG_ID", "recorded_time", "map_value", "cls"]])
        W("    done %s" % os.path.basename(p))
        Save()
    bp = pd.concat(frames, ignore_index=True)
    del frames
    bp = bp[bp["cls"].isin(["art", "nibp"])]
    bp["tmin"] = ((bp["recorded_time"] - pd.Timestamp("1970-01-01"))
                  .dt.total_seconds() // 60).astype("int64")
    bp = bp[["LOG_ID", "tmin", "map_value", "cls"]]
    bp.to_parquet(BPFILE, index=False)
    W("  saved -> %s  (%s rows)" % (BPFILE, format(len(bp), ",")))
W("")
W("  rows by class: %s" % bp["cls"].value_counts().to_dict())
W("")
Save()

# ==================================================================
# C. pairing
# ==================================================================
W("=" * 66)
W(" C. pairing")
W("=" * 66)
Save()

art = (bp.loc[bp["cls"] == "art", ["LOG_ID", "tmin", "map_value"]]
         .rename(columns={"map_value": "map_art", "tmin": "art_t"})
         .reset_index(drop=True))
nib = (bp.loc[bp["cls"] == "nibp", ["LOG_ID", "tmin", "map_value"]]
         .rename(columns={"map_value": "map_nibp"})
         .reset_index(drop=True))
nib = nib.reset_index(drop=True)
nib["nid"] = np.arange(len(nib), dtype="int64")
W("  invasive readings     : %s" % format(len(art), ","))
W("  non-invasive readings : %s" % format(len(nib), ","))
W("")
Save()

for TOL in (5, 10):
    W("-" * 66)
    W(" TOLERANCE +/- %d min" % TOL)
    W("-" * 66)
    t1 = time.time()
    matched = np.zeros(len(nib), dtype=bool)
    frames = []
    for off in sorted(range(-TOL, TOL + 1), key=abs):
        sub = nib.loc[~matched, ["nid", "LOG_ID", "tmin", "map_nibp"]]
        if not len(sub):
            break
        left = sub.copy()
        left["art_t"] = left["tmin"] + off
        m = left.merge(art, on=["LOG_ID", "art_t"], how="inner")
        if len(m):
            m["gap"] = abs(off)
            frames.append(m)
            matched[m["nid"].values] = True
    if not frames:
        W("  no pairs at this tolerance")
        W("")
        continue
    pairs = pd.concat(frames, ignore_index=True)
    del frames
    W("  paired readings          : %s" % format(len(pairs), ","))
    W("  LOG_IDs with >=1 pair    : %s" % format(pairs["LOG_ID"].nunique(), ","))
    W("  share of NIBP paired     : %.1f %%" % (100.0 * len(pairs) / len(nib)))
    W("")
    for T in (65, 60, 55):
        al = int((pairs["map_art"] < T).sum())
        nl = int((pairs["map_nibp"] < T).sum())
        oc = int(((pairs["map_nibp"] >= T) & (pairs["map_art"] < T)).sum())
        fa = int(((pairs["map_nibp"] < T) & (pairs["map_art"] >= T)).sum())
        co = int(((pairs["map_nibp"] < T) & (pairs["map_art"] < T)).sum())
        W("  threshold %d mmHg" % T)
        W("     invasive-low         : %s" % format(al, ","))
        W("     MISSED (cuff normal) : %s  (%.1f%%)"
          % (format(oc, ","), (100.0 * oc / al) if al else 0.0))
        W("     cuff-low, art normal : %s  (%.1f%% of cuff-low)"
          % (format(fa, ","), (100.0 * fa / nl) if nl else 0.0))
        W("     concordant low       : %s" % format(co, ","))
        W("")
    d = (pairs["map_nibp"] - pairs["map_art"]).astype(float)
    W("  agreement (NIBP - invasive)")
    W("     bias %.2f  SD %.2f  95%% limits %.2f to %.2f"
      % (d.mean(), d.std(), d.mean() - 1.96 * d.std(), d.mean() + 1.96 * d.std()))
    W("     NIBP>inv : %.1f %%" % (100.0 * (d > 0).mean()))
    W("")
    W("  (took %.0f s)" % (time.time() - t1))
    W("")
    Save()
    if TOL == 5:
        pairs[["LOG_ID", "tmin", "art_t", "gap", "map_nibp", "map_art"]].to_parquet(PAIRF, index=False)
        W("  saved -> %s" % PAIRF)
        W("")
        Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
