# -*- coding: utf-8 -*-
"""
Step 5 - MIMIC-IV: extract ABPm/NBPm, pair them, quantify occult hypotension.

MIMIC-IV is the cleanest of the three databases (well-defined itemids,
raw values). If the phenomenon is real it should show here with a
physiologically plausible Bland-Altman bias (+/- 2-5 mmHg).

Run:  python E:\\claude-tools\\step5_mimic.py
"""
import os
import time
import traceback
import numpy as np
import pandas as pd

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step5_mimic_report.txt")
BPFILE  = os.path.join(DERIVED, "mimic_bp.parquet")
PAIRF   = os.path.join(DERIVED, "mimic_pairs_t5.parquet")

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
W(" STEP 5 - MIMIC-IV paired blood pressure")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

CE = r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\chartevents.csv.gz"
ITEM_ABP = 220052
ITEM_NBP = 220181

# ==================================================================
# A. extract
# ==================================================================
W("=" * 66)
W(" A. extract ABPm / NBPm from chartevents")
W("=" * 66)
Save()
try:
    if os.path.exists(BPFILE):
        W("  cached parquet found, loading: %s" % BPFILE)
        bp = pd.read_parquet(BPFILE)
        W("  loaded %s rows" % format(len(bp), ","))
    else:
        W("  file: %s (%.1f MB)" % (CE, os.path.getsize(CE) / 1e6))
        W("  scanning...")
        W("")
        Save()

        parts = []
        scanned = 0
        t1 = time.time()
        reader = pd.read_csv(
            CE, compression="gzip", chunksize=3_000_000,
            usecols=["stay_id", "charttime", "itemid", "valuenum"],
            dtype={"stay_id": "int64", "itemid": "int32", "valuenum": "float64"},
            parse_dates=["charttime"],
        )
        for chunk in reader:
            scanned += len(chunk)
            sub = chunk[chunk["itemid"].isin((ITEM_ABP, ITEM_NBP))]
            if len(sub):
                parts.append(sub)
            W("    scanned %14s | kept %9s | %.0fs"
              % (format(scanned, ","),
                 format(sum(len(x) for x in parts), ","), time.time() - t1))
            Save()

        W("")
        bp = pd.concat(parts, ignore_index=True)
        del parts
        W("  scanned total : %s" % format(scanned, ","))
        W("  kept rows     : %s" % format(len(bp), ","))
        Save()

        bp["tmin"] = ((bp["charttime"] - pd.Timestamp("1970-01-01"))
                      .dt.total_seconds() // 60).astype("int64")
        bp = bp[["stay_id", "charttime", "tmin", "itemid", "valuenum"]]
        bp.to_parquet(BPFILE, index=False)
        W("  saved -> %s" % BPFILE)
        W("")
        Save()

    W("  rows by itemid: %s" % bp["itemid"].value_counts().to_dict())
    W("  stays with ABPm: %s" % format(bp.loc[bp.itemid == ITEM_ABP, "stay_id"].nunique(), ","))
    W("  stays with NBPm: %s" % format(bp.loc[bp.itemid == ITEM_NBP, "stay_id"].nunique(), ","))
    W("")
    Save()

except Exception:
    W("  extraction FAILED:")
    W(traceback.format_exc())
    Save()
    raise SystemExit

# ==================================================================
# B. plausibility filter
# ==================================================================
W("=" * 66)
W(" B. plausibility filter")
W("=" * 66)
W("  raw value ranges:")
for iid, nm in ((ITEM_ABP, "ABPm"), (ITEM_NBP, "NBPm")):
    s = bp.loc[bp.itemid == iid, "valuenum"]
    W("    %s  n=%s  min=%.1f  p1=%.1f  median=%.1f  p99=%.1f  max=%.1f"
      % (nm, format(len(s), ","), s.min(), s.percentile(1) if hasattr(s, "percentile") else np.percentile(s, 1),
         s.median(), np.percentile(s, 99), s.max()))
W("")
n0 = len(bp)
bp = bp[(bp["valuenum"] >= 20) & (bp["valuenum"] <= 200)]
W("  after keeping 20-200 mmHg : %s rows (dropped %s)"
  % (format(len(bp), ","), format(n0 - len(bp), ",")))
W("")
Save()

art = (bp.loc[bp.itemid == ITEM_ABP, ["stay_id", "tmin", "valuenum"]]
         .rename(columns={"valuenum": "map_art", "tmin": "art_t"})
         .sort_values(["stay_id", "art_t"]).reset_index(drop=True))
nib = (bp.loc[bp.itemid == ITEM_NBP, ["stay_id", "tmin", "valuenum"]]
         .rename(columns={"valuenum": "map_nibp"})
         .sort_values(["stay_id", "tmin"]).reset_index(drop=True))
nib = nib.reset_index(drop=True)
nib["nid"] = np.arange(len(nib), dtype="int64")

W("  invasive readings    : %s" % format(len(art), ","))
W("  non-invasive readings: %s" % format(len(nib), ","))
W("")
Save()

# ==================================================================
# C. pairing
# ==================================================================
W("=" * 66)
W(" C. pairing (1-minute offsets)")
W("=" * 66)
W("")
Save()

for TOL in (5, 10, 15):
    W("-" * 66)
    W(" TOLERANCE +/- %d min" % TOL)
    W("-" * 66)
    t1 = time.time()

    matched = np.zeros(len(nib), dtype=bool)
    frames = []
    for off in sorted(range(-TOL, TOL + 1), key=abs):
        sub = nib.loc[~matched, ["nid", "stay_id", "tmin", "map_nibp"]]
        if not len(sub):
            break
        left = sub.copy()
        left["art_t"] = left["tmin"] + off
        m = left.merge(art, on=["stay_id", "art_t"], how="inner")
        if len(m):
            m["gap"] = abs(off)
            frames.append(m)
            matched[m["nid"].values] = True

    if not frames:
        W("  no pairs")
        W("")
        continue

    pairs = pd.concat(frames, ignore_index=True)
    del frames
    n_pairs = len(pairs)
    W("  paired readings          : %s" % format(n_pairs, ","))
    W("  stays with >=1 pair      : %s" % format(pairs["stay_id"].nunique(), ","))
    W("  share of NIBP paired     : %.1f %%" % (100.0 * n_pairs / len(nib)))
    W("  gap distribution         : %s"
      % pairs["gap"].value_counts().sort_index().head(8).to_dict())
    W("")

    for THR in (65, 60, 55):
        occ = int(((pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)).sum())
        con = int(((pairs["map_nibp"] < THR) & (pairs["map_art"] < THR)).sum())
        fal = int(((pairs["map_nibp"] < THR) & (pairs["map_art"] >= THR)).sum())
        bn = int(((pairs["map_nibp"] >= THR) & (pairs["map_art"] >= THR)).sum())
        al = int((pairs["map_art"] < THR).sum())
        W("  threshold %d mmHg" % THR)
        W("     invasive low                  : %s" % format(al, ","))
        W("     concordant low                : %s  (%.1f%%)"
          % (format(con, ","), (100.0 * con / al) if al else 0.0))
        W("     OCCULT (NIBP normal, art low) : %s  (%.1f%%)"
          % (format(occ, ","), (100.0 * occ / al) if al else 0.0))
        W("     NIBP low, art normal          : %s" % format(fal, ","))
        W("     both normal                   : %s" % format(bn, ","))
        W("")

    d = (pairs["map_nibp"] - pairs["map_art"]).astype(float)
    bias, sd = float(d.mean()), float(d.std())
    W("  agreement (NIBP - invasive)")
    W("     bias       : %+.2f mmHg   <== should be +2 to +5 if the method is sound" % bias)
    W("     SD         : %.2f mmHg" % sd)
    W("     95%% limits : %+.2f to %+.2f" % (bias - 1.96 * sd, bias + 1.96 * sd))
    W("     NIBP>inv   : %.1f %%" % (100.0 * float((d > 0).mean())))
    W("")

    THR = 65
    pairs["art_low"] = pairs["map_art"] < THR
    pairs["occult"] = (pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)
    per = pairs.groupby("stay_id").agg(n=("occult", "size"),
                                       low=("art_low", "sum"),
                                       occ=("occult", "sum"))
    wl = per[per["low"] > 0]
    W("  per ICU stay (threshold 65)")
    W("     stays with >=1 pair     : %s" % format(len(per), ","))
    W("     ... with >=1 art-low    : %s" % format(len(wl), ","))
    if len(wl):
        fr = wl["occ"] / wl["low"]
        W("     ALL hypotensive missed  : %s" % format(int((fr >= 1).sum()), ","))
        W("     NONE missed             : %s" % format(int((fr <= 0).sum()), ","))
        W("     median missed share     : %.0f %%" % (100.0 * float(fr.median())))
    W("")
    W("  (took %.0f s)" % (time.time() - t1))
    W("")
    Save()

    if TOL == 5:
        keep = pairs[["stay_id", "tmin", "art_t", "gap", "map_nibp", "map_art"]]
        keep.to_parquet(PAIRF, index=False)
        W("  saved +/-5min pairs -> %s (%s rows)" % (PAIRF, format(len(keep), ",")))
        W("")
        Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
