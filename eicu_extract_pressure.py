# -*- coding: utf-8 -*-
"""
Step 11 - eICU-CRD: extract nurseCharting BP, build pairs, quantify discordance.

eICU stores BP in the long-format nurseCharting table:
  "Non-Invasive BP Mean"   celltypevalname
  "Invasive BP Mean"       celltypevalname

This replicates the MIMIC-IV analysis in an independent, 208-hospital cohort.

Run:  python E:\\claude-tools\\step11_eicu.py
"""
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step11_eicu_report.txt")
NC      = r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\nurseCharting.csv.gz"
WIDEF   = os.path.join(DERIVED, "eicu_bp_wide.parquet")
PAIRF   = os.path.join(DERIVED, "eicu_pairs_t5.parquet")
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
W(" STEP 11 - eICU-CRD nurseCharting blood pressure")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

# ==================================================================
# A. extract
# ==================================================================
if os.path.exists(WIDEF):
    wide = pd.read_parquet(WIDEF)
    W("  cached wide table: %s rows" % format(len(wide), ","))
else:
    W("  file: %s (%.1f MB)" % (NC, os.path.getsize(NC) / 1e6))
    target = {"Non-Invasive BP Mean": "nibp_m",
              "Invasive BP Mean": "art_m",
              "Non-Invasive BP Systolic": "nibp_s",
              "Invasive BP Systolic": "art_s"}
    parts = []
    scanned = 0
    kept = 0
    t1 = time.time()
    reader = pd.read_csv(
        NC, compression="gzip", chunksize=4_000_000,
        usecols=["patientunitstayid", "nursingchartoffset",
                 "nursingchartcelltypevalname", "nursingchartvalue"],
        dtype={"patientunitstayid": "int64", "nursingchartoffset": "int64",
               "nursingchartcelltypevalname": "string",
               "nursingchartvalue": "string"},
    )
    for chunk in reader:
        scanned += len(chunk)
        sub = chunk[chunk["nursingchartcelltypevalname"].isin(target.keys())]
        if len(sub):
            kept += len(sub)
            parts.append(sub)
        if scanned % 20_000_000 < 4_000_000:
            W("    scanned %13s | kept %10s | %.0fs"
              % (format(scanned, ","), format(kept, ","), time.time() - t1))
            Save()
    W("")
    W("  scanned %s rows" % format(scanned, ","))
    W("  kept    %s BP rows" % format(kept, ","))
    W("")
    Save()

    raw = pd.concat(parts, ignore_index=True)
    del parts
    raw["val"] = pd.to_numeric(raw["nursingchartvalue"], errors="coerce")
    raw["item"] = raw["nursingchartcelltypevalname"].map(target)
    raw = raw.dropna(subset=["val"])
    raw = raw[(raw["val"] >= 20) & (raw["val"] <= 200)]
    W("  after numeric + range filter: %s rows" % format(len(raw), ","))
    W("  by item: %s" % raw["item"].value_counts().to_dict())
    W("")
    Save()

    # collapse duplicates, then pivot long -> wide
    raw = (raw.groupby(["patientunitstayid", "nursingchartoffset", "item"],
                       observed=True)["val"].mean().reset_index())
    wide = raw.pivot_table(index=["patientunitstayid", "nursingchartoffset"],
                           columns="item", values="val",
                           aggfunc="mean").reset_index()
    wide.columns.name = None
    wide.to_parquet(WIDEF, index=False)
    W("  wide table: %s rows" % format(len(wide), ","))
    W("  saved -> %s" % WIDEF)
    W("")
    Save()

W("  columns: %s" % ", ".join(wide.columns.tolist()))
for c in wide.columns:
    if c != "patientunitstayid" and c != "nursingchartoffset":
        W("    %-8s non-null %s" % (c, format(int(wide[c].notna().sum()), ",")))
W("")
Save()

# ==================================================================
# B. per-stay availability
# ==================================================================
W("=" * 66)
W(" B. availability")
W("=" * 66)
g = wide.groupby("patientunitstayid")
has_art = g["art_m"].apply(lambda s: s.notna().any())
has_nib = g["nibp_m"].apply(lambda s: s.notna().any())
both = has_art & has_nib
W("  stays with invasive BP mean     : %s" % format(int(has_art.sum()), ","))
W("  stays with non-invasive BP mean : %s" % format(int(has_nib.sum()), ","))
W("  stays with BOTH                 : %s   <== eICU core cohort" % format(int(both.sum()), ","))
W("")
Save()

# ==================================================================
# C. pairing
# ==================================================================
W("=" * 66)
W(" C. pairing (1-min offsets)")
W("=" * 66)
W("")
Save()

art = (wide.loc[wide["art_m"].notna(), ["patientunitstayid", "nursingchartoffset", "art_m"]]
         .rename(columns={"nursingchartoffset": "art_t", "art_m": "map_art"})
         .reset_index(drop=True))
nib = (wide.loc[wide["nibp_m"].notna(), ["patientunitstayid", "nursingchartoffset", "nibp_m"]]
         .rename(columns={"nibp_m": "map_nibp"})
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
        sub = nib.loc[~matched, ["nid", "patientunitstayid",
                                 "nursingchartoffset", "map_nibp"]]
        if not len(sub):
            break
        left = sub.copy()
        left["art_t"] = left["nursingchartoffset"] + off
        m = left.merge(art, on=["patientunitstayid", "art_t"], how="inner")
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
    W("  paired readings          : %s" % format(len(pairs), ","))
    W("  stays with >=1 pair      : %s" % format(pairs["patientunitstayid"].nunique(), ","))
    W("  share of NIBP paired     : %.1f %%" % (100.0 * len(pairs) / len(nib)))
    W("")

    for T in (65, 60, 55):
        al = int((pairs["map_art"] < T).sum())
        nl = int((pairs["map_nibp"] < T).sum())
        oc = int(((pairs["map_nibp"] >= T) & (pairs["map_art"] < T)).sum())
        fa = int(((pairs["map_nibp"] < T) & (pairs["map_art"] >= T)).sum())
        co = int(((pairs["map_nibp"] < T) & (pairs["map_art"] < T)).sum())
        W("  threshold %d mmHg" % T)
        W("     invasive-low           : %s" % format(al, ","))
        W("     MISSED (cuff normal)   : %s  (%.1f%%)"
          % (format(oc, ","), (100.0 * oc / al) if al else 0.0))
        W("     cuff-low, art normal   : %s  (%.1f%% of cuff-low)"
          % (format(fa, ","), (100.0 * fa / nl) if nl else 0.0))
        W("     concordant low         : %s" % format(co, ","))
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
        pairs[["patientunitstayid", "nursingchartoffset", "art_t", "gap",
               "map_nibp", "map_art"]].to_parquet(PAIRF, index=False)
        W("  saved -> %s" % PAIRF)
        W("")
        Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
