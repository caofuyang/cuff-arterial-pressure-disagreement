# -*- coding: utf-8 -*-
"""
Step 3b - blood pressure value granularity check (memory-frugal).

Changes vs step3:
  * accumulate value_counts() instead of raw arrays  -> tiny memory
  * report file is flushed to disk after EVERY section
  * every section wrapped separately

Run:  python E:\\claude-tools\\step3_value_check.py
"""
import os
import glob
import time
import traceback
import numpy as np
import pandas as pd

PROJ   = r"E:\occult_htn"
AUDIT  = os.path.join(PROJ, "audit")
REPORT = os.path.join(AUDIT, "step3_value_check_report.txt")
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
        print("could not save report: %r" % e)


class Counter(object):
    """Memory-frugal exact value counts."""
    def __init__(self):
        self.d = {}

    def add(self, series):
        vc = series.dropna().value_counts()
        for v, c in vc.items():
            self.d[v] = self.d.get(v, 0) + int(c)

    @property
    def n(self):
        return sum(self.d.values())

    @property
    def distinct(self):
        return len(self.d)

    def top(self, k=12):
        items = sorted(self.d.items(), key=lambda x: -x[1])[:k]
        return items

    def describe(self, name, k=12):
        if not self.d:
            W("  %s : no data" % name)
            return None
        nd = self.distinct
        n = self.n
        vals = np.array(list(self.d.keys()), dtype=float)
        W("  --- %s ---" % name)
        W("    n            : %s" % format(n, ","))
        W("    n distinct   : %s" % format(nd, ","))
        W("    min / max    : %.2f / %.2f" % (vals.min(), vals.max()))
        W("    all integer  : %s" % bool((vals % 1 == 0).all()))
        top = self.top(k)
        W("    top values   : %s" % ", ".join("%.1f(x%s)" % (v, format(c, ","))
                                               for v, c in top))
        share = 100.0 * sum(c for _, c in top) / n
        W("    top %d share  : %.1f %%" % (k, share))
        W("    VERDICT      : %s"
          % ("SUSPECT (quantised)" if nd < 100 else "looks RAW (continuous)"))
        W("")
        return nd


t0 = time.time()
W("=" * 68)
W(" STEP 3b - BP value granularity (memory-frugal)")
W("=" * 68)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

verdicts = {}

# ==================================================================
# 1. MOVER
# ==================================================================
W("=" * 68)
W(" 1. MOVER")
W("=" * 68)
Save()
try:
    mdir = r"E:\MOVER\derived"
    parts = sorted(glob.glob(os.path.join(mdir, "epic_map_part*.parquet")))
    W("  epic_map_part files: %d" % len(parts))
    for p in parts:
        W("    %s" % os.path.basename(p))
    W("")
    Save()

    if parts:
        first = pd.read_parquet(parts[0])
        W("  columns of %s:" % os.path.basename(parts[0]))
        W("    %s" % ", ".join(first.columns.tolist()))
        W("    dtypes: %s" % first.dtypes.to_dict())
        W("    rows  : %s" % format(len(first), ","))
        W("")
        W("  head:")
        W(first.head(3).to_string())
        W("")
        Save()

        numcols = [c for c in first.columns
                   if pd.api.types.is_numeric_dtype(first[c])
                   and first[c].nunique() > 2]
        W("  numeric columns: %s" % numcols)
        W("")

        # pick the column most likely to be MAP (values mostly 10-250)
        valcol = None
        best = -1.0
        for c in numcols:
            frac = float(first[c].between(10, 250).mean())
            if frac > best:
                best, valcol = frac, c
        W("  chosen value column: %r  (%.0f%% of values in 10-250)" % (valcol, best * 100))
        W("")
        Save()

        if valcol:
            ctr = Counter()
            for p in parts:
                d = pd.read_parquet(p, columns=[valcol])
                ctr.add(d[valcol])
                W("    read %s (%s rows)" % (os.path.basename(p), format(len(d), ",")))
            W("")
            verdicts["MOVER"] = ctr.describe("MOVER epic_map '%s'" % valcol)
        Save()

    fl = r"E:\MOVER\data\flowsheets_cleaned\flowsheet_part1.csv"
    if os.path.exists(fl):
        W("  raw flowsheet present: %s (%.1f MB)"
          % (fl, os.path.getsize(fl) / 1e6))
        head = pd.read_csv(fl, nrows=3000)
        W("    columns: %s" % ", ".join(head.columns.tolist())[:500])
        W("")
        Save()
except Exception:
    W("  MOVER section FAILED:")
    W(traceback.format_exc())
    Save()

W("")
Save()

# ==================================================================
# 2. MIMIC-IV
# ==================================================================
W("=" * 68)
W(" 2. MIMIC-IV chartevents  (ABPm=220052, NBPm=220181)")
W("=" * 68)
Save()
try:
    ce = r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\chartevents.csv.gz"
    if not os.path.exists(ce):
        W("  not found: %s" % ce)
    else:
        W("  file size: %.1f MB" % (os.path.getsize(ce) / 1e6))
        W("  scanning (chunks of 2M, counts only - low memory)...")
        W("")
        Save()

        c_abp = Counter()
        c_nbp = Counter()
        scanned = 0
        t1 = time.time()
        budget = 1500

        reader = pd.read_csv(
            ce, compression="gzip", chunksize=2_000_000,
            usecols=["itemid", "valuenum"],
            dtype={"itemid": "int32", "valuenum": "float64"},
        )
        for chunk in reader:
            scanned += len(chunk)
            c_abp.add(chunk.loc[chunk["itemid"] == 220052, "valuenum"])
            c_nbp.add(chunk.loc[chunk["itemid"] == 220181, "valuenum"])
            W("    scanned %14s | ABPm %9s (%4d uniq) | NBPm %9s (%4d uniq) | %.0fs"
              % (format(scanned, ","), format(c_abp.n, ","), c_abp.distinct,
                 format(c_nbp.n, ","), c_nbp.distinct, time.time() - t1))
            Save()
            if time.time() - t1 > budget:
                W("    time budget reached, stopping early")
                break

        W("")
        W("  scanned total: %s rows" % format(scanned, ","))
        W("")
        if c_abp.n:
            verdicts["MIMIC-ABPm"] = c_abp.describe("MIMIC-IV ABPm (220052)")
        if c_nbp.n:
            verdicts["MIMIC-NBPm"] = c_nbp.describe("MIMIC-IV NBPm (220181)")
except Exception:
    W("  MIMIC section FAILED:")
    W(traceback.format_exc())
Save()

W("")
Save()

# ==================================================================
# 3. eICU
# ==================================================================
W("=" * 68)
W(" 3. eICU-CRD vitalPeriodic")
W("=" * 68)
Save()
try:
    vp = r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\vitalPeriodic.csv.gz"
    if not os.path.exists(vp):
        W("  not found: %s" % vp)
    else:
        W("  file size: %.1f MB" % (os.path.getsize(vp) / 1e6))
        cols = pd.read_csv(vp, compression="gzip", nrows=5).columns.tolist()
        W("  columns: %s" % ", ".join(cols))
        W("")
        Save()

        c_sys = Counter()
        c_non = Counter()
        scanned = 0
        reader = pd.read_csv(vp, compression="gzip", chunksize=2_000_000)
        for chunk in reader:
            scanned += len(chunk)
            if "systemicmean" in chunk.columns:
                c_sys.add(chunk["systemicmean"])
            if "noninvasivemean" in chunk.columns:
                c_non.add(chunk["noninvasivemean"])
            W("    scanned %12s | systemic %9s (%4d uniq) | noninvasive %9s (%4d uniq)"
              % (format(scanned, ","), format(c_sys.n, ","), c_sys.distinct,
                 format(c_non.n, ","), c_non.distinct))
            Save()

        W("")
        W("  scanned total: %s rows" % format(scanned, ","))
        W("")
        if c_sys.n:
            verdicts["eICU-systemicmean"] = c_sys.describe("eICU systemicmean (invasive)")
        if c_non.n:
            verdicts["eICU-noninvasivemean"] = c_non.describe("eICU noninvasivemean")
except Exception:
    W("  eICU section FAILED:")
    W(traceback.format_exc())
Save()

W("")

# ==================================================================
# SUMMARY
# ==================================================================
W("=" * 68)
W(" SUMMARY")
W("=" * 68)
W("")
W("  %-24s %-10s %s" % ("variable", "distinct", "verdict"))
for k in sorted(verdicts):
    v = verdicts[k]
    W("  %-24s %-10s %s" % (k, format(v, ",") if v else "n/a",
                            "SUSPECT" if (v or 0) < 100 else "OK"))
W("")
W("  Reference: INSPIRE art_mbp = 20 distinct, nibp_mbp = 21 distinct.")
W("  < 100 distinct  -> quantised, absolute thresholds unusable")
W("  > 200 distinct  -> raw, suitable")
W("")
W("=" * 68)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 68)
Save()
print("")
print("Report saved to: %s" % REPORT)
