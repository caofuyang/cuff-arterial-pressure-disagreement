# -*- coding: utf-8 -*-
"""
Step 14 - is MOVER's high discordance real, or an artefact of the time lag?

Repeats the discordance analysis on all three databases using ONLY pairs
whose readings fall in the same minute (gap = 0).

  if MOVER's missed rate collapses towards the ICU value -> time-lag artefact
  if it stays high                                      -> genuine OR property

Run:  python E:\\claude-tools\\step14_gap0.py
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
REPORT  = os.path.join(AUDIT, "step14_gap0_report.txt")
THR     = 65

FILES = [
    ("MOVER (OR)",     os.path.join(DERIVED, "mover_pairs_t5.parquet"),  "LOG_ID"),
    ("MIMIC-IV (ICU)", os.path.join(DERIVED, "mimic_pairs_t5.parquet"),  "stay_id"),
    ("eICU (ICU)",     os.path.join(DERIVED, "eicu_pairs_t5.parquet"),   "patientunitstayid"),
]

os.makedirs(AUDIT, exist_ok=True)
lines = []


def W(t=""):
    s = str(t)
    print(s, flush=True)
    lines.append(s)


def Save():
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


sys.excepthook = lambda t, v, tb: (lines.append("".join(traceback.format_exception(t, v, tb))), Save())


def stats(p, key):
    if not len(p):
        return None
    d = (p["map_nibp"] - p["map_art"]).astype(float)
    al = int((p["map_art"] < THR).sum())
    nl = int((p["map_nibp"] < THR).sum())
    oc = int(((p["map_nibp"] >= THR) & (p["map_art"] < THR)).sum())
    fa = int(((p["map_nibp"] < THR) & (p["map_art"] >= THR)).sum())
    return dict(n=len(p), cases=p[key].nunique(), bias=d.mean(), sd=d.std(),
                missed=(100.0 * oc / al) if al else np.nan,
                fa=(100.0 * fa / nl) if nl else np.nan,
                hi=100.0 * (d > 0).mean(), invlow=al)


t0 = time.time()
W("=" * 70)
W(" STEP 14 - discordance at gap = 0 (same minute) vs all gaps")
W("=" * 70)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

rows_all, rows_g0 = [], []
for label, path, key in FILES:
    W("=" * 70)
    W(" %s" % label)
    W("=" * 70)
    if not os.path.exists(path):
        W("  file missing: %s" % path)
        W("")
        continue
    p = pd.read_parquet(path)
    W("  total pairs: %s   cases: %s"
      % (format(len(p), ","), format(p[key].nunique(), ",")))
    W("  gap distribution: %s"
      % p["gap"].value_counts().sort_index().to_dict())
    W("")

    a = stats(p, key)
    z = stats(p[p["gap"] == 0], key)
    g0share = 100.0 * (p["gap"] == 0).mean()
    W("  --- ALL gaps ---")
    if a:
        W("    n=%s  cases=%s  bias %+.2f  SD %.2f  missed %.1f%%  false-alarm %.1f%%  NIBP>inv %.1f%%"
          % (format(a["n"], ","), format(a["cases"], ","), a["bias"], a["sd"],
             a["missed"], a["fa"], a["hi"]))
    W("  --- gap = 0 only (%.1f%% of pairs) ---" % g0share)
    if z:
        W("    n=%s  cases=%s  bias %+.2f  SD %.2f  missed %.1f%%  false-alarm %.1f%%  NIBP>inv %.1f%%"
          % (format(z["n"], ","), format(z["cases"], ","), z["bias"], z["sd"],
             z["missed"], z["fa"], z["hi"]))
    W("")
    if a:
        a = dict(a); a["db"] = label; rows_all.append(a)
    if z:
        z = dict(z); z["db"] = label; rows_g0.append(z)
    Save()

W("=" * 70)
W(" SUMMARY")
W("=" * 70)
W("")
W("  ALL GAPS")
W("  %-16s %10s %10s %8s %8s %9s %9s" % ("database", "pairs", "cases", "bias", "SD", "missed%", "FA%"))
for r in rows_all:
    W("  %-16s %10s %10s %+8.2f %8.2f %9.1f %9.1f"
      % (r["db"], format(r["n"], ","), format(r["cases"], ","), r["bias"], r["sd"],
         r["missed"], r["fa"]))
W("")
W("  SAME-MINUTE ONLY (gap = 0)")
W("  %-16s %10s %10s %8s %8s %9s %9s" % ("database", "pairs", "cases", "bias", "SD", "missed%", "FA%"))
for r in rows_g0:
    W("  %-16s %10s %10s %+8.2f %8.2f %9.1f %9.1f"
      % (r["db"], format(r["n"], ","), format(r["cases"], ","), r["bias"], r["sd"],
         r["missed"], r["fa"]))
W("")
W("  INTERPRETATION")
W("    - if MOVER's missed%% falls from ~74%% towards ~44%% when restricted")
W("      to same-minute pairs -> the OR discordance was mostly time-lag")
W("    - if it stays high      -> discordance is genuinely larger in the OR")
W("")
W("=" * 70)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 70)
Save()
print("")
print("Report saved to: %s" % REPORT)
