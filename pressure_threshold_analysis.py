# -*- coding: utf-8 -*-
"""
Step 23 - same-minute pairing at all three thresholds, for the Results table.

Fills the gaps left by step14 (which reported gap=0 only at 65 mmHg).

Run:  python E:\\claude-tools\\step23_thresholds.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step23_thresholds_report.txt")

FILES = [
    ("MOVER (OR)",     os.path.join(DERIVED, "mover_pairs_t5.parquet"), "LOG_ID"),
    ("MIMIC-IV (ICU)", os.path.join(DERIVED, "mimic_pairs_t5.parquet"), "stay_id"),
    ("eICU (ICU)",     os.path.join(DERIVED, "eicu_pairs_t5.parquet"),  "patientunitstayid"),
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


sys.excepthook = lambda t, v, tb: (Save(),)

W("=" * 74)
W(" STEP 23 - same-minute pairing, all thresholds")
W("=" * 74)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

summary = []
for label, path, key in FILES:
    W("=" * 74)
    W(" %s" % label)
    W("=" * 74)
    if not os.path.exists(path):
        W("  missing")
        W("")
        continue
    p = pd.read_parquet(path)
    p = p[p["gap"] == 0]
    d = (p["map_nibp"] - p["map_art"]).astype(float)
    W("  same-minute pairs : %s" % format(len(p), ","))
    W("  cases             : %s" % format(p[key].nunique(), ","))
    W("  mean difference   : %+.2f mmHg   SD %.2f" % (d.mean(), d.std()))
    W("  95%% limits        : %+.2f to %+.2f" % (d.mean() - 1.96 * d.std(),
                                                 d.mean() + 1.96 * d.std()))
    W("  cuff > arterial   : %.1f %%" % (100.0 * (d > 0).mean()))
    W("")
    W("  %-8s %12s %12s %10s %12s %10s" %
      ("thr", "arterial-low", "cuff-low", "missed%", "unconfirmed", "cuff-low%"))
    for T in (65, 60, 55):
        al = int((p["map_art"] < T).sum())
        nl = int((p["map_nibp"] < T).sum())
        ms = int(((p["map_nibp"] >= T) & (p["map_art"] < T)).sum())
        uc = int(((p["map_nibp"] < T) & (p["map_art"] >= T)).sum())
        W("  <%-7d %12s %12s %9.1f%% %12s %9.1f%%"
          % (T, format(al, ","), format(nl, ","),
             (100.0 * ms / al) if al else np.nan,
             format(uc, ","),
             (100.0 * uc / nl) if nl else np.nan))
        summary.append(dict(db=label, thr=T, pairs=len(p), cases=p[key].nunique(),
                            al=al, nl=nl, missed=(100.0 * ms / al) if al else np.nan,
                            unconf=(100.0 * uc / nl) if nl else np.nan))
    W("")
    W("  threshold 60 and 55 for reference, +/-5min pairing (step13/11/5):")
    Save()

W("=" * 74)
W(" TIDY TABLE FOR THE MANUSCRIPT (same-minute pairing)")
W("=" * 74)
W("")
W("  %-16s %6s %10s %10s %9s %12s %10s" %
  ("database", "thr", "pairs", "cases", "missed%", "unconf_n", "unconf%"))
for r in summary:
    W("  %-16s %6d %10s %10s %9.1f %12s %9.1f"
      % (r["db"], r["thr"], format(r["pairs"], ","), format(r["cases"], ","),
         r["missed"], format(r["al"], ","), r["unconf"]))
W("")
W("=" * 74)
W(" DONE")
W("=" * 74)
Save()
print("")
print("Report saved to: %s" % REPORT)
