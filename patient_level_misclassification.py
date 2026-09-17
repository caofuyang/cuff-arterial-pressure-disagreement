# -*- coding: utf-8 -*-
"""
Step 19 - patient-level misclassification: how many patients would be
          completely missed (or falsely alarmed) by one modality alone?

"44% of readings disagree" is abstract. This converts it into:
  - of patients who had arterial hypotension, how many never showed a
    single low cuff reading?
  - of patients whose cuff showed hypotension, how many had no arterial
    hypotension at all?

Run:  python E:\\claude-tools\\step19_misclass.py
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
REPORT  = os.path.join(AUDIT, "step19_misclass_report.txt")

FILES = [
    ("MOVER (OR)",     os.path.join(DERIVED, "mover_pairs_t5.parquet"), "LOG_ID", "tmin"),
    ("MIMIC-IV (ICU)", os.path.join(DERIVED, "mimic_pairs_t5.parquet"), "stay_id", "tmin"),
    ("eICU (ICU)",     os.path.join(DERIVED, "eicu_pairs_t5.parquet"),  "patientunitstayid", "nursingchartoffset"),
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

t0 = time.time()
W("=" * 72)
W(" STEP 19 - patient-level misclassification")
W("=" * 72)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
W("  For every case, ask: did the arterial line ever show hypotension?")
W("                       did the cuff ever show hypotension?")
W("")
Save()

for label, path, key, tcol in FILES:
    W("=" * 72)
    W(" %s" % label)
    W("=" * 72)
    if not os.path.exists(path):
        W("  missing: %s" % path)
        W("")
        continue
    p = pd.read_parquet(path)
    if "gap" in p.columns:
        p = p[p["gap"] == 0]
    W("  same-minute paired readings: %s" % format(len(p), ","))
    W("")

    for MINP in (5,):
        g = p.groupby(key)
        agg = g.agg(n=("map_art", "size"),
                    min_art=("map_art", "min"),
                    min_nib=("map_nibp", "min"))
        s = agg[agg["n"] >= MINP].copy()
        W("  --- cases with >= %d paired readings : %s ---" % (MINP, format(len(s), ",")))
        W("")

        for THR in (65, 60, 55):
            art_low_any = s["min_art"] < THR
            nib_low_any = s["min_nib"] < THR
            n_art = int(art_low_any.sum())
            n_nib = int(nib_low_any.sum())
            missed_all = int((art_low_any & (~nib_low_any)).sum())
            unconfirmed = int((nib_low_any & (~art_low_any)).sum())
            W("    threshold MAP < %d mmHg" % THR)
            W("      cases whose ARTERIAL pressure fell below %d : %s" % (THR, format(n_art, ",")))
            W("        ... of whom the CUFF never once showed low    : %s  (%.1f%%)"
              % (format(missed_all, ","), (100.0 * missed_all / n_art) if n_art else 0.0))
            W("      cases whose CUFF pressure fell below %d      : %s" % (THR, format(n_nib, ",")))
            W("        ... of whom the ARTERIAL line never showed low: %s  (%.1f%%)"
              % (format(unconfirmed, ","), (100.0 * unconfirmed / n_nib) if n_nib else 0.0))
            W("")

        # how deep does the completely-missed group actually get?
        miss = s[(s["min_art"] < 65) & (s["min_nib"] >= 65)]
        if len(miss):
            W("    depth among cases whose hypotension was ENTIRELY invisible to the cuff:")
            W("      n = %s" % format(len(miss), ","))
            for lo, hi, nm in ((0, 55, "<55"), (55, 60, "55-60"), (60, 65, "60-65")):
                c = int(((miss["min_art"] >= lo) & (miss["min_art"] < hi)).sum())
                W("        lowest arterial MAP %-6s : %s  (%.1f%%)"
                  % (nm, format(c, ","), 100.0 * c / len(miss)))
            W("      median lowest arterial MAP: %.0f mmHg" % miss["min_art"].median())
        W("")
        Save()

    # overall (not restricted)
    g = p.groupby(key)
    agg = g.agg(n=("map_art", "size"), min_art=("map_art", "min"), min_nib=("map_nibp", "min"))
    W("  --- ALL cases regardless of pair count : %s ---" % format(len(agg), ","))
    art_low = agg["min_art"] < 65
    nib_low = agg["min_nib"] < 65
    na, nn = int(art_low.sum()), int(nib_low.sum())
    ma = int((art_low & (~nib_low)).sum())
    uc = int((nib_low & (~art_low)).sum())
    W("    arterial low : %s   entirely cuff-invisible: %s (%.1f%%)"
      % (format(na, ","), format(ma, ","), (100.0 * ma / na) if na else 0.0))
    W("    cuff low     : %s   arterial-unconfirmed  : %s (%.1f%%)"
      % (format(nn, ","), format(uc, ","), (100.0 * uc / nn) if nn else 0.0))
    W("")
    Save()

W("=" * 72)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 72)
Save()
print("")
print("Report saved to: %s" % REPORT)
