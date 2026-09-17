# -*- coding: utf-8 -*-
"""
Step 16 - MOVER prognosis: is the inverted positive control caused by the
          post-op baseline rule?

Restricts to cases with a genuine PRE-operative baseline creatinine and
re-tests the positive control.

Run:  python E:\\claude-tools\\step16_mover_sens.py
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
REPORT  = os.path.join(AUDIT, "step16_mover_sens_report.txt")
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


sys.excepthook = lambda t, v, tb: (lines.append("".join(traceback.format_exception(t, v, tb))), Save())

W("=" * 66)
W(" STEP 16 - MOVER prognosis sensitivity")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

pairs = pd.read_parquet(os.path.join(DERIVED, "mover_pairs_t5.parquet"))
pairs["art_low"] = pairs["map_art"] < THR
pairs["occult"] = (pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)
pairs["conc"] = (pairs["map_nibp"] < THR) & (pairs["map_art"] < THR)
pairs = pairs.sort_values(["LOG_ID", "tmin"]).reset_index(drop=True)

rows = []
for lid, g in pairs.groupby("LOG_ID", sort=False):
    t = g["tmin"].values.astype("float64")
    a = g["map_art"].values.astype("float64")
    oc = g["occult"].values
    cc = g["conc"].values
    n = len(g)
    dt = np.diff(t) if n >= 2 else np.array([])
    dt = np.append(dt, dt[-1]) if n >= 2 else np.array([30.0])
    dt = np.clip(dt, 0, 60)
    defi = np.maximum(THR - a, 0.0)
    rows.append(dict(LOG_ID=lid, n_pairs=n, monitored_min=float(dt.sum()),
                     b_all=float((defi * dt).sum()),
                     b_occ=float((defi * dt * oc).sum()),
                     b_con=float((defi * dt * cc).sum()),
                     min_art=float(a.min()), n_low=int(g["art_low"].sum()),
                     mean_art=float(a.mean())))
exp = pd.DataFrame(rows)
for c in [c for c in exp.columns if c.startswith("b_")]:
    exp[c + "_ph"] = exp[c] / (exp["monitored_min"] / 60.0).replace(0, np.nan)

aki = pd.read_parquet(os.path.join(DERIVED, "mover_aki.parquet"))
d = exp.merge(aki, on="LOG_ID", how="inner")
W("  all cases with pairs + AKI : %s  (AKI %.1f%%)"
  % (format(len(d), ","), 100.0 * d["aki"].mean()))
W("")
Save()

try:
    import statsmodels.api as sm
except Exception:
    W("  statsmodels missing")
    Save(); raise SystemExit


def _fit(y, X):
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    mod = sm.Logit(y, X)
    for kw in ({"disp": 0}, {"disp": False}, {}):
        try:
            return mod.fit(**kw)
        except TypeError:
            continue
    raise RuntimeError("fit failed")


def one(df, col, label, cov=("baseline", "n_pairs")):
    s = df.dropna(subset=["aki", col] + list(cov)).copy()
    X = s[[col] + list(cov)].astype(float)
    X[col] /= 10.0
    X = sm.add_constant(X)
    m = _fit(s["aki"].astype(int), X)
    b, se = m.params[col], m.bse[col]
    W("    %-30s OR %.3f (%.3f-%.3f)  p=%.4f  n=%s"
      % (label, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
         m.pvalues[col], format(len(s), ",")))


for name, sub in (("ALL (n_pairs>=5)",
                   d[d["n_pairs"] >= 5]),
                  ("PRE-OP BASELINE ONLY, n_pairs>=5",
                   d[(d["n_pairs"] >= 5) & (d["base_src"] == "pre")]),
                  ("PRE-OP BASELINE + n_cr>=3, n_pairs>=5",
                   d[(d["n_pairs"] >= 5) & (d["base_src"] == "pre") & (d["n_cr"] >= 3)])):
    W("=" * 66)
    W(" %s" % name)
    W("=" * 66)
    W("  n=%s   AKI %.1f%%" % (format(len(sub), ","), 100.0 * sub["aki"].mean() if len(sub) else np.nan))
    W("")
    if len(sub) < 100:
        W("  too few")
        W("")
        continue
    one(sub, "b_all_ph", "POSITIVE CONTROL total burden")
    one(sub, "b_occ_ph", "occult burden")
    one(sub, "b_con_ph", "concordant burden")
    one(sub, "min_art", "lowest arterial MAP (per mmHg)", cov=("baseline", "n_pairs"))
    W("")
    Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - 0) if False else " DONE")
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
