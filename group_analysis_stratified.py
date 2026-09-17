# -*- coding: utf-8 -*-
"""
Step 30 - address the methodological asymmetry in the four-group analysis.

The continuous exposure (lowest arterial MAP) was defended against the
"more readings mechanically lower the minimum" objection by stratification.
The four-group analysis uses the same minimum but was not. This script:

  1. stratifies the four-group analysis by number of paired readings
  2. repeats it at different cohort thresholds (>=2, >=5, >=10, >=20)
  3. repeats it using the 5th percentile instead of the minimum
  4. adds sex to the adjustment set
  5. stratifies by sex (effect modification)

Run:  python E:\\claude-tools\\step30_groups_stratified.py
"""
import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step30_groups_stratified_report.txt")

os.makedirs(AUDIT, exist_ok=True)
lines = []


def W(t=""):
    s = str(t); print(s, flush=True); lines.append(s)


def Save():
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


sys.excepthook = lambda t, v, tb: (lines.append("".join(traceback.format_exception(t, v, tb))), Save())


def _fit(y, X):
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    if "const" not in X.columns:
        X = sm.add_constant(X)
    mod = sm.Logit(y, X)
    for kw in ({"method": "lbfgs", "maxiter": 2000, "disp": 0},
               {"method": "lbfgs", "maxiter": 2000},
               {"disp": 0}, {}):
        try:
            m = mod.fit(**kw)
            try:
                if m.mle_retvals.get("converged", True) is False:
                    continue
            except Exception:
                pass
            return m
        except TypeError:
            continue
    raise RuntimeError("fit failed")


def metrics(pairs, key):
    rows = []
    for sid, g in pairs.groupby(key, sort=False):
        a = g["map_art"].values.astype("float64")
        n = g["map_nibp"].values.astype("float64")
        rows.append(dict(**{key: sid}, n_pairs=len(g),
                         min_art=float(a.min()), min_nib=float(n.min()),
                         p05_art=float(np.percentile(a, 5)),
                         p05_nib=float(np.percentile(n, 5))))
    return pd.DataFrame(rows)


def assign_groups(d, art_col, nib_col, thr=65):
    return np.where((d[art_col] < thr) & (d[nib_col] < thr), "both",
           np.where(d[art_col] < thr, "art_only",
           np.where(d[nib_col] < thr, "cuff_only", "neither")))


# ------------------------------------------------------------------ load
SETS = []
try:
    pr = pd.read_parquet(os.path.join(DERIVED, "mimic_pairs_t5.parquet")); pr = pr[pr["gap"] == 0]
    ak = pd.read_parquet(os.path.join(DERIVED, "mimic_aki.parquet"))
    lo = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\icustays.csv.gz",
                     compression="gzip", usecols=["stay_id", "los"])
    lo["stay_id"] = lo["stay_id"].astype("int64")
    pt = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\hosp\patients.csv.gz",
                     compression="gzip", usecols=["subject_id", "gender"])
    pt["female"] = pt["gender"].astype(str).str.upper().str.startswith("F").astype(float)
    base = (ak.merge(lo.drop_duplicates("stay_id"), on="stay_id", how="left")
              .merge(pt[["subject_id", "female"]], on="subject_id", how="left"))
    SETS.append(("MIMIC-IV (ICU)", metrics(pr, "stay_id"), base, "stay_id"))
except Exception:
    W("MIMIC failed"); W(traceback.format_exc()); Save()

try:
    pr = pd.read_parquet(os.path.join(DERIVED, "eicu_pairs_t5.parquet")); pr = pr[pr["gap"] == 0]
    ak = pd.read_parquet(os.path.join(DERIVED, "eicu_aki.parquet"))
    pt = pd.read_csv(r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\patient.csv.gz",
                     compression="gzip",
                     usecols=["patientunitstayid", "unitdischargeoffset", "gender"])
    pt["los"] = pt["unitdischargeoffset"] / 1440.0
    pt["female"] = pt["gender"].astype(str).str.upper().str.startswith("F").astype(float)
    base = ak.merge(pt[["patientunitstayid", "los", "female"]], on="patientunitstayid", how="left")
    SETS.append(("eICU-CRD (ICU)", metrics(pr, "patientunitstayid"), base, "patientunitstayid"))
except Exception:
    W("eICU failed"); W(traceback.format_exc()); Save()

W("=" * 78)
W(" STEP 30 - four-group analysis, stratified and re-specified")
W("=" * 78)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()


def run(df, cov, label, minp=5, width=0):
    s = df[df["n_pairs"] >= minp].dropna(subset=["aki"]).copy()
    s["g_art"] = (s["grp"] == "art_only").astype(float)
    s["g_cuff"] = (s["grp"] == "cuff_only").astype(float)
    s["g_both"] = (s["grp"] == "both").astype(float)
    cols = ["g_art", "g_cuff", "g_both"] + cov
    s = s.dropna(subset=cols)
    if len(s) < 150 or s["g_art"].sum() < 20:
        W("  %-34s too few (n=%s)" % (label, len(s)))
        return
    X = s[cols].astype(float).copy()
    for c in cov:
        if c in ("baseline", "los", "n_pairs"):
            mu, sd = X[c].mean(), X[c].std()
            X[c] = (X[c] - mu) / (sd if sd > 0 else 1.0)
    m = _fit(s["aki"].astype(int), X)
    b, se = m.params["g_art"], m.bse["g_art"]
    W("  %-34s n=%8s  AKI %5.1f%%  OR %5.3f (%.3f-%.3f)  p=%.4f"
      % (label, format(len(s), ","), 100.0 * s["aki"].mean(),
         np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues["g_art"]))
    Save()


for label, met, base, key in SETS:
    d = met.merge(base, on=key, how="inner")
    d["grp"] = assign_groups(d, "min_art", "min_nib")
    d["grp05"] = assign_groups(d, "p05_art", "p05_nib")

    W("=" * 78)
    W(" %s" % label)
    W("=" * 78)
    W("")

    W("  1. STRATIFIED BY NUMBER OF PAIRED READINGS  (ref = neither; adj baseline+los+pairs)")
    for lo_, hi_, nm in ((5, 8, "5-8"), (9, 14, "9-14"), (15, 24, "15-24"),
                         (25, 49, "25-49"), (50, 10**9, ">=50")):
        sub = d[(d["n_pairs"] >= lo_) & (d["n_pairs"] <= hi_)].copy()
        run(sub, ["baseline", "los", "n_pairs"], "n_pairs %s" % nm, minp=0)
    W("")

    W("  2. VARYING THE COHORT THRESHOLD")
    for mp in (2, 5, 10, 20):
        run(d, ["baseline", "los", "n_pairs"], "at least %d paired readings" % mp, minp=mp)
    W("")

    W("  3. GROUPS DEFINED BY 5th PERCENTILE INSTEAD OF MINIMUM")
    d2 = d.copy()
    d2["grp"] = d2["grp05"]
    run(d2, ["baseline", "los", "n_pairs"], "5th percentile, >=5 pairs", minp=5)
    W("")

    W("  4. ADDING SEX TO THE ADJUSTMENT SET")
    run(d, ["baseline", "los", "n_pairs", "female"], "adjusted for sex", minp=5)
    W("")

    W("  5. STRATIFIED BY SEX")
    for fv, nm in ((1.0, "female"), (0.0, "male")):
        sub = d[d["female"] == fv].copy()
        run(sub, ["baseline", "los", "n_pairs"], nm, minp=5)
    W("")
    Save()

W("=" * 78)
W(" DONE")
W("=" * 78)
Save()
print("")
print("Report saved to: %s" % REPORT)
