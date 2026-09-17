# -*- coding: utf-8 -*-
"""
Step 17 - eICU internal comparison: does admission source explain the
          operating-room vs ICU discordance gap?

This removes the database/setting confounding: same hospitals, same EHR,
same measurement practices - only the unit of origin differs.

  A. unitadmitsource distribution
  B. discordance metrics by admission source
  C. AKI outcome by source, and whether the burden association differs

Run:  python E:\\claude-tools\\step17_eicu_source.py
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
REPORT  = os.path.join(AUDIT, "step17_eicu_source_report.txt")
D       = r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0"
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

t0 = time.time()
W("=" * 70)
W(" STEP 17 - eICU: discordance by ICU admission source")
W("=" * 70)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

pat = pd.read_csv(os.path.join(D, "patient.csv.gz"), compression="gzip",
                  usecols=["patientunitstayid", "unitadmitsource",
                           "unittype", "unitdischargeoffset"])
W("=" * 70)
W(" A. admission source distribution")
W("=" * 70)
vc = pat["unitadmitsource"].value_counts(dropna=False)
for k, v in vc.items():
    W("    %-40s %s" % (str(k)[:40], format(v, ",")))
W("")
Save()

src = pat["unitadmitsource"].astype(str).str.lower()
pat["grp"] = np.where(src.str.contains("operating", na=False), "OR",
             np.where(src.str.contains("recovery", na=False), "Recovery",
             np.where(src.str.contains("emergency", na=False), "ED",
             np.where(src.str.contains("floor|ward", na=False), "Floor", "Other"))))
W("  grouped:")
for k, v in pat["grp"].value_counts().items():
    W("    %-10s %s" % (k, format(v, ",")))
W("")
Save()

# ==================================================================
# B. discordance by source
# ==================================================================
pairs = pd.read_parquet(os.path.join(DERIVED, "eicu_pairs_t5.parquet"))
pairs["art_low"] = pairs["map_art"] < THR
pairs["missed"] = (pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)
pairs["falarm"] = (pairs["map_nibp"] < THR) & (pairs["map_art"] >= THR)
pairs["diff"] = pairs["map_nibp"] - pairs["map_art"]
pairs = pairs.merge(pat[["patientunitstayid", "grp"]], on="patientunitstayid", how="left")

W("=" * 70)
W(" B. discordance by admission source (gap = 0 pairs only)")
W("=" * 70)
W("")
p0 = pairs[pairs["gap"] == 0]
W("  %-10s %10s %8s %8s %9s %9s %9s" %
  ("source", "pairs", "cases", "bias", "SD", "missed%", "FA%"))
W("  " + "-" * 66)
for gname in ("OR", "Recovery", "ED", "Floor", "Other"):
    s = p0[p0["grp"] == gname]
    if len(s) < 500:
        continue
    al = int(s["art_low"].sum())
    nl = int((s["map_nibp"] < THR).sum())
    oc = int(s["missed"].sum())
    fa = int(s["falarm"].sum())
    W("  %-10s %10s %8s %+8.2f %8.2f %9.1f %9.1f"
      % (gname, format(len(s), ","), format(s["patientunitstayid"].nunique(), ","),
         s["diff"].mean(), s["diff"].std(),
         (100.0 * oc / al) if al else np.nan,
         (100.0 * fa / nl) if nl else np.nan))
W("")
Save()

# ==================================================================
# C. outcome by source
# ==================================================================
aki = pd.read_parquet(os.path.join(DERIVED, "eicu_aki.parquet"))

rows = []
for sid, g in pairs.groupby("patientunitstayid", sort=False):
    t = g["nursingchartoffset"].values.astype("float64")
    a = g["map_art"].values.astype("float64")
    oc = g["missed"].values
    cc = ((g["map_nibp"] < THR) & (g["map_art"] < THR)).values
    n = len(g)
    dt = np.diff(t) if n >= 2 else np.array([])
    dt = np.append(dt, dt[-1]) if n >= 2 else np.array([30.0])
    dt = np.clip(dt, 0, 60)
    defi = np.maximum(THR - a, 0.0)
    rows.append(dict(patientunitstayid=sid, n_pairs=n, monitored_min=float(dt.sum()),
                     b_all=float((defi * dt).sum()),
                     b_occ=float((defi * dt * oc).sum()),
                     b_con=float((defi * dt * cc).sum()),
                     min_art=float(a.min())))
exp = pd.DataFrame(rows)
for c in [c for c in exp.columns if c.startswith("b_")]:
    exp[c + "_ph"] = exp[c] / (exp["monitored_min"] / 60.0).replace(0, np.nan)

d = (exp.merge(aki, on="patientunitstayid", how="inner")
        .merge(pat[["patientunitstayid", "grp"]], on="patientunitstayid", how="left"))
sub = d[d["n_pairs"] >= 5].copy()

W("=" * 70)
W(" C. AKI outcome by admission source  (n_pairs >= 5)")
W("=" * 70)
W("")
W("  %-10s %8s %8s %9s %10s" % ("source", "cases", "AKI", "AKI%", "median pairs"))
W("  " + "-" * 60)
for gname in ("OR", "Recovery", "ED", "Floor", "Other"):
    s = sub[sub["grp"] == gname]
    if len(s) < 100:
        continue
    W("  %-10s %8s %8s %9.1f %10.0f"
      % (gname, format(len(s), ","), format(int(s["aki"].sum()), ","),
         100.0 * s["aki"].mean(), s["n_pairs"].median()))
W("")
Save()

try:
    import statsmodels.api as sm
except Exception:
    W("  statsmodels missing"); Save(); raise SystemExit


def _fit(y, X):
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    mod = sm.Logit(y, X)
    for kw in ({"disp": 0}, {"disp": False}, {}):
        try:
            return mod.fit(**kw)
        except TypeError:
            continue
    raise RuntimeError("fit failed")


COV = ["baseline", "n_pairs"]


def show(df, cols, label, scale=10.0):
    s = df.dropna(subset=["aki"] + cols + COV).copy()
    if len(s) < 100:
        W("    %-28s too few (n=%s)" % (label, len(s)))
        return
    X = s[cols + COV].astype(float)
    for c in cols:
        X[c] = X[c] / scale
    X = sm.add_constant(X)
    m = _fit(s["aki"].astype(int), X)
    W("    %-28s n=%s  AKI %.1f%%" % (label, format(len(s), ","), 100.0 * s["aki"].mean()))
    for c in cols:
        b, se = m.params[c], m.bse[c]
        W("        %-12s OR %.3f (%.3f-%.3f)  p=%.4f"
          % (c, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[c]))
    Save()


W("=" * 70)
W(" D. does the burden-AKI association differ by source?   (per 10 units)")
W("=" * 70)
W("")
for gname in ("OR", "Recovery", "ED", "Floor", "Other"):
    s = sub[sub["grp"] == gname]
    if len(s) < 300:
        continue
    show(s, ["b_all_ph"], "%s : total burden" % gname)
    show(s, ["b_occ_ph", "b_con_ph"], "%s : occult + concordant" % gname)
    W("")
W("")

# interaction test: OR vs non-OR
W("  --- interaction test: does the source modify the burden-AKI association? ---")
s = sub[sub["grp"].isin(["OR", "ED", "Floor", "Recovery"])].dropna(
    subset=["aki", "b_all_ph", "baseline", "n_pairs"]).copy()
s["is_or"] = (s["grp"] == "OR").astype(float)
s["burden"] = s["b_all_ph"] / 10.0
X = sm.add_constant(s[["burden", "is_or", "baseline", "n_pairs"]].astype(float))
X["burden_x_or"] = X["burden"] * X["is_or"]
try:
    m = _fit(s["aki"].astype(int), X)
    b, se = m.params["burden_x_or"], m.bse["burden_x_or"]
    W("    interaction burden x OR : OR %.3f (%.3f-%.3f)  p=%.4f  n=%s"
      % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
         m.pvalues["burden_x_or"], format(len(s), ",")))
    b, se = m.params["is_or"], m.bse["is_or"]
    W("    OR admission (main)     : OR %.3f (%.3f-%.3f)  p=%.4f"
      % (np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues["is_or"]))
except Exception as e:
    W("    interaction failed: %r" % e)
W("")
Save()

W("=" * 70)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 70)
Save()
print("")
print("Report saved to: %s" % REPORT)
