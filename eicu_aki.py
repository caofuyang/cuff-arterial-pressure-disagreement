# -*- coding: utf-8 -*-
"""
Step 12 - eICU: KDIGO AKI + the same burden / joint-model analysis as MIMIC.

Baseline creatinine rule is harmonised with MIMIC: lowest creatinine within
the first 24 h of ICU admission (eICU has no reliable pre-ICU labs).

Run:  python E:\\claude-tools\\step12_eicu_aki.py
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
REPORT  = os.path.join(AUDIT, "step12_eicu_aki_report.txt")
D       = r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0"
PAIRF   = os.path.join(DERIVED, "eicu_pairs_t5.parquet")
CRFILE  = os.path.join(DERIVED, "eicu_creatinine.parquet")
AKIF    = os.path.join(DERIVED, "eicu_aki.parquet")
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
W(" STEP 12 - eICU AKI outcome + association")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

# ==================================================================
# A. creatinine
# ==================================================================
W("=" * 66)
W(" A. eICU creatinine (lab table)")
W("=" * 66)
Save()
if os.path.exists(CRFILE):
    cr = pd.read_parquet(CRFILE)
    W("  cached: %s rows" % format(len(cr), ","))
else:
    lab = os.path.join(D, "lab.csv.gz")
    W("  file: %s (%.1f MB)" % (lab, os.path.getsize(lab) / 1e6))
    parts = []
    scanned = 0
    reader = pd.read_csv(
        lab, compression="gzip", chunksize=4_000_000,
        usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
        dtype={"patientunitstayid": "int64", "labresultoffset": "int64",
               "labname": "string", "labresult": "float64"})
    for chunk in reader:
        scanned += len(chunk)
        sub = chunk[chunk["labname"].str.lower().str.contains("creatinine", na=False)]
        if len(sub):
            parts.append(sub)
        if scanned % 20_000_000 < 4_000_000:
            W("    scanned %13s | kept %9s"
              % (format(scanned, ","), format(sum(len(x) for x in parts), ",")))
            Save()
    cr = pd.concat(parts, ignore_index=True)
    del parts
    W("")
    W("  labname values kept: %s" % cr["labname"].value_counts().to_dict())
    cr = cr.rename(columns={"labresultoffset": "offset", "labresult": "cr"})
    cr = cr.dropna(subset=["cr"])
    cr = cr[(cr["cr"] > 0) & (cr["cr"] < 50)]
    cr = cr[["patientunitstayid", "offset", "cr"]].drop_duplicates()
    cr.to_parquet(CRFILE, index=False)
    W("  creatinine rows: %s   stays: %s"
      % (format(len(cr), ","), format(cr["patientunitstayid"].nunique(), ",")))
    W("  saved -> %s" % CRFILE)
W("")
Save()

# ==================================================================
# B. patient table
# ==================================================================
W("=" * 66)
W(" B. patient table")
W("=" * 66)
pat = pd.read_csv(os.path.join(D, "patient.csv.gz"), compression="gzip",
                  usecols=["patientunitstayid", "uniquepid", "age", "gender",
                           "unitadmitsource", "unitdischargeoffset",
                           "hospitaldischargestatus", "hospitaldischargeoffset"])
W("  rows: %s" % format(len(pat), ","))
W("")


def parse_age(a):
    try:
        s = str(a).strip()
        if s.startswith(">"):
            return 90.0
        return float(s)
    except Exception:
        return np.nan


pat["age_n"] = pat["age"].map(parse_age)
pat["female"] = (pat["gender"].astype(str).str.strip().str.upper()
                 .str.startswith("F")).astype(int)
pat["los_days"] = pat["unitdischargeoffset"] / 1440.0
W("  age parsed: %s   female: %s"
  % (format(int(pat["age_n"].notna().sum()), ","), format(int(pat["female"].sum()), ",")))
W("")
Save()

# ==================================================================
# C. KDIGO (baseline = lowest creatinine in first 24 h)
# ==================================================================
W("=" * 66)
W(" C. KDIGO AKI (baseline = min creatinine in first 24 h)")
W("=" * 66)
Save()
if os.path.exists(AKIF):
    aki = pd.read_parquet(AKIF)
    W("  cached AKI: %s rows" % format(len(aki), ","))
else:
    cr = cr.sort_values(["patientunitstayid", "offset"]).reset_index(drop=True)
    grp = {sid: (g["offset"].values.astype("int64"), g["cr"].values.astype("float64"))
           for sid, g in cr.groupby("patientunitstayid", sort=False)}
    rows = []
    t1 = time.time()
    for i, r in enumerate(pat.itertuples(index=False)):
        sid = r.patientunitstayid
        if sid not in grp:
            continue
        off, val = grp[sid]
        m24 = (off >= 0) & (off <= 1440)
        if not m24.any():
            continue
        base = float(val[m24].min())
        w = (off >= 0) & (off <= 7 * 1440)
        if not w.any():
            continue
        ow, vw = off[w], val[w]
        aki_7d = bool((vw >= base * 1.5).any()) if base > 0 else False
        aki_48 = False
        for k in range(len(vw)):
            lo = ow[k] - 2880
            prev = vw[(ow >= lo) & (ow <= ow[k])]
            if len(prev) and (vw[k] - prev.min()) >= 0.3:
                aki_48 = True
                break
        rows.append(dict(patientunitstayid=sid, baseline=base, cr_max=float(vw.max()),
                         n_cr=int(len(vw)), aki_7d=aki_7d, aki_48=aki_48,
                         aki=bool(aki_7d or aki_48)))
        if (i + 1) % 40000 == 0:
            W("    %s stays (%.0fs)" % (format(i + 1, ","), time.time() - t1))
            Save()
    aki = pd.DataFrame(rows)
    aki.to_parquet(AKIF, index=False)
    W("  computed %s stays" % format(len(aki), ","))
    W("  saved -> %s" % AKIF)
W("")
if len(aki):
    W("  aki_7d %s | aki_48h %s | AKI any %s (%.1f%%)"
      % (format(int(aki['aki_7d'].sum()), ","), format(int(aki['aki_48'].sum()), ","),
         format(int(aki['aki'].sum()), ","), 100.0 * aki["aki"].mean()))
    W("  median baseline creatinine: %.2f mg/dL" % aki["baseline"].median())
W("")
Save()

# ==================================================================
# D. burden + models
# ==================================================================
W("=" * 66)
W(" D. burden and joint models")
W("=" * 66)
Save()
pairs = pd.read_parquet(PAIRF)
pairs["art_low"] = pairs["map_art"] < THR
pairs["occult"] = (pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)
pairs["conc"] = (pairs["map_nibp"] < THR) & (pairs["map_art"] < THR)
pairs = pairs.sort_values(["patientunitstayid", "nursingchartoffset"]).reset_index(drop=True)

rows = []
for sid, g in pairs.groupby("patientunitstayid", sort=False):
    t = g["nursingchartoffset"].values.astype("float64")
    a = g["map_art"].values.astype("float64")
    oc = g["occult"].values
    cc = g["conc"].values
    n = len(g)
    dt = np.diff(t) if n >= 2 else np.array([])
    dt = np.append(dt, dt[-1]) if n >= 2 else np.array([30.0])
    dt = np.clip(dt, 0, 60)
    deficit = np.maximum(THR - a, 0.0)
    shallow = (a >= 60) & (a < 65)
    rows.append(dict(
        patientunitstayid=sid, n_pairs=n, monitored_min=float(dt.sum()),
        b_all=float((deficit * dt).sum()),
        b_occ=float((deficit * dt * oc).sum()),
        b_con=float((deficit * dt * cc).sum()),
        b_occ_sh=float((deficit * dt * oc * shallow).sum()),
        b_con_sh=float((deficit * dt * cc * shallow).sum()),
        min_art=float(a.min()), n_low=int(g["art_low"].sum())))
exp = pd.DataFrame(rows)
for c in [c for c in exp.columns if c.startswith("b_")]:
    exp[c + "_ph"] = exp[c] / (exp["monitored_min"] / 60.0).replace(0, np.nan)

d = (exp.merge(aki, on="patientunitstayid", how="inner")
        .merge(pat[["patientunitstayid", "age_n", "female", "los_days"]],
               on="patientunitstayid", how="left"))
d = d.rename(columns={"age_n": "anchor_age", "los_days": "los"})
W("  merged: %s stays | AKI %.1f%%" % (format(len(d), ","), 100.0 * d["aki"].mean()))
sub = d[d["n_pairs"] >= 5].copy()
W("  >=5 pairs: %s stays | AKI %.1f%%" % (format(len(sub), ","), 100.0 * sub["aki"].mean()))
W("")
Save()

try:
    import statsmodels.api as sm
except Exception:
    W("  statsmodels missing")
    Save(); raise SystemExit


def _fit(y, X):
    # drop constant columns - they make the Hessian singular
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    mod = sm.Logit(y, X)
    for kw in ({"disp": 0}, {"disp": False}, {}):
        try:
            return mod.fit(**kw)
        except TypeError:
            continue
    raise RuntimeError("fit failed")


COV = ["baseline", "los", "n_pairs", "anchor_age", "female"]


def joint(df, c1, c2, label, extra=None):
    cols = [c1, c2] + (extra or []) + COV
    s = df.dropna(subset=["aki"] + cols).copy()
    X = s[[c1, c2] + (extra or []) + COV].astype(float)
    X[c1] /= 10.0
    X[c2] /= 10.0
    X = sm.add_constant(X)
    m = _fit(s["aki"].astype(int), X)
    W("  --- %s   (n=%s) ---" % (label, format(len(s), ",")))
    for c in (c1, c2):
        b, se = m.params[c], m.bse[c]
        W("      %-14s OR %.3f (%.3f-%.3f)  p=%.4f"
          % (c, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[c]))
    if extra:
        for c in extra:
            b, se = m.params[c], m.bse[c]
            W("      [extra] %-6s OR %.3f (%.3f-%.3f)  p=%.4f"
              % (c, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[c]))
    try:
        v = m.cov_params()
        diff = m.params[c1] - m.params[c2]
        se = float(np.sqrt(v.loc[c1, c1] + v.loc[c2, c2] - 2 * v.loc[c1, c2]))
        from math import erfc
        W("      Wald: diff %+.4f  z %+.2f  p=%.4f"
          % (diff, diff / se, float(erfc(abs(diff / se) / np.sqrt(2)))))
    except Exception as e:
        W("      Wald failed: %r" % e)
    W("")
    Save()


W("  --- single-exposure models ---")
for c, lab in (("b_all_ph", "total burden (positive control)"),
               ("b_occ_ph", "occult burden"),
               ("b_con_ph", "concordant burden")):
    s = sub.dropna(subset=["aki", c] + COV).copy()
    X = s[[c] + COV].astype(float)
    X[c] /= 10.0
    X = sm.add_constant(X)
    m = _fit(s["aki"].astype(int), X)
    b, se = m.params[c], m.bse[c]
    W("    %-34s OR %.3f (%.3f-%.3f)  p=%.4f  n=%s"
      % (lab, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se),
         m.pvalues[c], format(len(s), ",")))
W("")
Save()

W("  --- joint models ---")
joint(sub, "b_occ_ph", "b_con_ph", "ALL depths")
joint(sub, "b_occ_sh_ph", "b_con_sh_ph", "SHALLOW only (60-65)")
joint(sub, "b_occ_ph", "b_con_ph", "ALL depths + min arterial MAP", extra=["min_art"])

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
