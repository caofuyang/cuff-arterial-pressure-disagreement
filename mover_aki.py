# -*- coding: utf-8 -*-
"""
Step 15 - MOVER: postoperative AKI + the same burden / joint models.

  A. surgery times (patient_information.csv)
  B. creatinine (patient_labs.csv, LOINC 2160-0)
  C. KDIGO AKI: baseline before anaesthesia start, follow-up 7 days after
  D. burden and joint models, same specification as MIMIC / eICU

Run:  python E:\\claude-tools\\step15_mover_aki.py
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
REPORT  = os.path.join(AUDIT, "step15_mover_aki_report.txt")
INFO    = r"E:\MOVER\data\EPIC_EMR\EMR\patient_information.csv"
LABS    = r"E:\MOVER\data\EPIC_EMR\EMR\patient_labs.csv"
PAIRF   = os.path.join(DERIVED, "mover_pairs_t5.parquet")
CRFILE  = os.path.join(DERIVED, "mover_creatinine.parquet")
AKIF    = os.path.join(DERIVED, "mover_aki.parquet")
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
W("=" * 66)
W(" STEP 15 - MOVER postoperative AKI + association")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

# ==================================================================
# A. surgery times
# ==================================================================
W("=" * 66)
W(" A. patient_information")
W("=" * 66)
info = pd.read_csv(INFO, low_memory=False)
W("  rows: %s" % format(len(info), ","))
W("  columns: %s" % ", ".join(info.columns.tolist()))
for c in ("AN_START_DATETIME", "AN_STOP_DATETIME", "IN_OR_DTTM", "OUT_OR_DTTM"):
    if c in info.columns:
        info[c] = pd.to_datetime(info[c], errors="coerce")
        W("  %-18s non-null %s  min %s  max %s"
          % (c, format(int(info[c].notna().sum()), ","),
             info[c].min(), info[c].max()))
W("")
info = info.drop_duplicates("LOG_ID")
W("  unique LOG_ID: %s" % format(len(info), ","))
W("")
Save()

# ==================================================================
# B. creatinine
# ==================================================================
W("=" * 66)
W(" B. creatinine (LOINC 2160-0)")
W("=" * 66)
Save()
if os.path.exists(CRFILE):
    cr = pd.read_parquet(CRFILE)
    W("  cached: %s rows" % format(len(cr), ","))
else:
    W("  file: %s (%.1f MB)" % (LABS, os.path.getsize(LABS) / 1e6))
    parts = []
    scanned = 0
    reader = pd.read_csv(LABS, chunksize=4_000_000, low_memory=False,
                         usecols=["LOG_ID", "Lab Code", "Lab Name",
                                  "Observation Value", "Collection Datetime"])
    for chunk in reader:
        scanned += len(chunk)
        m = (chunk["Lab Code"].astype(str).str.strip() == "2160-0")
        sub = chunk[m]
        if len(sub):
            parts.append(sub)
        if scanned % 20_000_000 < 4_000_000:
            W("    scanned %13s | kept %9s"
              % (format(scanned, ","), format(sum(len(x) for x in parts), ",")))
            Save()
    cr = pd.concat(parts, ignore_index=True)
    del parts
    W("")
    W("  scanned %s rows" % format(scanned, ","))
    cr = cr.rename(columns={"Observation Value": "cr",
                            "Collection Datetime": "dt"})
    cr["cr"] = pd.to_numeric(cr["cr"], errors="coerce")
    cr["dt"] = pd.to_datetime(cr["dt"], errors="coerce")
    cr = cr.dropna(subset=["cr", "dt"])
    cr = cr[(cr["cr"] > 0) & (cr["cr"] < 50)]
    cr = cr[["LOG_ID", "dt", "cr"]].drop_duplicates()
    cr.to_parquet(CRFILE, index=False)
    W("  creatinine rows: %s  LOG_IDs: %s"
      % (format(len(cr), ","), format(cr["LOG_ID"].nunique(), ",")))
    W("  saved -> %s" % CRFILE)
W("")
Save()

# ==================================================================
# C. KDIGO
# ==================================================================
W("=" * 66)
W(" C. KDIGO AKI")
W("=" * 66)
Save()
if os.path.exists(AKIF):
    aki = pd.read_parquet(AKIF)
    W("  cached: %s rows" % format(len(aki), ","))
else:
    times = info[["LOG_ID", "AN_START_DATETIME", "AN_STOP_DATETIME"]].dropna(
        subset=["AN_START_DATETIME"])
    tmap = {r.LOG_ID: (r.AN_START_DATETIME, r.AN_STOP_DATETIME)
            for r in times.itertuples(index=False)}
    grp = {k: (g["dt"].values, g["cr"].values.astype("float64"))
           for k, g in cr.groupby("LOG_ID", sort=False)}

    rows = []
    for lid, (t_start, t_stop) in tmap.items():
        if lid not in grp:
            continue
        dt, v = grp[lid]
        d64 = dt.astype("datetime64[ns]").astype("int64")
        a64 = np.int64(np.datetime64(pd.Timestamp(t_start), "ns").astype("int64"))
        if pd.notna(t_stop):
            b64 = np.int64(np.datetime64(pd.Timestamp(t_stop), "ns").astype("int64"))
        else:
            b64 = a64

        pre = d64 < a64
        if pre.any():
            base = float(v[pre].min())
            src = "pre"
        else:
            post24 = (d64 >= b64) & (d64 <= b64 + 86400 * 10**9)
            if not post24.any():
                continue
            base = float(v[post24].min())
            src = "post24h"

        w = (d64 >= b64) & (d64 <= b64 + 7 * 86400 * 10**9)
        if not w.any():
            continue
        dw, vw = d64[w], v[w]
        aki_7d = bool((vw >= base * 1.5).any()) if base > 0 else False
        aki_48 = False
        for k in range(len(vw)):
            lo = dw[k] - 48 * 3600 * 10**9
            prev = vw[(dw >= lo) & (dw <= dw[k])]
            if len(prev) and (vw[k] - prev.min()) >= 0.3:
                aki_48 = True
                break
        rows.append(dict(LOG_ID=lid, baseline=base, base_src=src,
                         cr_max=float(vw.max()), n_cr=int(len(vw)),
                         aki_7d=aki_7d, aki_48=aki_48, aki=bool(aki_7d or aki_48)))
    aki = pd.DataFrame(rows)
    aki.to_parquet(AKIF, index=False)
    W("  computed %s cases" % format(len(aki), ","))
    W("  saved -> %s" % AKIF)
W("")
if len(aki):
    W("  aki_7d %s | aki_48h %s | AKI any %s (%.1f%%)"
      % (format(int(aki['aki_7d'].sum()), ","), format(int(aki['aki_48'].sum()), ","),
         format(int(aki['aki'].sum()), ","), 100.0 * aki["aki"].mean()))
    W("  baseline source: %s" % aki["base_src"].value_counts().to_dict())
    W("  median baseline creatinine: %.2f mg/dL" % aki["baseline"].median())
W("")
Save()

# ==================================================================
# D. burden + models
# ==================================================================
W("=" * 66)
W(" D. burden and models")
W("=" * 66)
Save()
pairs = pd.read_parquet(PAIRF)
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
    deficit = np.maximum(THR - a, 0.0)
    shallow = (a >= 60) & (a < 65)
    rows.append(dict(LOG_ID=lid, n_pairs=n, monitored_min=float(dt.sum()),
                     b_all=float((deficit * dt).sum()),
                     b_occ=float((deficit * dt * oc).sum()),
                     b_con=float((deficit * dt * cc).sum()),
                     b_occ_sh=float((deficit * dt * oc * shallow).sum()),
                     b_con_sh=float((deficit * dt * cc * shallow).sum()),
                     min_art=float(a.min()), n_low=int(g["art_low"].sum())))
exp = pd.DataFrame(rows)
for c in [c for c in exp.columns if c.startswith("b_")]:
    exp[c + "_ph"] = exp[c] / (exp["monitored_min"] / 60.0).replace(0, np.nan)

meta = info[["LOG_ID", "ASA_RATING_C", "SEX"]].copy()
if "BIRTH_DATE" in info.columns:
    meta["BIRTH_DATE"] = pd.to_datetime(info["BIRTH_DATE"], errors="coerce")
d = exp.merge(aki, on="LOG_ID", how="inner").merge(meta, on="LOG_ID", how="left")
d["asa"] = pd.to_numeric(d["ASA_RATING_C"], errors="coerce")
W("  merged: %s cases | AKI %.1f%%" % (format(len(d), ","), 100.0 * d["aki"].mean()))
sub = d[d["n_pairs"] >= 5].copy()
W("  >=5 pairs: %s cases | AKI %.1f%%" % (format(len(sub), ","), 100.0 * sub["aki"].mean()))
W("  ASA available: %s" % format(int(sub["asa"].notna().sum()), ","))
W("")
Save()

try:
    import statsmodels.api as sm
except Exception:
    W("  statsmodels missing")
    Save(); raise SystemExit


def _fit(y, X):
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    X = X[keep]
    mod = sm.Logit(y, X)
    for kw in ({"disp": 0}, {"disp": False}, {}):
        try:
            return mod.fit(**kw)
        except TypeError:
            continue
    raise RuntimeError("fit failed")


COV = ["baseline", "n_pairs"]


def show(df, c1, c2, label, extra=None):
    cols = [c1, c2] + (extra or []) + COV
    s = df.dropna(subset=["aki"] + cols).copy()
    X = s[[c1, c2] + (extra or []) + COV].astype(float)
    X[c1] /= 10.0
    X[c2] /= 10.0
    X = sm.add_constant(X)
    m = _fit(s["aki"].astype(int), X)
    W("  --- %s  (n=%s, AKI %.1f%%) ---" % (label, format(len(s), ","), 100.0 * s["aki"].mean()))
    for c in (c1, c2):
        b, se = m.params[c], m.bse[c]
        W("      %-12s OR %.3f (%.3f-%.3f)  p=%.4f"
          % (c, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[c]))
    if extra:
        for c in extra:
            b, se = m.params[c], m.bse[c]
            W("      [extra] %-6s OR %.3f (%.3f-%.3f) p=%.4f"
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


W("  --- single exposure ---")
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
show(sub, "b_occ_ph", "b_con_ph", "ALL depths")
show(sub, "b_occ_sh_ph", "b_con_sh_ph", "SHALLOW (60-65)")
show(sub, "b_occ_ph", "b_con_ph", "ALL + min arterial MAP", extra=["min_art"])

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
