# -*- coding: utf-8 -*-
"""
Step 18 - is the hypotension-burden / AKI association real, or a product of
          the modelling choices?

Varies one dimension at a time from a reference specification:
  covariates   baseline / +n_pairs / +los / +age+sex
  exposure     area below threshold / minutes below / count of low readings
  cohort       >=2 / >=5 / >=10 pairs
  pairing      all gaps / same minute only

If the association survives across the grid it is worth reporting.
If it flips sign, it is a modelling artefact and must be dropped.

Run:  python E:\\claude-tools\\step18_sensitivity.py
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
REPORT  = os.path.join(AUDIT, "step18_sensitivity_report.txt")
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

try:
    import statsmodels.api as sm
except Exception:
    W("statsmodels missing"); Save(); raise SystemExit


def _fit(y, X):
    X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]
    mod = sm.Logit(y, X)
    for kw in ({"disp": 0}, {"disp": False}, {}):
        try:
            return mod.fit(**kw)
        except TypeError:
            continue
    raise RuntimeError("fit failed")


def build_metrics(pairs, key, tcol):
    """per-case burden metrics from a pairs table"""
    rows = []
    for sid, g in pairs.groupby(key, sort=False):
        t = g[tcol].values.astype("float64")
        a = g["map_art"].values.astype("float64")
        n = len(g)
        dt = np.diff(t) if n >= 2 else np.array([])
        dt = np.append(dt, dt[-1]) if n >= 2 else np.array([30.0])
        dt = np.clip(dt, 0, 60)
        low = a < THR
        defi = np.maximum(THR - a, 0.0)
        hrs = max(dt.sum() / 60.0, 1e-6)
        rows.append(dict(**{key: sid}, n_pairs=n, monitored_min=float(dt.sum()),
                          m_area=float((defi * dt).sum()) / hrs,
                          m_time=float((dt * low).sum()) / hrs,
                          m_count=float(low.sum()) / hrs))
    return pd.DataFrame(rows)


def run_grid(label, metrics, base_df, key):
    d = metrics.merge(base_df, on=key, how="inner")
    if "female" not in d.columns:
        d["female"] = 0.0
    variants = []
    # reference: area, baseline+n_pairs, >=5, all pairs
    for cov_name, cov in (("baseline", ["baseline"]),
                          ("baseline+pairs", ["baseline", "n_pairs"]),
                          ("baseline+los+pairs", ["baseline", "los", "n_pairs"]),
                          ("full", ["baseline", "los", "n_pairs", "age", "female"])):
        variants.append(("cov=%s" % cov_name, "m_area", cov, 5, None))
    for m in ("m_area", "m_time", "m_count"):
        variants.append(("exposure=%s" % m, m, ["baseline", "n_pairs"], 5, None))
    for c in (2, 5, 10):
        variants.append(("cohort>=%d" % c, "m_area", ["baseline", "n_pairs"], c, None))
    variants.append(("gap=0", "m_area", ["baseline", "n_pairs"], 5, 0))

    W("=" * 74)
    W(" %s" % label)
    W("=" * 74)
    W("  %-24s %10s %8s %9s %20s %8s" %
      ("specification", "n", "AKI%", "OR/10", "95% CI", "p"))
    W("  " + "-" * 72)

    meta = None
    for name, expo, cov, minp, gap in variants:
        dd = d
        if gap is not None and meta is not None:
            mm = build_metrics(meta[meta["gap"] == gap], key,
                               "tmin" if key == "stay_id" else
                               ("nursingchartoffset" if key == "patientunitstayid" else "tmin"))
            dd = mm.merge(base_df, on=key, how="inner")
            if "female" not in dd.columns:
                dd["female"] = 0.0
        s = dd[(dd["n_pairs"] >= minp)].dropna(subset=["aki", expo] + cov).copy()
        if len(s) < 200:
            W("  %-24s too few" % name)
            continue
        X = s[[expo] + cov].astype(float)
        X[expo] = X[expo] / 10.0
        X = sm.add_constant(X)
        try:
            m = _fit(s["aki"].astype(int), X)
        except Exception as e:
            W("  %-24s FAILED %r" % (name, e))
            continue
        b, se = m.params[expo], m.bse[expo]
        W("  %-24s %10s %8.1f %9.3f  %8.3f-%-8.3f %8.4f"
          % (name, format(len(s), ","), 100.0 * s["aki"].mean(),
             np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[expo]))
        Save()
    W("")
    Save()


t0 = time.time()
W("=" * 74)
W(" STEP 18 - sensitivity grid for the burden / AKI association")
W("=" * 74)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
W("  exposure unit = per 10 units of the metric per monitored hour")
W("  OR > 1 means more hypotension -> more AKI")
W("")
Save()

# ---------------- MIMIC ----------------
try:
    pairs = pd.read_parquet(os.path.join(DERIVED, "mimic_pairs_t5.parquet"))
    aki = pd.read_parquet(os.path.join(DERIVED, "mimic_aki.parquet"))
    losd = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\icustays.csv.gz",
                       compression="gzip", usecols=["stay_id", "los"])
    losd["stay_id"] = losd["stay_id"].astype("int64")
    pat = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\hosp\patients.csv.gz",
                      compression="gzip", usecols=["subject_id", "gender", "anchor_age"])
    base = (aki.merge(losd.drop_duplicates("stay_id"), on="stay_id", how="left")
                .merge(pat, on="subject_id", how="left"))
    base["age"] = base["anchor_age"]
    base["female"] = (base["gender"].astype(str).str.upper().str.startswith("F")).astype(float)
    met = build_metrics(pairs, "stay_id", "tmin")
    meta = pairs
    run_grid("MIMIC-IV (ICU)", met, base, "stay_id")
except Exception:
    W("MIMIC FAILED:"); W(traceback.format_exc()); Save()

# ---------------- eICU ----------------
try:
    pairs = pd.read_parquet(os.path.join(DERIVED, "eicu_pairs_t5.parquet"))
    aki = pd.read_parquet(os.path.join(DERIVED, "eicu_aki.parquet"))
    pat = pd.read_csv(r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\patient.csv.gz",
                      compression="gzip",
                      usecols=["patientunitstayid", "age", "gender", "unitdischargeoffset"])
    pat["age"] = pd.to_numeric(
        pat["age"].astype(str).str.replace(">", "", regex=False).str.strip(),
        errors="coerce")
    pat["los"] = pat["unitdischargeoffset"] / 1440.0
    pat["female"] = (pat["gender"].astype(str).str.upper().str.startswith("F")).astype(float)
    base = aki.merge(pat, on="patientunitstayid", how="left")
    met = build_metrics(pairs, "patientunitstayid", "nursingchartoffset")
    meta = pairs
    run_grid("eICU-CRD (ICU)", met, base, "patientunitstayid")
except Exception:
    W("eICU FAILED:"); W(traceback.format_exc()); Save()

W("=" * 74)
W(" HOW TO READ")
W("=" * 74)
W("  If ORs stay above 1 across every row -> real association, report it.")
W("  If ORs cross 1 depending on covariates -> artefact, drop the outcome part.")
W("")
W("=" * 74)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 74)
Save()
print("")
print("Report saved to: %s" % REPORT)
