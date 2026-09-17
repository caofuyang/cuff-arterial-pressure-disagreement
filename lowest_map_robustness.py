# -*- coding: utf-8 -*-
"""
Step 20 - is the "lowest arterial MAP" signal robust, or an artefact of how
          many readings a case happens to have?

Key concern: the more paired readings a case has, the lower its observed
minimum will be - and more readings also means a sicker, longer stay.
This step tests the signal within strata of reading count and against two
alternative summaries (5th percentile, time-weighted mean).

Run:  python E:\\claude-tools\\step20_minmap.py
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
REPORT  = os.path.join(AUDIT, "step20_minmap_report.txt")
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


def metrics(pairs, key, tcol):
    rows = []
    for sid, g in pairs.groupby(key, sort=False):
        t = g[tcol].values.astype("float64")
        a = g["map_art"].values.astype("float64")
        n = len(g)
        dt = np.diff(t) if n >= 2 else np.array([])
        dt = np.append(dt, dt[-1]) if n >= 2 else np.array([30.0])
        dt = np.clip(dt, 0, 60)
        twa = float((a * dt).sum() / max(dt.sum(), 1e-6))
        rows.append(dict(**{key: sid}, n_pairs=n, monitored_min=float(dt.sum()),
                          min_art=float(a.min()),
                          p05_art=float(np.percentile(a, 5)),
                          mean_art=float(a.mean()),
                          twa_art=twa))
    return pd.DataFrame(rows)


def table(label, df, expo, cov, minp, extra_note=""):
    s = df[df["n_pairs"] >= minp].dropna(subset=["aki", expo] + cov).copy()
    if len(s) < 200 or s["aki"].nunique() < 2:
        W("    %-34s too few" % label)
        return
    X = s[[expo] + cov].astype(float)
    X = sm.add_constant(X)
    try:
        m = _fit(s["aki"].astype(int), X)
    except Exception as e:
        W("    %-34s FAILED %r" % (label, e))
        return
    b, se = m.params[expo], m.bse[expo]
    W("    %-34s n=%8s  OR %.3f (%.3f-%.3f)  p=%.4f"
      % (label, format(len(s), ","), np.exp(b),
         np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[expo]))
    Save()


t0 = time.time()
W("=" * 78)
W(" STEP 20 - is 'lowest arterial MAP' robust?")
W("=" * 78)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
W("  exposure = arterial MAP in mmHg (OR per 1 mmHg; OR>1 = higher MAP, less AKI)")
W("")
Save()

DATASETS = []

# MIMIC
try:
    pr = pd.read_parquet(os.path.join(DERIVED, "mimic_pairs_t5.parquet"))
    ak = pd.read_parquet(os.path.join(DERIVED, "mimic_aki.parquet"))
    lo = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\icustays.csv.gz",
                     compression="gzip", usecols=["stay_id", "los"])
    lo["stay_id"] = lo["stay_id"].astype("int64")
    pt = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\hosp\patients.csv.gz",
                     compression="gzip", usecols=["subject_id", "gender", "anchor_age"])
    base = ak.merge(lo.drop_duplicates("stay_id"), on="stay_id", how="left").merge(pt, on="subject_id", how="left")
    base["age"] = base["anchor_age"]
    base["female"] = base["gender"].astype(str).str.upper().str.startswith("F").astype(float)
    DATASETS.append(("MIMIC-IV (ICU)", metrics(pr, "stay_id", "tmin"), base, "stay_id", pr, "tmin"))
except Exception:
    W("MIMIC failed"); W(traceback.format_exc()); Save()

# eICU
try:
    pr = pd.read_parquet(os.path.join(DERIVED, "eicu_pairs_t5.parquet"))
    ak = pd.read_parquet(os.path.join(DERIVED, "eicu_aki.parquet"))
    pt = pd.read_csv(r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\patient.csv.gz",
                     compression="gzip", usecols=["patientunitstayid", "age", "gender", "unitdischargeoffset"])
    pt["age"] = pd.to_numeric(pt["age"].astype(str).str.replace(">", "", regex=False).str.strip(), errors="coerce")
    pt["los"] = pt["unitdischargeoffset"] / 1440.0
    pt["female"] = pt["gender"].astype(str).str.upper().str.startswith("F").astype(float)
    base = ak.merge(pt, on="patientunitstayid", how="left")
    DATASETS.append(("eICU-CRD (ICU)", metrics(pr, "patientunitstayid", "nursingchartoffset"),
                     base, "patientunitstayid", pr, "nursingchartoffset"))
except Exception:
    W("eICU failed"); W(traceback.format_exc()); Save()

# MOVER
try:
    pr = pd.read_parquet(os.path.join(DERIVED, "mover_pairs_t5.parquet"))
    ak = pd.read_parquet(os.path.join(DERIVED, "mover_aki.parquet"))
    base = ak.copy()
    base["los"] = np.nan
    base["age"] = np.nan
    base["female"] = 0.0
    DATASETS.append(("MOVER (OR)", metrics(pr, "LOG_ID", "tmin"), base, "LOG_ID", pr, "tmin"))
except Exception:
    W("MOVER failed"); W(traceback.format_exc()); Save()

for label, met, base, key, raw, tcol in DATASETS:
    d = met.merge(base, on=key, how="inner")
    for c in ("los", "age", "female"):
        if c not in d.columns:
            d[c] = np.nan
    W("=" * 78)
    W(" %s   (n=%s)" % (label, format(len(d), ",")))
    W("=" * 78)
    W("")

    W("  --- A. minimum arterial MAP, varying covariates (n_pairs>=5, all gaps) ---")
    table("baseline only", d, "min_art", ["baseline"], 5)
    table("+ n_pairs", d, "min_art", ["baseline", "n_pairs"], 5)
    table("+ los + n_pairs", d, "min_art", ["baseline", "los", "n_pairs"], 5)
    table("full", d, "min_art", ["baseline", "los", "n_pairs", "age", "female"], 5)
    W("")

    W("  --- B. alternative summaries (baseline + n_pairs, n_pairs>=5) ---")
    table("minimum MAP", d, "min_art", ["baseline", "n_pairs"], 5)
    table("5th percentile MAP", d, "p05_art", ["baseline", "n_pairs"], 5)
    table("time-weighted mean MAP", d, "twa_art", ["baseline", "n_pairs"], 5)
    W("")

    W("  --- C. varying cohort size (minimum MAP, baseline + n_pairs) ---")
    for mp in (2, 5, 10, 20):
        table("n_pairs >= %d" % mp, d, "min_art", ["baseline", "n_pairs"], mp)
    W("")

    W("  --- D. WITHIN STRATA of reading count (the key test) ---")
    W("       more readings mechanically lower the observed minimum;")
    W("       if the signal survives inside narrow strata it is not that artefact")
    dd = d[d["n_pairs"] >= 5].copy()
    for lo_, hi_, nm in ((5, 8, "5-8"), (9, 14, "9-14"), (15, 24, "15-24"),
                         (25, 49, "25-49"), (50, 10**9, ">=50")):
        sub = dd[(dd["n_pairs"] >= lo_) & (dd["n_pairs"] <= hi_)]
        table("n_pairs %s" % nm, sub, "min_art", ["baseline"], 0)
    W("")

    W("  --- E. same-minute pairs only ---")
    mm = metrics(raw[raw["gap"] == 0], key, tcol)
    d0 = mm.merge(base, on=key, how="inner")
    for c in ("los", "age", "female"):
        if c not in d0.columns:
            d0[c] = np.nan
    table("min MAP, gap=0", d0, "min_art", ["baseline", "n_pairs"], 5)
    table("mean MAP, gap=0", d0, "twa_art", ["baseline", "n_pairs"], 5)
    W("")
    Save()

W("=" * 78)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 78)
Save()
print("")
print("Report saved to: %s" % REPORT)
