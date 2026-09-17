# -*- coding: utf-8 -*-
"""
Step 25 - four-group adjusted odds ratios, with a convergence check.

step24 returned degenerate estimates (OR = 1.000, p = 1.000) for eICU in the
adjusted models. Likely cause: covariates on wildly different scales
(baseline ~1 mg/dL, n_pairs up to several hundred) causing L-BFGS to stall and
statsmodels to return the starting values without raising.

This version standardises the continuous covariates, tries several optimisers,
and explicitly reports whether the fit converged.

Run:  python E:\\claude-tools\\step25_groups2.py
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
REPORT  = os.path.join(AUDIT, "step25_groups2_report.txt")

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


def fit_checked(y, X):
    """Fit and report whether it actually converged."""
    if "const" not in X.columns:
        X = sm.add_constant(X)
    attempts = [
        dict(method="lbfgs", maxiter=2000, pgtol=1e-10),
        dict(method="bfgs", maxiter=2000),
        dict(method="newton", maxiter=200),
        dict(),
    ]
    last = None
    for kw in attempts:
        try:
            m = sm.Logit(y, X).fit(disp=0, **kw)
            conv = True
            try:
                conv = bool(m.mle_retvals.get("converged", True))
            except Exception:
                conv = None
            if conv is False:
                last = (m, kw, False)
                continue
            return m, kw, conv
        except Exception as e:
            last = (None, kw, "failed: %r" % e)
    return last


def metrics(pairs, key):
    rows = []
    for sid, g in pairs.groupby(key, sort=False):
        rows.append(dict(**{key: sid}, n_pairs=len(g),
                         min_art=float(g["map_art"].min()),
                         min_nib=float(g["map_nibp"].min())))
    return pd.DataFrame(rows)


SETS = []
try:
    pr = pd.read_parquet(os.path.join(DERIVED, "mimic_pairs_t5.parquet")); pr = pr[pr["gap"] == 0]
    ak = pd.read_parquet(os.path.join(DERIVED, "mimic_aki.parquet"))
    lo = pd.read_csv(r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\icustays.csv.gz",
                     compression="gzip", usecols=["stay_id", "los"])
    lo["stay_id"] = lo["stay_id"].astype("int64")
    SETS.append(("MIMIC-IV (ICU)", metrics(pr, "stay_id"),
                 ak.merge(lo.drop_duplicates("stay_id"), on="stay_id", how="left"), "stay_id"))
except Exception:
    W("MIMIC failed"); W(traceback.format_exc()); Save()

try:
    pr = pd.read_parquet(os.path.join(DERIVED, "eicu_pairs_t5.parquet")); pr = pr[pr["gap"] == 0]
    ak = pd.read_parquet(os.path.join(DERIVED, "eicu_aki.parquet"))
    pt = pd.read_csv(r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0\patient.csv.gz",
                     compression="gzip", usecols=["patientunitstayid", "unitdischargeoffset"])
    pt["los"] = pt["unitdischargeoffset"] / 1440.0
    SETS.append(("eICU (ICU)", metrics(pr, "patientunitstayid"),
                 ak.merge(pt, on="patientunitstayid", how="left"), "patientunitstayid"))
except Exception:
    W("eICU failed"); W(traceback.format_exc()); Save()

W("=" * 78)
W(" STEP 25 - four-group adjusted ORs, with convergence check")
W("=" * 78)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

for label, met, base, key in SETS:
    d = met.merge(base, on=key, how="inner")
    d = d[d["n_pairs"] >= 5].copy()
    cov = [c for c in ("baseline", "los", "n_pairs") if c in d.columns
           and d[c].notna().sum() > 100]

    for THR in (65, 55):
        W("=" * 78)
        W(" %s   threshold %d" % (label, THR))
        W("=" * 78)

        g = pd.Series("neither", index=d.index)
        g[(d["min_nib"] < THR) & (d["min_art"] >= THR)] = "cuff_only"
        g[(d["min_art"] < THR) & (d["min_nib"] >= THR)] = "art_only"
        g[(d["min_art"] < THR) & (d["min_nib"] < THR)] = "both"
        d["grp"] = g

        W("  %-12s %8s %7s %8s" % ("group", "n", "AKI", "AKI%"))
        for nm in ("neither", "cuff_only", "art_only", "both"):
            s = d[d["grp"] == nm]
            if len(s):
                W("  %-12s %8s %7s %8.1f" % (nm, format(len(s), ","),
                                             format(int(s["aki"].sum()), ","),
                                             100.0 * s["aki"].mean()))
        W("")

        s = d.dropna(subset=["aki"] + cov).copy()
        for nm, col in (("cuff_only", "g_cuff"), ("art_only", "g_art"), ("both", "g_both")):
            s[col] = (s["grp"] == nm).astype(float)

        # standardise continuous covariates
        sc = {}
        for c in cov:
            mu, sd = s[c].mean(), s[c].std()
            sc[c] = (mu, sd)
            s["z_" + c] = (s[c] - mu) / (sd if sd > 0 else 1.0)

        W("  covariate scales: " + "; ".join(
            "%s mean=%.2f sd=%.2f" % (c, sc[c][0], sc[c][1]) for c in cov))
        W("")

        dummies = ["g_cuff", "g_art", "g_both"]
        zcov = ["z_" + c for c in cov]

        m, kw, conv = fit_checked(s["aki"].astype(int), s[dummies + zcov].astype(float))
        W("  ADJUSTED (n=%s)   optimizer=%s   converged=%s"
          % (format(len(s), ","), kw, conv))
        if m is None:
            W("    ALL FITS FAILED")
        else:
            for c, nm in zip(dummies, ("cuff only", "ARTERIAL ONLY", "both")):
                b, se = m.params[c], m.bse[c]
                W("    %-14s OR %.3f (%.3f-%.3f)  p=%.4f"
                  % (nm, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m.pvalues[c]))
            try:
                W("    log-likelihood %.1f   pseudo-R2 %.4f"
                  % (m.llf, m.prsquared))
            except Exception:
                pass
        W("")

        m0, kw0, conv0 = fit_checked(s["aki"].astype(int), s[dummies].astype(float))
        W("  CRUDE (n=%s)   optimizer=%s   converged=%s" % (format(len(s), ","), kw0, conv0))
        if m0 is not None:
            for c, nm in zip(dummies, ("cuff only", "ARTERIAL ONLY", "both")):
                b, se = m0.params[c], m0.bse[c]
                W("    %-14s OR %.3f (%.3f-%.3f)  p=%.4f"
                  % (nm, np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), m0.pvalues[c]))
        W("")
        Save()

W("=" * 78)
W(" DONE")
W("=" * 78)
Save()
print("")
print("Report saved to: %s" % REPORT)
