# -*- coding: utf-8 -*-
"""
Step 7 - MIMIC-IV: KDIGO acute kidney injury + linkage to occult hypotension.

  A. extract serum creatinine (labevents itemid 50912)
  B. load icustays
  C. KDIGO AKI within 7 days of ICU admission
  D. link with the paired-BP cohort and cross-tabulate

Run:  python E:\\claude-tools\\step7_aki.py
"""
import os
import time
import traceback
import numpy as np
import pandas as pd

PROJ    = r"E:\occult_htn"
DERIVED = os.path.join(PROJ, "derived")
AUDIT   = os.path.join(PROJ, "audit")
REPORT  = os.path.join(AUDIT, "step7_aki_report.txt")
CRFILE  = os.path.join(DERIVED, "mimic_creatinine.parquet")
STAYF   = os.path.join(DERIVED, "mimic_icustays.parquet")
AKIF    = os.path.join(DERIVED, "mimic_aki.parquet")
PAIRF   = os.path.join(DERIVED, "mimic_pairs_primary.parquet")

os.makedirs(AUDIT, exist_ok=True)
lines = []


def W(t=""):
    s = str(t)
    print(s, flush=True)
    lines.append(s)


def Save():
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print("save failed: %r" % e)


t0 = time.time()
W("=" * 66)
W(" STEP 7 - MIMIC-IV AKI outcome")
W("=" * 66)
W("start : %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
W("")
Save()

LAB = r"E:\MIMIC-IV\data\mimic-iv-3.1\hosp\labevents.csv.gz"
ICU = r"E:\MIMIC-IV\data\mimic-iv-3.1\icu\icustays.csv.gz"
ITEM_CR = 50912

# ==================================================================
# A. creatinine
# ==================================================================
W("=" * 66)
W(" A. serum creatinine (itemid %d)" % ITEM_CR)
W("=" * 66)
Save()
try:
    if os.path.exists(CRFILE):
        cr = pd.read_parquet(CRFILE)
        W("  cached: %s rows" % format(len(cr), ","))
    else:
        W("  file: %s (%.1f MB)" % (LAB, os.path.getsize(LAB) / 1e6))
        W("  scanning...")
        Save()
        parts = []
        scanned = 0
        t1 = time.time()
        reader = pd.read_csv(
            LAB, compression="gzip", chunksize=4_000_000,
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
            dtype={"subject_id": "int64", "hadm_id": "Int64",
                   "itemid": "int32", "valuenum": "float64"},
            parse_dates=["charttime"],
        )
        for chunk in reader:
            scanned += len(chunk)
            sub = chunk[chunk["itemid"] == ITEM_CR]
            if len(sub):
                parts.append(sub)
            if scanned % 20_000_000 < 4_000_000:
                W("    scanned %13s | kept %9s | %.0fs"
                  % (format(scanned, ","),
                     format(sum(len(x) for x in parts), ","), time.time() - t1))
                Save()
        cr = pd.concat(parts, ignore_index=True)
        del parts
        cr = cr[["subject_id", "hadm_id", "charttime", "valuenum"]]
        cr = cr[cr["valuenum"].notna() & (cr["valuenum"] > 0) & (cr["valuenum"] < 50)]
        cr = cr.drop_duplicates()
        cr.to_parquet(CRFILE, index=False)
        W("")
        W("  scanned %s rows" % format(scanned, ","))
        W("  kept    %s creatinine rows" % format(len(cr), ","))
        W("  saved -> %s" % CRFILE)
        Save()
    W("  subjects: %s" % format(cr["subject_id"].nunique(), ","))
    W("")
    Save()
except Exception:
    W("  FAILED:"); W(traceback.format_exc()); Save(); raise SystemExit

# ==================================================================
# B. icustays
# ==================================================================
W("=" * 66)
W(" B. ICU stays")
W("=" * 66)
try:
    st = pd.read_csv(ICU, compression="gzip",
                     usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
                     parse_dates=["intime", "outtime"])
    W("  stayed rows: %s" % format(len(st), ","))
    st["stay_id"] = st["stay_id"].astype("int64")
    W("  LOS median (days): %.2f" % st["los"].median())
    W("")
    Save()
except Exception:
    W("  FAILED:"); W(traceback.format_exc()); Save(); raise SystemExit

# ==================================================================
# C. KDIGO
# ==================================================================
W("=" * 66)
W(" C. KDIGO AKI within 7 days of ICU admission")
W("=" * 66)
Save()

if os.path.exists(AKIF):
    aki = pd.read_parquet(AKIF)
    W("  cached AKI table: %s rows" % format(len(aki), ","))
else:
    # index creatinine by subject
    cr = cr.sort_values(["subject_id", "charttime"]).reset_index(drop=True)
    grp = {}
    for sid, g in cr.groupby("subject_id", sort=False):
        grp[sid] = (g["charttime"].values.astype("datetime64[s]").astype("int64"),
                    g["valuenum"].values.astype("float64"))

    rows = []
    t1 = time.time()
    for i, r in enumerate(st.itertuples(index=False)):
        sid = r.subject_id
        if sid not in grp:
            continue
        tt, vv = grp[sid]
        t_in = np.int64(pd.Timestamp(r.intime).to_datetime64().astype("datetime64[s]").astype("int64"))
        t_out = np.int64(pd.Timestamp(r.outtime).to_datetime64().astype("datetime64[s]").astype("int64"))

        # baseline: lowest creatinine in [intime - 7d, intime)
        m7 = (tt >= t_in - 7 * 86400) & (tt < t_in)
        if m7.any():
            base = float(vv[m7].min())
            base_src = "pre7d"
        else:
            m2 = (tt >= t_in) & (tt <= t_in + 86400)
            if not m2.any():
                continue
            base = float(vv[m2].min())
            base_src = "first24h"

        # 7-day window after ICU admission
        w = (tt >= t_in) & (tt <= t_in + 7 * 86400)
        if not w.any():
            continue
        tw, vw = tt[w], vv[w]

        aki_7d = bool((vw >= base * 1.5).any()) if base > 0 else False

        # 48h rise >= 0.3 mg/dL : compare each value with the min of preceding 48h
        aki_48 = False
        for k in range(len(vw)):
            lo = tw[k] - 48 * 3600
            prev = vw[(tw >= lo) & (tw <= tw[k])]
            if len(prev) and (vw[k] - prev.min()) >= 0.3:
                aki_48 = True
                break

        rows.append(dict(stay_id=r.stay_id, subject_id=sid,
                         baseline=base, base_src=base_src,
                         cr_max=float(vw.max()), n_cr=int(len(vw)),
                         aki_7d=aki_7d, aki_48=aki_48, aki=bool(aki_7d or aki_48)))
        if (i + 1) % 20000 == 0:
            W("    %s stays processed (%.0fs)" % (format(i + 1, ","), time.time() - t1))
            Save()

    aki = pd.DataFrame(rows)
    aki.to_parquet(AKIF, index=False)
    W("  computed for %s stays" % format(len(aki), ","))
    W("  saved -> %s" % AKIF)
W("")

if len(aki):
    W("  AKI definition breakdown:")
    W("    aki_7d (>=1.5x baseline) : %s" % format(int(aki["aki_7d"].sum()), ","))
    W("    aki_48h (>=0.3 mg/dL rise): %s" % format(int(aki["aki_48"].sum()), ","))
    W("    AKI (either)             : %s  (%.1f%%)"
      % (format(int(aki["aki"].sum()), ","), 100.0 * aki["aki"].mean()))
    W("")
    W("    baseline source: %s" % aki["base_src"].value_counts().to_dict())
    W("    median baseline creatinine: %.2f mg/dL" % aki["baseline"].median())
W("")
Save()

# ==================================================================
# D. link with pairs
# ==================================================================
W("=" * 66)
W(" D. link: occult hypotension burden vs AKI")
W("=" * 66)
Save()
try:
    pairs = pd.read_parquet(PAIRF)
    THR = 65
    pairs["art_low"] = pairs["map_art"] < THR
    pairs["occult"] = (pairs["map_nibp"] >= THR) & (pairs["map_art"] < THR)
    pairs["falarm"] = (pairs["map_nibp"] < THR) & (pairs["map_art"] >= THR)
    per = pairs.groupby("stay_id").agg(
        n_pairs=("occult", "size"),
        n_art_low=("art_low", "sum"),
        n_occult=("occult", "sum"),
        n_falarm=("falarm", "sum"),
        mean_art=("map_art", "mean"),
        min_art=("map_art", "min"),
    ).reset_index()

    m = per.merge(aki, on="stay_id", how="inner")
    W("  stays with pairs AND AKI data: %s" % format(len(m), ","))
    W("")
    if len(m):
        low = m[m["n_art_low"] > 0].copy()
        W("  --- among stays with >=1 invasively-low reading (n=%s) ---" % format(len(low), ","))
        low["occult_frac"] = low["n_occult"] / low["n_art_low"]

        q = pd.cut(low["occult_frac"], [-0.001, 0.001, 0.5, 0.999, 1.001],
                   labels=["none missed", "<=50% missed", "50-99% missed", "all missed"])
        tab = low.groupby(q, observed=False).agg(
            n=("aki", "size"), aki=("aki", "sum"), rate=("aki", "mean"))
        tab["rate_pct"] = (100.0 * tab["rate"]).round(1)
        W(tab[["n", "aki", "rate_pct"]].to_string())
        W("")
        W("  --- continuous, per 10-percentage-point increase in missed share ---")
        W("    crude AKI rate where no hypotension was missed : %.1f %%"
          % (100.0 * low.loc[low["occult_frac"] <= 0.001, "aki"].mean())
          if (low["occult_frac"] <= 0.001).any() else "    (none)")
        W("    crude AKI rate where all hypotension was missed: %.1f %%"
          % (100.0 * low.loc[low["occult_frac"] >= 0.999, "aki"].mean())
          if (low["occult_frac"] >= 0.999).any() else "    (none)")
        W("")
        m.to_parquet(os.path.join(DERIVED, "mimic_pairs_aki.parquet"), index=False)
        W("  saved -> %s" % os.path.join(DERIVED, "mimic_pairs_aki.parquet"))
except Exception:
    W("  D FAILED:"); W(traceback.format_exc())
W("")
Save()

W("=" * 66)
W(" DONE in %.0f s" % (time.time() - t0))
W("=" * 66)
Save()
print("")
print("Report saved to: %s" % REPORT)
