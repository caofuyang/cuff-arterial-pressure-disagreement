# Disagreement between non-invasive and invasive blood pressure monitoring

Analysis code supporting the manuscript:

> **Disagreement between non-invasive and invasive blood pressure monitoring in
> the operating room and the intensive care unit: a multi-database analysis of
> 1.36 million paired measurements**

---

## What this code does

The study pairs non-invasive (cuff) and invasive (arterial line) mean arterial
pressure readings recorded in the same minute, in three publicly available
databases, and asks three questions:

1. how large is the disagreement, and does its direction differ between the
   operating room and intensive care?
2. how many patients with arterial hypotension never had a single low cuff
   reading, and how deep was that unshown hypotension?
3. was hypotension the cuff never showed associated with acute kidney injury?

---

## Data sources

**The source data are not included in this repository.** They are governed by
data use agreements that prohibit redistribution and must be obtained from
their original providers:

| Database | Provider | Access |
|---|---|---|
| MOVER | UC Irvine | https://mover.ics.uci.edu/ — free, requires signing a data usage agreement |
| MIMIC-IV v3.1 | PhysioNet | https://physionet.org/content/mimiciv/ — credentialed access |
| eICU-CRD v2.0 | PhysioNet | https://physionet.org/content/eicu-crd/ — credentialed access |

PhysioNet credentialed access requires completion of CITI training, verification
of identity, and signature of the PhysioNet Credentialed Health Data Use
Agreement for each project.

---

## Requirements

```
python >= 3.10
pip install -r requirements.txt
```

---

## Before running anything

Every script begins with a short block of absolute paths. **Edit those paths to
match your own installation.** `config.py` lists the values used in the original
analysis as a reference.

The scripts are written to be run in order within each folder. Derived
intermediate files are written to the working directory, and later scripts read
them from there, so a full run from an empty working directory will take
considerably longer than a partial one.

---

## Repository structure

```
01_data_extraction/
  mimic_extract_pressure.py      extract arterial (item 220052) and cuff
                                 (item 220181) mean pressure from chartevents
  mimic_aki.py                   KDIGO acute kidney injury from labevents
  eicu_extract_pressure.py       extract "Invasive BP Mean" and "Non-invasive
                                 BP Mean" from the nurseCharting table
  eicu_aki.py                    KDIGO acute kidney injury from the lab table
  mover_extract_pressure.py      extract labelled arterial and cuff mean
                                 pressure from the EPIC flowsheet records;
                                 also documents why the unlabelled generic
                                 "MAP (mmHg)" field was excluded
  mover_aki.py                   postoperative acute kidney injury from the
                                 EPIC laboratory records (used only for the
                                 sensitivity analysis reported in
                                 Supplementary Table S3)

02_analysis/
  value_granularity_check.py     verify that pressure values are recorded at
                                 full resolution in each database (this check
                                 is what excluded the INSPIRE database from
                                 the study; see below)
  pairing_method_comparison.py   compare same-minute, nearest-within-5-minutes
                                 and window-averaged pairing strategies
  same_minute_vs_wider_windows.py  sensitivity of the agreement estimates to
                                 the matching window
  patient_level_misclassification.py
                                 cases whose arterial hypotension was never
                                 once shown by the cuff, and the depth of that
                                 unshown hypotension
  lowest_map_robustness.py       the lowest arterial MAP as a continuous
                                 exposure, across all modelling specifications
  pressure_threshold_analysis.py reading-level discordance at 65, 60 and
                                 55 mmHg (Tables 1 and 2, Figure 2)
  hypotension_group_analysis.py  four-group comparison of acute kidney injury
                                 (Table 3, Figure 3)
  group_analysis_stratified.py   the four-group analysis stratified by number
                                 of paired readings, by cohort threshold, by
                                 an alternative group definition using the 5th
                                 percentile, and by sex (Supplementary Table S5)
  cumulative_burden_sensitivity.py
                                 cumulative area below threshold per monitored
                                 hour, across all specifications (Supplementary
                                 Table S2); this exposure did not behave
                                 consistently and is not reported as a primary
                                 result
  mover_outcome_sensitivity.py   the MOVER outcome analysis and its failure of
                                 the positive control (Supplementary Table S3)
  eicu_admission_source.py       whether discordance differs by ICU admission
                                 source within eICU-CRD

03_figures/
  make_figures.py                generate Figures 1-3
```

---

## A note on INSPIRE

An earlier version of this project also used the INSPIRE perioperative database
(Seoul National University Hospital). It was excluded after
`value_granularity_check.py` showed that its intraoperative blood pressure
values are recorded at very low resolution — 20 distinct values for invasive
mean pressure and 21 for non-invasive, against 181-493 in the databases used
here. That is consistent with the percentile categorisation applied for
de-identification. Absolute thresholds such as 65 mmHg cannot be applied to
such data. The check is included in this repository because the same problem
may affect other users of that dataset.

---

## Statistical notes

Two numerical issues arose during the analysis and are relevant to anyone
re-running it:

1. **`statsmodels` does not add an intercept automatically.** A logistic model
   without one returns group-specific odds rather than odds ratios. These look
   plausible — all positive, sensible magnitude — and the only way to detect the
   problem is to check the coefficients against the crude event rates.
2. **L-BFGS can stall without converging when continuous covariates differ
   widely in scale**, and `statsmodels` then returns the starting values,
   producing odds ratios of exactly 1.000. Continuous covariates are standardised
   before fitting, and convergence is checked explicitly.

---

## Licence

Code released under the MIT Licence (see `LICENSE`).

The source data remain subject to their own licences and are **not** covered by
this licence.

---

## Citation

If you use this code, please cite the manuscript above.
