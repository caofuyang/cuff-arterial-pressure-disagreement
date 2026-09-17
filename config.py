# -*- coding: utf-8 -*-
"""
Path configuration.

Every analysis script in this repository begins with a small block of
absolute paths pointing at (a) the source databases and (b) a working
directory for derived files. Edit the values below to match your own
installation, then update the corresponding lines at the top of each script.

IMPORTANT
---------
The three source databases are not redistributed with this repository.
They must be obtained from their original providers under the applicable
data use agreements:

  MOVER      https://mover.ics.uci.edu/
  MIMIC-IV   https://physionet.org/content/mimiciv/
  eICU-CRD   https://physionet.org/content/eicu-crd/
"""

# ---------------------------------------------------------------- sources
MOVER_DERIVED      = r"E:\MOVER\derived"
MOVER_EPIC_EMR     = r"E:\MOVER\data\EPIC_EMR\EMR"
MIMIC_ROOT         = r"E:\MIMIC-IV\data\mimic-iv-3.1"
EICU_ROOT          = r"E:\eICU-CRD\data\eicu-collaborative-research-database-2.0"

# ------------------------------------------------------- working directory
# All derived files (parquet, reports) are written here.
WORKDIR = r"E:\occult_htn"
