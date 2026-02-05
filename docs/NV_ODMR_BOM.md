# NV ODMR / OPU BOM (Jan 2026) -- Local Staging

This repo has a locally staged bill-of-materials spreadsheet:

- Source (outside repo): `~/Playground/HeadOfPreparedness/NV_ODMR_OPU_BOM_Jan2026.xlsx`
- Local copy (gitignored): `data/external/nv_odmr/NV_ODMR_OPU_BOM_Jan2026.xlsx`

The local copy is **not committed** (it may be personal/private). The filename + hash are recorded in:

- `data/external/PROVENANCE.local.json`

To refresh hashes after updating local spreadsheets, run:

- `python3 bin/record_external_hashes.py`

## Contents (sheet summary)

- `BOM` (24x14): line items, vendors, URLs, notes, and estimated totals.
- `Known_Gaps` (9x7): explicit gaps and proposed actions (prioritized).
- `PeerReviewed_Sources` (19x9): citations supporting design decisions (OPU optics, ODMR builds, detectors, etc.).
