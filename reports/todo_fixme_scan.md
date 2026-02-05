# TODO/FIXME Scan (2026-02-01)

Command used (excluding venv/caches and large artifact dirs):

`rg -n "\\b(TODO|FIXME)\\b" -S`

## High-signal hotspots

- Claims pipeline backlog: `docs/CLAIMS_TASKS.md` (many rows still `TODO`)
- Global tracker: `docs/TODO.md`
- Targeted mechanism stubs:
  - `docs/C026_MASS_GAP_MECHANISM.md` (Status: TODO)
  - `docs/C027_DEFF_HORIZON_TEST.md` (Status: TODO)
- Tensor-network engine TODOs:
  - `src/tensor_networks/contraction.py`
  - `src/tensor_networks/gradients.py`
  - `src/tensor_networks/mera.py`

## Notes

- This scan is a navigation aid only. Treat each TODO as a scoped hypothesis-to-check task.
- For repo policy, TODO lines are allowed to be imperative ("must verify ...") and do not count as overclaims.
