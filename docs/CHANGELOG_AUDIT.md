<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/docs_root_narratives.toml -->

# Audit Changelog

## 2026-01-27

- Initialized local Git repo (`main`) and committed repository state.
- Added explicit sedenion zero-divisor identity test and a math validation report:
  - `tests/test_cayley_dickson_properties.py`
  - `docs/MATH_VALIDATION_REPORT.md`
- Added minimal group-theory helper for `|PSL(2,7)|` order:
  - `src/gemini_physics/group_theory.py`
  - `tests/test_group_theory.py`
- Hardened provenance hashing for `data/external/`:
  - `bin/record_external_hashes.py`
  - `make provenance` writes `data/external/PROVENANCE.local.json`
