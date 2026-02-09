<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml, registry/requirements_narrative.toml -->

# Requirements: Algebra Engine (Cayley-Dickson, de Marrais, Reggiani)

These components live under `src/gemini_physics/` and are exercised by the unit tests.

Install:
```bash
make install
make test
```

Notes:
- The core algebra replication/validation code depends only on the base install extras (NumPy/SciPy/SymPy/Numba).
- Keep `PYTHONWARNINGS=error` enabled in scripts/CI; treat warnings as errors.
- Artifact generation entrypoints:
  - `make artifacts-boxkites` (de Marrais exports)
  - `make artifacts-reggiani` (Reggiani exports)
  - `make artifacts-motifs` (motif census baseline: 16D/32D + plot)
  - `make artifacts-motifs-big` (motif census extended: 64D/128D exact + 256D sampled + plot)

## Optional: advanced quaternion + math foundations (system + pip)

The repo's default workflow uses an isolated venv (`make install`). If you want to use additional
math foundations, install them into the venv with pip (recommended) even if you also have them
installed system-wide (Arch/AUR).

### Quaternion / geometry

- euclid3
  - Arch pkg: `python-euclid3`
  - pip: `euclid3`
  - import: `euclid3`
- numpy-quaternion
  - Arch pkg: `python-numpy-quaternion`
  - pip: `numpy-quaternion`
  - import: `quaternion`
- pyquaternion
  - Arch pkg: `python-pyquaternion`
  - pip: `pyquaternion`
  - import: `pyquaternion`
- quaternionic (provider-dependent)
  - Arch pkg: `python-quaternionic`
  - pip: `quaternionic` (verify on install)
  - import: usually `quaternionic` (verify on install)

### Multidimensional numerics / inference

- findiff (finite-difference derivatives, N dimensions)
  - Arch pkg: `python-findiff`
  - pip: `findiff`
  - import: `findiff`
- ipfn (iterative proportional fitting, N dimensions)
  - Arch pkg: `python-ipfn`
  - pip: `ipfn`
  - import: `ipfn`
- MultiNest Python bindings
  - Arch pkg: `python-multinest`
  - pip: commonly `pymultinest` (verify on install)
  - import: commonly `pymultinest` (verify on install)
- typedunits (units + dimensions with static checking)
  - Arch pkg: `python-typedunits`
  - pip: `typedunits` (verify on install)
  - import: `typedunits` (verify on install)
- mutatorMath (piecewise linear interpolation in multiple dimensions)
  - Arch pkg: `python-mutatormath-git`
  - pip: `MutatorMath` / `mutatorMath` (verify on install)
  - import: commonly `mutatorMath` (verify on install)

### Utility (non-math but sometimes needed in pipelines)

- unicodedata2
  - Arch pkg: `python-unicodedata2`
  - pip: `unicodedata2`
  - import: `unicodedata2`
- defcon / fontMath (font tooling)
  - Arch pkgs: `python-defcon`, `python-fontmath`
  - pip: `defcon`, `fontMath`
  - imports: `defcon`, `fontMath`
  - warning: some transitive deps currently emit deprecation warnings on Python 3.14; avoid
    importing these in `PYTHONWARNINGS=error` test paths unless the warnings are resolved upstream.

## Deprecation note: pyfilesystem2 (`fs`) importing `pkg_resources`

On this machine, the system `fs` package currently calls `pkg_resources.declare_namespace(...)`
at import time. Newer `pkg_resources` emits a deprecation warning, which becomes a hard error under
`PYTHONWARNINGS=error`.

Workarounds (prefer reproducible venv-based ones):
- Use a venv and pin `setuptools<81` (avoids the warning, but does not remove the underlying
  dependency on `pkg_resources`).
- Patch `fs` to avoid `pkg_resources` at import time (recommended if you must use `fs`):
  `make patch-pyfilesystem2`
  This applies the small `pkgutil.extend_path(...)` replacement in venv site-packages.

Important:
- This repo's `make` targets create an isolated venv via `python3 -m venv venv` which does NOT
  see system site-packages by default.
- To use these from the venv, either:
  - install the pip equivalents into `venv` (preferred for reproducibility), or
  - create the venv with system packages enabled (less reproducible):
    `python3 -m venv --system-site-packages venv`

Suggested venv install (edit as needed for your environment):
```bash
venv/bin/pip install euclid3 numpy-quaternion pyquaternion quaternionic findiff ipfn
venv/bin/pip install typedunits pymultinest
```

Sanity-check note (this machine, system `python3` = 3.14.2, verified 2026-01-28):
- Venv (`venv/bin/python3`): none of these are installed by default.
- System `python3` import matrix with `PYTHONWARNINGS=error`:
  - OK: `euclid3`, `quaternion` (numpy-quaternion), `pyquaternion`, `mutatorMath`, `ipfn`, `unicodedata2`, `fontMath`
  - Missing: `quaternionic`, `typedunits`, `findiff`
  - NOT warnings-as-errors safe: `defcon` (raises a `pkg_resources` deprecation warning at import time)

If you need deterministic availability inside the repo build, install the pip equivalents into the venv.
