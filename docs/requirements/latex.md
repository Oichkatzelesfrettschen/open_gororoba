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

# Requirements: LaTeX (PDF Build)

To compile the LaTeX sources under `docs/latex/`, install a TeX distribution (e.g. TeX Live) and ensure `latexmk` is available.

```bash
make latex
```

Notes:
- The Makefile checks for `latexmk` on PATH.
- Output is written under `docs/latex/out/`.
- Treat LaTeX as an opt-in toolchain: it is not part of `make smoke` or `make check`.
