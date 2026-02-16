#!/usr/bin/env python3
"""Deprecated Wave 5 Batch 4 verifier shim for execution-planning registries."""

from __future__ import annotations

import importlib.util
from pathlib import Path


DEPRECATION_GUIDANCE = (
    "DEPRECATED: src/verification/verify_wave5_batch4_registries.py is a legacy alias. "
    "Use src/verification/verify_registry_execution_planning.py."
)


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_canonical_verify_registry_execution_planning",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load canonical script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    print(DEPRECATION_GUIDANCE)
    canonical = Path(__file__).with_name("verify_registry_execution_planning.py")
    return int(_load_main(canonical)())


if __name__ == "__main__":
    raise SystemExit(main())
