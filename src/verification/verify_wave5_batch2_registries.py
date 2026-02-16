#!/usr/bin/env python3
"""Deprecated compatibility shim for canonical evidence-provenance registry verification.

Use: src/verification/verify_registry_evidence_provenance.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_main(path: Path):
    spec = importlib.util.spec_from_file_location("_canonical_verify_registry_evidence_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load canonical verification script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    canonical = Path(__file__).with_name("verify_registry_evidence_provenance.py")
    print(
        "DEPRECATED: src/verification/verify_wave5_batch2_registries.py is a compatibility shim. "
        "Use src/verification/verify_registry_evidence_provenance.py.",
        file=sys.stderr,
    )
    return _load_main(canonical)()


if __name__ == "__main__":
    raise SystemExit(main())
