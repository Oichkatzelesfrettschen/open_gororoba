#!/usr/bin/env python3
"""Fail if /simulation/trace/rho_mean contains non-finite values."""

import sys
from pathlib import Path

import h5py
import numpy as np


def check_file(path: Path) -> bool:
    if not path.exists():
        print(f"[FAIL] {path}: file not found")
        return False

    try:
        with h5py.File(path, "r") as f:
            if "/simulation/trace/rho_mean" not in f:
                print(f"[FAIL] {path}: dataset /simulation/trace/rho_mean not found")
                return False
            data = np.asarray(f["/simulation/trace/rho_mean"])
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[FAIL] {path}: unable to read HDF5 ({exc})")
        return False

    total = int(data.size)
    finite_count = int(np.isfinite(data).sum())
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    if finite_count != total:
        print(
            f"[FAIL] {path}: finite={finite_count}/{total}, nan={nan_count}, inf={inf_count}"
        )
        return False

    print(f"[OK]   {path}: finite={finite_count}/{total}, nan={nan_count}, inf={inf_count}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: check_rho_mean_finite.py <h5-path> [<h5-path> ...]")
        return 2

    all_ok = True
    for raw in sys.argv[1:]:
        all_ok = check_file(Path(raw)) and all_ok

    if all_ok:
        print("FINITE_GATE: PASS")
        return 0

    print("FINITE_GATE: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
