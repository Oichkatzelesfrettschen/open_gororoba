"""
Smoke test for the external-source operational-contract verifier.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "src/verification/verify_external_source_operational_contracts.py"
pytestmark = pytest.mark.smoke


def test_external_source_operational_contracts_pass() -> None:
    env = dict(os.environ)
    env["PYTHONWARNINGS"] = "error"
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, (
        "Verifier exited non-zero.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "OK: external-source operational contracts verified." in proc.stdout
