"""
Smoke test for the execution-planning verifier script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "src/verification/verify_registry_execution_planning.py"
pytestmark = pytest.mark.smoke


def test_execution_planning_verifier_script_passes() -> None:
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
    assert "OK: execution-planning registry lane verified" in proc.stdout
