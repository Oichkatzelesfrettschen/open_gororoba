"""
Smoke test for the dedicated C-010/C-011 verifier script.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPO_ROOT / "src/verification/verify_c010_c011_theses.py"


def test_c010_c011_verifier_script_passes() -> None:
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
    assert "C-010 OK:" in proc.stdout
    assert "C-011 OK:" in proc.stdout
