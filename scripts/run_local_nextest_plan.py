#!/usr/bin/env python3
"""Run a package-aware local nextest plan.

The local push gate should not behave like a near-CI rebuild for bin-heavy CLI
crates. This runner executes one package at a time, selecting only the library
tests plus explicit integration tests discovered on disk for that package.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATES_DIR = REPO_ROOT / "crates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-jobs", required=True)
    parser.add_argument("--test-threads", required=True)
    parser.add_argument("--filterset", default="")
    parser.add_argument("packages", nargs="+")
    return parser.parse_args()


def package_root(package: str) -> Path:
    return CRATES_DIR / package


def has_library(package: str) -> bool:
    return (package_root(package) / "src" / "lib.rs").is_file()


def has_inline_tests(package: str) -> bool:
    src_root = package_root(package) / "src"
    if not src_root.is_dir():
        return False

    for path in src_root.rglob("*.rs"):
        if path.parts[: len((src_root / "bin").parts)] == (src_root / "bin").parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "#[test]" in text or "#[cfg(test)]" in text or "mod tests" in text:
            return True
    return False


def integration_tests(package: str) -> list[str]:
    tests_dir = package_root(package) / "tests"
    if not tests_dir.is_dir():
        return []
    return sorted(path.stem for path in tests_dir.glob("*.rs"))


def run_package(
    package: str,
    build_jobs: str,
    test_threads: str,
    filterset: str,
) -> int:
    has_lib = has_library(package)
    has_lib_tests = has_lib and has_inline_tests(package)
    tests = integration_tests(package)
    if not has_lib_tests and not tests:
        print(
            f"[local-nextest] skip {package}: no inline lib tests and no integration tests"
        )
        return 0

    command = [
        "cargo",
        "nextest",
        "run",
        "--build-jobs",
        build_jobs,
        "--test-threads",
        test_threads,
        "-p",
        package,
    ]
    if has_lib_tests:
        command.append("--lib")
    for test_name in tests:
        command.extend(["--test", test_name])
    if filterset:
        command.extend(["-E", filterset])

    selected_targets = []
    if has_lib_tests:
        selected_targets.append("lib")
    selected_targets.extend(f"test:{name}" for name in tests)
    print(
        f"[local-nextest] run {package}: "
        + (", ".join(selected_targets) if selected_targets else "(none)")
    )
    result = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy())
    return result.returncode


def main() -> int:
    args = parse_args()
    for package in args.packages:
        exit_code = run_package(
            package=package,
            build_jobs=args.build_jobs,
            test_threads=args.test_threads,
            filterset=args.filterset,
        )
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
