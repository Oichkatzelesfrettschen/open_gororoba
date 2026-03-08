#!/usr/bin/env python3
"""Run a package-aware local nextest plan.

The local push gate should not behave like a near-CI rebuild for bin-heavy CLI
crates. This runner groups packages with identical test target signatures so we
can amortize Cargo startup costs while still avoiding unnecessary bin-heavy
coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRATES_DIR = REPO_ROOT / "crates"
INLINE_TEST_MARKERS = ("#[test]", "#[cfg(test)]", "mod tests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-jobs", required=True)
    parser.add_argument("--test-threads", required=True)
    parser.add_argument("--filterset", default="")
    parser.add_argument(
        "--timing-json-out",
        help="append machine-readable JSONL timing records to this path",
    )
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

    bin_root = src_root / "bin"
    for path in src_root.rglob("*.rs"):
        if bin_root in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                continue
            if any(line.startswith(marker) for marker in INLINE_TEST_MARKERS):
                return True
    return False


def integration_tests(package: str) -> list[str]:
    tests_dir = package_root(package) / "tests"
    if not tests_dir.is_dir():
        return []
    return sorted(path.stem for path in tests_dir.glob("*.rs"))


def package_plan(package: str) -> tuple[bool, list[str]] | None:
    has_lib = has_library(package)
    has_lib_tests = has_lib and has_inline_tests(package)
    tests = integration_tests(package)
    if not has_lib_tests and not tests:
        return None
    return has_lib_tests, tests


class TimingRecorder:
    def __init__(self, output_path: str | None) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.total_start = time.perf_counter()
        self.run_count = 0
        self.skip_count = 0

    def write(self, record: dict[str, object]) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def record_skip(self, package: str, reason: str) -> None:
        self.skip_count += 1
        self.write(
            {
                "kind": "skip",
                "package": package,
                "reason": reason,
            }
        )

    def record_run(
        self,
        *,
        packages: list[str],
        targets: dict[str, list[str]],
        command: list[str],
        returncode: int,
        elapsed_sec: float,
    ) -> None:
        self.run_count += 1
        self.write(
            {
                "kind": "run",
                "packages": packages,
                "targets": targets,
                "command": command,
                "returncode": returncode,
                "elapsed_sec": round(elapsed_sec, 6),
            }
        )

    def record_summary(self, returncode: int) -> None:
        self.write(
            {
                "kind": "summary",
                "run_count": self.run_count,
                "skip_count": self.skip_count,
                "returncode": returncode,
                "total_elapsed_sec": round(time.perf_counter() - self.total_start, 6),
            }
        )


def build_command(
    *,
    packages: list[str],
    has_lib_tests: bool,
    tests: list[str],
    build_jobs: str,
    test_threads: str,
    filterset: str,
) -> tuple[list[str], dict[str, list[str]]]:
    command = [
        "cargo",
        "nextest",
        "run",
        "--build-jobs",
        build_jobs,
        "--test-threads",
        test_threads,
    ]
    if has_lib_tests:
        command.append("--lib")
    for package in packages:
        command.extend(["-p", package])
    for test_name in tests:
        command.extend(["--test", test_name])
    if filterset:
        command.extend(["-E", filterset])

    selected_targets: dict[str, list[str]] = {}
    for package in packages:
        targets: list[str] = []
        if has_lib_tests:
            targets.append("lib")
        targets.extend(f"test:{name}" for name in tests)
        selected_targets[package] = targets
    return command, selected_targets


def run_command(
    *,
    packages: list[str],
    command: list[str],
    selected_targets: dict[str, list[str]],
    timing: TimingRecorder,
) -> int:
    for package, targets in selected_targets.items():
        print(
            f"[local-nextest] run {package}: "
            + (", ".join(targets) if targets else "(none)"),
            flush=True,
        )
    start = time.perf_counter()
    result = subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy())
    elapsed_sec = time.perf_counter() - start
    timing.record_run(
        packages=packages,
        targets=selected_targets,
        command=command,
        returncode=result.returncode,
        elapsed_sec=elapsed_sec,
    )
    return result.returncode


def main() -> int:
    args = parse_args()
    timing = TimingRecorder(args.timing_json_out)
    grouped_plans: OrderedDict[tuple[bool, tuple[str, ...]], list[str]] = OrderedDict()

    for package in args.packages:
        plan = package_plan(package)
        if plan is None:
            reason = "no inline lib tests and no integration tests"
            print(f"[local-nextest] skip {package}: {reason}", flush=True)
            timing.record_skip(package, reason)
            continue
        signature = (plan[0], tuple(plan[1]))
        grouped_plans.setdefault(signature, []).append(package)

    exit_code = 0
    for (has_lib_tests, tests_tuple), packages in grouped_plans.items():
        command, selected_targets = build_command(
            packages=packages,
            has_lib_tests=has_lib_tests,
            tests=list(tests_tuple),
            build_jobs=args.build_jobs,
            test_threads=args.test_threads,
            filterset=args.filterset,
        )
        exit_code = run_command(
            packages=packages,
            command=command,
            selected_targets=selected_targets,
            timing=timing,
        )
        if exit_code != 0:
            timing.record_summary(exit_code)
            return exit_code

    timing.record_summary(exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
