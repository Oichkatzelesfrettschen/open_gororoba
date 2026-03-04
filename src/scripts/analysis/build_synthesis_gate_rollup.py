#!/usr/bin/env python3
"""Run the synthesis execution contract and write a dated TOML rollup."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATE_TOKEN = "2026_02_14"
DEFAULT_REPORT_TEMPLATE = "reports/synthesis_execution_contract_{date_token}.toml"


@dataclass(frozen=True)
class ContractStep:
    step_id: str
    label: str
    argv: tuple[str, ...]


@dataclass
class StepResult:
    step_id: str
    label: str
    command: str
    exit_code: int
    outcome: str
    started_utc: str
    finished_utc: str
    duration_seconds: float
    stdout_excerpt: str
    stderr_excerpt: str
    skip_reason: str = ""


def _q(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ascii_safe(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _excerpt(text: str, limit: int = 1200) -> str:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return ""
    if len(normalized) > limit:
        normalized = normalized[:limit] + "\n...[truncated]..."
    return _ascii_safe(normalized)


def _normalize_date_token(raw: str) -> tuple[str, str]:
    cleaned = raw.strip()
    if not cleaned:
        raise ValueError("date token cannot be empty")
    parsed = dt.date.fromisoformat(cleaned.replace("_", "-"))
    return parsed.isoformat(), parsed.strftime("%Y_%m_%d")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _render_report(
    *,
    date_iso: str,
    date_token: str,
    rollup_path: str,
    results: list[StepResult],
) -> str:
    success_count = sum(1 for result in results if result.outcome == "pass")
    failure_count = sum(1 for result in results if result.outcome == "fail")
    skipped_count = sum(1 for result in results if result.outcome == "skipped")
    overall_outcome = "pass" if failure_count == 0 and skipped_count == 0 else "fail"
    first_failure = next((result for result in results if result.outcome == "fail"), None)

    lines: list[str] = [
        "# Synthesis execution contract rollup.",
        "",
        "[report]",
        f"id = {_q(f'synthesis-execution-contract-{date_iso}')}",
        f"date = {_q(date_iso)}",
        f"date_token = {_q(date_token)}",
        'status = "completed"',
        'target = "synthesis-execution-contract"',
        f"rollup_path = {_q(rollup_path)}",
        f"overall_outcome = {_q(overall_outcome)}",
        f"all_commands_passed = {_bool(overall_outcome == 'pass')}",
        f"success_count = {success_count}",
        f"failure_count = {failure_count}",
        f"skipped_count = {skipped_count}",
        "",
        "[commands]",
        'governance_gate = "make governance-gate"',
        'registry_acceptance_gate = "make registry-acceptance-gate"',
        'typed_policy_strict = "make registry-verify-typed-policy-error"',
        'project_counter_sync_check = "cargo run --release --bin project-counter-sync -- --check"',
        "",
    ]
    if first_failure is None:
        lines.extend(
            [
                "[summary]",
                'first_failure_step = ""',
                'first_failure_command = ""',
                "",
            ]
        )
    else:
        lines.extend(
            [
                "[summary]",
                f"first_failure_step = {_q(first_failure.step_id)}",
                f"first_failure_command = {_q(first_failure.command)}",
                "",
            ]
        )

    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                "[[step]]",
                f"order = {index}",
                f"id = {_q(result.step_id)}",
                f"label = {_q(result.label)}",
                f"command = {_q(result.command)}",
                f"exit_code = {result.exit_code}",
                f"outcome = {_q(result.outcome)}",
                f"started_utc = {_q(result.started_utc)}",
                f"finished_utc = {_q(result.finished_utc)}",
                f"duration_seconds = {result.duration_seconds:.3f}",
            ]
        )
        if result.skip_reason:
            lines.append(f"skip_reason = {_q(result.skip_reason)}")
        lines.append(f"stdout_excerpt = {_q(result.stdout_excerpt)}")
        lines.append(f"stderr_excerpt = {_q(result.stderr_excerpt)}")
        lines.append("")

    rendered = "\n".join(lines).rstrip() + "\n"
    try:
        rendered.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"non-ASCII content in rollup output: {exc}") from exc
    return rendered


def _run_contract(repo_root: Path) -> list[StepResult]:
    # Group A: pure Python/Make targets -- safe to parallelize.
    group_a = [
        ContractStep(
            step_id="governance_gate",
            label="governance-gate",
            argv=("make", "governance-gate"),
        ),
        ContractStep(
            step_id="registry_acceptance_gate",
            label="registry-acceptance-gate",
            argv=("make", "registry-acceptance-gate"),
        ),
    ]
    # Group B: cargo run --release targets -- must run sequentially to avoid
    # cargo build-lock contention.
    group_b = [
        ContractStep(
            step_id="typed_policy_strict",
            label="registry-verify-typed-policy-error",
            argv=("make", "registry-verify-typed-policy-error"),
        ),
        ContractStep(
            step_id="project_counter_sync_check",
            label="project-counter-sync --check",
            argv=(
                "cargo",
                "run",
                "--release",
                "--bin",
                "project-counter-sync",
                "--",
                "--check",
            ),
        ),
    ]

    def _run_one(step: ContractStep) -> StepResult:
        command = shlex.join(step.argv)
        started_utc = _utc_now()
        t0 = time.monotonic()
        completed = subprocess.run(
            list(step.argv),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.monotonic() - t0
        return StepResult(
            step_id=step.step_id,
            label=step.label,
            command=command,
            exit_code=completed.returncode,
            outcome="pass" if completed.returncode == 0 else "fail",
            started_utc=started_utc,
            finished_utc=_utc_now(),
            duration_seconds=duration,
            stdout_excerpt=_excerpt(completed.stdout),
            stderr_excerpt=_excerpt(completed.stderr),
        )

    # Phase 1: run Group A in parallel; collect all results before gating.
    results: list[StepResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        future_map = {pool.submit(_run_one, s): s for s in group_a}
        a_results: list[StepResult] = []
        for future in concurrent.futures.as_completed(future_map):
            a_results.append(future.result())
    # Restore original step order for deterministic TOML output.
    order = {s.step_id: i for i, s in enumerate(group_a)}
    a_results.sort(key=lambda r: order[r.step_id])
    results.extend(a_results)

    # Phase 2: run Group B sequentially (cargo build lock contention risk).
    for step in group_b:
        results.append(_run_one(step))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthesis execution contract checks and emit TOML rollup."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to repository root (default: current directory).",
    )
    parser.add_argument(
        "--date-token",
        default=DEFAULT_DATE_TOKEN,
        help="Date token (YYYY_MM_DD or YYYY-MM-DD).",
    )
    parser.add_argument(
        "--report-path",
        default="",
        help="Explicit report path. Defaults to reports/synthesis_execution_contract_<date>.toml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    date_iso, date_token = _normalize_date_token(str(args.date_token))
    if args.report_path:
        report_path = Path(args.report_path)
        if not report_path.is_absolute():
            report_path = repo_root / report_path
    else:
        report_path = repo_root / DEFAULT_REPORT_TEMPLATE.format(date_token=date_token)
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    results = _run_contract(repo_root)
    rollup_display_path = _display_path(report_path, repo_root)
    rendered = _render_report(
        date_iso=date_iso,
        date_token=date_token,
        rollup_path=rollup_display_path,
        results=results,
    )
    report_path.write_text(rendered, encoding="utf-8")

    failure_count = sum(1 for result in results if result.outcome == "fail")
    skipped_count = sum(1 for result in results if result.outcome == "skipped")
    overall_outcome = "pass" if failure_count == 0 and skipped_count == 0 else "fail"

    print(f"synthesis-execution-contract: {overall_outcome}")
    print(f"rollup: {rollup_display_path}")
    return 0 if overall_outcome == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
