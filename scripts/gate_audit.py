#!/usr/bin/env python3
"""Run keep-going gate audits and capture per-step logs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

TAIL_LINE_COUNT = 20


def find_repo_root() -> Path:
    probe = Path.cwd().resolve()
    while probe != probe.parent:
        if (probe / ".git").exists():
            return probe
        probe = probe.parent
    return Path(__file__).resolve().parent.parent


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def render_tail_block(text: str) -> list[str]:
    lines = text.splitlines()
    if not lines:
        return ["(no log output)"]
    tail = lines[-TAIL_LINE_COUNT:]
    if len(lines) > len(tail):
        return [f"... ({len(lines) - len(tail)} earlier line(s) omitted)"] + tail
    return tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for logs and summary output. Defaults to reports/gates/YYYY-MM-DD/HHMMSS."
        ),
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    timestamp = dt.datetime.now().strftime("%Y-%m-%d/%H%M%S")
    if args.output_dir:
        output_dir = (repo_root / args.output_dir).resolve()
    else:
        output_dir = (repo_root / "reports" / "gates" / timestamp).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = [
        ("gate-local", ["./makew", "gate-local"]),
        ("gate-ci-python", ["./makew", "gate-ci-python"]),
        ("gate-ci-rust", ["./makew", "gate-ci-rust"]),
        ("nextest-list", ["cargo", "nextest", "list", "--workspace", "--tests"]),
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "error")
    env.setdefault("CARGO_HOME", str(repo_root / ".cache" / "cargo-home"))
    env.setdefault("CARGO_TARGET_DIR", str(repo_root / ".cache" / "gate-target"))

    summary_lines = [
        f"# Gate Audit ({dt.datetime.now().isoformat(timespec='seconds')})",
        "",
        f"Output directory: `{output_dir.relative_to(repo_root)}`",
        "",
        "| Step | Exit Code | Log |",
        "| --- | ---: | --- |",
    ]
    failures = 0
    step_rows: list[dict[str, object]] = []

    for name, command in commands:
        log_path = output_dir / f"{name}.log"
        proc = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        combined_output = proc.stdout + proc.stderr
        log_text = "\n".join(
            [
                f"# Step: {name}",
                f"# Command: {format_command(command)}",
                f"# Exit Code: {proc.returncode}",
                "",
                combined_output,
            ]
        )
        log_path.write_text(log_text, encoding="utf-8")
        summary_lines.append(
            f"| `{name}` | `{proc.returncode}` | `{log_path.relative_to(repo_root)}` |"
        )
        step_rows.append(
            {
                "name": name,
                "exit_code": proc.returncode,
                "log": log_path.relative_to(repo_root).as_posix(),
            }
        )

        summary_lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"Exit code: `{proc.returncode}`",
                "",
                "```text",
                *render_tail_block(combined_output),
                "```",
            ]
        )

        if proc.returncode != 0:
            failures += 1

    summary_lines.extend(
        [
            "",
            (f"Gate audit failed in {failures} step(s)." if failures else "Gate audit passed."),
            "",
            "Review the per-step logs for full output.",
            "",
        ]
    )

    summary_path = output_dir / "summary.md"
    summary_text = "\n".join(summary_lines)
    summary_path.write_text(summary_text, encoding="utf-8")

    latest_summary_path = repo_root / "reports" / "gates" / "LATEST.md"
    latest_manifest_path = repo_root / "reports" / "gates" / "latest.json"
    latest_summary_path.write_text(summary_text, encoding="utf-8")
    latest_manifest_path.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "output_dir": output_dir.relative_to(repo_root).as_posix(),
                "summary": summary_path.relative_to(repo_root).as_posix(),
                "failure_count": failures,
                "steps": step_rows,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {summary_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
