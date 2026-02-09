#!/usr/bin/env python3
"""
Export human-facing markdown mirrors from authoritative TOML registries.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tomllib


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _header(title: str) -> list[str]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        f"# {title}",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: TOML registry files under registry/ -->",
        f"<!-- Generated at: {now} -->",
        "",
    ]


def export_insights(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/insights.toml")
    insights = sorted(data.get("insight", []), key=lambda x: x.get("id", ""))
    lines = _header("Insights Registry Mirror")
    lines.append("Authoritative source: `registry/insights.toml`.")
    lines.append("")
    lines.append(f"Total insights: {len(insights)}")
    lines.append("")
    for row in insights:
        claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
        lines.append(f"## {row.get('id', 'I-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Date: {row.get('date', '')}")
        lines.append(f"- Status: {row.get('status', '')}")
        lines.append(f"- Sprint: {row.get('sprint', '')}")
        lines.append(f"- Claims: {claims}")
        lines.append("")
        lines.append(row.get("summary", "").strip())
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_experiments(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/experiments.toml")
    experiments = sorted(data.get("experiment", []), key=lambda x: x.get("id", ""))
    lines = _header("Experiments Registry Mirror")
    lines.append("Authoritative source: `registry/experiments.toml`.")
    lines.append("")
    lines.append(f"Total experiments: {len(experiments)}")
    lines.append("")
    for row in experiments:
        claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
        outputs = ", ".join(row.get("output", [])) if row.get("output") else "(none)"
        lines.append(f"## {row.get('id', 'E-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Binary: `{row.get('binary', '')}`")
        lines.append(f"- Input: {row.get('input', '')}")
        lines.append(f"- Output: {outputs}")
        lines.append(f"- Deterministic: `{row.get('deterministic', False)}`")
        if "seed" in row:
            lines.append(f"- Seed: `{row.get('seed')}`")
        lines.append(f"- GPU: `{row.get('gpu', False)}`")
        lines.append(f"- Claims: {claims}")
        lines.append("")
        lines.append("Method:")
        lines.append(row.get("method", "").strip())
        lines.append("")
        lines.append("Run command:")
        lines.append("```bash")
        lines.append(row.get("run", "").strip())
        lines.append("```")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_roadmap(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/roadmap.toml")
    roadmap = data.get("roadmap", {})
    sections = data.get("roadmap", {})
    workstreams = data.get("workstream", [])
    lines = _header("Roadmap Registry Mirror")
    lines.append("Authoritative source: `registry/roadmap.toml`.")
    lines.append("")
    lines.append(f"- Consolidated date: {roadmap.get('consolidated_date', '')}")
    lines.append(f"- Source markdown: `{roadmap.get('source_markdown', '')}`")
    lines.append(f"- Status: `{roadmap.get('status', '')}`")
    lines.append("")
    lines.append("## Companion Docs")
    lines.append("")
    for item in roadmap.get("companion_docs", []):
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Workstreams")
    lines.append("")
    for ws in workstreams:
        lines.append(f"### {ws.get('id', 'WS-???')}: {ws.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Priority: `{ws.get('priority', '')}`")
        lines.append(f"- Status: `{ws.get('status', '')}`")
        lines.append(f"- Description: {ws.get('description', '')}")
        lines.append("- Primary outputs:")
        for out in ws.get("primary_outputs", []):
            lines.append(f"  - `{out}`")
        lines.append("")
    if "sections" in data:
        _ = sections
    _write(out_path, "\n".join(lines))


def export_todo(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/todo.toml")
    todo = data.get("todo", {})
    sprint = data.get("sprint", {})
    tasks = data.get("task", [])
    lines = _header("TODO Registry Mirror")
    lines.append("Authoritative source: `registry/todo.toml`.")
    lines.append("")
    lines.append(f"- Updated: {todo.get('updated', '')}")
    lines.append(f"- Sprint: {sprint.get('id', '')} ({sprint.get('name', '')})")
    lines.append(f"- Sprint status: `{sprint.get('status', '')}`")
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    for task in tasks:
        lines.append(f"### {task.get('id', 'TASK-???')}: {task.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Status: `{task.get('status', '')}`")
        lines.append("- Evidence:")
        for ev in task.get("evidence", []):
            lines.append(f"  - `{ev}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_next_actions(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/next_actions.toml")
    meta = data.get("next_actions", {})
    actions = data.get("action", [])
    lines = _header("Next Actions Registry Mirror")
    lines.append("Authoritative source: `registry/next_actions.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Status: `{meta.get('status', '')}`")
    lines.append("")
    lines.append("## Priority Queue")
    lines.append("")
    for action in actions:
        lines.append(
            f"### {action.get('id', 'NA-???')} ({action.get('priority', 'P?')}): "
            f"{action.get('title', '(untitled)')}"
        )
        lines.append("")
        lines.append(f"- Status: `{action.get('status', '')}`")
        lines.append(f"- Description: {action.get('description', '')}")
        lines.append("- References:")
        for ref in action.get("references", []):
            lines.append(f"  - `{ref}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_requirements(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/requirements.toml")
    req = data.get("requirements", {})
    modules = data.get("module", [])
    gaps = data.get("coverage_gap", [])
    lines = _header("Requirements Registry Mirror")
    lines.append("Authoritative source: `registry/requirements.toml`.")
    lines.append("")
    lines.append(f"- Updated: {req.get('updated', '')}")
    lines.append(f"- Python recommended: `{req.get('python_recommended', '')}`")
    lines.append(f"- Python allowed: `{req.get('python_allowed', '')}`")
    lines.append(f"- Primary markdown: `{req.get('primary_markdown', '')}`")
    lines.append("")
    lines.append("## Modules")
    lines.append("")
    for module in modules:
        lines.append(f"### {module.get('id', 'REQ-???')}: {module.get('name', '(unnamed)')}")
        lines.append("")
        lines.append(f"- Status: `{module.get('status', '')}`")
        lines.append(f"- Markdown: `{module.get('markdown', '')}`")
        targets = module.get("install_targets", [])
        if targets:
            lines.append("- Install targets:")
            for target in targets:
                lines.append(f"  - `{target}`")
        lines.append("")
    lines.append("## Coverage Gaps")
    lines.append("")
    for gap in gaps:
        lines.append(f"### {gap.get('id', 'REQ-GAP-???')}: {gap.get('area', '(area)')}")
        lines.append("")
        lines.append(f"- Status: `{gap.get('status', '')}`")
        lines.append(f"- Description: {gap.get('description', '')}")
        lines.append(f"- Proposed resolution: {gap.get('proposed_resolution', '')}")
        lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_tasks(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_tasks.toml")
    meta = data.get("claims_tasks", {})
    sections = data.get("section", [])
    tasks = data.get("task", [])
    lines = _header("Claims Tasks Registry Mirror")
    lines.append("Authoritative source: `registry/claims_tasks.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown: `{meta.get('source_markdown', '')}`")
    lines.append(f"- Task count: {meta.get('task_count', len(tasks))}")
    lines.append(f"- Section count: {meta.get('section_count', len(sections))}")
    lines.append(
        f"- Canonical status task count: {meta.get('canonical_status_task_count', '')}"
    )
    lines.append(
        f"- Noncanonical status task count: {meta.get('noncanonical_status_task_count', '')}"
    )
    lines.append("")
    lines.append("## Sections")
    lines.append("")
    for section in sections:
        lines.append(f"- {section.get('id', '')}: {section.get('name', '')} ({section.get('task_count', 0)} tasks)")
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    for task in sorted(tasks, key=lambda row: int(row.get("order_index", 0))):
        lines.append(
            f"### {task.get('id', 'CTASK-???')} ({task.get('claim_id', 'C-???')}, {task.get('status_token', 'UNKNOWN')})"
        )
        lines.append("")
        lines.append(f"- Section: {task.get('section', '')}")
        lines.append(f"- Source line: {task.get('source_line', '')}")
        lines.append(f"- Status raw: {task.get('status_raw', '')}")
        lines.append(f"- Canonical: `{task.get('status_canonical', False)}`")
        lines.append("")
        lines.append(task.get("task", "").strip())
        lines.append("")
        artifacts = task.get("output_artifacts", [])
        if artifacts:
            lines.append("Output artifacts:")
            for artifact in artifacts:
                lines.append(f"- `{artifact}`")
            lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_domains(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_domains.toml")
    meta = data.get("claims_domains", {})
    domains = data.get("domain", [])
    claim_domains = data.get("claim_domain", [])
    lines = _header("Claims Domains Registry Mirror")
    lines.append("Authoritative source: `registry/claims_domains.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source CSV: `{meta.get('source_csv', '')}`")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Domain file count: {meta.get('domain_file_count', len(domains))}")
    lines.append(f"- Claim count: {meta.get('claim_count', len(claim_domains))}")
    lines.append("")
    lines.append("## Domain Files")
    lines.append("")
    for row in sorted(domains, key=lambda item: item.get("id", "")):
        lines.append(f"### {row.get('id', '')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Declared count: {row.get('declared_count', 0)}")
        lines.append(f"- CSV claim count: {row.get('csv_claim_count', 0)}")
        lines.append(f"- Markdown claim count: {row.get('markdown_claim_count', 0)}")
        lines.append(f"- Count match: `{row.get('count_match', False)}`")
        lines.append(f"- Mapping match: `{row.get('mapping_match', False)}`")
        lines.append("")
    lines.append("## Claim Crosswalk")
    lines.append("")
    for row in sorted(claim_domains, key=lambda item: item.get("claim_id", "")):
        lines.append(f"- {row.get('claim_id', '')}: csv={row.get('domains_csv', [])}, markdown={row.get('domains_markdown', [])}, match={row.get('domain_sets_match', False)}")
    lines.append("")
    _write(out_path, "\n".join(lines))


def export_claim_tickets(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claim_tickets.toml")
    meta = data.get("claim_tickets", {})
    tickets = data.get("ticket", [])
    lines = _header("Claim Tickets Registry Mirror")
    lines.append("Authoritative source: `registry/claim_tickets.toml`.")
    lines.append("")
    lines.append(f"- Updated: {meta.get('updated', '')}")
    lines.append(f"- Source markdown glob: `{meta.get('source_markdown_glob', '')}`")
    lines.append(f"- Ticket count: {meta.get('ticket_count', len(tickets))}")
    lines.append("")
    lines.append("## Tickets")
    lines.append("")
    for row in sorted(tickets, key=lambda item: item.get("id", "")):
        lines.append(f"### {row.get('id', 'TICKET-???')}: {row.get('title', '(untitled)')}")
        lines.append("")
        lines.append(f"- Source markdown: `{row.get('source_markdown', '')}`")
        lines.append(f"- Kind: `{row.get('ticket_kind', '')}`")
        lines.append(f"- Owner: {row.get('owner', '')}")
        lines.append(f"- Created: {row.get('created', '')}")
        lines.append(f"- Status: `{row.get('status_token', '')}` ({row.get('status_raw', '')})")
        lines.append(
            f"- Claim range: {row.get('claim_range_start', 0)}..{row.get('claim_range_end', 0)}"
        )
        lines.append(
            f"- Checkbox progress: done={row.get('done_checkboxes', 0)}, open={row.get('open_checkboxes', 0)}"
        )
        claims = row.get("claims_referenced", [])
        if claims:
            lines.append(f"- Claims referenced ({len(claims)}): {', '.join(claims)}")
        backlog = row.get("backlog_reports", [])
        if backlog:
            lines.append("- Backlog reports:")
            for item in backlog:
                lines.append(f"  - `{item}`")
        checks = row.get("acceptance_checks", [])
        if checks:
            lines.append("- Acceptance checks:")
            for check in checks:
                lines.append(f"  - `{check}`")
        lines.append("")
    _write(out_path, "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/generated",
        help="Output markdown mirror directory.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    export_insights(repo_root, out_dir / "INSIGHTS_REGISTRY_MIRROR.md")
    export_experiments(repo_root, out_dir / "EXPERIMENTS_REGISTRY_MIRROR.md")
    export_roadmap(repo_root, out_dir / "ROADMAP_REGISTRY_MIRROR.md")
    export_todo(repo_root, out_dir / "TODO_REGISTRY_MIRROR.md")
    export_next_actions(repo_root, out_dir / "NEXT_ACTIONS_REGISTRY_MIRROR.md")
    export_requirements(repo_root, out_dir / "REQUIREMENTS_REGISTRY_MIRROR.md")
    export_claims_tasks(repo_root, out_dir / "CLAIMS_TASKS_REGISTRY_MIRROR.md")
    export_claims_domains(repo_root, out_dir / "CLAIMS_DOMAINS_REGISTRY_MIRROR.md")
    export_claim_tickets(repo_root, out_dir / "CLAIM_TICKETS_REGISTRY_MIRROR.md")

    print(f"Wrote TOML-driven markdown mirrors to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
