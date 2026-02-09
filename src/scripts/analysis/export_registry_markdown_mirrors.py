#!/usr/bin/env python3
"""
Export human-facing markdown mirrors from authoritative TOML registries.
"""

from __future__ import annotations

import argparse
import csv
import tomllib
from io import StringIO
from pathlib import Path

CHECK_MODE = False
CHANGED_PATHS: list[str] = []


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    _assert_ascii(text, str(path))
    if CHECK_MODE:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if existing != text:
            CHANGED_PATHS.append(str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _header(title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: TOML registry files under registry/ -->",
        "",
    ]


def _claim_sort_key(claim_id: str) -> int:
    if claim_id.startswith("C-") and claim_id[2:].isdigit():
        return int(claim_id[2:])
    return 999999


def _pipe_escape(text: str) -> str:
    return text.replace("|", "\\|")


def _render_csv(rows: list[list[str]]) -> str:
    buf = StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _load_optional_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    return _load_toml(path)


def _narrative_map(data: dict) -> tuple[str, dict[str, str]]:
    section = data.get("insights_narrative") or data.get("experiments_narrative") or {}
    preamble = str(section.get("preamble_markdown", "")).strip()
    mapping: dict[str, str] = {}
    for row in data.get("entry", []):
        entry_id = str(row.get("id", "")).strip()
        body = str(row.get("body_markdown", "")).strip()
        if entry_id:
            mapping[entry_id] = body
    return preamble, mapping


def _single_overlay_body(data: dict, section_key: str) -> str:
    section = data.get(section_key, {})
    return str(section.get("body_markdown", "")).strip()


def _legacy_header(sources: list[str]) -> list[str]:
    return [
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        f"<!-- Source of truth: {', '.join(sources)} -->",
        "",
    ]


def _legacy_lines_from_body(body: str, fallback_title: str, fallback_lines: list[str]) -> list[str]:
    lines: list[str] = []
    if body:
        lines.extend(body.splitlines())
    else:
        lines.append(f"# {fallback_title}")
        lines.append("")
        lines.extend(fallback_lines)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


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


def export_insights_legacy(repo_root: Path, out_path: Path) -> None:
    registry_data = _load_toml(repo_root / "registry/insights.toml")
    narrative_data = _load_optional_toml(repo_root / "registry/insights_narrative.toml")
    preamble, body_by_id = _narrative_map(narrative_data)
    insights = sorted(registry_data.get("insight", []), key=lambda x: x.get("id", ""))

    lines: list[str] = []
    if preamble:
        lines.extend(preamble.splitlines())
    else:
        lines.extend(
            [
                "# Insights",
                "",
                "Source-of-truth policy:",
                "- Authoritative machine-readable registry: `registry/insights.toml`",
                "- Narrative overlay registry: `registry/insights_narrative.toml`",
                "- TOML-driven markdown mirror: `docs/generated/INSIGHTS_REGISTRY_MIRROR.md`",
                "- This file is generated from TOML sources.",
                "",
            ]
        )
    lines.append("")

    for row in insights:
        insight_id = str(row.get("id", "I-???"))
        title = str(row.get("title", "(untitled)"))
        lines.append(f"## {insight_id}: {title}")
        lines.append("")
        body = body_by_id.get(insight_id, "").strip()
        if body:
            lines.extend(body.splitlines())
        else:
            claims = ", ".join(row.get("claims", [])) if row.get("claims") else "(none)"
            lines.append(f"Date: {row.get('date', '')}")
            lines.append(f"Status: {row.get('status', '')}")
            lines.append(f"Claims: {claims}")
            lines.append("")
            lines.append(str(row.get("summary", "")).strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    _write(out_path, "\n".join(lines) + "\n")


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


def export_experiments_legacy(repo_root: Path, out_path: Path) -> None:
    registry_data = _load_toml(repo_root / "registry/experiments.toml")
    narrative_data = _load_optional_toml(repo_root / "registry/experiments_narrative.toml")
    preamble, body_by_id = _narrative_map(narrative_data)
    experiments = sorted(registry_data.get("experiment", []), key=lambda x: x.get("id", ""))

    lines: list[str] = []
    if preamble:
        lines.extend(preamble.splitlines())
    else:
        lines.extend(
            [
                "# Experiments Portfolio Shortlist",
                "",
                "Source-of-truth policy:",
                "- Authoritative machine-readable registry: `registry/experiments.toml`",
                "- Narrative overlay registry: `registry/experiments_narrative.toml`",
                "- TOML-driven markdown mirror: `docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md`",
                "- This file is generated from TOML sources.",
                "",
            ]
        )
    lines.append("")

    for row in experiments:
        experiment_id = str(row.get("id", "E-???"))
        title = str(row.get("title", "(untitled)"))
        lines.append(f"## {experiment_id}: {title}")
        lines.append("")
        body = body_by_id.get(experiment_id, "").strip()
        if body:
            lines.extend(body.splitlines())
        else:
            lines.append(f"Method: {row.get('method', '')}")
            lines.append(f"Input: {row.get('input', '')}")
            outputs = ", ".join(row.get("output", [])) if row.get("output") else "(none)"
            lines.append(f"Output: {outputs}")
            lines.append("Run:")
            lines.append("```bash")
            lines.append(str(row.get("run", "")).strip())
            lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    while lines and not lines[-1].strip():
        lines.pop()
    _write(out_path, "\n".join(lines) + "\n")


def export_roadmap(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/roadmap.toml")
    roadmap = data.get("roadmap", {})
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
    _write(out_path, "\n".join(lines))


def export_roadmap_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/roadmap_narrative.toml")
    body = _single_overlay_body(narrative, "roadmap_narrative")
    fallback = [
        (
            "This file is generated from `registry/roadmap.toml` "
            "and `registry/roadmap_narrative.toml`."
        ),
        "",
        "See the structured mirror at `docs/generated/ROADMAP_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/roadmap.toml", "registry/roadmap_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "ROADMAP", fallback))
    _write(out_path, "\n".join(lines) + "\n")


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


def export_todo_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/todo_narrative.toml")
    body = _single_overlay_body(narrative, "todo_narrative")
    fallback = [
        "This file is generated from `registry/todo.toml` and `registry/todo_narrative.toml`.",
        "",
        "See the structured mirror at `docs/generated/TODO_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/todo.toml", "registry/todo_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "TODO", fallback))
    _write(out_path, "\n".join(lines) + "\n")


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


def export_next_actions_legacy(repo_root: Path, out_path: Path) -> None:
    narrative = _load_optional_toml(repo_root / "registry/next_actions_narrative.toml")
    body = _single_overlay_body(narrative, "next_actions_narrative")
    fallback = [
        (
            "This file is generated from `registry/next_actions.toml` "
            "and `registry/next_actions_narrative.toml`."
        ),
        "",
        "See the structured mirror at `docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md`.",
        "",
    ]
    lines = _legacy_header(["registry/next_actions.toml", "registry/next_actions_narrative.toml"])
    lines.extend(_legacy_lines_from_body(body, "NEXT ACTIONS", fallback))
    _write(out_path, "\n".join(lines) + "\n")


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


def export_requirements_legacy(repo_root: Path) -> None:
    req_data = _load_toml(repo_root / "registry/requirements.toml")
    narrative = _load_optional_toml(repo_root / "registry/requirements_narrative.toml")
    narrative_rows = narrative.get("document", [])
    body_by_path = {
        str(row.get("path", "")).strip(): str(row.get("body_markdown", "")).strip()
        for row in narrative_rows
        if str(row.get("path", "")).strip()
    }
    title_by_path = {
        str(row.get("path", "")).strip(): str(row.get("title", "")).strip()
        for row in narrative_rows
        if str(row.get("path", "")).strip()
    }

    req_meta = req_data.get("requirements", {})
    primary_markdown = str(req_meta.get("primary_markdown", "docs/REQUIREMENTS.md"))
    target_paths = {"REQUIREMENTS.md", primary_markdown}
    for module in req_data.get("module", []):
        markdown_path = str(module.get("markdown", "")).strip()
        if markdown_path:
            target_paths.add(markdown_path)

    module_by_markdown = {
        str(module.get("markdown", "")).strip(): module for module in req_data.get("module", [])
    }

    sources = ["registry/requirements.toml", "registry/requirements_narrative.toml"]
    for rel_path in sorted(target_paths):
        body = body_by_path.get(rel_path, "")
        title = title_by_path.get(rel_path, Path(rel_path).stem.replace("-", " ").title())
        module = module_by_markdown.get(rel_path, {})
        fallback: list[str] = [
            f"This file is generated from `{sources[0]}` and `{sources[1]}`.",
            "",
            "See the structured mirror at `docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md`.",
            "",
        ]
        if module:
            fallback.append(f"- Module ID: `{module.get('id', '')}`")
            fallback.append(f"- Module name: `{module.get('name', '')}`")
            fallback.append(f"- Status: `{module.get('status', '')}`")
            targets = module.get("install_targets", [])
            if targets:
                fallback.append("- Install targets:")
                for target in targets:
                    fallback.append(f"  - `{target}`")
                fallback.append("")

        lines = _legacy_header(sources)
        lines.extend(_legacy_lines_from_body(body, title, fallback))
        _write(repo_root / rel_path, "\n".join(lines) + "\n")


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
        section_id = section.get("id", "")
        section_name = section.get("name", "")
        section_count = section.get("task_count", 0)
        lines.append(f"- {section_id}: {section_name} ({section_count} tasks)")
    lines.append("")
    lines.append("## Tasks")
    lines.append("")
    for task in sorted(tasks, key=lambda row: int(row.get("order_index", 0))):
        task_id = task.get("id", "CTASK-???")
        claim_id = task.get("claim_id", "C-???")
        status_token = task.get("status_token", "UNKNOWN")
        lines.append(f"### {task_id} ({claim_id}, {status_token})")
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


def export_claims_tasks_legacy(repo_root: Path, out_path: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_tasks.toml")
    sections = data.get("section", [])
    tasks = data.get("task", [])

    by_section: dict[str, list[dict]] = {}
    for task in tasks:
        by_section.setdefault(str(task.get("section", "unscoped")), []).append(task)

    lines = [
        "# Claims -> Tasks Tracker (Generated Mirror)",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claims_tasks.toml -->",
        "",
        "This file is generated from `registry/claims_tasks.toml`.",
        "",
    ]

    for section in sorted(
        sections, key=lambda row: int(row.get("id", "CTS-999").split("-")[1])
    ):
        name = str(section.get("name", "unscoped"))
        section_tasks = sorted(
            by_section.get(name, []), key=lambda row: int(row.get("order_index", 0))
        )
        lines.append(f"## {name}")
        lines.append("")
        if not section_tasks:
            lines.append("No task rows currently in this section.")
            lines.append("")
            continue
        lines.append("| Claim ID | Task | Output artifact(s) | Status |")
        lines.append("|---|---|---|---|")
        for task in section_tasks:
            claim_id = str(task.get("claim_id", ""))
            task_text = _pipe_escape(str(task.get("task", "")).strip())
            artifacts = task.get("output_artifacts", [])
            artifact_cell = ", ".join(f"`{item}`" for item in artifacts) if artifacts else "(none)"
            status = str(task.get("status_token", ""))
            lines.append(f"| {claim_id} | {task_text} | {artifact_cell} | {status} |")
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
        claim_id = row.get("claim_id", "")
        domains_csv = row.get("domains_csv", [])
        domains_markdown = row.get("domains_markdown", [])
        domain_sets_match = row.get("domain_sets_match", False)
        lines.append(
            f"- {claim_id}: csv={domains_csv}, markdown={domains_markdown}, "
            f"match={domain_sets_match}"
        )
    lines.append("")
    _write(out_path, "\n".join(lines))


def export_claims_domains_legacy(repo_root: Path) -> None:
    data = _load_toml(repo_root / "registry/claims_domains.toml")
    domains = sorted(data.get("domain", []), key=lambda row: row.get("id", ""))
    claim_domains = sorted(
        data.get("claim_domain", []), key=lambda row: row.get("claim_id", "")
    )
    entries = data.get("domain_entry", [])

    index_lines = [
        "# Claims by domain",
        "",
        "<!-- AUTO-GENERATED: DO NOT EDIT -->",
        "<!-- Source of truth: registry/claims_domains.toml -->",
        "",
        "See also: docs/CLAIMS_DOMAIN_TAXONOMY.md",
        "",
    ]
    for row in domains:
        domain_id = str(row.get("id", ""))
        source_markdown = str(
            row.get("source_markdown", f"docs/claims/by_domain/{domain_id}.md")
        )
        count = int(row.get("markdown_claim_count", 0))
        index_lines.append(f"- `{domain_id}` ({count}): `{source_markdown}`")
    index_lines.append("")
    _write(repo_root / "docs/claims/INDEX.md", "\n".join(index_lines))

    csv_rows = [["claim_id", "domains"]]
    for row in claim_domains:
        domains_csv = row.get("domains_csv", [])
        csv_rows.append(
            [str(row.get("claim_id", "")), ";".join(str(item) for item in domains_csv)]
        )
    _write(repo_root / "docs/claims/CLAIMS_DOMAIN_MAP.csv", _render_csv(csv_rows))

    entries_by_domain: dict[str, list[dict]] = {}
    for entry in entries:
        entries_by_domain.setdefault(str(entry.get("domain", "")), []).append(entry)

    for row in domains:
        domain_id = str(row.get("id", ""))
        path = Path(
            str(row.get("source_markdown", f"docs/claims/by_domain/{domain_id}.md"))
        )
        domain_entries = sorted(
            entries_by_domain.get(domain_id, []),
            key=lambda item: _claim_sort_key(str(item.get("claim_id", ""))),
        )
        lines = [
            f"# Claims: {domain_id}",
            "",
            "<!-- AUTO-GENERATED: DO NOT EDIT -->",
            "<!-- Source of truth: registry/claims_domains.toml -->",
            "",
            f"Count: {len(domain_entries)}",
            "",
        ]
        for entry in domain_entries:
            claim_id = str(entry.get("claim_id", ""))
            status_text = str(entry.get("status_text", ""))
            status_date = str(entry.get("status_date", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            where_stated = [str(item) for item in entry.get("where_stated", [])]
            status_blob = f"{status_text}, {status_date}" if status_date else status_text
            lines.append(f"- Hypothesis {claim_id} ({status_blob}): {summary}")
            if where_stated:
                where_joined = ", ".join(f"`{item}`" for item in where_stated)
                lines.append(f"  - Where stated: {where_joined}")
            lines.append("")

        _write(repo_root / path, "\n".join(lines))


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
        done = row.get("done_checkboxes", 0)
        open_checkboxes = row.get("open_checkboxes", 0)
        lines.append(f"- Checkbox progress: done={done}, open={open_checkboxes}")
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
    global CHECK_MODE

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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if any mirror would change.",
    )
    parser.add_argument(
        "--legacy-claims-sync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also regenerate legacy claims-support markdown/csv mirrors from TOML.",
    )
    args = parser.parse_args()

    CHECK_MODE = bool(args.check)

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    if not CHECK_MODE:
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
    export_insights_legacy(repo_root, repo_root / "docs/INSIGHTS.md")
    export_experiments_legacy(
        repo_root, repo_root / "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md"
    )
    export_roadmap_legacy(repo_root, repo_root / "docs/ROADMAP.md")
    export_todo_legacy(repo_root, repo_root / "docs/TODO.md")
    export_next_actions_legacy(repo_root, repo_root / "docs/NEXT_ACTIONS.md")
    export_requirements_legacy(repo_root)

    if args.legacy_claims_sync:
        export_claims_tasks_legacy(repo_root, repo_root / "docs/CLAIMS_TASKS.md")
        export_claims_domains_legacy(repo_root)

    if CHECK_MODE:
        if CHANGED_PATHS:
            print("ERROR: TOML-driven mirrors are stale. Regenerate with make registry.")
            for path in sorted(set(CHANGED_PATHS)):
                print(path)
            return 1
        print("OK: TOML-driven mirrors are fresh.")
        return 0

    print(f"Wrote TOML-driven markdown mirrors to {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
