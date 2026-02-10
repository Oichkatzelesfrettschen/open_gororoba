#!/usr/bin/env python3
"""
Build a full markdown inventory registry (tracked, untracked, ignored, archived).

This inventory is intentionally broader than registry/knowledge_sources.toml:
- includes ignored and untracked markdown
- classifies generated vs non-generated
- recommends aggressive TOML-first migration actions without data loss
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import re
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", flags=re.M)
CLAIM_RE = re.compile(r"\bC-\d{3}\b")
INSIGHT_RE = re.compile(r"\bI-\d{3}\b")
EXPERIMENT_RE = re.compile(r"\bE-\d{3}\b")

GENERATED_MARKERS = (
    "AUTO-GENERATED",
    "Source of truth:",
    "This file is generated from",
    "DO NOT EDIT",
)

MANUAL_EXCEPTIONS: set[str] = set()

THIRD_PARTY_PATTERNS = (
    ".pytest_cache/README.md",
    "*/site-packages/*/LICENSE.md",
    "*/site-packages/*/licenses/*.md",
)

GENERATED_PATTERNS = (
    "docs/generated/*.md",
    "docs/CLAIMS_EVIDENCE_MATRIX.md",
    "docs/BIBLIOGRAPHY.md",
    "docs/INSIGHTS.md",
    "docs/EXPERIMENTS_PORTFOLIO_SHORTLIST.md",
    "docs/ROADMAP.md",
    "docs/TODO.md",
    "docs/NEXT_ACTIONS.md",
    "docs/CLAIMS_TASKS.md",
    "docs/claims/INDEX.md",
    "docs/claims/by_domain/*.md",
    "docs/tickets/*.md",
    "docs/tickets/INDEX.md",
    "docs/book/src/*.md",
    "docs/book/src/*/*.md",
    "docs/book/src/*/*/*.md",
    "docs/external_sources/*.md",
    "docs/external_sources/INDEX.md",
    "docs/theory/*.md",
    "docs/theory/INDEX.md",
    "docs/engineering/*.md",
    "docs/engineering/INDEX.md",
    "docs/research/*.md",
    "docs/monograph/*.md",
    "docs/convos/*.md",
    "data/artifacts/ALGEBRAIC_FOUNDATIONS.md",
    "data/artifacts/BIBLIOGRAPHY.md",
    "data/artifacts/FINAL_REPORT.md",
    "data/artifacts/QUANTUM_REPORT.md",
    "data/artifacts/SIMULATION_REPORT.md",
    "data/artifacts/extracted_equations.md",
    "data/artifacts/reality_check_and_synthesis.md",
    "reports/*.md",
    "REQUIREMENTS.md",
    "docs/REQUIREMENTS.md",
    "docs/requirements/*.md",
    "NAVIGATOR.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "curated/README.md",
    "curated/01_theory_frameworks/README_COQ.md",
    "data/csv/README.md",
    "data/artifacts/README.md",
)

DESTINATION_OVERRIDES = {
    "docs/generated/BIBLIOGRAPHY_REGISTRY_MIRROR.md": "registry/bibliography.toml",
    "docs/generated/BOOK_DOCS_REGISTRY_MIRROR.md": "registry/book_docs.toml",
    "docs/generated/CLAIMS_DOMAINS_REGISTRY_MIRROR.md": "registry/claims_domains.toml",
    "docs/generated/CLAIMS_REGISTRY_MIRROR.md": "registry/claims.toml",
    "docs/generated/CLAIMS_TASKS_REGISTRY_MIRROR.md": "registry/claims_tasks.toml",
    "docs/generated/CLAIM_TICKETS_REGISTRY_MIRROR.md": "registry/claim_tickets.toml",
    "docs/generated/DATA_ARTIFACT_NARRATIVES_REGISTRY_MIRROR.md": "registry/data_artifact_narratives.toml",
    "docs/generated/DOCS_CONVOS_REGISTRY_MIRROR.md": "registry/docs_convos.toml",
    "docs/generated/DOCS_ROOT_NARRATIVES_REGISTRY_MIRROR.md": "registry/docs_root_narratives.toml",
    "docs/generated/ENTRYPOINT_DOCS_REGISTRY_MIRROR.md": "registry/entrypoint_docs.toml",
    "docs/generated/EXPERIMENTS_REGISTRY_MIRROR.md": "registry/experiments.toml",
    "docs/generated/EXTERNAL_SOURCES_REGISTRY_MIRROR.md": "registry/external_sources.toml",
    "docs/generated/INSIGHTS_REGISTRY_MIRROR.md": "registry/insights.toml",
    "docs/generated/KNOWLEDGE_MIGRATION_PLAN_REGISTRY_MIRROR.md": "registry/knowledge_migration_plan.toml",
    "docs/generated/MARKDOWN_GOVERNANCE_REGISTRY_MIRROR.md": "registry/markdown_governance.toml",
    "docs/generated/NAVIGATOR_REGISTRY_MIRROR.md": "registry/navigator.toml",
    "docs/generated/NEXT_ACTIONS_REGISTRY_MIRROR.md": "registry/next_actions.toml",
    "docs/generated/REPORTS_NARRATIVES_REGISTRY_MIRROR.md": "registry/reports_narratives.toml",
    "docs/generated/REQUIREMENTS_REGISTRY_MIRROR.md": "registry/requirements.toml",
    "docs/generated/RESEARCH_NARRATIVES_REGISTRY_MIRROR.md": "registry/research_narratives.toml",
    "docs/generated/ROADMAP_REGISTRY_MIRROR.md": "registry/roadmap.toml",
    "docs/generated/TODO_REGISTRY_MIRROR.md": "registry/todo.toml",
    "docs/claims/INDEX.md": "registry/claims_domains.toml",
}


@dataclass(frozen=True)
class Doc:
    path: str
    git_status: str
    title: str
    size_bytes: int
    line_count: int
    sha256: str
    claim_refs: int
    insight_refs: int
    experiment_refs: int
    archived: bool
    generated_declared: bool
    generated_pattern: bool
    generated: bool
    manual_exception: bool
    third_party: bool
    toml_destination: str
    classification: str
    migration_action: str
    migration_priority: str
    rationale: str


def _ascii_safe(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ord(ch) <= 127:
            out.append(ch)
        else:
            out.append(f"<U+{ord(ch):04X}>")
    return "".join(out)


def _esc(text: str) -> str:
    s = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{_ascii_safe(s)}"'


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {''.join(bad[:20])!r}")


def _git_paths(root: Path, args: list[str]) -> set[str]:
    out = subprocess.check_output(["git", *args], cwd=root, text=True)
    return {line.strip() for line in out.splitlines() if line.strip()}


def _all_filesystem_markdown(root: Path) -> set[str]:
    out: set[str] = set()
    for path in root.rglob("*.md"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/"):
            continue
        out.add(rel)
    return out


def _is_archived(path: str) -> bool:
    return path.startswith("archive/") or path.startswith("docs/archive/")


def _is_generated_pattern(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in GENERATED_PATTERNS)


def _is_third_party(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in THIRD_PARTY_PATTERNS)


def _first_title(text: str, fallback: str) -> str:
    m = HEADING_RE.search(text)
    if m:
        return m.group(1).strip()
    return fallback


def _declared_generated(text: str) -> bool:
    lines = text.splitlines()[:80]
    for line in lines:
        if any(marker in line for marker in GENERATED_MARKERS):
            return True
    return False


def _iter_registry_refs(root: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    non_destination_registries = {
        "registry/knowledge_migration_plan.toml",
        "registry/markdown_inventory.toml",
        "registry/markdown_corpus_registry.toml",
        "registry/toml_inventory.toml",
        "registry/wave4_roadmap.toml",
        "registry/markdown_origin_audit.toml",
        "registry/markdown_owner_map.toml",
    }

    def add(path: str, src: str) -> None:
        path = path.strip()
        if not path.endswith(".md"):
            return
        refs.setdefault(path, set()).add(src)

    def walk(obj: object, src: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lk = key.lower()
                if lk in {"source_markdown", "markdown", "output_markdown", "path", "primary_markdown"}:
                    if isinstance(value, str):
                        add(value, src)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                add(item, src)
                elif lk in {"generated_mirror"}:
                    if isinstance(value, str):
                        add(value, src)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                add(item, src)
                elif lk in {"source_markdown_glob", "source_markdown_globs"}:
                    if isinstance(value, str):
                        for path in root.glob(value):
                            if path.is_file():
                                add(path.relative_to(root).as_posix(), src)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str):
                                for path in root.glob(item):
                                    if path.is_file():
                                        add(path.relative_to(root).as_posix(), src)
                else:
                    walk(value, src)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, src)

    for reg in sorted((root / "registry").glob("*.toml")):
        reg_rel = reg.relative_to(root).as_posix()
        if reg_rel in non_destination_registries:
            continue
        data = tomllib.loads(reg.read_text(encoding="utf-8"))
        walk(data, reg_rel)
    return refs


def _priority_rank(priority: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}.get(priority, 5)


def _choose_destination(candidates: set[str]) -> str:
    if not candidates:
        return ""
    sorted_candidates = sorted(candidates)
    preferred = [
        item for item in sorted_candidates if item != "registry/knowledge_migration_plan.toml"
    ]
    if preferred:
        canonical = [item for item in preferred if not item.endswith("_narrative.toml")]
        if canonical:
            return canonical[0]
        return preferred[0]
    return sorted_candidates[0]


def _classify(path: str, text: str, toml_destination: str) -> tuple[str, str, str, str]:
    third_party = _is_third_party(path)
    manual_exception = path in MANUAL_EXCEPTIONS
    generated_declared = _declared_generated(text)
    generated_pattern = _is_generated_pattern(path)
    generated = generated_declared or generated_pattern
    high_information = (
        path.startswith("docs/")
        or path.startswith("data/artifacts/")
        or path.startswith("reports/")
        or path.startswith("NAVIGATOR.md")
        or path.startswith("REQUIREMENTS.md")
    )

    if third_party:
        return (
            "third_party_markdown",
            "ignore_vendor",
            "none",
            "Third-party or tool cache markdown; not a project knowledge source.",
        )
    if manual_exception:
        return (
            "manual_exception",
            "keep_manual_exception",
            "none",
            "Explicitly retained manual entrypoint/readme exception.",
        )
    if generated and toml_destination:
        return (
            "toml_published_markdown",
            "keep_generated_mirror",
            "low",
            "Published markdown generated from TOML destination.",
        )
    if generated and not toml_destination:
        return (
            "generated_artifact",
            "keep_generated_artifact",
            "low",
            "Generated artifact markdown with no migration requirement.",
        )
    if toml_destination:
        return (
            "toml_destination_exists_manual_markdown",
            "port_body_to_toml_and_lock_mirror",
            "high" if high_information else "medium",
            "Markdown has TOML destination but still carries manual content; lock to TOML flow.",
        )
    return (
        "unbacked_manual_markdown",
        "migrate_to_new_registry",
        "critical" if high_information else "high",
        "Manual markdown without TOML destination; migrate aggressively.",
    )


def _build_doc(
    root: Path, path: str, git_status: str, refs: dict[str, set[str]]
) -> Doc:
    full = root / path
    text = full.read_text(encoding="utf-8", errors="ignore")
    title = _first_title(text, Path(path).stem)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    lines = text.count("\n") + (1 if text else 0)
    toml_destination = DESTINATION_OVERRIDES.get(path, _choose_destination(refs.get(path, set())))
    classification, action, priority, rationale = _classify(path, text, toml_destination)
    return Doc(
        path=path,
        git_status=git_status,
        title=_ascii_safe(title),
        size_bytes=full.stat().st_size,
        line_count=lines,
        sha256=sha,
        claim_refs=len(set(CLAIM_RE.findall(text))),
        insight_refs=len(set(INSIGHT_RE.findall(text))),
        experiment_refs=len(set(EXPERIMENT_RE.findall(text))),
        archived=_is_archived(path),
        generated_declared=_declared_generated(text),
        generated_pattern=_is_generated_pattern(path),
        generated=_declared_generated(text) or _is_generated_pattern(path),
        manual_exception=path in MANUAL_EXCEPTIONS,
        third_party=_is_third_party(path),
        toml_destination=toml_destination,
        classification=classification,
        migration_action=action,
        migration_priority=priority,
        rationale=_ascii_safe(rationale),
    )


def _render(docs: list[Doc]) -> str:
    tracked = sum(1 for d in docs if d.git_status == "tracked")
    untracked = sum(1 for d in docs if d.git_status == "untracked")
    ignored = sum(1 for d in docs if d.git_status == "ignored")
    fs_only = sum(1 for d in docs if d.git_status == "filesystem_only")
    generated = sum(1 for d in docs if d.generated)
    non_generated = len(docs) - generated
    archived = sum(1 for d in docs if d.archived)
    third_party = sum(1 for d in docs if d.third_party)
    manual_ex = sum(1 for d in docs if d.manual_exception)
    unbacked = sum(1 for d in docs if d.classification == "unbacked_manual_markdown")
    toml_backed_manual = sum(1 for d in docs if d.classification == "toml_backed_manual_markdown")

    lines: list[str] = []
    lines.append("# Full markdown inventory registry (TOML-first governance support).")
    lines.append("# Generated by src/scripts/analysis/build_markdown_inventory_registry.py")
    lines.append("")
    lines.append("[markdown_inventory]")
    lines.append('generated_at = "deterministic"')
    lines.append("authoritative = true")
    lines.append(f"document_count = {len(docs)}")
    lines.append(f"tracked_count = {tracked}")
    lines.append(f"untracked_count = {untracked}")
    lines.append(f"ignored_count = {ignored}")
    lines.append(f"filesystem_only_count = {fs_only}")
    lines.append(f"generated_count = {generated}")
    lines.append(f"non_generated_count = {non_generated}")
    lines.append(f"archived_count = {archived}")
    lines.append(f"third_party_count = {third_party}")
    lines.append(f"manual_exception_count = {manual_ex}")
    lines.append(f"unbacked_manual_count = {unbacked}")
    lines.append(f"toml_backed_manual_count = {toml_backed_manual}")
    lines.append("")

    queue = sorted(
        [
            d
            for d in docs
            if d.migration_action
            in {"migrate_to_new_registry", "port_body_to_toml_and_lock_mirror"}
        ],
        key=lambda d: (_priority_rank(d.migration_priority), -d.line_count, d.path),
    )

    for i, d in enumerate(queue[:60], start=1):
        lines.append("[[migration_queue]]")
        lines.append(f"rank = {i}")
        lines.append(f"path = {_esc(d.path)}")
        lines.append(f"migration_priority = {_esc(d.migration_priority)}")
        lines.append(f"migration_action = {_esc(d.migration_action)}")
        if d.toml_destination:
            lines.append(f"toml_destination = {_esc(d.toml_destination)}")
        lines.append(f"line_count = {d.line_count}")
        lines.append(f"rationale = {_esc(d.rationale)}")
        lines.append("")

    for i, d in enumerate(
        sorted(docs, key=lambda x: (_priority_rank(x.migration_priority), x.path)), start=1
    ):
        lines.append("[[document]]")
        lines.append(f"id = {_esc(f'MDI-{i:04d}')}")
        lines.append(f"path = {_esc(d.path)}")
        lines.append(f"title = {_esc(d.title)}")
        lines.append(f"git_status = {_esc(d.git_status)}")
        lines.append(f"archived = {'true' if d.archived else 'false'}")
        lines.append(f"generated_declared = {'true' if d.generated_declared else 'false'}")
        lines.append(f"generated_pattern = {'true' if d.generated_pattern else 'false'}")
        lines.append(f"generated = {'true' if d.generated else 'false'}")
        lines.append(f"manual_exception = {'true' if d.manual_exception else 'false'}")
        lines.append(f"third_party = {'true' if d.third_party else 'false'}")
        lines.append(f"classification = {_esc(d.classification)}")
        lines.append(f"migration_action = {_esc(d.migration_action)}")
        lines.append(f"migration_priority = {_esc(d.migration_priority)}")
        if d.toml_destination:
            lines.append(f"toml_destination = {_esc(d.toml_destination)}")
        lines.append(f"rationale = {_esc(d.rationale)}")
        lines.append(f"size_bytes = {d.size_bytes}")
        lines.append(f"line_count = {d.line_count}")
        lines.append(f"sha256 = {_esc(d.sha256)}")
        lines.append(f"claim_ref_count = {d.claim_refs}")
        lines.append(f"insight_ref_count = {d.insight_refs}")
        lines.append(f"experiment_ref_count = {d.experiment_refs}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--out",
        default="registry/markdown_inventory.toml",
        help="Output TOML path.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    tracked = _git_paths(root, ["ls-files", "*.md"])
    untracked = _git_paths(root, ["ls-files", "--others", "--exclude-standard", "*.md"])
    ignored = _git_paths(
        root, ["ls-files", "--others", "--ignored", "--exclude-standard", "*.md"]
    )
    fs_all = _all_filesystem_markdown(root)

    all_paths = sorted(tracked | untracked | ignored | fs_all)
    refs = _iter_registry_refs(root)

    docs: list[Doc] = []
    for path in all_paths:
        if path in tracked:
            git_status = "tracked"
        elif path in untracked:
            git_status = "untracked"
        elif path in ignored:
            git_status = "ignored"
        else:
            git_status = "filesystem_only"
        docs.append(_build_doc(root, path, git_status, refs))

    text = _render(docs)
    out_path = root / args.out
    _assert_ascii(text, str(out_path))
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} with {len(docs)} markdown records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
