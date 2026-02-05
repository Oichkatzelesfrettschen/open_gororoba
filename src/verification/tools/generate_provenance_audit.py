#!/usr/bin/env python3
"""
Generate a provenance audit report.

Focus:
- data/external provenance coverage (PROVENANCE.local.json)
- README-advertised artifact hashes vs actual files
- README artifact presence in ARTIFACTS_MANIFEST.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadmeArtifact:
    file_path: str
    sha256: str


def _find_repo_root() -> Path:
    p = Path.cwd().resolve()
    while p != p.parent:
        if (p / ".git").exists():
            return p
        p = p.parent
    return Path(__file__).resolve().parents[3]


def _sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_external_provenance(prov_path: Path) -> set[str]:
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    files: set[str] = set()
    if isinstance(prov, dict) and "hashes" in prov and "root" in prov:
        root = str(prov["root"])
        hashes = prov["hashes"]
        if isinstance(hashes, list):
            for item in hashes:
                if not isinstance(item, dict):
                    continue
                rel = item.get("path")
                if isinstance(rel, str) and rel:
                    files.add(f"{root.rstrip('/')}/{rel.lstrip('/')}")
        return files

    mapping = prov.get("files", prov) if isinstance(prov, dict) else {}
    if isinstance(mapping, dict):
        files = set(mapping.keys())
    return files


def _scan_external_coverage(repo_root: Path) -> tuple[list[str], list[str]]:
    external_dir = repo_root / "data" / "external"
    prov_path = external_dir / "PROVENANCE.local.json"
    if not external_dir.exists() or not prov_path.exists():
        return [str(external_dir)], [str(prov_path)]

    listed = _load_external_provenance(prov_path)
    missing: list[str] = []
    empty: list[str] = []
    for p in external_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(repo_root).as_posix()
        # Ignore json metadata/provenance files; they are covered separately.
        if rel.endswith(".json"):
            continue
        if p.stat().st_size == 0:
            empty.append(rel)
        if rel not in listed:
            missing.append(rel)
    return missing, empty


def _parse_readme_artifacts(repo_root: Path) -> list[ReadmeArtifact]:
    readme = repo_root / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8", errors="replace")

    # Minimal parse for the documented pattern:
    # - File: `path`
    # - SHA256: `hex`
    file_re = re.compile(r"^- File:\\s+`([^`]+)`", flags=re.MULTILINE)
    sha_re = re.compile(r"^- SHA256:\\s+`([0-9a-f]{64})`", flags=re.MULTILINE)

    files = file_re.findall(text)
    shas = sha_re.findall(text)
    out: list[ReadmeArtifact] = []
    for fp, sha in zip(files, shas, strict=False):
        out.append(ReadmeArtifact(file_path=fp, sha256=sha))
    return out


def _load_artifacts_manifest(repo_root: Path) -> set[str]:
    manifest_path = repo_root / "data" / "artifacts" / "ARTIFACTS_MANIFEST.csv"
    if not manifest_path.exists():
        return set()
    out: set[str] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = (row.get("artifact_path") or "").strip()
            if p:
                out.add(p)
    return out


def _render(
    repo_root: Path,
    missing_external: list[str],
    empty_external: list[str],
    readme_artifacts: list[ReadmeArtifact],
    manifest_paths: set[str],
) -> str:
    today = _dt.date.today().isoformat()
    lines: list[str] = []
    lines.append(f"# Provenance Audit ({today})")
    lines.append("")
    lines.append("Goal: keep external caches and advertised artifacts auditable and reproducible.")
    lines.append("")

    lines.append("## data/external coverage")
    lines.append("")
    if missing_external or empty_external:
        if missing_external:
            lines.append(
                f"- Missing from `data/external/PROVENANCE.local.json`: {len(missing_external)}"
            )
        if empty_external:
            lines.append(f"- Empty external files (0 bytes): {len(empty_external)}")
        lines.append("")
        for rel in sorted(missing_external)[:200]:
            lines.append(f"- MISSING: `{rel}`")
        for rel in sorted(empty_external)[:200]:
            lines.append(f"- EMPTY: `{rel}`")
        lines.append("")
    else:
        lines.append("- OK: external files appear covered and non-empty.")
        lines.append("")

    lines.append("## README advertised artifacts")
    lines.append("")
    if not readme_artifacts:
        lines.append("(No README artifact entries found.)")
        lines.append("")
        return "\n".join(lines)

    lines.append("| file | sha256 (README) | sha256 (actual) | in manifest |")
    lines.append("| --- | --- | --- | --- |")
    for ra in readme_artifacts:
        fp = repo_root / ra.file_path
        actual = "missing"
        if fp.exists() and fp.is_file():
            actual = _sha256_file(fp)
        in_manifest = "yes" if ra.file_path in manifest_paths else "no"
        lines.append(
            f"| `{ra.file_path}` | `{ra.sha256}` | `{actual}` | {in_manifest} |"
        )
    lines.append("")

    lines.append("## Checklist")
    lines.append("")
    lines.append("- [ ] Every file under `data/external/` is covered by")
    lines.append("  `PROVENANCE.local.json` (or a subdir PROVENANCE.json) and is non-empty.")
    lines.append("- [ ] Every README-advertised artifact exists and has the documented SHA256.")
    lines.append("- [ ] Every README-advertised artifact is listed in")
    lines.append("  `data/artifacts/ARTIFACTS_MANIFEST.csv`.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/audit_provenance.md",
        help="Output markdown path.",
    )
    args = parser.parse_args()

    repo_root = _find_repo_root()
    missing_external, empty_external = _scan_external_coverage(repo_root)
    readme_artifacts = _parse_readme_artifacts(repo_root)
    manifest_paths = _load_artifacts_manifest(repo_root)

    out_path = (repo_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _render(
            repo_root=repo_root,
            missing_external=missing_external,
            empty_external=empty_external,
            readme_artifacts=readme_artifacts,
            manifest_paths=manifest_paths,
        ),
        encoding="utf-8",
    )
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
