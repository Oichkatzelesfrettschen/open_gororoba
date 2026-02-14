#!/usr/bin/env python3
"""
Fetch resolved sensational primary-source URLs and record deterministic traces.

Reads:
- reports/sensational_primary_source_mapping_2026_02_14.toml

Writes:
- data/papers/sensational_primary/2026_02_14/*
- data/papers/sensational_primary/2026_02_14/manifest.tsv
- data/papers/sensational_primary/2026_02_14/provenance.toml
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import tomllib
from urllib.parse import urlparse


USER_AGENT = (
    "open_gororoba-primary-fetch/2026-02-14 "
    "(research provenance capture; +https://github.com/eirikr/open_gororoba)"
)
CHALLENGE_SNIPPETS = (
    "Just a moment...",
    "Enable JavaScript and cookies to continue",
    "__cf_chl_opt",
)


@dataclass(frozen=True)
class Target:
    item_id: str
    headline: str
    role: str
    url: str


@dataclass
class Result:
    item_id: str
    headline: str
    role: str
    url: str
    status: str
    http_status: str
    content_type: str
    output_path: str
    sha256: str
    size_bytes: int
    error: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _escape(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _infer_ext(url: str, content_type: str) -> str:
    lower = url.lower()
    if lower.endswith(".pdf") or "/pdf/" in lower:
        return ".pdf"
    if "application/pdf" in content_type.lower():
        return ".pdf"
    return ".html"


def _parse_content_type(header_path: Path) -> str:
    content_type = ""
    for line in header_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lower().startswith("content-type:"):
            content_type = line.split(":", 1)[1].strip()
    return content_type


def _detect_challenge(path: Path) -> bool:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:12000]
    return any(token in sample for token in CHALLENGE_SNIPPETS)


def _fetch(url: str, output_path: Path, timeout_s: int) -> tuple[str, str, str]:
    with tempfile.NamedTemporaryFile(delete=False, prefix="primary-src-header-", suffix=".txt") as tmp:
        header_file = Path(tmp.name)
    cmd = [
        "curl",
        "-L",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_s),
        "-A",
        USER_AGENT,
        "-D",
        str(header_file),
        "-o",
        str(output_path),
        "-w",
        "%{http_code}",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    http_status = proc.stdout.strip() or "000"
    content_type = _parse_content_type(header_file)
    header_file.unlink(missing_ok=True)
    err = proc.stderr.strip()
    return http_status, content_type, err


def _load_targets(mapping_path: Path) -> list[Target]:
    data = tomllib.loads(mapping_path.read_text(encoding="utf-8"))
    rows = data.get("item", [])
    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            continue
        if str(row.get("primary_source_status", "")).strip() != "resolved":
            continue
        headline = str(row.get("headline", "")).strip()
        primary_url = str(row.get("primary_source_url", "")).strip()
        candidates = [str(item).strip() for item in row.get("candidate_primary_urls", []) if str(item).strip()]
        ordered_urls = []
        if primary_url:
            ordered_urls.append(("primary", primary_url))
        for url in candidates:
            ordered_urls.append(("candidate", url))
        # Add arXiv PDF fetch when abs URL exists.
        for role, url in list(ordered_urls):
            parsed = urlparse(url)
            if parsed.netloc.endswith("arxiv.org") and parsed.path.startswith("/abs/"):
                arxiv_id = parsed.path.split("/abs/", 1)[1]
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                ordered_urls.append(("candidate_pdf", pdf_url))
        for role, url in ordered_urls:
            key = (item_id, url)
            if key in seen:
                continue
            seen.add(key)
            targets.append(Target(item_id=item_id, headline=headline, role=role, url=url))
    return targets


def _render_manifest(rows: list[Result]) -> str:
    header = (
        "item_id\theadline\trole\tstatus\thttp_status\tcontent_type\tsha256\tsize_bytes\toutput_path\turl\terror"
    )
    lines = [header]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.item_id,
                    row.headline,
                    row.role,
                    row.status,
                    row.http_status,
                    row.content_type,
                    row.sha256,
                    str(row.size_bytes),
                    row.output_path,
                    row.url,
                    row.error.replace("\t", " ").replace("\n", " "),
                ]
            )
        )
    rendered = "\n".join(lines) + "\n"
    _assert_ascii(rendered, "sensational manifest")
    return rendered


def _render_provenance(mapping_rel: str, root_rel: str, rows: list[Result]) -> str:
    ok_count = sum(1 for row in rows if row.status == "ok")
    blocked_count = sum(1 for row in rows if row.status.startswith("blocked"))
    fail_count = len(rows) - ok_count - blocked_count
    lines: list[str] = []
    lines.append("# Generated by src/scripts/analysis/fetch_sensational_primary_sources.py")
    lines.append("[batch]")
    lines.append('id = "SENSATIONAL-PRIMARY-FETCH-2026-02-14"')
    lines.append(f"generated_at_utc = {_escape(_utc_now())}")
    lines.append(f"mapping_path = {_escape(mapping_rel)}")
    lines.append(f"output_root = {_escape(root_rel)}")
    lines.append(f"target_count = {len(rows)}")
    lines.append(f"ok_count = {ok_count}")
    lines.append(f"blocked_count = {blocked_count}")
    lines.append(f"failed_count = {fail_count}")
    lines.append("")
    for row in rows:
        lines.append("[[artifact]]")
        lines.append(f"item_id = {_escape(row.item_id)}")
        lines.append(f"headline = {_escape(row.headline)}")
        lines.append(f"role = {_escape(row.role)}")
        lines.append(f"url = {_escape(row.url)}")
        lines.append(f"status = {_escape(row.status)}")
        lines.append(f"http_status = {_escape(row.http_status)}")
        lines.append(f"content_type = {_escape(row.content_type)}")
        lines.append(f"output_path = {_escape(row.output_path)}")
        lines.append(f"sha256 = {_escape(row.sha256)}")
        lines.append(f"size_bytes = {row.size_bytes}")
        lines.append(f"error = {_escape(row.error)}")
        lines.append("")
    rendered = "\n".join(lines)
    _assert_ascii(rendered, "sensational provenance")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--mapping",
        default="reports/sensational_primary_source_mapping_2026_02_14.toml",
        help="Mapping TOML path.",
    )
    parser.add_argument(
        "--output-root",
        default="data/papers/sensational_primary/2026_02_14",
        help="Output root path.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="Per-URL timeout in seconds.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    mapping_path = root / args.mapping
    if not mapping_path.exists():
        raise SystemExit(f"ERROR: missing mapping file: {mapping_path}")

    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(mapping_path)
    results: list[Result] = []

    for idx, target in enumerate(targets, start=1):
        base = _slug(f"{target.item_id}-{target.role}-{idx}")
        temp_file = output_root / f"{base}.tmp"
        http_status, content_type, fetch_err = _fetch(target.url, temp_file, args.timeout_seconds)
        status = "failed"
        out_rel = ""
        sha = ""
        size = 0
        error = fetch_err

        if temp_file.exists() and temp_file.stat().st_size > 0:
            ext = _infer_ext(target.url, content_type)
            final_path = output_root / f"{base}{ext}"
            temp_file.replace(final_path)
            out_rel = final_path.relative_to(root).as_posix()
            size = final_path.stat().st_size
            sha = _sha256(final_path)
            challenge_page = ext == ".html" and _detect_challenge(final_path)
            if http_status.startswith("2"):
                if challenge_page:
                    status = "blocked_bot_challenge"
                    if not error:
                        error = "challenge_page_detected"
                else:
                    status = "ok"
                    error = ""
            else:
                if challenge_page:
                    status = "blocked_bot_challenge"
                    if not error:
                        error = "challenge_page_detected"
                else:
                    status = "failed_http"
                if not error:
                    error = f"http_status_{http_status}"
        else:
            temp_file.unlink(missing_ok=True)
            if not error:
                error = f"http_status_{http_status}"

        results.append(
            Result(
                item_id=target.item_id,
                headline=target.headline,
                role=target.role,
                url=target.url,
                status=status,
                http_status=http_status,
                content_type=content_type,
                output_path=out_rel,
                sha256=sha,
                size_bytes=size,
                error=error,
            )
        )

    manifest_text = _render_manifest(results)
    provenance_text = _render_provenance(
        mapping_rel=mapping_path.relative_to(root).as_posix(),
        root_rel=output_root.relative_to(root).as_posix(),
        rows=results,
    )
    (output_root / "manifest.tsv").write_text(manifest_text, encoding="utf-8")
    (output_root / "provenance.toml").write_text(provenance_text, encoding="utf-8")

    print(
        "Fetched sensational primary sources: "
        f"targets={len(results)} "
        f"ok={sum(1 for row in results if row.status == 'ok')} "
        f"blocked={sum(1 for row in results if row.status.startswith('blocked'))} "
        f"failed={sum(1 for row in results if row.status.startswith('failed'))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
