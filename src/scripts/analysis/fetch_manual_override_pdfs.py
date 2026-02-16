#!/usr/bin/env python3
"""Fetch direct-PDF candidates from manual mirror overrides.

This is a narrow, reproducible fetch lane meant to turn "working direct PDF" hints
into cached PDFs under data/external/intake/... with a pdf_success_added.tsv ledger.

Inputs:
- registry/manual_mirror_overrides_2026_02_15.toml

Outputs:
- data/external/intake/2026_02_15_manual_override_fetch/pdf_success/*.pdf
- data/external/intake/2026_02_15_manual_override_fetch/pdf_success_added.tsv
- data/external/intake/2026_02_15_manual_override_fetch/fetch_attempts.tsv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


URL_RE = re.compile(r"^https?://", re.IGNORECASE)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)


def _normalize_url(url: str) -> str:
    value = str(url).strip()
    if not value:
        return ""
    value = value.strip("`")
    value = value.lstrip("(<[{\"'")
    value = value.rstrip(">)]}\"'")
    value = value.rstrip("`")
    value = value.rstrip(".,;:")
    return value.strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = _normalize_url(value)
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _is_direct_pdf_candidate(url: str) -> bool:
    u = url.lower()
    if not URL_RE.match(u):
        return False
    # Obvious.
    if u.endswith(".pdf"):
        return True
    # Common: /pdf endpoints (MDPI, IOP, etc.)
    if u.endswith("/pdf") or u.endswith("/pdf/"):
        return True
    if "/pdf?" in u:
        return True
    # Common publisher patterns.
    if "/pdf/" in u:
        return True
    if "/doi/pdf/" in u:
        return True
    if "/doi/epdf/" in u:
        return True
    if "tandfonline.com/doi/pdf/" in u:
        return True
    if "tandfonline.com/doi/epdf/" in u:
        return True
    # APS: /pr/pdf/<doi> etc.
    if "journals.aps.org/" in u and "/pdf/" in u:
        return True
    # Royal Society: /doi/pdf/<doi>
    if "royalsocietypublishing.org" in u and "/doi/pdf/" in u:
        return True
    # arXiv PDFs may not end with .pdf (rare); allow /pdf/.
    if "arxiv.org/pdf/" in u:
        return True
    # Common institutional repository endpoints.
    if "/bitstreams/" in u and u.endswith("/download"):
        return True
    if u.endswith("/download") or "/download?" in u:
        return True
    # OpenReview uses /pdf?id=... endpoints.
    if "openreview.net/pdf" in u:
        return True
    return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_slug(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "unknown"


def _extract_doi(url: str) -> str:
    for match in DOI_RE.finditer(url):
        return match.group(0)
    return ""


def _name_for(url: str, override_id: str) -> str:
    doi = _extract_doi(url)
    if doi:
        return _safe_slug(f"{override_id}_{doi}") + ".pdf"
    # Host + path fragment.
    cleaned = url.replace("https://", "").replace("http://", "")
    cleaned = cleaned.split("?", 1)[0]
    cleaned = cleaned[:160]
    return _safe_slug(f"{override_id}_{cleaned}") + ".pdf"


def _curl_fetch(url: str, out_path: Path, timeout_s: int) -> tuple[str, str, str, bool, str]:
    headers = out_path.with_suffix(out_path.suffix + ".headers.txt")
    err = out_path.with_suffix(out_path.suffix + ".curl.err")

    for p in (headers, err):
        p.write_text("", encoding="utf-8")

    cmd = [
        "curl",
        "-L",
        "--connect-timeout",
        "8",
        "--max-time",
        str(timeout_s),
        "--compressed",
        "-A",
        USER_AGENT,
        "-D",
        str(headers),
        "-o",
        str(out_path),
        "-w",
        "%{http_code}\t%{content_type}\t%{size_download}",
        url,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err.open("wb"), check=False)
    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    parts = raw.split("\t")
    http_code = parts[0].strip() if len(parts) > 0 else ""
    content_type = parts[1].strip() if len(parts) > 1 else ""
    size_download = parts[2].strip() if len(parts) > 2 else ""

    is_pdf = False
    note = ""
    try:
        with out_path.open("rb") as f:
            is_pdf = f.read(5) == b"%PDF-"
    except OSError:
        note = "body_unreadable"

    if proc.returncode != 0 and not http_code:
        note = f"curl_exit_{proc.returncode}"

    status = "pdf_ok" if http_code.startswith("2") and is_pdf else ("ok_nonpdf" if http_code.startswith("2") else (f"http_{http_code}" if http_code else "unknown"))
    if note:
        status = "unknown"
    return http_code or "000", content_type, size_download or "0", is_pdf, status, note


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--overrides",
        default="registry/manual_mirror_overrides_2026_02_15.toml",
        help="Manual mirror overrides TOML.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/external/intake/2026_02_15_manual_override_fetch",
        help="Intake output directory.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=25,
        help="Per-request max time.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    overrides_path = repo_root / args.overrides
    if not overrides_path.exists():
        raise SystemExit(f"ERROR: missing overrides file: {overrides_path}")

    out_dir = repo_root / args.out_dir
    pdf_dir = out_dir / "pdf_success"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    data = tomllib.loads(overrides_path.read_text(encoding="utf-8"))
    overrides = data.get("mirror_override", [])

    attempts_path = out_dir / "fetch_attempts.tsv"
    added_path = out_dir / "pdf_success_added.tsv"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    urls: list[tuple[str, str]] = []
    for entry in overrides:
        oid = str(entry.get("id", "")).strip() or "OVERRIDE"
        for url in entry.get("urls", []) or []:
            u = _normalize_url(url)
            if not URL_RE.match(u):
                continue
            if not _is_direct_pdf_candidate(u):
                continue
            urls.append((oid, u))

    urls = [(oid, u) for oid, u in urls if u]
    # De-dupe by URL, keeping first override id.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for oid, u in urls:
        if u in seen:
            continue
        seen.add(u)
        unique.append((oid, u))

    # Avoid re-downloading identical content when already present in pdf_dir.
    existing_sha_to_name: dict[str, str] = {}
    for path in pdf_dir.glob("*.pdf"):
        try:
            existing_sha_to_name[_sha256(path)] = path.name
        except OSError:
            continue

    attempts_rows: list[list[str]] = []
    added_rows: list[list[str]] = []

    for idx, (oid, url) in enumerate(unique, start=1):
        tmp = pdf_dir / f"tmp_{idx:03d}.bin"
        http_code, content_type, size_download, is_pdf, status, note = _curl_fetch(
            url,
            tmp,
            args.timeout_seconds,
        )

        canonical_name = ""
        sha = ""
        size = "0"
        add_note = ""

        if status == "pdf_ok" and is_pdf and tmp.exists():
            sha = _sha256(tmp)
            size = str(tmp.stat().st_size)
            existing = existing_sha_to_name.get(sha)
            if existing:
                canonical_name = existing
                add_note = "duplicate_sha_existing_not_copied"
                tmp.unlink(missing_ok=True)
            else:
                canonical_name = _name_for(url, oid)
                final = pdf_dir / canonical_name
                # Ensure unique filename.
                if final.exists():
                    canonical_name = f"{final.stem}_{sha[:8]}.pdf"
                    final = pdf_dir / canonical_name
                tmp.replace(final)
                existing_sha_to_name[sha] = canonical_name

        else:
            # Keep non-PDF trace for debugging, but don't pretend it's a PDF.
            if tmp.exists() and tmp.stat().st_size == 0:
                tmp.unlink(missing_ok=True)

        attempts_rows.append(
            [
                oid,
                url,
                http_code,
                content_type,
                size_download,
                "1" if is_pdf else "0",
                status,
                note,
                now,
            ]
        )

        if canonical_name:
            added_rows.append(
                [
                    oid,
                    canonical_name,
                    sha,
                    size,
                    url,
                    add_note,
                ]
            )

    with attempts_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "override_id",
                "url",
                "http_code",
                "content_type",
                "size_download",
                "is_pdf",
                "status",
                "note",
                "checked_at",
            ]
        )
        w.writerows(attempts_rows)

    with added_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "canonical_pdf_name", "sha256", "size_download", "source_url", "note"])
        w.writerows(added_rows)

    print(f"Wrote attempts: {attempts_path.relative_to(repo_root).as_posix()} urls={len(unique)}")
    print(f"Wrote pdf ledger: {added_path.relative_to(repo_root).as_posix()} pdfs={len(added_rows)}")
    print(f"PDF directory: {pdf_dir.relative_to(repo_root).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
