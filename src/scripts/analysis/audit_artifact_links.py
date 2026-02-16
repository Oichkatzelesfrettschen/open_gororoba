#!/usr/bin/env python3
"""
Audit all links listed in registry/artifact_source_of_truth.toml.

Outputs a normalized TSV with one row per URL:
- url
- http_code
- content_type
- size_download
- is_pdf
- status (pdf_ok|ok_nonpdf|http_<code>|unknown)
- note
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import tomllib


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


def _status_from(code: str, is_pdf: bool) -> str:
    if code.startswith("2") and is_pdf:
        return "pdf_ok"
    if code.startswith("2"):
        return "ok_nonpdf"
    if code:
        return f"http_{code}"
    return "unknown"


def _read_urls(source_path: Path) -> list[str]:
    data = tomllib.loads(source_path.read_text(encoding="utf-8"))
    urls: set[str] = set()
    for artifact in data.get("artifact", []):
        for url in artifact.get("all_links", []):
            value = _normalize_url(url)
            if value.startswith("http://") or value.startswith("https://"):
                urls.add(value)
    return sorted(urls)


def _audit_one(url: str, temp_dir: Path, timeout_s: int) -> dict[str, str]:
    slug = (
        url.replace("https://", "")
        .replace("http://", "")
        .replace("/", "_")
        .replace("?", "_")
        .replace("&", "_")
    )
    slug = slug[:120]
    body = temp_dir / f"{slug}.bin"
    headers = temp_dir / f"{slug}.headers.txt"
    err = temp_dir / f"{slug}.curl.err"

    for path in (body, headers, err):
        path.write_text("", encoding="utf-8")

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
        str(body),
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
        with body.open("rb") as handle:
            magic = handle.read(5)
            is_pdf = magic == b"%PDF-"
    except OSError:
        note = "body_unreadable"

    status = _status_from(http_code, is_pdf)
    if not note and proc.returncode != 0 and not http_code:
        note = f"curl_exit_{proc.returncode}"

    return {
        "url": url,
        "http_code": http_code or "000",
        "content_type": content_type,
        "size_download": size_download or "0",
        "is_pdf": "1" if is_pdf else "0",
        "status": status,
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--source",
        default="registry/artifact_source_of_truth.toml",
        help="Input source-of-truth registry.",
    )
    parser.add_argument(
        "--out",
        default="data/external/intake/global_link_audit_2026_02_15/link_audit_results.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=20,
        help="Per-request max time.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_path = repo_root / args.source
    out_path = repo_root / args.out
    if not source_path.exists():
        raise SystemExit(f"ERROR: missing source registry: {source_path}")

    urls = _read_urls(source_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = out_path.parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_audit_one, url, temp_dir, args.timeout_seconds): url
            for url in urls
        }
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda r: r["url"])
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
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
        for row in rows:
            writer.writerow(
                [
                    row["url"],
                    row["http_code"],
                    row["content_type"],
                    row["size_download"],
                    row["is_pdf"],
                    row["status"],
                    row["note"],
                    checked_at,
                ]
            )

    # Keep temp artifacts for provenance.
    print(
        f"Wrote link audit TSV: {out_path.relative_to(repo_root).as_posix()} "
        f"urls={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
