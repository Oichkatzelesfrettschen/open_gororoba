#!/usr/bin/env python3
"""
Fetch a TOML-declared research intake batch with provenance metadata.

Workflow:
1) Read registry/research_intake_*.toml
2) Download each normalized URL with curl fallback to wget
3) If both fail and firefox is available, capture a headless screenshot
4) Write data/external/intake/.../provenance*.toml and index*.tsv

Python-first policy:
- This script is the canonical ingestion entrypoint.
- SHA256 hashing uses Rust via pyo3 (`gororoba_py.py_sha256_file`) when available.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # Optional pyo3 bridge for hashing.
    import gororoba_py  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    gororoba_py = None  # type: ignore


USER_AGENT = (
    "open_gororoba-intake-bot/2026-02-14 "
    "(research provenance fetch; +https://github.com/eirikr/open_gororoba)"
)

SAFE_SUFFIXES = {
    ".pdf",
    ".txt",
    ".csv",
    ".json",
    ".xml",
    ".zip",
    ".gz",
    ".tar",
    ".tgz",
    ".html",
    ".htm",
}


@dataclass
class FetchResult:
    entry_id: str
    topic: str
    resource_class: str
    source_url: str
    method: str
    status: str
    output_path: str
    sha256: str
    size_bytes: int
    fetched_at_utc: str
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


def _sha256_python(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path, backend: str) -> str:
    if backend not in {"auto", "pyo3", "python"}:
        raise SystemExit("ERROR: --sha-backend must be one of auto,pyo3,python")

    if backend in {"auto", "pyo3"}:
        if gororoba_py is not None and hasattr(gororoba_py, "py_sha256_file"):
            try:
                out = str(gororoba_py.py_sha256_file(str(path)))
                if out and len(out) == 64:
                    return out
            except Exception as exc:
                if backend == "pyo3":
                    raise SystemExit(f"ERROR: pyo3 sha256 bridge failed: {exc}") from exc
        elif backend == "pyo3":
            raise SystemExit("ERROR: gororoba_py.py_sha256_file is unavailable")

    return _sha256_python(path)


def _guess_suffix(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in SAFE_SUFFIXES:
        return suffix
    return ".html"


def _run(cmd: list[str], timeout_s: int = 0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=(timeout_s if timeout_s > 0 else None),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        if "timeout" not in stderr.lower():
            stderr = (stderr + ";timeout_expired").strip(";")
        return subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)


def _fetch_with_curl(url: str, dest: Path, timeout_s: int) -> tuple[bool, str]:
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout_s),
        "-A",
        USER_AGENT,
        "-o",
        str(dest),
        "-w",
        "%{http_code}",
        url,
    ]
    proc = _run(cmd)
    if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
        return True, proc.stdout.strip() or "200"
    return False, (proc.stderr.strip() or proc.stdout.strip() or "curl_failed")


def _fetch_with_wget(url: str, dest: Path, timeout_s: int) -> tuple[bool, str]:
    cmd = [
        "wget",
        "--quiet",
        "--tries=2",
        "--timeout",
        str(timeout_s),
        "--user-agent",
        USER_AGENT,
        "-O",
        str(dest),
        url,
    ]
    proc = _run(cmd)
    if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
        return True, "wget_ok"
    return False, (proc.stderr.strip() or proc.stdout.strip() or "wget_failed")


def _fetch_with_firefox_screenshot(url: str, dest_png: Path) -> tuple[bool, str]:
    firefox = shutil.which("firefox")
    if firefox is None:
        return False, "firefox_not_installed"
    with tempfile.TemporaryDirectory(prefix="firefox-intake-profile-") as profile:
        cmd = [
            "timeout",
            "--signal=TERM",
            "25",
            firefox,
            "--headless",
            "--no-remote",
            "--profile",
            profile,
            "--screenshot",
            str(dest_png),
            url,
        ]
        proc = _run(cmd, timeout_s=30)
        if proc.returncode == 0 and dest_png.exists() and dest_png.stat().st_size > 0:
            return True, "firefox_screenshot_ok"
        if proc.returncode == 124:
            return False, "firefox_timeout"
        return False, (proc.stderr.strip() or proc.stdout.strip() or "firefox_failed")


def _to_toml(batch_id: str, registry_path: str, root_path: str, rows: list[FetchResult]) -> str:
    ok_count = sum(1 for row in rows if row.status == "ok")
    fail_count = sum(1 for row in rows if row.status != "ok")
    lines: list[str] = []
    lines.append("# Generated by src/scripts/analysis/fetch_research_intake_batch.py")
    lines.append("[batch]")
    lines.append(f"id = {_escape(batch_id)}")
    lines.append(f"generated_at_utc = {_escape(_utc_now())}")
    lines.append(f"registry_path = {_escape(registry_path)}")
    lines.append(f"output_root = {_escape(root_path)}")
    lines.append(f"entry_count = {len(rows)}")
    lines.append(f"success_count = {ok_count}")
    lines.append(f"failure_count = {fail_count}")
    lines.append("")
    for row in rows:
        lines.append("[[artifact]]")
        lines.append(f"id = {_escape(row.entry_id)}")
        lines.append(f"topic = {_escape(row.topic)}")
        lines.append(f"resource_class = {_escape(row.resource_class)}")
        lines.append(f"source_url = {_escape(row.source_url)}")
        lines.append(f"method = {_escape(row.method)}")
        lines.append(f"status = {_escape(row.status)}")
        lines.append(f"output_path = {_escape(row.output_path)}")
        lines.append(f"sha256 = {_escape(row.sha256)}")
        lines.append(f"size_bytes = {row.size_bytes}")
        lines.append(f"fetched_at_utc = {_escape(row.fetched_at_utc)}")
        lines.append(f"error = {_escape(row.error)}")
        lines.append("")
    rendered = "\n".join(lines)
    _assert_ascii(rendered, "provenance.toml")
    return rendered


def _to_tsv(rows: list[FetchResult]) -> str:
    header = (
        "id\ttopic\tresource_class\tstatus\tmethod\tsize_bytes\tsha256\toutput_path\terror"
    )
    body = [header]
    for row in rows:
        body.append(
            "\t".join(
                [
                    row.entry_id,
                    row.topic,
                    row.resource_class,
                    row.status,
                    row.method,
                    str(row.size_bytes),
                    row.sha256,
                    row.output_path,
                    row.error.replace("\t", " ").replace("\n", " "),
                ]
            )
        )
    rendered = "\n".join(body) + "\n"
    _assert_ascii(rendered, "index.tsv")
    return rendered


def _entry_url(entry: dict[str, Any]) -> str:
    normalized = str(entry.get("normalized_url", "")).strip()
    if normalized:
        return normalized
    return str(entry.get("original_url", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--registry",
        default="registry/research_intake_2026_02_14.toml",
        help="Input intake registry.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of entries to fetch (0 means all).",
    )
    parser.add_argument(
        "--ids",
        default="",
        help="Optional comma-separated entry IDs to fetch (example: RI-004,RI-018).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Network timeout per request.",
    )
    parser.add_argument(
        "--firefox-fallback",
        action="store_true",
        help="Capture screenshot with firefox if curl/wget fail.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output files (for example: _retry_firefox).",
    )
    parser.add_argument(
        "--sha-backend",
        default="auto",
        help="Checksum backend: auto, pyo3, or python.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    registry_path = root / args.registry
    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    meta = data.get("research_intake", {})
    batch_id = str(meta.get("id", "unknown-batch"))
    entries: list[dict[str, Any]] = list(data.get("entry", []))
    if args.ids.strip():
        wanted = {token.strip() for token in args.ids.split(",") if token.strip()}
        entries = [entry for entry in entries if str(entry.get("id", "")).strip() in wanted]
    if args.limit > 0:
        entries = entries[: args.limit]

    output_root = root / str(meta.get("download_root", "data/external/intake/unnamed"))
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[FetchResult] = []

    for entry in entries:
        entry_id = str(entry.get("id", "missing-id"))
        topic = str(entry.get("topic", "unknown"))
        resource_class = str(entry.get("resource_class", "unknown"))
        url = _entry_url(entry)
        suffix = _guess_suffix(url)
        output = raw_dir / f"{entry_id}{suffix}"
        error = ""
        method = "none"
        status = "failed"

        ok, detail = _fetch_with_curl(url, output, args.timeout_seconds)
        if ok:
            method = "curl"
            status = "ok"
        else:
            error = f"curl:{detail}"
            if output.exists():
                output.unlink()
            ok_wget, detail_wget = _fetch_with_wget(url, output, args.timeout_seconds)
            if ok_wget:
                method = "wget"
                status = "ok"
                error = ""
            else:
                error = f"{error};wget:{detail_wget}"
                if output.exists():
                    output.unlink()
                if args.firefox_fallback:
                    screenshot = raw_dir / f"{entry_id}.png"
                    ok_firefox, detail_firefox = _fetch_with_firefox_screenshot(url, screenshot)
                    if ok_firefox:
                        output = screenshot
                        method = "firefox_headless_screenshot"
                        status = "ok"
                        error = ""
                    else:
                        error = f"{error};firefox:{detail_firefox}"

        if output.exists():
            digest = _sha256(output, args.sha_backend)
            size = output.stat().st_size
            output_rel = output.relative_to(root).as_posix()
        else:
            digest = ""
            size = 0
            output_rel = ""

        rows.append(
            FetchResult(
                entry_id=entry_id,
                topic=topic,
                resource_class=resource_class,
                source_url=url,
                method=method,
                status=status,
                output_path=output_rel,
                sha256=digest,
                size_bytes=size,
                fetched_at_utc=_utc_now(),
                error=error,
            )
        )

    provenance = _to_toml(
        batch_id=batch_id,
        registry_path=registry_path.relative_to(root).as_posix(),
        root_path=output_root.relative_to(root).as_posix(),
        rows=rows,
    )
    suffix = str(args.output_suffix)
    provenance_name = f"provenance{suffix}.toml" if suffix else "provenance.toml"
    index_name = f"index{suffix}.tsv" if suffix else "index.tsv"
    (output_root / provenance_name).write_text(provenance, encoding="utf-8")
    (output_root / index_name).write_text(_to_tsv(rows), encoding="utf-8")
    print(
        f"Fetched {len(rows)} entries: "
        f"{sum(1 for r in rows if r.status == 'ok')} ok / "
        f"{sum(1 for r in rows if r.status != 'ok')} failed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
