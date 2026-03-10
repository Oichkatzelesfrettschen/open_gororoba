#!/usr/bin/env python3
"""Fetch selected SOHO products with deterministic manifests.

This fetcher intentionally starts narrow:

- CELIAS Proton Monitor mission-long bundles for inner-boundary solar wind data.
- LASCO level-0.5 day directories for CME / coronagraph event workflows.

It writes JSON manifests compatible with the repo's provenance workflow and
degrades to metadata-only or partial manifests when a requested endpoint fails.
"""

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

USER_AGENT = "gororoba-fetch/0.1 (research)"
OUTPUT_DIR = Path("data/external/soho")

CELIAS_PRODUCTS = {
    "celias_pm_5min_bundle": {
        "filename": "CELIAS_Proton_Monitor_5min.tar.gz",
        "url": "https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz",
        "description": "SOHO CELIAS Proton Monitor 5-minute mission-long TXT bundle.",
    },
    "celias_pm_cdf_bundle": {
        "filename": "CELIAS_Proton_Monitor_5min-30s-CDF.tar.gz",
        "url": "https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min-30s-CDF.tar.gz",
        "description": (
            "SOHO CELIAS Proton Monitor 5-minute plus 30-second "
            "mission-long CDF bundle."
        ),
    },
}

SOHO_METADATA_URLS = {
    "gsfc_index": "https://soho.nascom.nasa.gov/data/archive/index_gsfc.html",
    "gsfc_archive": "https://soho.nascom.nasa.gov/data/archive.html",
    "esa_cmdline": "https://www.cosmos.esa.int/web/soho/command-line",
    "lasco_direct": "https://lasco-www.nrl.navy.mil/index.php?p=get_data",
}

LASCO_ROOT = "https://umbra.nascom.nasa.gov/pub/lasco_level05"
HREF_RE = re.compile(r'href="([^"]+)"')


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def build_file_entry(url: str, path: Path, status: str) -> dict[str, object]:
    entry: dict[str, object] = {"url": url, "path": str(path), "status": status}
    if not path.exists():
        return entry
    data = path.read_bytes()
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["filename"] = path.name
    return entry


def fetch_file(url: str, out: Path, *, skip_existing: bool) -> dict[str, object]:
    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped")
    try:
        data = fetch_bytes(url)
    except (URLError, OSError, TimeoutError):
        return {"url": url, "path": str(out), "status": "failed"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return build_file_entry(url, out, "fetched")


def fetch_metadata_trace(meta_dir: Path) -> list[dict[str, object]]:
    sidecars = []
    meta_dir.mkdir(parents=True, exist_ok=True)
    for key, url in SOHO_METADATA_URLS.items():
        suffix = ".html"
        out = meta_dir / f"{key}{suffix}"
        sidecars.append(fetch_file(url, out, skip_existing=False))
    return sidecars


def discover_lasco_children(url: str) -> list[str]:
    html = fetch_bytes(url).decode("utf-8", errors="replace")
    children = []
    for href in HREF_RE.findall(html):
        if href.startswith("?") or href.startswith("/pub/lasco_level05/") or href == "../":
            continue
        children.append(href)
    return children


def fetch_celias_bundle(product: str, *, skip_existing: bool) -> dict[str, object]:
    spec = CELIAS_PRODUCTS[product]
    subdir = OUTPUT_DIR / "celias"
    subdir.mkdir(parents=True, exist_ok=True)
    meta = fetch_metadata_trace(subdir / "metadata")
    entry = fetch_file(spec["url"], subdir / spec["filename"], skip_existing=skip_existing)
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    counts[entry["status"]] += 1
    status = "ready" if entry["status"] in {"fetched", "skipped"} else "metadata_only"
    return {
        "status": status,
        "reason": None if status == "ready" else "bundle_unreachable",
        "counts": counts,
        "files": [entry],
        "metadata": {
            "product": product,
            "description": spec["description"],
            "metadata_sidecars": meta,
        },
    }


def fetch_lasco_day(
    date_text: str, *, camera: str, max_files: int, skip_existing: bool
) -> dict[str, object]:
    date = dt.date.fromisoformat(date_text)
    yymmdd = date.strftime("%y%m%d")
    subdir = OUTPUT_DIR / "lasco" / "level05" / yymmdd
    subdir.mkdir(parents=True, exist_ok=True)
    meta = fetch_metadata_trace(subdir / "metadata")

    day_url = f"{LASCO_ROOT}/{yymmdd}/"
    cameras = [camera] if camera in {"c2", "c3"} else ["c2", "c3"]
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    day_index = fetch_file(day_url, subdir / "index.html", skip_existing=skip_existing)
    files.append(day_index)
    counts[day_index["status"]] += 1

    for cam in cameras:
        cam_url = f"{day_url}{cam}/"
        cam_dir = subdir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        index_entry = fetch_file(cam_url, cam_dir / "index.html", skip_existing=skip_existing)
        files.append(index_entry)
        counts[index_entry["status"]] += 1
        img_hdr_entry = fetch_file(
            f"{cam_url}img_hdr.txt", cam_dir / "img_hdr.txt", skip_existing=skip_existing
        )
        img_hdr_entry["camera"] = cam
        files.append(img_hdr_entry)
        counts[img_hdr_entry["status"]] += 1

        try:
            children = discover_lasco_children(cam_url)
        except (URLError, OSError, TimeoutError):
            continue

        fts_names = [name for name in children if name.endswith(".fts")]
        for name in fts_names[:max_files]:
            entry = fetch_file(f"{cam_url}{name}", cam_dir / name, skip_existing=skip_existing)
            entry["camera"] = cam
            files.append(entry)
            counts[entry["status"]] += 1

    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "metadata_only"
    return {
        "status": status,
        "reason": None if status == "ready" else "lasco_day_unreachable",
        "counts": counts,
        "files": files,
        "metadata": {
            "product": "lasco_lz_day",
            "date": date_text,
            "camera": camera,
            "max_files": max_files,
            "lasco_root": LASCO_ROOT,
            "metadata_sidecars": meta,
        },
    }


def write_manifest(name: str, payload: dict[str, object]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch selected SOHO products.")
    parser.add_argument(
        "--product",
        choices=["celias_pm_5min_bundle", "celias_pm_cdf_bundle", "lasco_lz_day"],
        required=True,
    )
    parser.add_argument(
        "--date",
        help="Date for lasco_lz_day in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--camera",
        choices=["c2", "c3", "all"],
        default="all",
        help="LASCO camera for lasco_lz_day (default: all).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2,
        help="Maximum number of sample FITS files per camera for lasco_lz_day.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.product == "lasco_lz_day":
        if not args.date:
            parser.error("--date is required for lasco_lz_day")
        payload = fetch_lasco_day(
            args.date,
            camera=args.camera,
            max_files=args.max_files,
            skip_existing=args.skip_existing,
        )
        manifest_name = f"lasco_lz_day_{args.date.replace('-', '')}"
    else:
        payload = fetch_celias_bundle(args.product, skip_existing=args.skip_existing)
        manifest_name = args.product

    manifest_path = write_manifest(manifest_name, payload)
    print(f"manifest: {manifest_path}")
    print(
        f"SOHO fetch: {payload['counts']['fetched']} fetched, "
        f"{payload['counts']['skipped']} skipped, {payload['counts']['failed']} failed"
    )
    return 1 if payload["status"] != "ready" else 0


if __name__ == "__main__":
    sys.exit(main())
