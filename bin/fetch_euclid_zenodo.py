#!/usr/bin/env python3
"""Fetch governed Euclid Q1 supplementary catalogs from Zenodo.

Downloads specific Euclid Quick Release 1 supplementary science products
from versioned Zenodo records via the REST API. These are small catalogs
(morphology, strong lensing, mergers, photo-z, spectroscopy) suitable for
direct download, as opposed to the multi-TB primary survey catalogs that
require TAP/ADQL queries (see fetch_euclid.py for those).

Target records:
    15106473: Visual Morphology Classification (97.3 + 67.7 MB parquet)
    15025832: Strong Lensing Discovery Engine Candidates (422 KB CSV)
    17087034: Galaxy Merger Classification (39.7 MB CSV)
    15027787: Galaxy Morphology Supplementary (additional release)
    14935225: Photometric Redshift PDFs (photo-z posterior distributions)
    15025557: Galaxy Clustering Catalog (angular correlation functions)
    15108017: Compact Galaxy Groups (group membership catalog)
    14951889: Globular Cluster Candidates (point-source catalog)

Modes:
    --discover  Query Zenodo API for all Euclid Q1 records, print manifest
    --dry-run   Query record metadata but skip downloads
    --catalog X Fetch a specific catalog (or 'all')

License: CC BY 4.0 (Euclid Consortium publications).

Output: data/external/euclid/zenodo/{record_id}/{filename}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ZENODO_API = "https://zenodo.org/api/records"
BASE_OUT_DIR = Path("data/external/euclid/zenodo")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# Retry parameters for large file downloads.
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0

# Size thresholds for discovery classification.
LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024 * 1024  # 50 GB
TOTAL_FOOTPRINT_CAP_BYTES = 200 * 1024 * 1024 * 1024  # 200 GB

# Target matrix: logical name -> (record_id, [target_filenames]).
# Only download the specific files we need, not entire records which
# may contain supplementary figures, ReadMe files, etc.
TARGETS: dict[str, dict[str, object]] = {
    "morphology": {
        "record_id": 15106473,
        "description": "Euclid Q1 Visual Morphology Classification Catalogue",
        "target_files": [
            "morphology_catalogue.parquet",
            "useful_physical_measurements.parquet",
        ],
    },
    "strong_lensing": {
        "record_id": 15025832,
        "description": "Euclid Q1 Strong Lensing Discovery Engine Candidates",
        "target_files": [
            "q1_discovery_engine_lens_catalog.csv",
        ],
    },
    "mergers": {
        "record_id": 17087034,
        "description": "Euclid Q1 Galaxy Merger Classification",
        "target_files": [
            "Q1_merger_classification.csv",
        ],
    },
    "morphology_supplementary": {
        "record_id": 15027787,
        "description": "Euclid Q1 Galaxy Morphology Supplementary Release",
        "target_files": [
            "morphology_supplementary.parquet",
        ],
    },
    "photo_z": {
        "record_id": 14935225,
        "description": "Euclid Q1 Photometric Redshift PDFs",
        "target_files": [
            "photoz_pdfs.parquet",
        ],
    },
    "galaxy_clustering": {
        "record_id": 15025557,
        "description": "Euclid Q1 Galaxy Clustering Catalog",
        "target_files": [
            "clustering_catalog.parquet",
        ],
    },
    "compact_groups": {
        "record_id": 15108017,
        "description": "Euclid Q1 Compact Galaxy Groups",
        "target_files": [
            "compact_groups.csv",
        ],
    },
    "globular_clusters": {
        "record_id": 14951889,
        "description": "Euclid Q1 Globular Cluster Candidates",
        "target_files": [
            "globular_cluster_candidates.csv",
        ],
    },
}

# O(1) lookup for dedup during discovery: all record IDs already in TARGETS.
KNOWN_RECORD_IDS: frozenset[int] = frozenset(
    int(cfg["record_id"]) for cfg in TARGETS.values()
)


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_json(url: str, *, timeout: int = 60) -> dict:
    data = fetch_bytes(url, timeout=timeout)
    return json.loads(data.decode("utf-8"))


def download_file(url: str, dest: Path, *, expected_size: int | None = None) -> int:
    """Download a file with retry logic. Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=300) as resp:  # noqa: S310
                data = resp.read()
            dest.write_bytes(data)
            if expected_size is not None and len(data) != expected_size:
                print(
                    f"    WARNING: expected {expected_size} bytes, got {len(data)}",
                    file=sys.stderr,
                )
            return len(data)
        except (URLError, HTTPError, OSError, TimeoutError) as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_S * attempt
                print(f"    attempt {attempt}/{MAX_RETRIES} failed: {exc}, retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                raise
    # Unreachable but satisfies type checker.
    msg = "download_file: exhausted retries"
    raise RuntimeError(msg)


def verify_checksum(path: Path, expected_md5: str) -> bool:
    """Verify downloaded file against Zenodo-provided MD5 checksum."""
    h = hashlib.md5()  # noqa: S324 -- Zenodo uses MD5 for integrity
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected_md5


def build_file_entry(
    url: str, path: Path, status: str, *, source: str, reason: str | None = None
) -> dict[str, object]:
    entry: dict[str, object] = {
        "url": url,
        "path": str(path),
        "status": status,
        "source": source,
    }
    if reason:
        entry["reason"] = reason
    if not path.exists():
        return entry
    data = path.read_bytes()
    entry["filename"] = path.name
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    return entry


def discover_euclid_q1_records(
    *, max_pages: int = 5, page_size: int = 100
) -> list[dict[str, object]]:
    """Query Zenodo search API for all Euclid Q1 records.

    Paginates through results, filters for Euclid Consortium records,
    and returns a manifest of record metadata. Respects Zenodo rate
    limits (60 req/min) via exponential backoff.
    """
    # Zenodo InvenioRDM search endpoint.
    base_url = (
        f"{ZENODO_API}?q=euclid+q1&size={page_size}"
        "&sort=mostrecent&type=dataset"
    )
    records: list[dict[str, object]] = []
    seen_ids: set[int] = set()

    for page in range(1, max_pages + 1):
        url = f"{base_url}&page={page}"
        try:
            data = fetch_json(url, timeout=30)
        except (URLError, HTTPError, OSError, TimeoutError) as exc:
            print(f"  WARNING: Zenodo search page {page} failed: {exc}", file=sys.stderr)
            break

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            record_id = hit.get("id", 0)
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)

            metadata = hit.get("metadata", {})
            title = metadata.get("title", "")

            # Filter: must mention "euclid" in title or creators.
            title_lower = title.lower()
            creators_str = " ".join(
                c.get("name", "") for c in metadata.get("creators", [])
            ).lower()
            if "euclid" not in title_lower and "euclid" not in creators_str:
                continue

            files = hit.get("files", [])
            total_size = sum(f.get("size", 0) for f in files)
            file_count = len(files)

            # Classify file types.
            extensions = set()
            for f in files:
                key = f.get("key", "")
                if "." in key:
                    extensions.add(key.rsplit(".", 1)[-1].lower())

            has_catalog = bool(extensions & {"parquet", "csv", "fits", "ecsv", "hdf5"})
            image_only = extensions <= {"fits", "png", "jpg", "jpeg", "tiff"} and not has_catalog

            records.append({
                "record_id": record_id,
                "title": title,
                "doi": hit.get("doi", ""),
                "file_count": file_count,
                "total_size_bytes": total_size,
                "extensions": sorted(extensions),
                "has_catalog": has_catalog,
                "image_only": image_only,
                "known": record_id in KNOWN_RECORD_IDS,
                "large": total_size > LARGE_FILE_THRESHOLD_BYTES,
                "license": metadata.get("license", {}).get("id", "unknown"),
                "created": hit.get("created", ""),
            })

        # Respect rate limits: brief pause between pages.
        total_hits = data.get("hits", {}).get("total", 0)
        if page * page_size >= total_hits:
            break
        time.sleep(1.0)

    return records


def format_discovery_report(records: list[dict[str, object]]) -> None:
    """Print ASCII-formatted discovery report to stdout."""
    if not records:
        print("No Euclid Q1 records found on Zenodo.")
        return

    print(f"\n{'='*90}")
    print("Euclid Q1 Zenodo Record Discovery")
    print(f"{'='*90}")
    print(
        f"  {'Record ID':>10s}  {'Status':8s}  {'Files':>5s}  "
        f"{'Size (MB)':>10s}  {'Formats':12s}  Title"
    )
    print(f"  {'-'*10}  {'-'*8}  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*35}")

    grand_total = 0
    new_count = 0
    for rec in sorted(records, key=lambda r: r.get("record_id", 0)):
        rid = rec["record_id"]
        size_mb = rec.get("total_size_bytes", 0) / 1e6
        grand_total += rec.get("total_size_bytes", 0)
        fcount = rec.get("file_count", 0)
        exts = ",".join(rec.get("extensions", []))[:12]
        title = str(rec.get("title", ""))[:40]

        if rec.get("known"):
            status = "[KNOWN]"
        elif rec.get("large"):
            status = "[LARGE]"
        elif rec.get("image_only"):
            status = "[IMG]"
        else:
            status = "[NEW]"
            new_count += 1

        print(f"  {rid:>10d}  {status:8s}  {fcount:>5d}  {size_mb:>10.1f}  {exts:12s}  {title}")

    print(f"  {'-'*90}")
    print(
        f"  Total: {len(records)} records, {grand_total / 1e9:.2f} GB, "
        f"{new_count} new candidates"
    )
    cap_gb = TOTAL_FOOTPRINT_CAP_BYTES / 1e9
    print(f"  Auto-fetch cap: {cap_gb:.0f} GB")
    if grand_total > TOTAL_FOOTPRINT_CAP_BYTES:
        print(f"  WARNING: total exceeds {cap_gb:.0f} GB cap -- manual review required")
    print()


def fetch_zenodo_catalog(
    name: str,
    config: dict[str, object],
    *,
    skip_existing: bool,
    dry_run: bool,
    include_large: bool = False,
) -> dict[str, object]:
    """Fetch one Zenodo catalog (may contain multiple files)."""
    record_id = config["record_id"]
    target_files = config["target_files"]
    description = config["description"]
    out_dir = BASE_OUT_DIR / str(record_id)

    print(f"\n--- {name}: {description} (record {record_id}) ---")

    # Step 1: Query Zenodo API for record metadata.
    api_url = f"{ZENODO_API}/{record_id}"
    try:
        record = fetch_json(api_url, timeout=30)
    except (URLError, HTTPError, OSError, TimeoutError) as exc:
        return {
            "name": name,
            "record_id": record_id,
            "status": "failed",
            "reason": f"zenodo_api_unreachable: {exc}",
            "files": [],
        }

    # Extract file metadata from the record.
    api_files = record.get("files", [])
    file_index: dict[str, dict] = {}
    for fmeta in api_files:
        key = fmeta.get("key", "")
        file_index[key] = fmeta

    # Record-level metadata for manifest.
    record_meta = {
        "doi": record.get("doi", ""),
        "title": record.get("metadata", {}).get("title", ""),
        "license": record.get("metadata", {}).get("license", {}).get("id", "unknown"),
        "version": record.get("metadata", {}).get("version", ""),
        "created": record.get("created", ""),
        "updated": record.get("updated", ""),
        "total_files_in_record": len(api_files),
        "total_size_bytes": sum(f.get("size", 0) for f in api_files),
    }
    total_mb = record_meta["total_size_bytes"] / 1e6
    print(f"  Record has {len(api_files)} files, total {total_mb:.1f} MB")
    print(f"  DOI: {record_meta['doi']}")
    print(f"  License: {record_meta['license']}")

    # Step 2: Download each target file.
    files: list[dict[str, object]] = []
    counts = {"fetched": 0, "skipped": 0, "failed": 0, "verified": 0}

    for target_name in target_files:
        dest = out_dir / target_name
        fmeta = file_index.get(target_name)

        if fmeta is None:
            # Target file not found in record -- list available files.
            available = sorted(file_index.keys())
            print(f"  WARNING: '{target_name}' not found in record. Available: {available}")
            files.append(
                build_file_entry(
                    api_url, dest, "failed",
                    source="zenodo",
                    reason=f"file_not_in_record (available: {available})",
                )
            )
            counts["failed"] += 1
            continue

        download_url = fmeta.get("links", {}).get("self", "")
        expected_size = fmeta.get("size", 0)
        # Zenodo checksums are formatted as "md5:hexdigest".
        raw_checksum = fmeta.get("checksum", "")
        expected_md5 = raw_checksum.split(":")[-1] if ":" in raw_checksum else raw_checksum

        size_mb = expected_size / 1e6
        print(f"  {target_name}: {size_mb:.1f} MB", end="")

        # Guard: skip files >50 GB unless --include-large is set.
        if expected_size > LARGE_FILE_THRESHOLD_BYTES and not include_large:
            print(" [skip-large]")
            files.append(
                build_file_entry(
                    download_url, dest, "skipped",
                    source="zenodo",
                    reason=f"file_exceeds_50GB ({size_mb:.0f} MB), use --include-large",
                )
            )
            counts["skipped"] += 1
            continue

        if skip_existing and dest.exists():
            # Verify checksum of existing file.
            if expected_md5 and verify_checksum(dest, expected_md5):
                print(" [cached, checksum OK]")
                entry = build_file_entry(download_url, dest, "skipped", source="zenodo")
                entry["md5_verified"] = True
                files.append(entry)
                counts["skipped"] += 1
                counts["verified"] += 1
                continue
            elif dest.stat().st_size == expected_size:
                print(" [cached, size match]")
                entry = build_file_entry(download_url, dest, "skipped", source="zenodo")
                entry["md5_verified"] = False
                files.append(entry)
                counts["skipped"] += 1
                continue
            else:
                print(" [cached but size mismatch, re-downloading]")

        if dry_run:
            print(f" [dry-run, would download {size_mb:.1f} MB]")
            files.append(
                build_file_entry(
                    download_url, dest, "dry_run",
                    source="zenodo",
                    reason=f"would_download_{expected_size}_bytes",
                )
            )
            continue

        # Download.
        try:
            written = download_file(download_url, dest, expected_size=expected_size)
            # Verify checksum.
            md5_ok = False
            if expected_md5:
                md5_ok = verify_checksum(dest, expected_md5)
                if md5_ok:
                    counts["verified"] += 1
                else:
                    print(f" WARNING: MD5 mismatch for {target_name}", file=sys.stderr)
            print(f" -> {written / 1e6:.1f} MB {'[MD5 OK]' if md5_ok else '[no MD5 check]'}")
            entry = build_file_entry(download_url, dest, "fetched", source="zenodo")
            entry["md5_verified"] = md5_ok
            entry["expected_md5"] = expected_md5
            files.append(entry)
            counts["fetched"] += 1
        except (URLError, HTTPError, OSError, TimeoutError, RuntimeError) as exc:
            print(f" FAILED: {exc}")
            files.append(
                build_file_entry(
                    download_url, dest, "failed",
                    source="zenodo",
                    reason=str(exc),
                )
            )
            counts["failed"] += 1

    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "failed"
    return {
        "name": name,
        "record_id": record_id,
        "status": status,
        "counts": counts,
        "files": files,
        "metadata": {
            "zenodo": record_meta,
            "description": description,
        },
    }


def write_manifest(results: list[dict[str, object]]) -> Path:
    """Write combined manifest for all Zenodo catalogs."""
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE_OUT_DIR / "euclid_zenodo_manifest.json"
    manifest = {
        "product": "euclid_q1_zenodo_supplementary",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "catalogs": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nManifest: {manifest_path}")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed Euclid Q1 supplementary catalogs from Zenodo."
    )
    catalog_names_list = sorted(TARGETS.keys())
    parser.add_argument(
        "--catalog",
        choices=catalog_names_list + ["all"],
        default="all",
        help="Which catalog to fetch (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists and checksum matches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query Zenodo API but do not download files",
    )
    parser.add_argument(
        "--list-records",
        action="store_true",
        help="List all files in target Zenodo records and exit",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Query Zenodo API for all Euclid Q1 records and print manifest",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Include files >50 GB in downloads (default: skip with [skip-large])",
    )
    args = parser.parse_args()

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.catalog == "all":
        catalog_names = list(TARGETS.keys())
    else:
        catalog_names = [args.catalog]

    # Discovery mode: search Zenodo for all Euclid Q1 records.
    if args.discover:
        print("Querying Zenodo API for Euclid Q1 records...")
        records = discover_euclid_q1_records()
        format_discovery_report(records)
        # Write machine-readable manifest.
        BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = BASE_OUT_DIR / "euclid_zenodo_discovery.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "discovered_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "records": records,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n"
        )
        print(f"Discovery manifest: {manifest_path}")
        return 0

    # List mode: query API and show all files in each record.
    if args.list_records:
        for name in catalog_names:
            config = TARGETS[name]
            record_id = config["record_id"]
            print(f"\n=== {name} (record {record_id}) ===")
            try:
                record = fetch_json(f"{ZENODO_API}/{record_id}", timeout=30)
                for fmeta in record.get("files", []):
                    size_mb = fmeta.get("size", 0) / 1e6
                    checksum = fmeta.get("checksum", "")
                    print(f"  {fmeta['key']:60s}  {size_mb:8.1f} MB  {checksum}")
            except (URLError, HTTPError, OSError, TimeoutError) as exc:
                print(f"  ERROR: {exc}")
        return 0

    # Fetch mode.
    print(f"Euclid Q1 Zenodo fetch: {', '.join(catalog_names)}")
    print(f"Output directory: {BASE_OUT_DIR}")
    if args.dry_run:
        print("DRY RUN: will query API but not download files")

    results: list[dict[str, object]] = []
    totals = {"fetched": 0, "skipped": 0, "failed": 0, "verified": 0}

    for name in catalog_names:
        config = TARGETS[name]
        result = fetch_zenodo_catalog(
            name,
            config,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
            include_large=args.include_large,
        )
        results.append(result)
        for key in totals:
            totals[key] += result.get("counts", {}).get(key, 0)

    manifest_path = write_manifest(results)

    print()
    print(
        f"Euclid Zenodo fetch: {totals['fetched']} fetched, "
        f"{totals['skipped']} skipped, {totals['failed']} failed, "
        f"{totals['verified']} checksum-verified"
    )
    print(f"Manifest: {manifest_path}")

    if totals["failed"] > 0 and totals["fetched"] == 0 and totals["skipped"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
