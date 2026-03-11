#!/usr/bin/env python3
"""Fetch governed Pioneer 10 & 11 data.

Canonical authority:
    NASA SPDF/CDAWeb/COHOWeb Pioneer merged hourly magnetic field + plasma data.

Operational reality on this host:
    Direct GSFC byte access is often blocked or intermittent, so this fetcher
    also stages browser-reachable metadata from catalog.data.gov, data.nasa.gov,
    and DOI landing pages when bytes cannot be acquired.

Reachable non-GSFC byte lanes:
    UCLA PDS/PPI serves several Pioneer encounter subsets over HTTPS. These are
    not interchangeable with the full annual merged hourly COHO/NSSDC products,
    so they are exposed as a separate `encounter_hourly_ppi` product family.

    AMDA HAPI provides MAG-only hourly data (p10-mag-full, p11-mag-full) in RTN.
    NO plasma data exists for Pioneer in AMDA. Output has valid B-field but
    NaN fill for density, speed, temperature.

Auto fallback chain: SPDF -> AMDA MAG-only -> metadata
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PIONEER10_BASE = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer10/merged/"
PIONEER11_BASE = "https://spdf.gsfc.nasa.gov/pub/data/pioneer/pioneer11/merged/"
CATALOG_SEARCH = "https://catalog.data.gov/api/3/action/package_search"
OUTPUT_DIR = Path("data/external/pioneer")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:136.0) Gecko/20100101 Firefox/136.0"

PIONEER_METADATA = {
    "p10": {
        "title": "Pioneer 10 hourly merged magnetic field and plasma data",
        "catalog_page": "https://catalog.data.gov/dataset/pioneer-10-hourly-merged-magnetic-field-and-plasma-data",
        "nasa_page": "https://data.nasa.gov/dataset/pioneer-10-hourly-merged-magnetic-field-and-plasma-data",
        "doi": "https://doi.org/10.48322/dn2j-fn46",
        "dataset_id": "P10_COHO1HR_MERGED_MAG_PLASMA",
        "base": PIONEER10_BASE,
        "label": "10",
    },
    "p11": {
        "title": "Pioneer 11 hourly merged magnetic field and plasma data",
        "catalog_page": "https://catalog.data.gov/dataset/pioneer-11-hourly-merged-magnetic-field-and-plasma-data",
        "nasa_page": "https://data.nasa.gov/dataset/pioneer-11-hourly-merged-magnetic-field-and-plasma-data",
        "doi": "https://doi.org/10.48322/vb0s-0w02",
        "dataset_id": "P11_COHO1HR_MERGED_MAG_PLASMA",
        "base": PIONEER11_BASE,
        "label": "11",
    },
}

PIONEER_PPI_PRODUCTS = {
    "p10_jupiter": {
        "spacecraft": "p10",
        "label": "Pioneer 10 Jupiter encounter merged hourly",
        "product": "encounter_hourly_ppi",
        "root": "https://pds-ppi.igpp.ucla.edu/data/P10-J-HVM_PA-4-SUMM-MERGED-1HR-V1.0/",
        "subdir": OUTPUT_DIR / "pioneer10" / "jupiter_encounter_ppi",
        "files": [
            "AAREADME.TXT",
            "CHECKSUMS.TXT",
            "ERRATA.TXT",
            "VOLDESC.CAT",
            "DATA/INFO.TXT",
            "DATA/P10_JUP_HVM_PA_1HR_MERGED.TAB",
            "DATA/p10_mgd.txt",
        ],
    },
    "p11_jupiter": {
        "spacecraft": "p11",
        "label": "Pioneer 11 Jupiter encounter merged hourly",
        "product": "encounter_hourly_ppi",
        "root": "https://pds-ppi.igpp.ucla.edu/data/P11-J-HVM-PA-4-SUMM-MERGED-1HR-V1.0/",
        "subdir": OUTPUT_DIR / "pioneer11" / "jupiter_encounter_ppi",
        "files": [
            "AAREADME.TXT",
            "CHECKSUMS.TXT",
            "ERRATA.TXT",
            "VOLDESC.CAT",
            "DATA/INFO.TXT",
            "DATA/P11_JUP_HVM_PLS_1HR_MERGED.TAB",
            "DATA/p11mgd.txt",
        ],
    },
    "p11_saturn": {
        "spacecraft": "p11",
        "label": "Pioneer 11 Saturn encounter merged hourly",
        "product": "encounter_hourly_ppi",
        "root": "https://pds-ppi.igpp.ucla.edu/data/P11-S-HVM-PA-4-MERGED-ENC-1HR-V1.0/",
        "subdir": OUTPUT_DIR / "pioneer11" / "saturn_encounter_ppi",
        "files": [
            "AAREADME.TXT",
            "CHECKSUMS.TXT",
            "ERRATA.TXT",
            "VOLDESC.CAT",
            "DATA/INFO.TXT",
            "DATA/P11_SAT_HVM_PA_1HR_MERGED.TAB",
            "DATA/p11mgd.txt",
        ],
    },
}

PIONEER_PPI_MERGED = {
    "p10": {
        "root": "https://pds-ppi.igpp.ucla.edu/data/p10-hvm-pa-crt-cal/data-merged-1hour/",
        "subdir": OUTPUT_DIR / "pioneer10" / "pds_ppi_merged",
        "prefix": "p10",
        "metadata_files": [
            "collection-data-hvm-pa-crt-1hour-1.0.csv",
            "collection-data-hvm-pa-crt-1hour-1.0.xml",
        ],
    },
    "p11": {
        "root": "https://pds-ppi.igpp.ucla.edu/data/p11-hvm-pa-crt-cal/data-merged-1hour/",
        "subdir": OUTPUT_DIR / "pioneer11" / "pds_ppi_merged",
        "prefix": "p11",
        "metadata_files": [
            "collection-data-hvm-pa-crt-1hour-1.0.csv",
            "collection-data-hvm-pa-crt-1hour-1.0.xml",
        ],
    },
}


AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"

# AMDA Pioneer MAG-only datasets. NO plasma data exists for Pioneer in AMDA.
# Orbit data limited to Jupiter encounters (p10-orb-jup, p11-orb-jup).
AMDA_PIONEER_DATASETS = {
    "p10": {"mag": "p10-mag-full"},
    "p11": {"mag": "p11-mag-full"},
}

# Fill values for Pioneer MAG-only output (matching pioneer.rs constants).
PIONEER_FILL_B = 9999.99
PIONEER_FILL_DENSITY = 999.9
PIONEER_FILL_SPEED = 9999.9
PIONEER_FILL_TEMP = 999999.0
PIONEER_FILL_DISTANCE = 999.999
PIONEER_FILL_LATLON = 999.99


def build_file_entry(url: str, path: Path, status: str, *, source: str) -> dict[str, object]:
    entry: dict[str, object] = {
        "url": url,
        "path": str(path),
        "status": status,
        "source": source,
    }
    if not path.exists():
        return entry

    data = path.read_bytes()
    entry["filename"] = path.name
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    if path.suffix.lower() in {".asc", ".txt", ".html", ".xml", ".json"}:
        entry["lines"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    return entry


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_text(url: str, *, timeout: int = 60) -> str:
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def fetch_file(url: str, out: Path, *, skip_existing: bool) -> dict[str, object]:
    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped", source="spdf")

    try:
        data = fetch_bytes(url)
    except (URLError, OSError) as exc:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": str(exc),
        }

    if len(data) < 100:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": f"unexpectedly_small:{len(data)}",
        }

    out.write_bytes(data)
    entry = build_file_entry(url, out, "fetched", source="spdf")
    print(f"  {out.name}: OK ({entry.get('lines', 0)} lines, {entry['bytes']} bytes) [SPDF]")
    return entry


def fetch_catalog_package(title: str) -> dict[str, object] | None:
    query = urlencode({"q": f'"{title}"'})
    url = f"{CATALOG_SEARCH}?{query}"
    try:
        payload = json.loads(fetch_text(url, timeout=30))
    except (URLError, OSError, json.JSONDecodeError):
        return None
    results = payload.get("result", {}).get("results", [])
    return results[0] if results else None


def fetch_metadata_trace(spacecraft: str, subdir: Path) -> dict[str, object]:
    meta = dict(PIONEER_METADATA[spacecraft])
    package = fetch_catalog_package(meta["title"])
    meta["catalog_package"] = package

    meta_dir = subdir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sidecars: list[dict[str, object]] = []
    for key, suffix in [
        ("catalog_page", "catalog.html"),
        ("nasa_page", "nasa.html"),
        ("doi", "doi.html"),
    ]:
        url = meta[key]
        out = meta_dir / f"{spacecraft}_{suffix}"
        try:
            data = fetch_bytes(url, timeout=30)
            out.write_bytes(data)
            sidecars.append(build_file_entry(url, out, "fetched", source="metadata"))
        except (URLError, OSError) as exc:
            sidecars.append(
                {
                    "url": url,
                    "path": str(out),
                    "status": "failed",
                    "source": "metadata",
                    "reason": str(exc),
                }
            )
    meta["sidecars"] = sidecars
    return meta


def write_manifest(
    spacecraft: str,
    start: int,
    end: int,
    payload: dict[str, object],
) -> Path:
    manifest_path = OUTPUT_DIR / f"{spacecraft}_{start}_{end}_manifest.json"
    manifest = {
        "product": "pioneer_merged_hourly",
        "spacecraft": spacecraft,
        "start_year": start,
        "end_year": end,
        **payload,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")
    return manifest_path


def fetch_amda_info(dataset_id: str) -> dict[str, object]:
    url = f"{AMDA_HAPI}/info?id={dataset_id}"
    return json.loads(fetch_text(url, timeout=60))


def fetch_amda_csv(dataset_id: str, start: str, end: str, *, timeout: int = 120) -> str:
    url = f"{AMDA_HAPI}/data?id={dataset_id}&time.min={start}&time.max={end}&format=csv"
    return fetch_text(url, timeout=timeout)


def parse_hapi_time(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(dt.timezone.utc)


def format_hapi_time(timestamp: dt.datetime) -> str:
    return timestamp.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    if value <= -1.0e30:
        return float("nan")
    return value


def floor_to_hour(timestamp: dt.datetime) -> dt.datetime:
    return timestamp.replace(minute=0, second=0, microsecond=0)


def fmt_or_fill(value: float, fill: float, decimals: int) -> str:
    if not math.isfinite(value):
        return f"{fill:.{decimals}f}"
    return f"{value:.{decimals}f}"


def median_or_nan(values: list[float]) -> float:
    valid = [v for v in values if math.isfinite(v)]
    if not valid:
        return float("nan")
    return float(statistics.median(valid))


def parse_pioneer_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse AMDA p10-mag-full or p11-mag-full CSV.

    Expected columns: timestamp, Br, Bt, Bn, |B| (RTN, hourly).
    Pioneer MAG data in AMDA is already hourly, but we aggregate to be safe.
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"br": [], "bt": [], "bn": [], "bmag": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < 5:
            continue
        bucket = floor_to_hour(parse_hapi_time(fields[0]))
        by_hour[bucket]["br"].append(parse_float(fields[1]))
        by_hour[bucket]["bt"].append(parse_float(fields[2]))
        by_hour[bucket]["bn"].append(parse_float(fields[3]))
        by_hour[bucket]["bmag"].append(parse_float(fields[4]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        br = median_or_nan(accum["br"])
        bt = median_or_nan(accum["bt"])
        bn = median_or_nan(accum["bn"])
        bmag = median_or_nan(accum["bmag"])
        if not math.isfinite(bmag) and all(math.isfinite(v) for v in (br, bt, bn)):
            bmag = math.sqrt(br * br + bt * bt + bn * bn)
        out[bucket] = {"br": br, "bt": bt, "bn": bn, "bmag": bmag}
    return out


def fetch_pioneer_amda_mag(
    spacecraft: str, years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Pioneer MAG-only data from AMDA HAPI.

    AMDA has MAG but NO plasma for Pioneer. Output is 13-column SPDF-style
    with NaN fill for density, speed, temperature, and position columns.
    """
    datasets = AMDA_PIONEER_DATASETS[spacecraft]
    label = PIONEER_METADATA[spacecraft]["label"]
    subdir = OUTPUT_DIR / f"pioneer{label}"
    subdir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    try:
        mag_info = fetch_amda_info(datasets["mag"])
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        for year in years:
            out = subdir / f"p{label}_{year}_amda_mag_only.asc"
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "source": "amda",
                    "year": year,
                    "reason": f"amda_info_unreachable:{exc}",
                }
            )
            counts["failed"] += 1
        return {
            "status": "metadata_only",
            "reason": "amda_info_unreachable",
            "counts": counts,
            "files": files,
            "metadata": {"amda": {"datasets": datasets, "info_error": str(exc)}},
        }

    mag_start = parse_hapi_time(str(mag_info["startDate"]))
    mag_stop = parse_hapi_time(str(mag_info["stopDate"]))

    for year in years:
        out = subdir / f"p{label}_{year}_amda_mag_only.asc"
        if skip_existing and out.exists():
            entry = build_file_entry("amda://derived", out, "skipped", source="amda")
            entry["year"] = year
            files.append(entry)
            counts["skipped"] += 1
            continue

        requested_start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        requested_end = dt.datetime(year, 12, 31, 23, 0, 0, tzinfo=dt.timezone.utc)
        start_dt = max(requested_start, mag_start)
        end_dt = min(requested_end, mag_stop)
        if start_dt > end_dt:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "source": "amda",
                    "year": year,
                    "reason": "amda_year_outside_dataset_range",
                }
            )
            counts["failed"] += 1
            continue

        start = format_hapi_time(start_dt)
        end = format_hapi_time(end_dt)
        try:
            mag_rows = parse_pioneer_mag_rows(
                fetch_amda_csv(datasets["mag"], start, end, timeout=120)
            )
        except (URLError, OSError, TimeoutError) as exc:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "source": "amda",
                    "year": year,
                    "reason": f"amda_unreachable:{exc}",
                }
            )
            counts["failed"] += 1
            continue

        lines: list[str] = []
        for bucket in sorted(mag_rows):
            if bucket.year != year:
                continue
            mag = mag_rows[bucket]
            # MAG-only output: position and plasma columns are NaN fill.
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{bucket.timetuple().tm_yday:03d}",
                    f"{bucket.hour:02d}",
                    fmt_or_fill(float("nan"), PIONEER_FILL_DISTANCE, 3),
                    fmt_or_fill(float("nan"), PIONEER_FILL_LATLON, 2),
                    fmt_or_fill(float("nan"), PIONEER_FILL_LATLON, 2),
                    fmt_or_fill(mag["bmag"], PIONEER_FILL_B, 2),
                    fmt_or_fill(mag["br"], PIONEER_FILL_B, 2),
                    fmt_or_fill(mag["bt"], PIONEER_FILL_B, 2),
                    fmt_or_fill(mag["bn"], PIONEER_FILL_B, 2),
                    fmt_or_fill(float("nan"), PIONEER_FILL_DENSITY, 1),
                    fmt_or_fill(float("nan"), PIONEER_FILL_SPEED, 1),
                    fmt_or_fill(float("nan"), PIONEER_FILL_TEMP, 1),
                ]
            )
            lines.append(line)

        if not lines:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "source": "amda",
                    "year": year,
                    "reason": "empty_amda_translation",
                }
            )
            counts["failed"] += 1
            continue

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entry = build_file_entry("amda://derived", out, "fetched", source="amda")
        entry["year"] = year
        entry["records"] = len(lines)
        entry["mag_provenance"] = "measured_amda_mag_full_rtn"
        entry["plasma_provenance"] = "unavailable_no_amda_pioneer_plasma"
        entry["position_provenance"] = "unavailable_no_cruise_orbit"
        entry["blocker_note"] = (
            "MAG-ONLY: AMDA has no Pioneer plasma datasets. "
            "Density, speed, temperature are NaN fill. "
            "Position unavailable (AMDA orbit limited to Jupiter encounter)."
        )
        files.append(entry)
        counts["fetched"] += 1
        print(f"  {out.name}: AMDA MAG-only OK ({len(lines)} rows)")

    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "metadata_only"
    return {
        "status": status,
        "counts": counts,
        "files": files,
        "metadata": {
            "amda": {
                "hapi_base": AMDA_HAPI,
                "datasets": datasets,
                "dataset_bounds": {
                    "mag": {
                        "start": str(mag_info["startDate"]),
                        "stop": str(mag_info["stopDate"]),
                    }
                },
                "derived_lane": {
                    "magnetic_field": "measured_mag_full_rtn",
                    "plasma": "unavailable",
                    "trajectory": "unavailable_cruise",
                },
            }
        },
    }


def fetch_pioneer_ppi_merged(
    spacecraft: str, years: range, *, skip_existing: bool
) -> dict[str, object]:
    product = PIONEER_PPI_MERGED[spacecraft]
    subdir = product["subdir"]
    subdir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    for rel_path in product["metadata_files"]:
        url = f"{product['root']}{rel_path}"
        out = subdir / rel_path
        entry = fetch_file(url, out, skip_existing=skip_existing)
        if entry["status"] in {"fetched", "skipped"}:
            entry["source"] = "pds_ppi"
        counts[str(entry["status"])] += 1
        files.append(entry)

    for year in years:
        for suffix in [".TAB", ".xml"]:
            fname = f"{product['prefix']}_{year}{suffix}"
            url = f"{product['root']}{fname}"
            out = subdir / fname
            entry = fetch_file(url, out, skip_existing=skip_existing)
            if entry["status"] in {"fetched", "skipped"}:
                entry["source"] = "pds_ppi"
                entry["year"] = year
            counts[str(entry["status"])] += 1
            files.append(entry)

    staged_bytes = counts["fetched"] > 0 or counts["skipped"] > 0
    manifest_path = write_manifest(
        spacecraft,
        years.start,
        years.stop - 1,
        {
            "status": "ready" if staged_bytes else "metadata_only",
            "counts": counts,
            "files": files,
            "metadata": {
                "annual_source_root": product["root"],
                "annual_source_kind": "pds_ppi_merged_hourly",
            },
            "notes": (
                "UCLA PDS/PPI annual merged hourly Pioneer lane. These annual "
                "TAB files are governed alternate science bytes for the merged "
                "hourly product family and are distinct from the narrower "
                "encounter bundles."
            ),
        },
    )
    return {
        "counts": counts,
        "status": "ready" if staged_bytes else "metadata_only",
        "manifest": str(manifest_path),
    }


def fetch_spacecraft(
    spacecraft: str, years: range, *, source: str, skip_existing: bool
) -> dict[str, object]:
    meta = PIONEER_METADATA[spacecraft]
    subdir = OUTPUT_DIR / ("pioneer10" if spacecraft == "p10" else "pioneer11")
    subdir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    if source == "amda":
        return fetch_pioneer_amda_mag(spacecraft, years, skip_existing=skip_existing)
    if source == "pds_ppi":
        return fetch_pioneer_ppi_merged(spacecraft, years, skip_existing=skip_existing)

    if source in {"auto", "spdf"}:
        for year in years:
            fname = f"p{meta['label']}_{year}_merged_hourly.asc"
            url = f"{meta['base']}{fname}"
            out = subdir / fname
            entry = fetch_file(url, out, skip_existing=skip_existing)
            counts[str(entry["status"])] += 1
            files.append(entry)

    staged_bytes = counts["fetched"] > 0 or counts["skipped"] > 0

    # Auto fallback: if SPDF failed, try AMDA MAG-only
    if not staged_bytes and source == "auto":
        print(f"  SPDF failed for Pioneer {spacecraft}, trying UCLA PDS/PPI annual merged...")
        ppi_result = fetch_pioneer_ppi_merged(spacecraft, years, skip_existing=skip_existing)
        if ppi_result["status"] == "ready":
            return ppi_result
        print(f"  SPDF failed for Pioneer {spacecraft}, trying AMDA MAG-only...")
        return fetch_pioneer_amda_mag(spacecraft, years, skip_existing=skip_existing)

    metadata = fetch_metadata_trace(spacecraft, subdir)
    status = "ready" if staged_bytes else "metadata_only"
    if not staged_bytes and source == "spdf":
        status = "metadata_only"

    manifest_path = write_manifest(
        spacecraft,
        years.start,
        years.stop - 1,
        {
            "status": status,
            "counts": counts,
            "files": files,
            "metadata": metadata,
            "notes": (
                "Canonical Pioneer merged hourly bytes live behind GSFC/CDAWeb/COHOWeb. "
                "On this host, metadata pages are browser-reachable with a Mozilla-style "
                "user agent even when direct GSFC byte serving is blocked."
            ),
        },
    )
    return {"counts": counts, "status": status, "manifest": str(manifest_path)}


def fetch_ppi_product(product_key: str, *, skip_existing: bool) -> dict[str, object]:
    product = PIONEER_PPI_PRODUCTS[product_key]
    subdir = product["subdir"]
    subdir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []
    for rel_path in product["files"]:
        url = f"{product['root']}{rel_path}"
        out = subdir / rel_path
        out.parent.mkdir(parents=True, exist_ok=True)
        entry = fetch_file(url, out, skip_existing=skip_existing)
        if entry["status"] in {"fetched", "skipped"}:
            entry["source"] = "pds_ppi"
        counts[str(entry["status"])] += 1
        files.append(entry)

    manifest_path = subdir / f"{product_key}_manifest.json"
    manifest = {
        "product": product["product"],
        "product_key": product_key,
        "spacecraft": product["spacecraft"],
        "status": "ready" if counts["failed"] == 0 else "partial",
        "counts": counts,
        "files": files,
        "source_root": product["root"],
        "notes": (
            "These reachable UCLA PDS/PPI bundles are encounter-window subsets "
            "of the original NSSDC Pioneer merged products. They are governed "
            "science bytes, but they are not interchangeable with the full "
            "annual COHO/NSSDC merged hourly lane."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")
    return {"counts": counts, "status": manifest["status"], "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed Pioneer 10 & 11 merged hourly and encounter data."
    )
    parser.add_argument(
        "--spacecraft",
        choices=["p10", "p11", "both"],
        default="p10",
        help="Which spacecraft (default: p10)",
    )
    parser.add_argument(
        "--start", type=int, default=1979, help="Start year (inclusive, default 1979)"
    )
    parser.add_argument("--end", type=int, default=1979, help="End year (inclusive, default 1979)")
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "amda", "metadata", "pds_ppi"],
        default="auto",
        help="Preferred source ladder (default: auto). AMDA provides MAG-only data.",
    )
    parser.add_argument(
        "--product",
        choices=["merged_hourly", "encounter_hourly_ppi"],
        default="merged_hourly",
        help="Which Pioneer product family to fetch (default: merged_hourly)",
    )
    parser.add_argument(
        "--encounter",
        choices=["p10_jupiter", "p11_jupiter", "p11_saturn", "all"],
        default="all",
        help="Encounter product when --product=encounter_hourly_ppi (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip direct byte download if the yearly file already exists",
    )
    args = parser.parse_args()

    if args.end < args.start:
        print("--end must be >= --start", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = {"fetched": 0, "skipped": 0, "failed": 0}
    statuses: list[str] = []
    if args.product == "merged_hourly":
        years = range(args.start, args.end + 1)
        targets = ["p10", "p11"] if args.spacecraft == "both" else [args.spacecraft]
        for spacecraft in targets:
            print(f"Fetching Pioneer {spacecraft.upper()} {args.start}-{args.end}...")
            result = fetch_spacecraft(
                spacecraft,
                years,
                source=args.source,
                skip_existing=args.skip_existing,
            )
            for key in totals:
                totals[key] += result["counts"][key]
            statuses.append(result["status"])
    else:
        targets = list(PIONEER_PPI_PRODUCTS.keys()) if args.encounter == "all" else [args.encounter]
        for product_key in targets:
            print(f"Fetching Pioneer encounter {product_key}...")
            result = fetch_ppi_product(product_key, skip_existing=args.skip_existing)
            for key in totals:
                totals[key] += result["counts"][key]
            statuses.append(result["status"])

    print()
    print(
        f"Pioneer fetch: {totals['fetched']} fetched, "
        f"{totals['skipped']} skipped, {totals['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if all(status == "metadata_only" for status in statuses) else 0


if __name__ == "__main__":
    sys.exit(main())
