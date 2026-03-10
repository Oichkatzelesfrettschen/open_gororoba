#!/usr/bin/env python3
"""Fetch Ulysses merged hourly data from NASA SPDF or AMDA HAPI fallback.

Ulysses is the ONLY spacecraft with polar solar wind measurements,
providing fast latitude scans from -80 to +80 deg heliographic.
Range: 1.3-5.4 AU. Operated 1990-2009.

Key science periods:
  First fast latitude scan:  1994-1995 (solar minimum)
  Second fast latitude scan: 2000-2001 (solar maximum)
  Third fast latitude scan:  2007-2008 (deep solar minimum)

SPDF source: https://spdf.gsfc.nasa.gov/pub/data/ulysses/
AMDA datasets (fallback when SPDF blocked):
  - ulys-bai-mom  : SWOOPS proton moments (density, speed, temperature)
  - ulys-fgm-rtn  : VHM/FGM magnetic field in RTN coordinates
  - ulys-orb-all  : Heliocentric orbit (r_au, lat_deg, lon_deg)

Sub-hourly cadence AMDA data aggregated to per-hour medians.
Output is 13-column SPDF-style merged format:
    YYYY DOY HH r_au lat lon Br Bt Bn |B| density speed temperature

Usage:
    python3 bin/fetch_ulysses.py                                  # SPDF 1994-1995
    python3 bin/fetch_ulysses.py --start 1990 --end 2009          # full mission
    python3 bin/fetch_ulysses.py --source amda --start 1994 --end 1995
    python3 bin/fetch_ulysses.py --skip-existing
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
from urllib.request import Request, urlopen

ULYSSES_MERGED_BASE = "https://spdf.gsfc.nasa.gov/pub/data/ulysses/merged/"
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"
OUTPUT_DIR = Path("data/external/ulysses")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# AMDA dataset IDs for Ulysses.
AMDA_DATASETS = {
    "orbit": "ulys-orb-all",
    "mag": "ulys-fgm-rtn",
    "plasma": "ulys-bai-mom",
}

# Fill values matching ULYSSES_LAYOUT in ulysses.rs.
FILL_B = 9999.9
FILL_DENSITY = 9999.9
FILL_SPEED = 9999.9
FILL_TEMP = 9999999.0
FILL_DISTANCE = 999.999
FILL_LATLON = 999.99


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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
    entry["lines"] = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    return entry


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_text(url: str, *, timeout: int = 60) -> str:
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# AMDA HAPI helpers
# ---------------------------------------------------------------------------

def fetch_amda_info(dataset_id: str) -> dict[str, object]:
    return json.loads(fetch_text(f"{AMDA_HAPI}/info?id={dataset_id}", timeout=60))


def fetch_amda_csv(dataset_id: str, start: str, end: str, *, timeout: int) -> str:
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


def normalize_lon_deg(lon_deg: float) -> float:
    if not math.isfinite(lon_deg):
        return float("nan")
    return lon_deg % 360.0


def fmt_or_fill(value: float, fill: float, decimals: int) -> str:
    if not math.isfinite(value):
        return f"{fill:.{decimals}f}"
    return f"{value:.{decimals}f}"


def median_or_nan(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value)]
    if not valid:
        return float("nan")
    return float(statistics.median(valid))


# ---------------------------------------------------------------------------
# AMDA row parsers (sub-hourly -> hourly median buckets)
# ---------------------------------------------------------------------------

def parse_orbit_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse ulys-orb-all CSV -> hourly-median orbit dict.

    Expected columns: timestamp, r_au, lat_hgi_deg, lon_hgi_deg
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"r_au": [], "lat_deg": [], "lon_deg": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ts = parse_hapi_time(parts[0].strip())
        except (ValueError, IndexError):
            continue
        bucket = floor_to_hour(ts)
        by_hour[bucket]["r_au"].append(parse_float(parts[1]))
        by_hour[bucket]["lat_deg"].append(parse_float(parts[2]))
        by_hour[bucket]["lon_deg"].append(parse_float(parts[3]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "r_au": median_or_nan(accum["r_au"]),
            "lat_deg": median_or_nan(accum["lat_deg"]),
            "lon_deg": normalize_lon_deg(median_or_nan(accum["lon_deg"])),
        }
    return out


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse ulys-fgm-rtn CSV -> hourly-median MAG dict.

    Expected columns: timestamp, Br, Bt, Bn, |B| (RTN coordinates).
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"br": [], "bt": [], "bn": [], "bmag": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            ts = parse_hapi_time(parts[0].strip())
        except (ValueError, IndexError):
            continue
        bucket = floor_to_hour(ts)
        by_hour[bucket]["br"].append(parse_float(parts[1]))
        by_hour[bucket]["bt"].append(parse_float(parts[2]))
        by_hour[bucket]["bn"].append(parse_float(parts[3]))
        by_hour[bucket]["bmag"].append(parse_float(parts[4]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "br": median_or_nan(accum["br"]),
            "bt": median_or_nan(accum["bt"]),
            "bn": median_or_nan(accum["bn"]),
            "bmag": median_or_nan(accum["bmag"]),
        }
    return out


def parse_plasma_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse ulys-bai-mom CSV -> hourly-median plasma dict.

    Expected columns: timestamp, density (cm^-3), speed (km/s), temperature (K).
    SWOOPS/BAI provides temperature directly (no thermal speed conversion needed).
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"density": [], "speed": [], "temp_k": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            ts = parse_hapi_time(parts[0].strip())
        except (ValueError, IndexError):
            continue
        bucket = floor_to_hour(ts)
        by_hour[bucket]["density"].append(parse_float(parts[1]))
        by_hour[bucket]["speed"].append(parse_float(parts[2]))
        by_hour[bucket]["temp_k"].append(parse_float(parts[3]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "density": median_or_nan(accum["density"]),
            "speed": median_or_nan(accum["speed"]),
            "temp_k": median_or_nan(accum["temp_k"]),
        }
    return out


# ---------------------------------------------------------------------------
# SPDF fetcher (original path)
# ---------------------------------------------------------------------------

def fetch_spdf_file(url: str, out: Path, *, skip_existing: bool) -> str:
    """Fetch a single SPDF file. Returns status string."""
    if skip_existing and out.exists():
        return "skipped"

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=60) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, OSError):
        return "failed"

    if len(data) < 100:
        return "failed"

    out.write_bytes(data)
    lines = sum(1 for _ in out.open())
    size = out.stat().st_size
    print(f"  {out.name}: SPDF OK ({lines} lines, {size} bytes)")
    return "fetched"


def fetch_ulysses_spdf(
    years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Ulysses merged hourly from SPDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    for year in years:
        fname = f"uly_{year}_merged_hourly.asc"
        url = f"{ULYSSES_MERGED_BASE}{fname}"
        out = OUTPUT_DIR / fname
        status = fetch_spdf_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        entry = build_file_entry(url, out, status, source="spdf")
        entry["year"] = year
        files.append(entry)
        if status == "failed":
            print(f"  Ulysses {year}: SPDF FAILED")

    return {
        "status": "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "failed",
        "counts": counts,
        "files": files,
    }


# ---------------------------------------------------------------------------
# AMDA fetcher
# ---------------------------------------------------------------------------

def fetch_ulysses_amda(
    years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Ulysses merged hourly from AMDA HAPI (fallback lane).

    Combines ulys-bai-mom (plasma) + ulys-fgm-rtn (MAG) + ulys-orb-all (orbit).
    Output: 13-column SPDF-style ASCII matching ULYSSES_LAYOUT column order:
        YYYY DOY HH r_au lat lon Br Bt Bn |B| density speed temperature
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    # Query AMDA info endpoints to get dataset time bounds.
    dataset_info: dict[str, dict[str, object]] = {}
    for key, dataset_id in AMDA_DATASETS.items():
        try:
            dataset_info[key] = fetch_amda_info(dataset_id)
        except (URLError, OSError) as exc:
            return {
                "status": "failed",
                "counts": counts,
                "files": files,
                "error": f"amda_info_unreachable:{dataset_id}:{exc}",
            }

    dataset_bounds = {
        key: {
            "start": info["startDate"],
            "stop": info["stopDate"],
        }
        for key, info in dataset_info.items()
    }

    overlap_start = max(parse_hapi_time(bounds["start"]) for bounds in dataset_bounds.values())
    overlap_end = min(parse_hapi_time(bounds["stop"]) for bounds in dataset_bounds.values())

    for year in years:
        out = OUTPUT_DIR / f"uly_{year}_amda_merged_hourly.asc"
        if skip_existing and out.exists():
            entry = build_file_entry("amda://derived", out, "skipped", source="amda")
            entry["year"] = year
            files.append(entry)
            counts["skipped"] += 1
            continue

        requested_start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        requested_end = dt.datetime(year, 12, 31, 23, 0, 0, tzinfo=dt.timezone.utc)
        start_dt = max(requested_start, overlap_start)
        end_dt = min(requested_end, overlap_end)
        if start_dt > end_dt:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "source": "amda",
                    "year": year,
                    "reason": "amda_year_outside_dataset_range",
                    "effective_start": format_hapi_time(start_dt),
                    "effective_end": format_hapi_time(end_dt),
                }
            )
            counts["failed"] += 1
            continue

        start = format_hapi_time(start_dt)
        end = format_hapi_time(end_dt)
        try:
            orbit_rows = parse_orbit_rows(
                fetch_amda_csv(AMDA_DATASETS["orbit"], start, end, timeout=120)
            )
            mag_rows = parse_mag_rows(
                fetch_amda_csv(AMDA_DATASETS["mag"], start, end, timeout=300)
            )
            plasma_rows = parse_plasma_rows(
                fetch_amda_csv(AMDA_DATASETS["plasma"], start, end, timeout=120)
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

        # Merge on hours that have orbit + plasma (MAG optional via .get).
        keys = sorted(set(orbit_rows) & set(plasma_rows))
        lines: list[str] = []
        first_bucket: dt.datetime | None = None
        last_bucket: dt.datetime | None = None
        for bucket in keys:
            if bucket.year != year:
                continue
            if first_bucket is None:
                first_bucket = bucket
            last_bucket = bucket
            orbit = orbit_rows[bucket]
            plasma = plasma_rows[bucket]
            mag = mag_rows.get(bucket, {})
            # Column order matches ULYSSES_LAYOUT:
            # year doy hour r_au lat lon Br Bt Bn |B| density speed temperature
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{bucket.timetuple().tm_yday:03d}",
                    f"{bucket.hour:02d}",
                    fmt_or_fill(orbit["r_au"], FILL_DISTANCE, 3),
                    fmt_or_fill(orbit["lat_deg"], FILL_LATLON, 2),
                    fmt_or_fill(orbit["lon_deg"], FILL_LATLON, 2),
                    fmt_or_fill(mag.get("br", float("nan")), FILL_B, 1),
                    fmt_or_fill(mag.get("bt", float("nan")), FILL_B, 1),
                    fmt_or_fill(mag.get("bn", float("nan")), FILL_B, 1),
                    fmt_or_fill(mag.get("bmag", float("nan")), FILL_B, 1),
                    fmt_or_fill(plasma["density"], FILL_DENSITY, 1),
                    fmt_or_fill(plasma["speed"], FILL_SPEED, 1),
                    fmt_or_fill(plasma["temp_k"], FILL_TEMP, 1),
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
                    "effective_start": start,
                    "effective_end": end,
                }
            )
            counts["failed"] += 1
            continue

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entry = build_file_entry("amda://derived", out, "fetched", source="amda")
        entry["year"] = year
        entry["records"] = len(lines)
        entry["datasets"] = dict(AMDA_DATASETS)
        entry["effective_start"] = (
            format_hapi_time(first_bucket) if first_bucket is not None else start
        )
        entry["effective_end"] = (
            format_hapi_time(last_bucket) if last_bucket is not None else end
        )
        entry["plasma_provenance"] = "measured_swoops_bai"
        entry["mag_provenance"] = "measured_vhm_fgm_rtn"
        entry["orbit_provenance"] = "measured_amda_orb"
        files.append(entry)
        counts["fetched"] += 1
        print(f"  {out.name}: AMDA OK ({len(lines)} rows)")

    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "metadata_only"
    return {
        "status": status,
        "counts": counts,
        "files": files,
        "metadata": {
            "amda": {
                "hapi_base": AMDA_HAPI,
                "datasets": AMDA_DATASETS,
                "dataset_bounds": {k: dict(v) for k, v in dataset_bounds.items()},
            },
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Ulysses merged hourly data from NASA SPDF or AMDA HAPI."
    )
    parser.add_argument(
        "--start", type=int, default=1994, help="Start year (inclusive, default 1994)"
    )
    parser.add_argument(
        "--end", type=int, default=1995, help="End year (inclusive, default 1995)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "amda"],
        default="auto",
        help="Data source: auto (SPDF then AMDA), spdf, or amda (default: auto)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    source = args.source
    print(f"Fetching Ulysses merged hourly {args.start}-{args.end} [source={source}]...")

    if source == "amda":
        result = fetch_ulysses_amda(years, skip_existing=args.skip_existing)
    elif source == "spdf":
        result = fetch_ulysses_spdf(years, skip_existing=args.skip_existing)
    else:
        # Auto: try SPDF first, fall back to AMDA on failure.
        result = fetch_ulysses_spdf(years, skip_existing=args.skip_existing)
        counts = result.get("counts", {})
        if isinstance(counts, dict) and counts.get("fetched", 0) == 0 and counts.get("skipped", 0) == 0:
            print("  SPDF failed, falling back to AMDA...")
            result = fetch_ulysses_amda(years, skip_existing=args.skip_existing)

    counts = result.get("counts", {})
    if isinstance(counts, dict):
        print()
        print(
            f"Ulysses fetch: {counts.get('fetched', 0)} fetched, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('failed', 0)} failed"
        )
    print(f"Output directory: {OUTPUT_DIR}")
    fetched = counts.get("fetched", 0) if isinstance(counts, dict) else 0
    skipped = counts.get("skipped", 0) if isinstance(counts, dict) else 0
    return 1 if fetched == 0 and skipped == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
