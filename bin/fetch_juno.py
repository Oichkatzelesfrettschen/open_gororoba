#!/usr/bin/env python3
"""Fetch Juno cruise phase merged hourly data from NASA SPDF or AMDA HAPI fallback.

Juno JADE/MAG cruise data covers 1-5 AU (2011-2016) before Jupiter orbit insertion.

Instruments:
  JADE: Jovian Auroral Distributions Experiment (McComas et al. 2017)
  MAG:  Fluxgate Magnetometer (Connerney et al. 2017)

SPDF source: https://spdf.gsfc.nasa.gov/pub/data/juno/
AMDA datasets (fallback when SPDF blocked):
  - juno-jadel5-protmom : JADE-L5 proton moments (density, speed, temperature)
  - juno-fgm-cruise60  : FGM 1-minute cruise MAG
  - juno-cruise-all    : Cruise phase ephemeris (r_au, lat, lon)

Output is 13-column SPDF-style merged format matching JUNO_CRUISE_LAYOUT:
    YYYY DOY HH r_au lat lon |B| Br Bt Bn density speed temperature
NOTE: |B| is BEFORE B-components (col 6), unlike Ulysses/Voyager where it follows.

B-field coordinate system: SE (Solar Ecliptic) during cruise phase.

Usage:
    python3 bin/fetch_juno.py                                    # SPDF 2011-2016
    python3 bin/fetch_juno.py --start 2013 --end 2015
    python3 bin/fetch_juno.py --source amda --skip-existing
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

JUNO_CRUISE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/juno/merged/"
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"
OUTPUT_DIR = Path("data/external/juno")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# AMDA dataset IDs for Juno cruise phase.
AMDA_DATASETS = {
    "orbit": "juno-cruise-all",
    "mag": "juno-fgm-cruise60",
    "plasma": "juno-jadel5-protmom",
}

# Fill values matching JUNO_CRUISE_LAYOUT in juno.rs.
FILL_B = 9999.99
FILL_DENSITY = 999.9
FILL_SPEED = 9999.9
FILL_TEMP = 999999.0
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
    """Parse juno-cruise-all CSV -> hourly-median orbit dict.

    Expected columns: timestamp, r_au, lat_deg, lon_deg
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
    """Parse juno-fgm-cruise60 CSV -> hourly-median MAG dict.

    Expected columns: timestamp, Bx, By, Bz, |B| (SE coordinates during cruise).
    FGM cruise data is 1-minute cadence, aggregated to hourly medians.
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"bx": [], "by": [], "bz": [], "bmag": []}
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
        by_hour[bucket]["bx"].append(parse_float(parts[1]))
        by_hour[bucket]["by"].append(parse_float(parts[2]))
        by_hour[bucket]["bz"].append(parse_float(parts[3]))
        by_hour[bucket]["bmag"].append(parse_float(parts[4]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "bx": median_or_nan(accum["bx"]),
            "by": median_or_nan(accum["by"]),
            "bz": median_or_nan(accum["bz"]),
            "bmag": median_or_nan(accum["bmag"]),
        }
    return out


def parse_plasma_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse juno-jadel5-protmom CSV -> hourly-median plasma dict.

    Expected columns: timestamp, density (cm^-3), speed (km/s), temperature (K or eV).
    JADE-L5 provides Level 5 proton moments.
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


def fetch_juno_spdf(
    years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Juno cruise merged hourly from SPDF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    for year in years:
        fname = f"juno_{year}_merged_hourly.asc"
        url = f"{JUNO_CRUISE_BASE}{fname}"
        out = OUTPUT_DIR / fname
        status = fetch_spdf_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        entry = build_file_entry(url, out, status, source="spdf")
        entry["year"] = year
        files.append(entry)
        if status == "failed":
            print(f"  Juno {year}: SPDF FAILED")

    return {
        "status": "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "failed",
        "counts": counts,
        "files": files,
    }


# ---------------------------------------------------------------------------
# AMDA fetcher
# ---------------------------------------------------------------------------

def fetch_juno_amda(
    years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Juno cruise merged hourly from AMDA HAPI.

    Combines juno-jadel5-protmom + juno-fgm-cruise60 + juno-cruise-all.
    Output: 13-column SPDF-style matching JUNO_CRUISE_LAYOUT column order:
        YYYY DOY HH r_au lat lon |B| Bx By Bz density speed temperature
    NOTE: |B| is column 6 (BEFORE B-components), b_is_se=true.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    # Query AMDA info endpoints.
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
        out = OUTPUT_DIR / f"juno_{year}_amda_merged_hourly.asc"
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
            # Column order matches JUNO_CRUISE_LAYOUT:
            # year doy hour r_au lat lon |B| Bx By Bz density speed temperature
            # NOTE: |B| at col 6 BEFORE B-components (Juno convention).
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{bucket.timetuple().tm_yday:03d}",
                    f"{bucket.hour:02d}",
                    fmt_or_fill(orbit["r_au"], FILL_DISTANCE, 3),
                    fmt_or_fill(orbit["lat_deg"], FILL_LATLON, 2),
                    fmt_or_fill(orbit["lon_deg"], FILL_LATLON, 2),
                    fmt_or_fill(mag.get("bmag", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bx", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("by", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bz", float("nan")), FILL_B, 2),
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
        entry["plasma_provenance"] = "measured_jade_l5_proton"
        entry["mag_provenance"] = "measured_fgm_cruise"
        entry["orbit_provenance"] = "measured_amda_cruise_orb"
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
        description="Fetch Juno cruise merged hourly data from NASA SPDF or AMDA HAPI."
    )
    parser.add_argument(
        "--start", type=int, default=2011, help="Start year (inclusive, default 2011)"
    )
    parser.add_argument(
        "--end", type=int, default=2016, help="End year (inclusive, default 2016)"
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
    print(f"Fetching Juno cruise hourly {args.start}-{args.end} [source={source}]...")

    if source == "amda":
        result = fetch_juno_amda(years, skip_existing=args.skip_existing)
    elif source == "spdf":
        result = fetch_juno_spdf(years, skip_existing=args.skip_existing)
    else:
        # Auto: SPDF -> AMDA fallback.
        result = fetch_juno_spdf(years, skip_existing=args.skip_existing)
        counts = result.get("counts", {})
        if isinstance(counts, dict) and counts.get("fetched", 0) == 0 and counts.get("skipped", 0) == 0:
            print("  SPDF failed, falling back to AMDA...")
            result = fetch_juno_amda(years, skip_existing=args.skip_existing)

    counts = result.get("counts", {})
    if isinstance(counts, dict):
        print()
        print(
            f"Juno fetch: {counts.get('fetched', 0)} fetched, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('failed', 0)} failed"
        )
    print(f"Output directory: {OUTPUT_DIR}")
    fetched = counts.get("fetched", 0) if isinstance(counts, dict) else 0
    skipped = counts.get("skipped", 0) if isinstance(counts, dict) else 0
    return 1 if fetched == 0 and skipped == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
