#!/usr/bin/env python3
"""Fetch WIND SWE (plasma) and MFI (magnetic field) data from NASA SPDF or AMDA HAPI.

WIND SWE key-parameter: yearly files with proton density, bulk speed,
thermal speed, flow direction at ~92-second cadence.
WIND MFI key-parameter: monthly hourly-averaged magnetic field in GSE.

Source (SWE): Ogilvie et al. (1995), Space Sci. Rev. 71, 55
Source (MFI): Lepping et al. (1995), Space Sci. Rev. 71, 207
SPDF (SWE): https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/
SPDF (MFI): https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/

AMDA datasets (fallback when SPDF blocked):
  - wnd-swe-kp : SWE key parameters (density, speed, temperature)
  - wnd-mfi-kp : MFI key parameters (Bx, By, Bz GSE, |B|)

Wind is at L1 (r ~ 1.0 AU, lat ~ 0 deg). Position is hardcoded in AMDA output.
B-field is in GSE (same as MFI native coordinate system).

AMDA output: 13-column SPDF-style merged format (new merged lane):
    YYYY DOY HH r_au lat lon |B| Bx By Bz density speed temperature
Output directory: data/external/wind/ (distinct from split wind_swe/ and wind_mfi/)

Usage:
    python3 bin/fetch_wind.py                                    # SPDF 2024
    python3 bin/fetch_wind.py --start 2020 --end 2024
    python3 bin/fetch_wind.py --source amda --start 2020 --end 2024
    python3 bin/fetch_wind.py --skip-existing
    python3 bin/fetch_wind.py --swe-only
    python3 bin/fetch_wind.py --mfi-only
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

SWE_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/"
MFI_BASE = "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/"
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"

SWE_DIR = Path("data/external/wind_swe")
MFI_DIR = Path("data/external/wind_mfi")
AMDA_DIR = Path("data/external/wind")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# AMDA dataset IDs for Wind.
AMDA_DATASETS = {
    "swe": "wnd-swe-kp",
    "mfi": "wnd-mfi-kp",
}

# Fill values for 13-col AMDA merged output.
FILL_B = 9999.99
FILL_DENSITY = 999.99
FILL_SPEED = 99999.9
FILL_TEMP = 9999999.0
FILL_DISTANCE = 999.999
FILL_LATLON = 999.99

# Wind at L1: fixed position.
WIND_R_AU = 1.0
WIND_LAT_DEG = 0.0


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
# AMDA row parsers
# ---------------------------------------------------------------------------

def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse wnd-mfi-kp CSV -> hourly-median MAG dict.

    Expected columns: timestamp, Bx, By, Bz (GSE), |B|.
    MFI KP is already hourly, but aggregate in case of sub-hourly samples.
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
    """Parse wnd-swe-kp CSV -> hourly-median plasma dict.

    Expected columns: timestamp, density (cm^-3), speed (km/s), temperature (K).
    SWE KP is at ~92s cadence, aggregated to hourly medians.
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
# SPDF fetcher (original paths)
# ---------------------------------------------------------------------------

def fetch_spdf_file(url: str, out: Path, *, skip_existing: bool) -> str:
    """Fetch a single file from SPDF. Returns status string."""
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


def fetch_swe(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch WIND SWE key-parameter yearly files."""
    SWE_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        fname = f"wind_kp_unspike{year}.txt"
        url = f"{SWE_BASE}{fname}"
        out = SWE_DIR / fname
        status = fetch_spdf_file(url, out, skip_existing=skip_existing)
        counts[status] += 1
        if status == "failed":
            print(f"  {fname}: SPDF FAILED")

    return counts


def fetch_mfi(years: range, *, skip_existing: bool) -> dict[str, int]:
    """Fetch WIND MFI key-parameter monthly hourly files."""
    MFI_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for year in years:
        for month in range(1, 13):
            fname = f"{year}{month:02d}_wind_mag_1hour.asc"
            url = f"{MFI_BASE}{fname}"
            out = MFI_DIR / fname
            status = fetch_spdf_file(url, out, skip_existing=skip_existing)
            counts[status] += 1
            if status == "failed":
                print(f"  {fname}: SPDF FAILED")

    return counts


def fetch_wind_spdf(
    years: range, *, skip_existing: bool, swe_only: bool, mfi_only: bool
) -> dict[str, object]:
    """Fetch WIND SWE + MFI from SPDF (original path)."""
    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    do_swe = not mfi_only
    do_mfi = not swe_only

    if do_swe:
        print(f"  Fetching WIND SWE (SPDF)...")
        swe = fetch_swe(years, skip_existing=skip_existing)
        for k in total:
            total[k] += swe[k]

    if do_mfi:
        print(f"  Fetching WIND MFI (SPDF)...")
        mfi = fetch_mfi(years, skip_existing=skip_existing)
        for k in total:
            total[k] += mfi[k]

    return {
        "status": "ready" if total["fetched"] > 0 or total["skipped"] > 0 else "failed",
        "counts": total,
    }


# ---------------------------------------------------------------------------
# AMDA fetcher (merged SWE+MFI lane)
# ---------------------------------------------------------------------------

def fetch_wind_amda(
    years: range, *, skip_existing: bool
) -> dict[str, object]:
    """Fetch Wind merged hourly from AMDA HAPI.

    Combines wnd-swe-kp (plasma) + wnd-mfi-kp (MAG).
    Wind at L1: position hardcoded at r=1.0 AU, lat=0.0 deg.
    Output: 13-column SPDF-style:
        YYYY DOY HH r_au lat lon |B| Bx By Bz density speed temperature
    B-field in GSE (same as MFI native). b_is_se=true equivalent.
    """
    AMDA_DIR.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    # Query AMDA info.
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
        out = AMDA_DIR / f"wind_{year}_amda_merged_hourly.asc"
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
            mag_rows = parse_mag_rows(
                fetch_amda_csv(AMDA_DATASETS["mfi"], start, end, timeout=300)
            )
            plasma_rows = parse_plasma_rows(
                fetch_amda_csv(AMDA_DATASETS["swe"], start, end, timeout=300)
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

        # Merge on hours that have either MAG or plasma (union, not intersection).
        # Wind at L1 has dense coverage; use union to preserve as much as possible.
        all_hours = sorted(set(mag_rows) | set(plasma_rows))
        lines: list[str] = []
        first_bucket: dt.datetime | None = None
        last_bucket: dt.datetime | None = None
        for bucket in all_hours:
            if bucket.year != year:
                continue
            mag = mag_rows.get(bucket, {})
            plasma = plasma_rows.get(bucket, {})
            # Skip hours with neither MAG nor plasma.
            if not mag and not plasma:
                continue
            if first_bucket is None:
                first_bucket = bucket
            last_bucket = bucket
            # Column order: year doy hour r_au lat lon |B| Bx By Bz density speed temp
            # Wind longitude varies with Earth's orbital position.
            # Approximate: lon = (DOY / 365.25) * 360 deg (ecliptic longitude).
            doy = bucket.timetuple().tm_yday
            approx_lon = (doy / 365.25) * 360.0
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{doy:03d}",
                    f"{bucket.hour:02d}",
                    f"{WIND_R_AU:.3f}",
                    f"{WIND_LAT_DEG:.2f}",
                    f"{approx_lon:.2f}",
                    fmt_or_fill(mag.get("bmag", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bx", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("by", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bz", float("nan")), FILL_B, 2),
                    fmt_or_fill(plasma.get("density", float("nan")), FILL_DENSITY, 2),
                    fmt_or_fill(plasma.get("speed", float("nan")), FILL_SPEED, 1),
                    fmt_or_fill(plasma.get("temp_k", float("nan")), FILL_TEMP, 1),
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
        entry["plasma_provenance"] = "measured_swe_kp"
        entry["mag_provenance"] = "measured_mfi_kp"
        entry["orbit_provenance"] = "hardcoded_l1_position"
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
        description="Fetch WIND SWE + MFI data from NASA SPDF or AMDA HAPI."
    )
    parser.add_argument(
        "--start", type=int, default=2024, help="Start year (inclusive, default 2024)"
    )
    parser.add_argument(
        "--end", type=int, default=2024, help="End year (inclusive, default 2024)"
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
    parser.add_argument(
        "--swe-only", action="store_true", help="Fetch only SWE plasma (SPDF mode only)"
    )
    parser.add_argument(
        "--mfi-only", action="store_true", help="Fetch only MFI B-field (SPDF mode only)"
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    source = args.source
    print(f"Fetching WIND data {args.start}-{args.end} [source={source}]...")

    if source == "amda":
        result = fetch_wind_amda(years, skip_existing=args.skip_existing)
    elif source == "spdf":
        result = fetch_wind_spdf(
            years,
            skip_existing=args.skip_existing,
            swe_only=args.swe_only,
            mfi_only=args.mfi_only,
        )
    else:
        # Auto: SPDF -> AMDA fallback.
        result = fetch_wind_spdf(
            years,
            skip_existing=args.skip_existing,
            swe_only=args.swe_only,
            mfi_only=args.mfi_only,
        )
        counts = result.get("counts", {})
        if isinstance(counts, dict) and counts.get("fetched", 0) == 0 and counts.get("skipped", 0) == 0:
            print("  SPDF failed, falling back to AMDA...")
            result = fetch_wind_amda(years, skip_existing=args.skip_existing)

    counts = result.get("counts", {})
    if isinstance(counts, dict):
        print()
        print(
            f"WIND fetch: {counts.get('fetched', 0)} fetched, "
            f"{counts.get('skipped', 0)} skipped, {counts.get('failed', 0)} failed"
        )
    print(f"Output directories: {SWE_DIR}, {MFI_DIR}, {AMDA_DIR}")
    fetched = counts.get("fetched", 0) if isinstance(counts, dict) else 0
    skipped = counts.get("skipped", 0) if isinstance(counts, dict) else 0
    return 1 if fetched == 0 and skipped == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
