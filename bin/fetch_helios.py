#!/usr/bin/env python3
"""Fetch governed Helios 1 & 2 inner-heliosphere hourly data.

Primary source intent:
    NASA SPDF Helios merged hourly magnetic field and plasma data.

Operational fallback on this host:
    AMDA/CDPP HAPI datasets combined into a governed derived hourly lane:
      - helios{1,2}-e1-all: proton corefit (density, speed, thermal_speed)
      - helios{1,2}-e3-all: MAG 6-sec average (Bx, By, Bz, |B|)
      - helios{1,2}-orb-all: orbit (r_au, lat, lon)

Thermal speed -> temperature conversion: T = m_p * v_th^2 / (2 * k_B).

The E1 proton corefit cadence is ~40s and the E3 MAG cadence is ~6s.
Both are aggregated to hourly medians for the merged output.

Coverage:
    Helios 1: 1974-12 to 1985-09  (perihelion 0.31 AU)
    Helios 2: 1976-01 to 1980-03  (perihelion 0.29 AU)

Output: 13-column SPDF-style ASCII matching the SpdfColumnLayout in
crates/data_core/src/catalogs/helios.rs.
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

AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"
BASE_OUT_DIR = Path("data/external/helios")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# Proton mass (kg) and Boltzmann constant (J/K) for v_th -> T conversion.
M_PROTON = 1.67262192e-27
K_BOLTZMANN = 1.380649e-23

# AMDA dataset IDs per spacecraft.
AMDA_DATASETS = {
    "helios1": {
        "plasma": "helios1-e1-all",
        "mag": "helios1-e3-all",
        "orbit": "helios1-orb-all",
    },
    "helios2": {
        "plasma": "helios2-e1-all",
        "mag": "helios2-e3-all",
        "orbit": "helios2-orb-all",
    },
}

# Mission time windows.
MISSION_BOUNDS = {
    "helios1": {"start_year": 1974, "end_year": 1985},
    "helios2": {"start_year": 1976, "end_year": 1980},
}

# Fill values matching crates/data_core/src/catalogs/helios.rs.
FILL_B = 9999.99
FILL_LAT_LON = 999.99
FILL_DISTANCE = 999.999
FILL_DENSITY = 999.9
FILL_SPEED = 9999.9
FILL_TEMP = 999999.0


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


def vth_to_temperature(v_th_kms: float) -> float:
    """Convert thermal speed (km/s) to proton temperature (K).

    T = m_p * v_th^2 / (2 * k_B), where v_th is in m/s.
    """
    if not math.isfinite(v_th_kms):
        return float("nan")
    v_th_ms = v_th_kms * 1000.0
    return M_PROTON * v_th_ms * v_th_ms / (2.0 * K_BOLTZMANN)


def parse_orbit_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse AMDA helios{1,2}-orb-all CSV.

    Expected columns: timestamp, ..., r_au, lat, lon (exact column
    positions depend on AMDA version; we parse all available floats
    and take the last 3 as r_au, lat_deg, lon_deg).
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"r_au": [], "lat_deg": [], "lon_deg": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        bucket = floor_to_hour(parse_hapi_time(fields[0]))
        # AMDA orbit datasets: timestamp, r_au, lat, lon
        by_hour[bucket]["r_au"].append(parse_float(fields[1]))
        by_hour[bucket]["lat_deg"].append(parse_float(fields[2]))
        by_hour[bucket]["lon_deg"].append(parse_float(fields[3]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "r_au": median_or_nan(accum["r_au"]),
            "lat_deg": median_or_nan(accum["lat_deg"]),
            "lon_deg": normalize_lon_deg(median_or_nan(accum["lon_deg"])),
        }
    return out


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse AMDA helios{1,2}-e3-all MAG CSV.

    Expected columns: timestamp, Bx, By, Bz, |B|.
    6-sec cadence -> per-hour median aggregation.
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"br": [], "bt": [], "bn": [], "bmag": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
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


def parse_plasma_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse AMDA helios{1,2}-e1-all proton corefit CSV.

    Expected columns: timestamp, density, speed, thermal_speed.
    ~40s cadence -> per-hour median aggregation.
    thermal_speed (km/s) -> temperature (K) via T = m_p * v_th^2 / (2*k_B).
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"density": [], "speed": [], "temp_k": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        bucket = floor_to_hour(parse_hapi_time(fields[0]))
        density = parse_float(fields[1])
        speed = parse_float(fields[2])
        v_th = parse_float(fields[3])
        temp_k = vth_to_temperature(v_th)
        by_hour[bucket]["density"].append(density)
        by_hour[bucket]["speed"].append(speed)
        by_hour[bucket]["temp_k"].append(temp_k)

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {
            "density": median_or_nan(accum["density"]),
            "speed": median_or_nan(accum["speed"]),
            "temp_k": median_or_nan(accum["temp_k"]),
        }
    return out


def fetch_helios_amda(
    spacecraft: str, years: range, *, skip_existing: bool
) -> dict[str, object]:
    datasets = AMDA_DATASETS[spacecraft]
    out_dir = BASE_OUT_DIR / spacecraft
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    # Fetch dataset metadata to determine actual time coverage.
    try:
        dataset_info = {
            key: fetch_amda_info(dataset_id) for key, dataset_id in datasets.items()
        }
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        for year in years:
            out = out_dir / f"{spacecraft}_{year}_amda_merged.asc"
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
        out = out_dir / f"{spacecraft}_{year}_amda_merged.asc"
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
                fetch_amda_csv(datasets["orbit"], start, end, timeout=120)
            )
            mag_rows = parse_mag_rows(
                fetch_amda_csv(datasets["mag"], start, end, timeout=300)
            )
            plasma_rows = parse_plasma_rows(
                fetch_amda_csv(datasets["plasma"], start, end, timeout=300)
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

        # Merge on hours where at least orbit AND plasma exist.
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
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{bucket.timetuple().tm_yday:03d}",
                    f"{bucket.hour:02d}",
                    fmt_or_fill(orbit["r_au"], FILL_DISTANCE, 3),
                    fmt_or_fill(orbit["lat_deg"], FILL_LAT_LON, 2),
                    fmt_or_fill(orbit["lon_deg"], FILL_LAT_LON, 2),
                    fmt_or_fill(mag.get("bmag", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("br", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bt", float("nan")), FILL_B, 2),
                    fmt_or_fill(mag.get("bn", float("nan")), FILL_B, 2),
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
        entry["datasets"] = dict(datasets)
        entry["effective_start"] = (
            format_hapi_time(first_bucket) if first_bucket is not None else start
        )
        entry["effective_end"] = (
            format_hapi_time(last_bucket) if last_bucket is not None else end
        )
        entry["plasma_provenance"] = "measured_e1_corefit"
        entry["mag_provenance"] = "measured_e3_mag_6sec"
        entry["orbit_provenance"] = "measured_orb"
        entry["temperature_derivation"] = "T = m_p * v_th^2 / (2 * k_B)"
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
                "datasets": datasets,
                "dataset_bounds": dataset_bounds,
                "derived_lane": {
                    "trajectory": "measured",
                    "magnetic_field": "measured_e3_6sec",
                    "plasma": "measured_e1_corefit",
                },
            }
        },
    }


def write_manifest(
    spacecraft: str, start: int, end: int, payload: dict[str, object]
) -> None:
    out_dir = BASE_OUT_DIR / spacecraft
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{spacecraft}_{start}_{end}_manifest.json"
    manifest = {
        "product": f"{spacecraft}_merged_hourly",
        "spacecraft": spacecraft,
        "start_year": start,
        "end_year": end,
        **payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed Helios 1 & 2 inner-heliosphere hourly data."
    )
    parser.add_argument(
        "--spacecraft",
        choices=["helios1", "helios2", "both"],
        default="helios1",
        help="Which spacecraft (default: helios1)",
    )
    parser.add_argument(
        "--start", type=int, default=1976, help="Start year (inclusive, default 1976)"
    )
    parser.add_argument(
        "--end", type=int, default=1980, help="End year (inclusive, default 1980)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "amda"],
        default="auto",
        help="Preferred source ladder (default: auto)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if a governed file for that year already exists",
    )
    args = parser.parse_args()

    if args.end < args.start:
        print("--end must be >= --start", file=sys.stderr)
        return 2

    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = ["helios1", "helios2"] if args.spacecraft == "both" else [args.spacecraft]
    totals = {"fetched": 0, "skipped": 0, "failed": 0}

    for spacecraft in targets:
        # Clamp requested years to mission bounds.
        bounds = MISSION_BOUNDS[spacecraft]
        effective_start = max(args.start, bounds["start_year"])
        effective_end = min(args.end, bounds["end_year"])
        if effective_start > effective_end:
            print(
                f"  {spacecraft}: requested {args.start}-{args.end} outside "
                f"mission window {bounds['start_year']}-{bounds['end_year']}, skipping"
            )
            continue

        years = range(effective_start, effective_end + 1)
        print(f"Fetching {spacecraft.upper()} {effective_start}-{effective_end} via AMDA...")

        result = fetch_helios_amda(spacecraft, years, skip_existing=args.skip_existing)
        for key in totals:
            totals[key] += result["counts"][key]
        write_manifest(spacecraft, effective_start, effective_end, result)

    print()
    print(
        f"Helios fetch: {totals['fetched']} fetched, "
        f"{totals['skipped']} skipped, {totals['failed']} failed"
    )
    print(f"Output directory: {BASE_OUT_DIR}")
    return 1 if totals["failed"] > 0 and totals["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
