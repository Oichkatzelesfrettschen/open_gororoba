#!/usr/bin/env python3
"""Fetch Parker Solar Probe (PSP) merged hourly data via AMDA HAPI.

Parker Solar Probe launched 2018-08-12, performs repeated perihelion passes
through the inner heliosphere (0.05-1.0 AU). SWEAP/SPC measures proton moments
and FIELDS/MAG provides high-cadence magnetic field data. Orbit data covers
the full mission in HCI coordinates.

AMDA/CDPP HAPI datasets combined into a governed derived hourly lane:
  - psp-spc-mom   : SWEAP/SPC proton moments (density, bulk speed, thermal speed)
  - psp-mag-1min  : FIELDS/MAG 1-minute RTN magnetic field
  - psp-orb-all   : Heliocentric orbit (r_au, lat_deg, lon_deg in HCI)

Temperature is derived from thermal speed:
    T = m_p * v_th^2 / (2 * k_B)
where m_p = 1.67262192e-27 kg, k_B = 1.380649e-23 J/K.

Sub-hourly cadence data (SPC and MAG) is aggregated to per-hour medians.
Output is 13-column SPDF-style merged format:
    YYYY DOY HH r_au lat lon |B| Br Bt Bn density speed temperature

Usage:
    python3 bin/fetch_psp.py                              # fetch 2022-2023
    python3 bin/fetch_psp.py --start 2019 --end 2024
    python3 bin/fetch_psp.py --source amda --skip-existing
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
OUT_DIR = Path("data/external/psp")
USER_AGENT = "gororoba-fetch/0.1 (research)"

# Proton mass and Boltzmann constant for v_th -> T conversion.
M_PROTON = 1.67262192e-27   # kg
K_BOLTZMANN = 1.380649e-23  # J/K

AMDA_DATASETS = {
    "orbit": "psp-orb-all",
    "mag": "psp-mag-1min",
    "plasma": "psp-spc-mom",
}


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


def round_to_hour(timestamp: dt.datetime) -> dt.datetime:
    shifted = timestamp + dt.timedelta(minutes=30)
    return shifted.replace(minute=0, second=0, microsecond=0)


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


def thermal_speed_to_temp_k(v_th: float) -> float:
    """Convert thermal speed (km/s) to temperature (K).

    T = m_p * v_th^2 / (2 * k_B)
    v_th arrives in km/s, convert to m/s first.
    """
    if not math.isfinite(v_th) or v_th <= 0.0:
        return float("nan")
    v_th_ms = v_th * 1.0e3
    return M_PROTON * v_th_ms * v_th_ms / (2.0 * K_BOLTZMANN)


def parse_orbit_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse psp-orb-all CSV.

    Expected columns: time, r_au, lat_deg, lon_deg (and possibly more).
    Orbit data is typically hourly or coarser; use round_to_hour.
    """
    rows: dict[dt.datetime, dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 4:
            continue
        bucket = round_to_hour(parse_hapi_time(fields[0]))
        rows[bucket] = {
            "r_au": parse_float(fields[1]),
            "lat_deg": parse_float(fields[2]),
            "lon_deg": normalize_lon_deg(parse_float(fields[3])),
        }
    return rows


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse psp-mag-1min CSV.

    Expected columns: time, Br, Bt, Bn, |B|.
    Sub-hourly cadence: aggregate to per-hour medians.
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
    """Parse psp-spc-mom CSV.

    Expected columns: time, density, bulk_speed, thermal_speed (and possibly more).
    Sub-hourly cadence: aggregate to per-hour medians.
    Temperature derived from thermal speed: T = m_p * v_th^2 / (2 * k_B).
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
        temp_k = thermal_speed_to_temp_k(v_th)
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


def fetch_psp_amda(years: range, *, skip_existing: bool) -> dict[str, object]:
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    try:
        dataset_info = {
            key: fetch_amda_info(dataset_id) for key, dataset_id in AMDA_DATASETS.items()
        }
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        for year in years:
            out = OUT_DIR / f"psp_{year}_amda_merged.asc"
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
            "metadata": {"amda": {"datasets": AMDA_DATASETS, "info_error": str(exc)}},
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
        out = OUT_DIR / f"psp_{year}_amda_merged.asc"
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
                    fmt_or_fill(orbit["r_au"], 999.999, 3),
                    fmt_or_fill(orbit["lat_deg"], 999.99, 2),
                    fmt_or_fill(orbit["lon_deg"], 999.99, 2),
                    fmt_or_fill(mag.get("bmag", float("nan")), 9999.99, 2),
                    fmt_or_fill(mag.get("br", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bt", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bn", float("nan")), 999.99, 2),
                    fmt_or_fill(plasma["density"], 999.9, 1),
                    fmt_or_fill(plasma["speed"], 9999.9, 1),
                    fmt_or_fill(plasma["temp_k"], 999999.0, 1),
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
        entry["plasma_provenance"] = "measured_sweap_spc"
        entry["mag_provenance"] = "measured_fields_mag"
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
                "dataset_bounds": dataset_bounds,
                "derived_lane": {
                    "trajectory": "measured",
                    "magnetic_field": "measured",
                    "plasma": "measured_sweap_spc",
                },
            }
        },
    }


def fetch_year_auto(year: int, *, skip_existing: bool) -> dict[str, object]:
    """Auto mode: PSP has no SPDF merged product, so delegate to AMDA directly."""
    existing = OUT_DIR / f"psp_{year}_amda_merged.asc"
    if skip_existing and existing.exists():
        return build_file_entry(
            "local://existing", existing, "skipped", source="amda", reason="skip_existing"
        )
    amda_result = fetch_psp_amda(range(year, year + 1), skip_existing=False)
    if amda_result["files"]:
        return amda_result["files"][0]
    return {
        "url": AMDA_HAPI,
        "path": str(existing),
        "status": "failed",
        "source": "amda",
        "reason": "unknown_auto_failure",
    }


def write_manifest(start: int, end: int, payload: dict[str, object]) -> None:
    manifest_path = OUT_DIR / f"psp_{start}_{end}_manifest.json"
    manifest = {
        "product": "psp_merged_hourly",
        "start_year": start,
        "end_year": end,
        **payload,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Parker Solar Probe merged hourly data via AMDA HAPI."
    )
    parser.add_argument(
        "--start", type=int, default=2022, help="Start year (inclusive, default 2022)"
    )
    parser.add_argument(
        "--end", type=int, default=2023, help="End year (inclusive, default 2023)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "amda"],
        default="auto",
        help="Preferred source (default: auto)",
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    if args.source == "amda":
        result = fetch_psp_amda(
            range(args.start, args.end + 1),
            skip_existing=args.skip_existing,
        )
        counts = result["counts"]
        files = result["files"]
        payload = result
    else:
        for year in range(args.start, args.end + 1):
            entry = fetch_year_auto(year, skip_existing=args.skip_existing)
            counts[str(entry["status"])] += 1
            files.append(entry)
        payload = {"status": "ready", "counts": counts, "files": files, "metadata": {}}

    write_manifest(args.start, args.end, payload)
    print()
    print(
        f"PSP fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Output directory: {OUT_DIR}")
    return 1 if counts["failed"] > 0 and counts["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
