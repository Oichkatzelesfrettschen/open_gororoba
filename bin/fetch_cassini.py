#!/usr/bin/env python3
"""Fetch governed Cassini cruise hourly data.

Primary source intent:
    NASA SPDF/CDAWeb Cassini cruise merged hourly products.

Operational fallback on this host:
    AMDA/CDPP HAPI datasets combined into a governed derived hourly lane:
      - `cass-orb-cruise` for heliocentric trajectory
      - `cass-mag-rtn60` for measured RTN magnetic field
      - `tao-cass-sw` for modeled solar-wind plasma context

The derived AMDA product is intentionally marked as hybrid:
trajectory and magnetic field are measurement-driven, while the plasma lane
comes from the TAO propagated solar-wind model rather than an onboard proton
moment product across the full cruise window.
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

SPDF_BASE = "https://spdf.gsfc.nasa.gov/pub/data/cassini/merged/"
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"
OUT_DIR = Path("data/external/cassini")
USER_AGENT = "gororoba-fetch/0.1 (research)"
EV_TO_K = 11604.51812

AMDA_DATASETS = {
    "orbit": "cass-orb-cruise",
    "mag": "cass-mag-rtn60",
    "plasma": "tao-cass-sw",
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


def parse_orbit_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    rows: dict[dt.datetime, dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 17:
            continue
        bucket = round_to_hour(parse_hapi_time(fields[0]))
        rows[bucket] = {
            "r_au": parse_float(fields[13]),
            "lat_deg": parse_float(fields[14]),
            "lon_deg": normalize_lon_deg(parse_float(fields[15])),
        }
    return rows


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"br": [], "bt": [], "bn": [], "bmag": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
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


def scalar_speed_from_components(v1: float, v2: float) -> float:
    valid = [value for value in (v1, v2) if math.isfinite(value)]
    if not valid:
        return float("nan")
    if len(valid) == 1:
        return abs(valid[0])
    return math.sqrt(valid[0] * valid[0] + valid[1] * valid[1])


def parse_tao_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"density": [], "speed": [], "temp_k": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 9:
            continue
        bucket = floor_to_hour(parse_hapi_time(fields[0]))
        density = parse_float(fields[1])
        speed = scalar_speed_from_components(parse_float(fields[2]), parse_float(fields[3]))
        temp_ev = parse_float(fields[4])
        temp_k = temp_ev * EV_TO_K if math.isfinite(temp_ev) else float("nan")
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


def fetch_spdf_year(year: int, *, skip_existing: bool) -> dict[str, object]:
    out = OUT_DIR / f"cassini_{year}_merged_hourly.asc"
    url = f"{SPDF_BASE}cassini_{year}_merged_hourly.asc"
    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped", source="spdf")

    try:
        data = fetch_bytes(url, timeout=60)
    except (URLError, OSError) as exc:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": str(exc),
        }

    if len(data) < 1024:
        return {
            "url": url,
            "path": str(out),
            "status": "failed",
            "source": "spdf",
            "reason": f"unexpectedly_small:{len(data)}",
        }

    out.write_bytes(data)
    entry = build_file_entry(url, out, "fetched", source="spdf")
    print(f"  {out.name}: OK ({entry['lines']} lines, {entry['bytes']} bytes) [SPDF]")
    return entry


def fetch_cassini_amda(years: range, *, skip_existing: bool) -> dict[str, object]:
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    try:
        dataset_info = {
            key: fetch_amda_info(dataset_id) for key, dataset_id in AMDA_DATASETS.items()
        }
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        for year in years:
            out = OUT_DIR / f"cassini_{year}_amda_cruise_hourly.asc"
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
        out = OUT_DIR / f"cassini_{year}_amda_cruise_hourly.asc"
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
            plasma_rows = parse_tao_rows(
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
                    fmt_or_fill(mag.get("bmag", float('nan')), 9999.99, 2),
                    fmt_or_fill(mag.get("br", float('nan')), 999.99, 2),
                    fmt_or_fill(mag.get("bt", float('nan')), 999.99, 2),
                    fmt_or_fill(mag.get("bn", float('nan')), 999.99, 2),
                    fmt_or_fill(plasma["density"], 999.9, 3),
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
        entry["plasma_provenance"] = "modeled_tao"
        entry["mag_provenance"] = "measured_amda_mag_rtn60"
        entry["orbit_provenance"] = "measured_amda_orb_cruise"
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
                    "plasma": "modeled_tao",
                },
            }
        },
    }


def fetch_year_auto(year: int, *, skip_existing: bool) -> dict[str, object]:
    existing = OUT_DIR / f"cassini_{year}_amda_cruise_hourly.asc"
    if skip_existing and existing.exists():
        return build_file_entry(
            "local://existing", existing, "skipped", source="amda", reason="skip_existing"
        )
    spdf_entry = fetch_spdf_year(year, skip_existing=False)
    if spdf_entry["status"] in {"fetched", "skipped"}:
        return spdf_entry
    amda_result = fetch_cassini_amda(range(year, year + 1), skip_existing=False)
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
    manifest_path = OUT_DIR / f"cassini_{start}_{end}_manifest.json"
    manifest = {
        "product": "cassini_cruise_hourly",
        "start_year": start,
        "end_year": end,
        **payload,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed Cassini cruise hourly data."
    )
    parser.add_argument(
        "--start", type=int, default=1997, help="Start year (inclusive, default 1997)"
    )
    parser.add_argument(
        "--end", type=int, default=2004, help="End year (inclusive, default 2004)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "amda"],
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    if args.source == "amda":
        result = fetch_cassini_amda(
            range(args.start, args.end + 1),
            skip_existing=args.skip_existing,
        )
        counts = result["counts"]
        files = result["files"]
        payload = result
    else:
        for year in range(args.start, args.end + 1):
            if args.source == "spdf":
                entry = fetch_spdf_year(year, skip_existing=args.skip_existing)
            else:
                entry = fetch_year_auto(year, skip_existing=args.skip_existing)
            counts[str(entry["status"])] += 1
            files.append(entry)
        payload = {"status": "ready", "counts": counts, "files": files, "metadata": {}}

    write_manifest(args.start, args.end, payload)
    print()
    print(
        f"Cassini fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Output directory: {OUT_DIR}")
    return 1 if counts["failed"] > 0 and counts["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
