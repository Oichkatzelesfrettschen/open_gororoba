#!/usr/bin/env python3
"""Fetch governed Solar Orbiter MAG + RPW partial data via AMDA HAPI.

PARTIAL LANE -- MAG and RPW electron density only.
Solar Orbiter SWA (Solar Wind Analyser) proton moments are NOT yet available
in AMDA as of 2026-03.  Consequently this fetcher produces files where:
  - |B|, Br, Bt, Bn  : measured (MAG LL 8-sec, median-aggregated to hourly)
  - density           : RPW electron density (10-sec, median-aggregated to hourly)
                        used as a PROXY for proton density (quasi-neutrality)
  - speed             : NaN fill (no SWA-PAS data in AMDA)
  - temperature       : NaN fill (no SWA-PAS data in AMDA)

The output uses 13-column SPDF-style format so downstream Rust parsers
(SpdfColumnLayout) can ingest it without modification.  The blocker_note
field in the manifest records the partial nature of this lane.

AMDA datasets used:
  solo-mag-8s         MAG LL 8-second cadence (Br, Bt, Bn in RTN, |B|)
  solo-rpw-density10s RPW electron density 10-second cadence

Solar Orbiter launched 2020-02-10, science operations from ~2020-06 onward.

Usage:
    python3 bin/fetch_solo.py                          # 2021-2023, AMDA
    python3 bin/fetch_solo.py --start 2021 --end 2022
    python3 bin/fetch_solo.py --skip-existing
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
OUT_DIR = Path("data/external/solo")
USER_AGENT = "gororoba-fetch/0.1 (research)"

AMDA_DATASETS = {
    "mag": "solo-mag-8s",
    "density": "solo-rpw-density10s",
}

BLOCKER_NOTE = (
    "PARTIAL: MAG + RPW electron density only. "
    "SWA proton plasma moments (speed, temperature, proton density) "
    "not available in AMDA as of 2026-03. "
    "Density column is RPW electron density used as quasi-neutrality proxy. "
    "Speed and temperature columns are NaN fill."
)


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


def fetch_text(url: str, *, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


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


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse solo-mag-8s HAPI CSV into hourly-median MAG records.

    Expected columns: timestamp, Br, Bt, Bn, |B|
    The 8-second cadence yields ~450 samples per hour; we take the median.
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


def parse_density_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    """Parse solo-rpw-density10s HAPI CSV into hourly-median electron density.

    Expected columns: timestamp, electron_density
    The 10-second cadence yields ~360 samples per hour; we take the median.
    This is RPW-derived ELECTRON density, used as a quasi-neutrality proxy
    for proton density.  Speed and temperature are not available.
    """
    by_hour: dict[dt.datetime, dict[str, list[float]]] = defaultdict(
        lambda: {"density": []}
    )
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        bucket = floor_to_hour(parse_hapi_time(fields[0]))
        by_hour[bucket]["density"].append(parse_float(fields[1]))

    out: dict[dt.datetime, dict[str, float]] = {}
    for bucket, accum in by_hour.items():
        out[bucket] = {"density": median_or_nan(accum["density"])}
    return out


def fetch_solo_amda(years: range, *, skip_existing: bool) -> dict[str, object]:
    """Fetch Solar Orbiter MAG + RPW density from AMDA, one file per year."""
    counts = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    try:
        dataset_info = {
            key: fetch_amda_info(dataset_id) for key, dataset_id in AMDA_DATASETS.items()
        }
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        for year in years:
            out = OUT_DIR / f"solo_{year}_amda_mag_density.asc"
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
        out = OUT_DIR / f"solo_{year}_amda_mag_density.asc"
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
            mag_rows = parse_mag_rows(
                fetch_amda_csv(AMDA_DATASETS["mag"], start, end, timeout=300)
            )
            density_rows = parse_density_rows(
                fetch_amda_csv(AMDA_DATASETS["density"], start, end, timeout=120)
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

        # Union of MAG and density hours -- we emit a row if EITHER has data.
        # Solar Orbiter has no separate orbit dataset in AMDA; position is
        # not available per-hour from these products.  We fill r_au/lat/lon.
        all_hours = sorted(set(mag_rows) | set(density_rows))
        lines: list[str] = []
        first_bucket: dt.datetime | None = None
        last_bucket: dt.datetime | None = None
        for bucket in all_hours:
            if bucket.year != year:
                continue
            if first_bucket is None:
                first_bucket = bucket
            last_bucket = bucket
            mag = mag_rows.get(bucket, {})
            dens = density_rows.get(bucket, {})
            line = " ".join(
                [
                    f"{bucket.year:04d}",
                    f"{bucket.timetuple().tm_yday:03d}",
                    f"{bucket.hour:02d}",
                    # r_au, lat_deg, lon_deg -- no orbit product, fill
                    fmt_or_fill(float("nan"), 999.999, 3),
                    fmt_or_fill(float("nan"), 999.99, 2),
                    fmt_or_fill(float("nan"), 999.99, 2),
                    # MAG
                    fmt_or_fill(mag.get("bmag", float("nan")), 9999.99, 2),
                    fmt_or_fill(mag.get("br", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bt", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bn", float("nan")), 999.99, 2),
                    # RPW electron density as proton density proxy
                    fmt_or_fill(dens.get("density", float("nan")), 999.9, 3),
                    # speed, temperature -- no SWA data, fill
                    fmt_or_fill(float("nan"), 9999.9, 1),
                    fmt_or_fill(float("nan"), 999999.0, 1),
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
        entry["mag_provenance"] = "measured_amda_solo_mag_8s"
        entry["density_provenance"] = "measured_amda_solo_rpw_density10s"
        entry["density_species"] = "electron"
        entry["blocker_note"] = BLOCKER_NOTE
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
                    "magnetic_field": "measured_mag_8s",
                    "density": "measured_rpw_electron_10s",
                    "speed": "unavailable_no_swa",
                    "temperature": "unavailable_no_swa",
                    "trajectory": "unavailable_no_orbit_product",
                },
                "blocker_note": BLOCKER_NOTE,
            }
        },
    }


def fetch_year_auto(year: int, *, skip_existing: bool) -> dict[str, object]:
    """Auto source simply delegates to AMDA (no SPDF product for Solo)."""
    existing = OUT_DIR / f"solo_{year}_amda_mag_density.asc"
    if skip_existing and existing.exists():
        return build_file_entry(
            "local://existing", existing, "skipped", source="amda", reason="skip_existing"
        )
    result = fetch_solo_amda(range(year, year + 1), skip_existing=False)
    if result["files"]:
        return result["files"][0]
    return {
        "url": AMDA_HAPI,
        "path": str(existing),
        "status": "failed",
        "source": "amda",
        "reason": "unknown_auto_failure",
    }


def write_manifest(start: int, end: int, payload: dict[str, object]) -> None:
    manifest_path = OUT_DIR / f"solo_{start}_{end}_manifest.json"
    manifest = {
        "product": "solo_mag_density_hourly",
        "start_year": start,
        "end_year": end,
        "blocker_note": BLOCKER_NOTE,
        **payload,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch governed Solar Orbiter MAG + RPW electron density "
            "(PARTIAL -- no SWA proton plasma)."
        ),
    )
    parser.add_argument(
        "--start", type=int, default=2021, help="Start year (inclusive, default 2021)"
    )
    parser.add_argument(
        "--end", type=int, default=2023, help="End year (inclusive, default 2023)"
    )
    parser.add_argument(
        "--source",
        choices=["auto", "amda"],
        default="auto",
        help="Source (default: auto). Only AMDA available for Solar Orbiter.",
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

    if args.source == "amda":
        result = fetch_solo_amda(
            range(args.start, args.end + 1),
            skip_existing=args.skip_existing,
        )
    else:
        # auto -- per-year with skip-existing support
        counts = {"fetched": 0, "skipped": 0, "failed": 0}
        files: list[dict[str, object]] = []
        for year in range(args.start, args.end + 1):
            entry = fetch_year_auto(year, skip_existing=args.skip_existing)
            counts[str(entry["status"])] += 1
            files.append(entry)
        result = {
            "status": "ready",
            "counts": counts,
            "files": files,
            "metadata": {},
        }

    write_manifest(args.start, args.end, result)
    counts = result["counts"]
    print()
    print(
        f"Solar Orbiter fetch: {counts['fetched']} fetched, "
        f"{counts['skipped']} skipped, {counts['failed']} failed"
    )
    print(f"Output directory: {OUT_DIR}")
    print(f"NOTE: {BLOCKER_NOTE}")
    return 1 if counts["failed"] > 0 and counts["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
