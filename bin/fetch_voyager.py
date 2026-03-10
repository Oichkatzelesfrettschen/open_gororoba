#!/usr/bin/env python3
"""Fetch Voyager 1 & 2 merged hourly data from NASA SPDF / Bartol / AMDA.

Voyager spacecraft provide the deepest heliospheric penetration:
  V1: launched 1977, 157 AU (2024), crossed termination shock 94 AU (2004)
  V2: launched 1977, 134 AU (2024), crossed termination shock 84 AU (2007)

Sources (auto fallback chain):
  1. SPDF merged hourly (13-col SE): https://spdf.gsfc.nasa.gov/pub/data/voyager/
  2. Bartol Research Institute (V2 only, 16-col RTN, 1977-1997):
     https://ftp.bartol.udel.edu/whm/Voyager/
  3. AMDA HAPI translation (plasma + MAG + ephemeris -> 13-col merged)

Usage:
    python3 bin/fetch_voyager.py                              # fetch V1 2020
    python3 bin/fetch_voyager.py --spacecraft both --start 2000 --end 2024
    python3 bin/fetch_voyager.py --spacecraft v2 --source bartol --start 1977 --end 1995
    python3 bin/fetch_voyager.py --skip-existing
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

VOYAGER1_BASE = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/merged/"
VOYAGER2_BASE = "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/merged/"
OUTPUT_DIR = Path("data/external/voyager")
USER_AGENT = "gororoba-fetch/0.1 (research)"
YEAR_FILE_RE = re.compile(r"^vy(?P<label>[12])_(?P<year>\d{4})\.asc$")
HREF_RE = re.compile(r'href="([^"]+)"')
AMDA_HAPI = "https://amda.irap.omp.eu/service/hapi"

VOYAGER_METADATA = {
    "v1": {
        "doi": "https://doi.org/10.48322/041p-cv36",
        "spase_html": "https://spase-metadata.org/NASA/NumericalData/Voyager1/MAGandPLS/PT1H.html",
        "spase_xml": "https://spase-metadata.org/NASA/NumericalData/Voyager1/MAGandPLS/PT1H.xml",
        "catalog_page": "https://catalog.data.gov/dataset/voyager-1-hourly-merged-magnetic-field-and-plasma-data",
        "dataset_id": "VOYAGER1_COHO1HR_MERGED_MAG_PLASMA",
        "hapi_template": "https://cdaweb.gsfc.nasa.gov/hapi/data?id=VOYAGER1_COHO1HR_MERGED_MAG_PLASMA&time.min={start}&time.max={end}",
    },
    "v2": {
        "doi": "https://doi.org/10.48322/ve5t-hf15",
        "spase_html": "https://spase-metadata.org/NASA/NumericalData/Voyager2/MAGandPLS/PT1H.html",
        "spase_xml": "https://spase-metadata.org/NASA/NumericalData/Voyager2/MAGandPLS/PT1H.xml",
        "catalog_page": "https://catalog.data.gov/dataset/voyager-2-hourly-merged-magnetic-field-and-plasma-data",
        "dataset_id": "VOYAGER2_COHO1HR_MERGED_MAG_PLASMA",
        "hapi_template": "https://cdaweb.gsfc.nasa.gov/hapi/data?id=VOYAGER2_COHO1HR_MERGED_MAG_PLASMA&time.min={start}&time.max={end}",
    },
}

AMDA_DATASETS = {
    "v1": {"pls": "vo1-pls-full", "mag": "vo1-mag-full", "ephem": "vo1-ephem-all"},
    "v2": {"pls": "vo2-pls-full", "mag": "vo2-mag-full", "ephem": "vo2-ephem-all"},
}

# Bartol Research Institute legacy archive (Voyager 2 only, 1977-1995).
# 16-column NSSDC/COHO format with 2-digit years and RTN B-field.
# Self-signed TLS certificate requires unverified context.
BARTOL_V2_BASE = "https://ftp.bartol.udel.edu/whm/Voyager/"
BARTOL_YEARS = range(77, 98)  # vy2_77.dat through vy2_97.dat


def _bartol_ssl_context() -> ssl.SSLContext:
    """Create an unverified SSL context for Bartol FTP (self-signed cert)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def fetch_bartol_file(url: str, out: Path, *, skip_existing: bool) -> dict[str, object]:
    """Fetch a single Bartol .dat file with TLS workaround."""
    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped")

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=60, context=_bartol_ssl_context()) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, OSError) as exc:
        return {"url": url, "path": str(out), "status": "failed", "reason": str(exc)}

    if len(data) < 100:
        return {"url": url, "path": str(out), "status": "failed", "bytes": len(data)}

    out.write_bytes(data)
    lines = sum(1 for _ in out.open())
    size = out.stat().st_size
    print(f"  {out.name}: Bartol OK ({lines} lines, {size} bytes)")
    entry = build_file_entry(url, out, "fetched")
    entry["lines"] = lines
    entry["source"] = "bartol"
    return entry


def fetch_voyager_bartol(
    years: range, *, skip_existing: bool, metadata: dict[str, object]
) -> dict[str, object]:
    """Fetch Voyager 2 hourly merged data from Bartol Research Institute archive.

    Bartol covers Voyager 2 only, 1977-1997 (2-digit year files vy2_77.dat..vy2_97.dat).
    16-column NSSDC/COHO format with RTN B-field.
    """
    subdir = OUTPUT_DIR / "voyager2" / "bartol"
    subdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []

    for year in years:
        yy = year % 100
        if yy not in BARTOL_YEARS:
            files.append(
                {
                    "url": BARTOL_V2_BASE,
                    "path": str(subdir / f"vy2_{yy:02d}.dat"),
                    "status": "failed",
                    "year": year,
                    "reason": "year_outside_bartol_range",
                    "source": "bartol",
                }
            )
            counts["failed"] += 1
            continue

        fname = f"vy2_{yy:02d}.dat"
        url = f"{BARTOL_V2_BASE}{fname}"
        out = subdir / fname
        entry = fetch_bartol_file(url, out, skip_existing=skip_existing)
        entry["year"] = year
        entry["source"] = "bartol"
        files.append(entry)
        counts[entry["status"]] += 1
        if entry["status"] == "failed":
            print(f"  Voyager 2 {year} (Bartol): FAILED")

    # Fetch the format documentation if present.
    doc_url = f"{BARTOL_V2_BASE}vy2mgd.txt"
    doc_out = subdir / "vy2mgd.txt"
    doc_entry = fetch_bartol_file(doc_url, doc_out, skip_existing=skip_existing)
    doc_entry["source"] = "bartol"
    doc_entry["type"] = "documentation"
    files.append(doc_entry)
    counts[doc_entry["status"]] += 1

    metadata = dict(metadata)
    metadata["bartol"] = {
        "base_url": BARTOL_V2_BASE,
        "format": "16-column NSSDC/COHO RTN, 2-digit year",
        "spacecraft": "voyager2",
        "coverage": "1977-1997",
        "tls_note": "self-signed cert, unverified context",
    }
    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "metadata_only"
    return {"counts": counts, "files": files, "metadata": metadata, "status": status}


def fetch_file(url: str, out: Path, *, skip_existing: bool) -> dict[str, object]:
    """Fetch a single file and return manifest metadata."""
    if skip_existing and out.exists():
        return build_file_entry(url, out, "skipped")

    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=60) as resp:  # noqa: S310
            data = resp.read()
    except (URLError, OSError):
        return {"url": url, "path": str(out), "status": "failed"}

    if len(data) < 100:
        return {"url": url, "path": str(out), "status": "failed", "bytes": len(data)}

    out.write_bytes(data)
    lines = sum(1 for _ in out.open())
    size = out.stat().st_size
    print(f"  {out.name}: OK ({lines} lines, {size} bytes)")
    entry = build_file_entry(url, out, "fetched")
    entry["lines"] = lines
    return entry


def build_file_entry(url: str, path: Path, status: str) -> dict[str, object]:
    """Build manifest metadata for a local file."""
    entry: dict[str, object] = {"url": url, "path": str(path), "status": status}
    if not path.exists():
        return entry

    data = path.read_bytes()
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["filename"] = path.name
    return entry


def fetch_text(url: str) -> str:
    """Fetch a small text page."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def fetch_metadata_trace(spacecraft: str, subdir: Path) -> dict[str, object]:
    """Fetch reachable metadata sidecars for Voyager merged products."""
    meta = dict(VOYAGER_METADATA[spacecraft])
    sidecars = []
    meta_dir = subdir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for key in ("spase_html", "spase_xml"):
        url = meta[key]
        suffix = ".html" if key.endswith("html") else ".xml"
        out = meta_dir / f"{spacecraft}_merged{suffix}"
        req = Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urlopen(req, timeout=30) as resp:  # noqa: S310
                data = resp.read()
            out.write_bytes(data)
            sidecars.append(build_file_entry(url, out, "fetched"))
        except (URLError, OSError):
            sidecars.append({"url": url, "path": str(out), "status": "failed"})
    meta["sidecars"] = sidecars
    return meta


def fetch_amda_csv(dataset_id: str, start: str, end: str) -> str:
    url = f"{AMDA_HAPI}/data?id={dataset_id}&time.min={start}&time.max={end}&format=csv"
    return fetch_text(url)


def fetch_amda_info(dataset_id: str) -> dict[str, object]:
    url = f"{AMDA_HAPI}/info?id={dataset_id}"
    return json.loads(fetch_text(url))


def parse_hapi_csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append([field.strip() for field in line.split(",")])
    return rows


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


def parse_pls_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    rows = {}
    for fields in parse_hapi_csv_rows(text):
        if len(fields) < 4:
            continue
        timestamp = parse_hapi_time(fields[0])
        rows[timestamp] = {
            "density": parse_float(fields[1]),
            "temperature": parse_float(fields[2]),
            "speed": parse_float(fields[3]),
        }
    return rows


def parse_mag_rows(text: str) -> dict[dt.datetime, dict[str, float]]:
    rows = {}
    for fields in parse_hapi_csv_rows(text):
        if len(fields) < 7:
            continue
        timestamp = parse_hapi_time(fields[0])
        rows[timestamp] = {
            "br": parse_float(fields[1]),
            "bt": parse_float(fields[2]),
            "bn": parse_float(fields[3]),
            "bmag": parse_float(fields[4]),
        }
    return rows


def parse_ephem_rows(text: str) -> list[dict[str, object]]:
    rows = []
    for fields in parse_hapi_csv_rows(text):
        if len(fields) < 10:
            continue
        timestamp = parse_hapi_time(fields[0])
        lon_value = parse_float(fields[8])
        lat_value = parse_float(fields[9])
        rows.append(
            {
                "time": timestamp,
                "r_au": parse_float(fields[7]),
                "lon_deg": lon_value,
                "lat_deg": lat_value,
            }
        )
    rows.sort(key=lambda row: row["time"])
    return rows


def normalize_lon_deg(lon_deg: float) -> float:
    if not math.isfinite(lon_deg):
        return float("nan")
    return lon_deg % 360.0


def interp_linear(a: float, b: float, frac: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return a * (1.0 - frac) + b * frac


def interp_longitude_deg(a: float, b: float, frac: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    delta = b - a
    if delta > 180.0:
        delta -= 360.0
    elif delta < -180.0:
        delta += 360.0
    return normalize_lon_deg(a + frac * delta)


def interpolate_ephem(
    ephem_rows: list[dict[str, object]],
    timestamp: dt.datetime,
) -> dict[str, float]:
    if not ephem_rows:
        return {"r_au": float("nan"), "lat_deg": float("nan"), "lon_deg": float("nan")}
    if timestamp <= ephem_rows[0]["time"]:
        return {
            "r_au": ephem_rows[0]["r_au"],
            "lat_deg": ephem_rows[0]["lat_deg"],
            "lon_deg": normalize_lon_deg(ephem_rows[0]["lon_deg"]),
        }
    if timestamp >= ephem_rows[-1]["time"]:
        return {
            "r_au": ephem_rows[-1]["r_au"],
            "lat_deg": ephem_rows[-1]["lat_deg"],
            "lon_deg": normalize_lon_deg(ephem_rows[-1]["lon_deg"]),
        }
    for left, right in zip(ephem_rows, ephem_rows[1:], strict=False):
        if left["time"] <= timestamp <= right["time"]:
            dt_total = (right["time"] - left["time"]).total_seconds()
            frac = (
                0.0
                if dt_total == 0
                else (timestamp - left["time"]).total_seconds() / dt_total
            )
            return {
                "r_au": interp_linear(left["r_au"], right["r_au"], frac),
                "lat_deg": interp_linear(left["lat_deg"], right["lat_deg"], frac),
                "lon_deg": interp_longitude_deg(left["lon_deg"], right["lon_deg"], frac),
            }
    return {"r_au": float("nan"), "lat_deg": float("nan"), "lon_deg": float("nan")}


def fmt_or_fill(value: float, fill: float, decimals: int) -> str:
    if not math.isfinite(value):
        return f"{fill:.{decimals}f}"
    return f"{value:.{decimals}f}"


def fetch_voyager_amda(
    spacecraft: str,
    years: range,
    *,
    skip_existing: bool,
    metadata: dict[str, object],
) -> dict[str, object]:
    datasets = AMDA_DATASETS[spacecraft]
    label = "1" if spacecraft == "v1" else "2"
    subdir = OUTPUT_DIR / f"voyager{label}"
    subdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []
    dataset_bounds: dict[str, dict[str, str]] = {}

    try:
        dataset_info = {
            key: fetch_amda_info(dataset_id)
            for key, dataset_id in datasets.items()
        }
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        metadata = dict(metadata)
        metadata["amda"] = {
            "hapi_base": AMDA_HAPI,
            "datasets": datasets,
            "info_error": str(exc),
        }
        for year in years:
            out = subdir / f"vy{label}_{year}_amda_merged_hourly.asc"
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "year": year,
                    "reason": f"amda_info_unreachable:{exc}",
                    "source": "amda",
                }
            )
            counts["failed"] += 1
        return {
            "files": files,
            "counts": counts,
            "metadata": metadata,
            "reason": None,
            "status": "ready",
        }

    for key, info in dataset_info.items():
        dataset_bounds[key] = {
            "start": info["startDate"],
            "stop": info["stopDate"],
        }

    for year in years:
        out = subdir / f"vy{label}_{year}_amda_merged_hourly.asc"
        if skip_existing and out.exists():
            entry = build_file_entry("amda://derived", out, "skipped")
            entry["year"] = year
            entry["source"] = "amda"
            files.append(entry)
            counts["skipped"] += 1
            continue

        requested_start = dt.datetime(year, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)
        requested_end = dt.datetime(year, 12, 31, 23, 0, 0, tzinfo=dt.timezone.utc)
        available_start = max(
            parse_hapi_time(bounds["start"]) for bounds in dataset_bounds.values()
        )
        available_end = min(
            parse_hapi_time(bounds["stop"]) for bounds in dataset_bounds.values()
        )
        start_dt = max(requested_start, available_start)
        end_dt = min(requested_end, available_end)
        if start_dt > end_dt:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "year": year,
                    "reason": "amda_year_outside_dataset_range",
                    "source": "amda",
                    "effective_start": format_hapi_time(start_dt),
                    "effective_end": format_hapi_time(end_dt),
                }
            )
            counts["failed"] += 1
            continue

        start = format_hapi_time(start_dt)
        end = format_hapi_time(end_dt)
        try:
            pls_rows = parse_pls_rows(fetch_amda_csv(datasets["pls"], start, end))
            mag_rows = parse_mag_rows(fetch_amda_csv(datasets["mag"], start, end))
            ephem_rows = parse_ephem_rows(fetch_amda_csv(datasets["ephem"], start, end))
        except (URLError, OSError, TimeoutError) as exc:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "year": year,
                    "reason": f"amda_unreachable:{exc}",
                    "source": "amda",
                }
            )
            counts["failed"] += 1
            continue

        lines = []
        for timestamp in sorted(pls_rows):
            ephem = interpolate_ephem(ephem_rows, timestamp)
            mag = mag_rows.get(timestamp, {})
            pls = pls_rows[timestamp]
            line = " ".join(
                [
                    f"{timestamp.year:04d}",
                    f"{timestamp.timetuple().tm_yday:03d}",
                    f"{timestamp.hour:02d}",
                    fmt_or_fill(ephem["r_au"], 999.999, 3),
                    fmt_or_fill(ephem["lat_deg"], 999.99, 2),
                    fmt_or_fill(ephem["lon_deg"], 999.99, 2),
                    fmt_or_fill(mag.get("bmag", float("nan")), 9999.99, 2),
                    fmt_or_fill(mag.get("br", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bt", float("nan")), 999.99, 2),
                    fmt_or_fill(mag.get("bn", float("nan")), 999.99, 2),
                    fmt_or_fill(pls["density"], 999.9, 3),
                    fmt_or_fill(pls["speed"], 9999.9, 1),
                    fmt_or_fill(pls["temperature"], 999999.0, 1),
                ]
            )
            lines.append(line)

        if not lines:
            files.append(
                {
                    "url": AMDA_HAPI,
                    "path": str(out),
                    "status": "failed",
                    "year": year,
                    "reason": "empty_amda_translation",
                    "source": "amda",
                }
            )
            counts["failed"] += 1
            continue

        out.write_text("\n".join(lines) + "\n")
        entry = build_file_entry("amda://derived", out, "fetched")
        entry["year"] = year
        entry["source"] = "amda"
        entry["records"] = len(lines)
        entry["datasets"] = datasets
        entry["effective_start"] = start
        entry["effective_end"] = end
        files.append(entry)
        counts["fetched"] += 1
        print(f"  {out.name}: AMDA OK ({len(lines)} rows)")

    metadata = dict(metadata)
    metadata["amda"] = {
        "hapi_base": AMDA_HAPI,
        "datasets": datasets,
        "dataset_bounds": dataset_bounds,
    }
    return {"counts": counts, "files": files, "metadata": metadata, "status": "ready"}


def discover_year_map(base: str, label: str) -> dict[int, str]:
    """Discover available Voyager merged files from the SPDF directory index."""
    html = fetch_text(base)
    year_map: dict[int, str] = {}
    for href in HREF_RE.findall(html):
        fname = href.rsplit("/", maxsplit=1)[-1]
        match = YEAR_FILE_RE.match(fname)
        if not match or match.group("label") != label:
            continue
        year_map[int(match.group("year"))] = fname

    return year_map


def fetch_voyager(
    spacecraft: str, years: range, *, skip_existing: bool, source: str
) -> dict[str, object]:
    """Fetch Voyager merged hourly files for the given spacecraft and years."""
    if spacecraft == "v1":
        base = VOYAGER1_BASE
        label = "1"
        subdir = OUTPUT_DIR / "voyager1"
    else:
        base = VOYAGER2_BASE
        label = "2"
        subdir = OUTPUT_DIR / "voyager2"

    subdir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}
    files: list[dict[str, object]] = []
    metadata = fetch_metadata_trace(spacecraft, subdir)

    if source == "amda":
        return fetch_voyager_amda(spacecraft, years, skip_existing=skip_existing, metadata=metadata)

    if source == "bartol":
        if spacecraft != "v2":
            print("  Bartol archive covers Voyager 2 only, skipping V1")
            return {
                "counts": {"fetched": 0, "skipped": 0, "failed": 0},
                "files": [],
                "metadata": metadata,
                "status": "metadata_only",
                "reason": "bartol_v2_only",
            }
        return fetch_voyager_bartol(years, skip_existing=skip_existing, metadata=metadata)

    try:
        discovered = discover_year_map(base, label)
    except (URLError, OSError) as exc:
        print(f"  failed to read index {base}: {exc}")
        if source == "auto":
            # Auto fallback chain: SPDF -> Bartol (V2, 1977-1997) -> AMDA
            if spacecraft == "v2":
                bartol_years = range(
                    max(years.start, 1977), min(years.stop, 1998)
                )
                if bartol_years:
                    print("  trying Bartol Research Institute archive (V2 1977-1997)...")
                    bartol_result = fetch_voyager_bartol(
                        bartol_years, skip_existing=skip_existing, metadata=metadata
                    )
                    if bartol_result["counts"]["fetched"] > 0:
                        return bartol_result
                    print("  Bartol also failed, falling back to AMDA HAPI...")
            print("  falling back to AMDA HAPI translation...")
            return fetch_voyager_amda(
                spacecraft,
                years,
                skip_existing=skip_existing,
                metadata=metadata,
            )
        return {
            "counts": {"fetched": 0, "skipped": 0, "failed": len(list(years))},
            "files": [],
            "metadata": metadata,
            "status": "metadata_only",
            "reason": "index_unreachable",
        }

    readme_name = f"00readme_v{label}.txt"
    readme_entry = fetch_file(
        f"{base}{readme_name}",
        subdir / readme_name,
        skip_existing=skip_existing,
    )
    files.append(readme_entry)
    counts[readme_entry["status"]] += 1

    for year in years:
        fname = discovered.get(year)
        if fname is None:
            print(f"  Voyager {label} {year}: FAILED (not present in SPDF index)")
            counts["failed"] += 1
            files.append(
                {
                    "url": base,
                    "path": str(subdir / f"vy{label}_{year}.asc"),
                    "status": "failed",
                    "year": year,
                    "reason": "missing_from_index",
                }
            )
            continue
        url = f"{base}{fname}"
        out = subdir / fname
        entry = fetch_file(url, out, skip_existing=skip_existing)
        entry["year"] = year
        files.append(entry)
        counts[entry["status"]] += 1
        if entry["status"] == "failed":
            print(f"  Voyager {label} {year}: FAILED")

    status = "ready" if counts["fetched"] > 0 or counts["skipped"] > 0 else "metadata_only"
    return {"counts": counts, "files": files, "metadata": metadata, "status": status}


def write_manifest(spacecraft: str, start: int, end: int, payload: dict[str, object]) -> None:
    """Write a deterministic fetch manifest."""
    manifest_path = OUTPUT_DIR / f"{spacecraft}_{start}_{end}_manifest.json"
    manifest = {
        "spacecraft": spacecraft,
        "start_year": start,
        "end_year": end,
        "status": payload.get("status", "ready"),
        "reason": payload.get("reason"),
        "counts": payload["counts"],
        "metadata": payload.get("metadata", {}),
        "files": payload["files"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"  manifest: {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Voyager 1 & 2 merged hourly data from NASA SPDF."
    )
    parser.add_argument(
        "--spacecraft",
        choices=["v1", "v2", "both"],
        default="v1",
        help="Which spacecraft (default: v1)",
    )
    parser.add_argument(
        "--start", type=int, default=2020, help="Start year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--end", type=int, default=2020, help="End year (inclusive, default 2020)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "amda", "bartol"],
        default="auto",
        help="Acquisition source: SPDF, Bartol (V2 1977-1997), AMDA HAPI, or auto fallback",
    )
    args = parser.parse_args()

    years = range(args.start, args.end + 1)
    targets = ["v1", "v2"] if args.spacecraft == "both" else [args.spacecraft]

    total: dict[str, int] = {"fetched": 0, "skipped": 0, "failed": 0}

    for sc in targets:
        label = "Voyager 1" if sc == "v1" else "Voyager 2"
        print(f"Fetching {label} merged hourly {args.start}-{args.end}...")
        result = fetch_voyager(sc, years, skip_existing=args.skip_existing, source=args.source)
        write_manifest(sc, args.start, args.end, result)
        for k in total:
            total[k] += result["counts"][k]

    print()
    print(
        f"Voyager fetch: {total['fetched']} fetched, "
        f"{total['skipped']} skipped, {total['failed']} failed"
    )
    print(f"Output directory: {OUTPUT_DIR}")
    return 1 if total["failed"] > 0 and total["fetched"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
