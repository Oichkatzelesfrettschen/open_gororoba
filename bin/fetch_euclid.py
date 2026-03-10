#!/usr/bin/env python3
"""Fetch governed Euclid Q1 catalog subsets via TAP/ADQL.

This is an ADQL client targeting the Euclid Q1 primary survey catalogs
(MER, PHZ, SPE) via the Table Access Protocol (IVOA TAP).

Unlike the Zenodo fetcher (fetch_euclid_zenodo.py) which downloads small
supplementary catalogs in bulk, this script issues ADQL queries for
targeted subsets of the multi-TB primary catalogs.

Primary targets:
    MER_FINAL_CATALOG: Source photometry, positions, morphology (30M rows total)
    PHZ_PF_OUTPUT_CATALOG: Photometric redshifts, PDFs (29.6M rows total)

TAP endpoints (fallback chain):
    1. ESA SAS: https://eas.esac.esa.int/tap-server/tap
    2. IRSA TAP: https://irsa.ipac.caltech.edu/TAP
    3. AWS S3: direct FITS download (no TAP, Tile ID required)

License: CC BY-NC 3.0 IGO
Credit: ESA Euclid/Euclid Consortium/NASA/Q1-2025
DOI: https://doi.org/10.57780/esa-2853f3b

Catalog registry: data/external/euclid_q1_catalogs.toml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_OUT_DIR = Path("data/external/euclid/tap")
USER_AGENT = "gororoba-fetch/0.1 (research; ADQL client)"

# TAP endpoints in priority order.
TAP_ENDPOINTS = {
    "esa": "https://eas.esac.esa.int/tap-server/tap",
    "irsa": "https://irsa.ipac.caltech.edu/TAP",
}

# Synchronous query size limit -- TAP servers typically cap at ~2M rows for sync.
# For larger results, use async. We stay well under this with spatial/tile constraints.
SYNC_MAX_ROWS = 500_000

# Retry parameters.
MAX_RETRIES = 3
RETRY_BACKOFF_S = 5.0

# Known catalog table names per endpoint.
# ESA and IRSA use different schema names for the same data.
TABLE_NAMES = {
    "esa": {
        "mer": "catalogue.mer_catalogue",
        "phz": "catalogue.phz_photo_z",
        "spe_redshift": "catalogue.phz_classification",
        "phz_classification": "catalogue.phz_classification",
    },
    "irsa": {
        "mer": "euclid_q1_mer_catalogue",
        "phz": "euclid_q1_phz_photo_z",
        "spe_redshift": "euclid_q1_spectro_zcatalog_spe_quality",
        "phz_classification": "euclid_q1_phz_classification",
    },
}

# Default columns for each catalog (cosmology-relevant subset).
# Full catalogs have 400+ columns; we select the science-critical ones.
MER_COLUMNS = [
    "object_id",
    "ra",
    "dec",
    "flux_vis_sersic",
    "fluxerr_vis_sersic",
    "flux_y_sersic",
    "flux_j_sersic",
    "flux_h_sersic",
    "flux_detection_total",
    "fluxerr_detection_total",
    "kron_radius",
    "ellipticity",
    "fwhm",
    "extended_flag",
    "point_like_flag",
    "flag_vis",
]

PHZ_COLUMNS = [
    "object_id",
    "phz_median",
    "phz_mode_1",
    "phz_mode_1_area",
    "phz_mode_2",
    "phz_70_int1",
    "phz_70_int2",
    "phz_90_int1",
    "phz_90_int2",
    "best_chi2",
    "phz_classification",
    "phz_flags",
    "phz_weight",
    "tom_bin_id",
]

SPE_COLUMNS = [
    "object_id",
    "phz_classification",
    "phz_flags",
]


def fetch_bytes(url: str, *, timeout: int = 60, method: str = "GET",
                data: bytes | None = None,
                content_type: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if content_type:
        headers["Content-Type"] = content_type
    req = Request(url, headers=headers, method=method, data=data)
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def tap_sync_query(
    endpoint_url: str,
    adql: str,
    *,
    fmt: str = "csv",
    maxrec: int = SYNC_MAX_ROWS,
    timeout: int = 300,
) -> str:
    """Execute a synchronous TAP query and return results as text.

    Parameters
    ----------
    endpoint_url : str
        TAP base URL (without /sync suffix).
    adql : str
        ADQL query string.
    fmt : str
        Output format: 'csv', 'votable', 'tsv'.
    maxrec : int
        Maximum records to return.
    timeout : int
        HTTP timeout in seconds.
    """
    sync_url = f"{endpoint_url}/sync"
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": fmt,
        "MAXREC": str(maxrec),
        "QUERY": adql,
    }
    body = urlencode(params).encode("utf-8")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = fetch_bytes(
                sync_url,
                timeout=timeout,
                method="POST",
                data=body,
                content_type="application/x-www-form-urlencoded",
            )
            return result.decode("utf-8", errors="replace")
        except (URLError, HTTPError, OSError, TimeoutError) as exc:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF_S * attempt
                print(f"    TAP attempt {attempt}/{MAX_RETRIES} failed: {exc}, retrying in {wait:.0f}s")
                time.sleep(wait)
            else:
                raise
    msg = "tap_sync_query: exhausted retries"
    raise RuntimeError(msg)


def tap_tables(endpoint_url: str, *, timeout: int = 60) -> str:
    """Query the TAP /tables endpoint for available tables."""
    url = f"{endpoint_url}/tables"
    return fetch_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def build_cone_adql(
    table: str,
    columns: list[str],
    ra_center: float,
    dec_center: float,
    radius_deg: float,
    *,
    maxrec: int = SYNC_MAX_ROWS,
) -> str:
    """Build an ADQL cone search query using a box approximation.

    Some TAP servers (IRSA) do not support ADQL geometric functions
    (CONTAINS/POINT/CIRCLE). We use a rectangular RA/DEC box instead,
    which is a valid approximation for small search radii. The box is
    slightly oversized at the corners but that is acceptable for our
    use case of fetching science subsets.

    For tables without ra/dec (like PHZ), we join against MER to get
    spatial coordinates.
    """
    import math
    col_list = ", ".join(columns)
    # Correct RA width for declination (cos(dec) foreshortening).
    cos_dec = math.cos(math.radians(dec_center))
    ra_half = radius_deg / max(cos_dec, 0.01)
    dec_half = radius_deg
    ra_min = ra_center - ra_half
    ra_max = ra_center + ra_half
    dec_min = dec_center - dec_half
    dec_max = dec_center + dec_half
    return (
        f"SELECT TOP {maxrec} {col_list} "
        f"FROM {table} "
        f"WHERE ra BETWEEN {ra_min} AND {ra_max} "
        f"AND dec BETWEEN {dec_min} AND {dec_max}"
    )


# Tables that lack ra/dec columns and need a JOIN via object_id.
TABLES_WITHOUT_COORDS = {"phz", "spe_redshift", "phz_classification"}


def build_tile_adql(
    table: str,
    columns: list[str],
    tile_id: str,
    *,
    maxrec: int = SYNC_MAX_ROWS,
) -> str:
    """Build an ADQL query constrained to a specific Tile ID."""
    col_list = ", ".join(columns)
    return (
        f"SELECT TOP {maxrec} {col_list} "
        f"FROM {table} "
        f"WHERE tile_index = '{tile_id}'"
    )


def build_count_adql(table: str) -> str:
    """Build a row count query for a table."""
    return f"SELECT COUNT(*) AS row_count FROM {table}"


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


def save_csv_result(
    text: str, dest: Path, *, adql: str, endpoint: str, catalog: str
) -> dict[str, object]:
    """Save TAP query result to CSV with metadata header."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Count rows (skip header).
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    rows = list(reader)

    # Write with metadata comment header.
    with dest.open("w", encoding="utf-8") as fh:
        fh.write(f"# catalog={catalog}\n")
        fh.write(f"# endpoint={endpoint}\n")
        fh.write(f"# fetched_at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fh.write(f"# adql={adql}\n")
        fh.write(f"# rows={len(rows)}\n")
        fh.write(f"# columns={len(header) if header else 0}\n")
        if header:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(rows)

    entry = build_file_entry(endpoint, dest, "fetched", source="tap")
    entry["rows"] = len(rows)
    entry["columns"] = len(header) if header else 0
    entry["catalog"] = catalog
    entry["adql"] = adql
    return entry


def try_endpoints(
    adql_by_endpoint: dict[str, str],
    dest: Path,
    catalog: str,
    *,
    maxrec: int = SYNC_MAX_ROWS,
    timeout: int = 300,
) -> dict[str, object]:
    """Try TAP query across endpoints in priority order."""
    for ep_name, adql in adql_by_endpoint.items():
        ep_url = TAP_ENDPOINTS[ep_name]
        print(f"  Trying {ep_name} ({ep_url})...")
        try:
            text = tap_sync_query(ep_url, adql, maxrec=maxrec, timeout=timeout)
            # Check for error responses (VOTable error or empty).
            if "<VOTABLE" in text[:200]:
                if "ERROR" in text[:2000]:
                    # Extract error message from VOTable.
                    import re
                    err_match = re.search(r"<INFO[^>]*value=\"([^\"]+)\"", text[:2000])
                    err_msg = err_match.group(1) if err_match else text[:300]
                    print(f"    {ep_name}: TAP error: {err_msg[:200]}")
                    continue
                # Some servers return VOTable even for FORMAT=csv on errors.
                print(f"    {ep_name}: unexpected VOTable response (expected CSV)")
                continue
            if len(text.strip()) < 10:
                print(f"    {ep_name}: empty response")
                continue
            entry = save_csv_result(
                text, dest, adql=adql, endpoint=ep_url, catalog=catalog
            )
            print(f"    {ep_name}: OK ({entry.get('rows', '?')} rows, {entry.get('columns', '?')} cols)")
            return entry
        except (URLError, HTTPError, OSError, TimeoutError, RuntimeError) as exc:
            print(f"    {ep_name}: FAILED ({exc})")
            continue

    return build_file_entry(
        "all_endpoints_failed", dest, "failed",
        source="tap",
        reason="all TAP endpoints failed or returned errors",
    )


def fetch_cone(
    catalog: str,
    ra: float,
    dec: float,
    radius_deg: float,
    *,
    maxrec: int = SYNC_MAX_ROWS,
) -> dict[str, object]:
    """Fetch a cone search subset from a catalog."""
    columns = {"mer": MER_COLUMNS, "phz": PHZ_COLUMNS, "spe_redshift": SPE_COLUMNS}.get(
        catalog, MER_COLUMNS
    )

    ra_str = f"{ra:.4f}"
    dec_str = f"{dec:.4f}"
    rad_str = f"{radius_deg:.4f}"
    dest = BASE_OUT_DIR / catalog / f"{catalog}_cone_{ra_str}_{dec_str}_r{rad_str}.csv"

    adql_by_endpoint = {}
    for ep_name in TAP_ENDPOINTS:
        table = TABLE_NAMES[ep_name].get(catalog)
        if not table:
            continue
        if catalog in TABLES_WITHOUT_COORDS:
            # Join against MER to get spatial coordinates.
            mer_table = TABLE_NAMES[ep_name].get("mer")
            if not mer_table:
                continue
            import math
            cos_dec = math.cos(math.radians(dec))
            ra_half = radius_deg / max(cos_dec, 0.01)
            col_list = ", ".join(f"t.{c}" for c in columns)
            adql = (
                f"SELECT TOP {maxrec} {col_list} "
                f"FROM {table} AS t "
                f"JOIN {mer_table} AS m ON t.object_id = m.object_id "
                f"WHERE m.ra BETWEEN {ra - ra_half} AND {ra + ra_half} "
                f"AND m.dec BETWEEN {dec - radius_deg} AND {dec + radius_deg}"
            )
            adql_by_endpoint[ep_name] = adql
        else:
            adql_by_endpoint[ep_name] = build_cone_adql(
                table, columns, ra, dec, radius_deg, maxrec=maxrec
            )

    return try_endpoints(adql_by_endpoint, dest, catalog, maxrec=maxrec)


def fetch_tile(
    catalog: str,
    tile_id: str,
    *,
    maxrec: int = SYNC_MAX_ROWS,
) -> dict[str, object]:
    """Fetch all rows for a specific Tile ID from a catalog."""
    columns = {"mer": MER_COLUMNS, "phz": PHZ_COLUMNS, "spe_redshift": SPE_COLUMNS}.get(
        catalog, MER_COLUMNS
    )

    dest = BASE_OUT_DIR / catalog / f"{catalog}_tile_{tile_id}.csv"

    adql_by_endpoint = {}
    for ep_name in TAP_ENDPOINTS:
        table = TABLE_NAMES[ep_name].get(catalog)
        if table:
            adql_by_endpoint[ep_name] = build_tile_adql(
                table, columns, tile_id, maxrec=maxrec
            )

    return try_endpoints(adql_by_endpoint, dest, catalog, maxrec=maxrec)


def fetch_count(catalog: str) -> dict[str, object]:
    """Query row count for a catalog across endpoints."""
    results: dict[str, object] = {"catalog": catalog}
    any_success = False
    for ep_name in TAP_ENDPOINTS:
        table = TABLE_NAMES[ep_name].get(catalog)
        if not table:
            continue
        ep_url = TAP_ENDPOINTS[ep_name]
        adql = build_count_adql(table)
        try:
            text = tap_sync_query(ep_url, adql, maxrec=10, timeout=120)
            # Parse CSV count result.
            lines = [line for line in text.strip().splitlines() if line.strip() and not line.startswith("#")]
            if len(lines) >= 2:
                count = lines[1].strip()
                results[ep_name] = {"table": table, "row_count": count}
                print(f"  {ep_name} ({table}): {count} rows")
                any_success = True
            else:
                results[ep_name] = {"table": table, "error": "unexpected_format"}
        except (URLError, HTTPError, OSError, TimeoutError, RuntimeError) as exc:
            results[ep_name] = {"table": table, "error": str(exc)}
            print(f"  {ep_name} ({table}): FAILED ({exc})")
    if not any_success:
        results["status"] = "failed"
    return results


def write_manifest(results: list[dict[str, object]], mode: str) -> Path:
    """Write manifest for TAP query results."""
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE_OUT_DIR / "euclid_tap_manifest.json"
    manifest = {
        "product": "euclid_q1_tap_subsets",
        "mode": mode,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoints": dict(TAP_ENDPOINTS),
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nManifest: {manifest_path}")
    return manifest_path


# Well-known Euclid Q1 field centers for convenience.
FIELD_CENTERS = {
    "edf-north": (269.73, 66.02),     # EDF-North (Euclid Deep Field North)
    "edf-south": (61.24, -48.42),      # EDF-South
    "edf-fornax": (54.0, -28.1),       # EDF-Fornax (Fornax cluster region)
    "ldn1641": (83.8, -6.5),           # LDN 1641 (Orion, performance verification)
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch governed Euclid Q1 catalog subsets via TAP/ADQL."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Cone search.
    cone_p = sub.add_parser("cone", help="Fetch a cone search subset")
    cone_p.add_argument("--catalog", choices=["mer", "phz", "spe_redshift"], default="mer")
    cone_p.add_argument("--ra", type=float, help="RA center (deg)")
    cone_p.add_argument("--dec", type=float, help="DEC center (deg)")
    cone_p.add_argument("--radius", type=float, default=0.1, help="Search radius (deg, default 0.1)")
    cone_p.add_argument("--field", choices=list(FIELD_CENTERS.keys()),
                        help="Use a known field center instead of --ra/--dec")
    cone_p.add_argument("--maxrec", type=int, default=SYNC_MAX_ROWS)

    # Tile query.
    tile_p = sub.add_parser("tile", help="Fetch all rows for a Tile ID")
    tile_p.add_argument("--catalog", choices=["mer", "phz", "spe_redshift"], default="mer")
    tile_p.add_argument("--tile-id", required=True, help="Euclid Tile ID")
    tile_p.add_argument("--maxrec", type=int, default=SYNC_MAX_ROWS)

    # Count query.
    count_p = sub.add_parser("count", help="Query row counts for catalogs")
    count_p.add_argument("--catalog", choices=["mer", "phz", "spe_redshift", "all"], default="all")

    # Raw ADQL.
    raw_p = sub.add_parser("raw", help="Execute a raw ADQL query")
    raw_p.add_argument("--adql", required=True, help="ADQL query string")
    raw_p.add_argument("--endpoint", choices=list(TAP_ENDPOINTS.keys()), default="irsa")
    raw_p.add_argument("--output", type=str, default="raw_query_result.csv", help="Output filename")
    raw_p.add_argument("--maxrec", type=int, default=SYNC_MAX_ROWS)

    args = parser.parse_args()
    BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    if args.command == "cone":
        if args.field:
            ra, dec = FIELD_CENTERS[args.field]
        elif args.ra is not None and args.dec is not None:
            ra, dec = args.ra, args.dec
        else:
            print("ERROR: provide --ra/--dec or --field", file=sys.stderr)
            return 2

        print(f"Euclid Q1 cone search: {args.catalog} at ({ra}, {dec}) r={args.radius} deg")
        result = fetch_cone(args.catalog, ra, dec, args.radius, maxrec=args.maxrec)
        results.append(result)

    elif args.command == "tile":
        print(f"Euclid Q1 tile query: {args.catalog} tile={args.tile_id}")
        result = fetch_tile(args.catalog, args.tile_id, maxrec=args.maxrec)
        results.append(result)

    elif args.command == "count":
        catalogs = ["mer", "phz", "spe_redshift"] if args.catalog == "all" else [args.catalog]
        print("Euclid Q1 row counts:")
        for cat in catalogs:
            result = fetch_count(cat)
            results.append(result)

    elif args.command == "raw":
        ep_url = TAP_ENDPOINTS[args.endpoint]
        dest = BASE_OUT_DIR / args.output
        print(f"Euclid Q1 raw ADQL query via {args.endpoint}:")
        print(f"  {args.adql}")
        try:
            text = tap_sync_query(ep_url, args.adql, maxrec=args.maxrec)
            entry = save_csv_result(
                text, dest, adql=args.adql, endpoint=ep_url, catalog="raw"
            )
            print(f"  OK ({entry.get('rows', '?')} rows)")
            results.append(entry)
        except (URLError, HTTPError, OSError, TimeoutError, RuntimeError) as exc:
            print(f"  FAILED: {exc}")
            results.append({"status": "failed", "reason": str(exc)})

    write_manifest(results, args.command)

    # Check for any failures.
    failed = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "failed")
    if failed > 0 and len(results) == failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
