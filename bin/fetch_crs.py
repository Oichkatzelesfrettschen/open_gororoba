#!/usr/bin/env python3
"""Fetch Voyager CRS (Cosmic Ray Subsystem) products with source fallbacks.

CRS data comes in two distinct product families:
  - rate_ascii: legacy SPDF count-rate ASCII files by cadence
  - daily_flux: calibrated daily proton-flux exports staged from CDAWeb/Open Data
  - jupiter_encounter: PDS/PPI CRS encounter-era flux bundles hosted at UCLA

The CLI keeps the provenance explicit in a deterministic manifest, including the
selected source kind and any fallback reason.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URLS = {
    1: "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager1/particle/crs/",
    2: "https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager2/particle/crs/",
}
HAPI_URLS = {
    1: "https://cdaweb.gsfc.nasa.gov/hapi/data",
    2: "https://cdaweb.gsfc.nasa.gov/hapi/data",
}
HAPI_DATASET_IDS = {
    1: "VOYAGER1_CRS_DAILY_FLUX",
    2: "VOYAGER2_CRS_DAILY_FLUX",
}
NOTESV_URLS = {
    1: "https://cdaweb.gsfc.nasa.gov/misc/NotesV.html#VOYAGER1_CRS_DAILY_FLUX",
    2: "https://cdaweb.gsfc.nasa.gov/misc/NotesV.html#VOYAGER2_CRS_DAILY_FLUX",
}
OPEN_DATA_PAGES = {
    1: "https://data.nasa.gov/dataset/Voyager-1-Cosmic-Ray-Subsystem-CRS-Proton-and-/5gej-xc7j",
    2: "https://data.nasa.gov/api/3/action/package_search?q=Voyager%202%20Cosmic%20Ray%20Subsystem%20%28CRS%29%20Proton%20and%20Helium%20Energy%20Fluxes",
}
CATALOG_DATA_GOV_SEARCH = "https://catalog.data.gov/api/3/action/package_search"
DATA_NASA_GOV_SEARCH = "https://data.nasa.gov/api/3/action/package_search"
PDS_PPI_ROOT = "https://pds-ppi.igpp.ucla.edu/data"
PDS_PPI_DATASETS = {
    1: {
        "dataset_id": "VG1-J-CRS-5-SUMM-FLUX-V1.0",
        "title": "Voyager 1 Jupiter CRS summary flux bundle",
        "start_year": 1979,
        "end_year": 1979,
    },
    2: {
        "dataset_id": "VG2-J-CRS-5-SUMM-FLUX-V1.0",
        "title": "Voyager 2 Jupiter CRS summary flux bundle",
        "start_year": 1979,
        "end_year": 1979,
    },
}

USER_AGENT = "gororoba-fetch/0.1 (research)"
HREF_RE = re.compile(r'href="([^"]+)"')
YEAR_RE = re.compile(r"(19|20)\d{2}")
DAILY_FLUX_METADATA = {
    1: {
        "title": (
            "Voyager 1 Cosmic Ray Subsystem (CRS) Proton and Helium "
            "Energy Fluxes, Level H3 (H3), Daily Data"
        ),
        "doi": "10.48322/rmq3-1z61",
        "spase_resource_id": "spase://NASA/NumericalData/Voyager1/CRS/Flux/EnergyFlux/HydrogenHelium/LevelH3/CDF/P1D",
    },
    2: {
        "title": (
            "Voyager 2 Cosmic Ray Subsystem (CRS) Proton and Helium "
            "Energy Fluxes, Level H3 (H3), Daily Data"
        ),
        "doi": "10.48322/f4xx-mq95",
        "spase_resource_id": "",
    },
}


def fetch_text(url: str) -> str:
    """Fetch a text page."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def fetch_json(url: str) -> dict:
    """Fetch and decode JSON."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


def discover_crs_files(spacecraft: int, cadence: str, years: range) -> dict[int, list[str]]:
    """Discover CRS files for a cadence by scraping the SPDF directory index."""
    base = BASE_URLS[spacecraft].rstrip("/") + f"/{cadence}/"
    html = fetch_text(base)
    files_by_year: dict[int, list[str]] = {year: [] for year in years}

    for href in HREF_RE.findall(html):
        fname = href.rsplit("/", maxsplit=1)[-1]
        if not fname.endswith(".asc"):
            continue
        match = YEAR_RE.search(fname)
        if not match:
            continue
        year = int(match.group(0))
        if year in files_by_year:
            files_by_year[year].append(fname)

    return files_by_year


def build_entry(url: str, dest: str, status: str) -> dict[str, object]:
    """Build manifest metadata for a fetched or existing file."""
    entry: dict[str, object] = {"url": url, "path": dest, "status": status}
    if not os.path.exists(dest):
        return entry
    with open(dest, "rb") as fh:
        data = fh.read()
    entry["bytes"] = len(data)
    entry["sha256"] = hashlib.sha256(data).hexdigest()
    entry["filename"] = os.path.basename(dest)
    return entry


def copy_inbox_files(
    src_dir: str,
    dest_dir: str,
    *,
    skip_existing: bool,
) -> list[dict[str, object]]:
    """Copy pre-staged files from an inbox directory into the repo layout."""
    if not os.path.isdir(src_dir):
        return []
    entries: list[dict[str, object]] = []
    for fname in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue
        dest = os.path.join(dest_dir, fname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if skip_existing and os.path.exists(dest):
            entry = build_entry(src, dest, "skipped")
        else:
            shutil.copy2(src, dest)
            entry = build_entry(src, dest, "copied")
        entry["source_kind"] = "inbox"
        entries.append(entry)
    return entries


def fetch_crs_file(url: str, dest: str, *, skip_existing: bool) -> dict[str, object]:
    """Fetch one CRS file and return manifest metadata."""
    if skip_existing and os.path.exists(dest):
        return build_entry(url, dest, "skipped")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 100:
            print(
                f"  [warn] {os.path.basename(dest)}: response too small "
                f"({len(data)} bytes), skipping"
            )
            return {"url": url, "path": dest, "status": "failed", "bytes": len(data)}
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(data)
        print(f"  [ok]   {os.path.basename(dest)}")
        return build_entry(url, dest, "fetched")
    except (urllib.error.URLError, OSError) as exc:
        print(f"  [fail] {os.path.basename(dest)}: {exc}")
        return {"url": url, "path": dest, "status": "failed", "reason": str(exc)}


def write_manifest(path: str, payload: dict[str, object]) -> None:
    """Write a deterministic manifest."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def probe_host(host: str, port: int = 443, timeout: float = 3.0) -> dict[str, object]:
    """Return a simple TCP reachability probe for HTTPS hosts."""
    result: dict[str, object] = {"host": host, "port": port, "reachable": False}
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        if infos:
            result["address"] = infos[0][4][0]
        with socket.create_connection((host, port), timeout=timeout):
            result["reachable"] = True
    except OSError as exc:
        result["reason"] = str(exc)
    return result


def package_search(base_url: str, query: str) -> list[dict]:
    """Query a CKAN package_search endpoint."""
    url = f"{base_url}?rows=10&q={urllib.parse.quote(query)}"
    data = fetch_json(url)
    return data.get("result", {}).get("results", [])


def extract_extra(package: dict, key: str) -> object | None:
    """Get one CKAN extras value."""
    for extra in package.get("extras", []):
        if extra.get("key") == key:
            return extra.get("value")
    return None


def discover_daily_flux_package(spacecraft: int, source: str) -> dict[str, object]:
    """Discover daily-flux metadata from reachable CKAN mirrors."""
    expected = DAILY_FLUX_METADATA[spacecraft]
    queries = [
        expected["title"],
        expected["doi"],
        f"Voyager {spacecraft} Cosmic Ray Subsystem (CRS) Proton and Helium Energy Fluxes",
    ]
    search_roots: list[tuple[str, str]]
    if source == "catalog_data_gov":
        search_roots = [("catalog_data_gov", CATALOG_DATA_GOV_SEARCH)]
    elif source == "open_data":
        search_roots = [("open_data", DATA_NASA_GOV_SEARCH)]
    else:
        search_roots = [("catalog_data_gov", CATALOG_DATA_GOV_SEARCH)]

    for source_kind, base_url in search_roots:
        for query in queries:
            try:
                packages = package_search(base_url, query)
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                continue
            for package in packages:
                title = package.get("title", "")
                identifier = extract_extra(package, "identifier")
                landing_page = extract_extra(package, "landingPage")
                doi_url = f"https://doi.org/{expected['doi']}"
                if (
                    title == expected["title"]
                    or identifier == doi_url
                    or landing_page == doi_url
                ):
                    return {
                        "source_kind": source_kind,
                        "package_id": package.get("id"),
                        "package_name": package.get("name"),
                        "title": title,
                        "identifier": identifier,
                        "landing_page": landing_page,
                        "modified": extract_extra(package, "modified"),
                        "publisher": extract_extra(package, "publisher"),
                        "resources": package.get("resources", []),
                    }
    return {
        "source_kind": "unresolved",
        "package_id": None,
        "package_name": None,
        "title": expected["title"],
        "identifier": f"https://doi.org/{expected['doi']}",
        "landing_page": f"https://doi.org/{expected['doi']}",
        "modified": None,
        "publisher": None,
        "resources": [],
    }


def fetch_hapi_csv(
    spacecraft: int,
    start_year: int,
    end_year: int,
    dest: str,
    *,
    skip_existing: bool = False,
) -> dict[str, object]:
    """Attempt a calibrated daily-flux export from CDAWeb HAPI."""
    query = urllib.parse.urlencode(
        {
            "id": HAPI_DATASET_IDS[spacecraft],
            "time.min": f"{start_year}-01-01T00:00:00Z",
            "time.max": f"{end_year + 1}-01-01T00:00:00Z",
            "format": "csv",
        }
    )
    url = f"{HAPI_URLS[spacecraft]}?{query}"
    return fetch_crs_file(url, dest, skip_existing=skip_existing)


def parse_pds_index_tab(text: str) -> list[dict[str, str]]:
    """Parse a small PDS INDEX.TAB file into per-product rows."""
    rows: list[dict[str, str]] = []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return rows
    header = [part.strip() for part in lines[0].split(",")]
    for line in lines[1:]:
        parts = [part.strip().strip('"') for part in line.split(",")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts, strict=False)))
    return rows


def fetch_pds_bundle(
    spacecraft: int,
    out_dir: str,
    *,
    skip_existing: bool,
) -> list[dict[str, object]]:
    """Fetch the PDS/PPI Jupiter encounter CRS bundle for one spacecraft."""
    dataset = PDS_PPI_DATASETS[spacecraft]
    dataset_root = f"{PDS_PPI_ROOT}/{dataset['dataset_id']}"
    index_url = f"{dataset_root}/INDEX/INDEX.TAB"
    index_lbl_url = f"{dataset_root}/INDEX/INDEX.LBL"
    readme_url = f"{dataset_root}/AAREADME.TXT"
    entries: list[dict[str, object]] = []

    bundle_dir = os.path.join(out_dir, dataset["dataset_id"])
    os.makedirs(bundle_dir, exist_ok=True)

    readme_path = os.path.join(bundle_dir, "AAREADME.TXT")
    index_tab_path = os.path.join(bundle_dir, "INDEX", "INDEX.TAB")
    index_lbl_path = os.path.join(bundle_dir, "INDEX", "INDEX.LBL")

    readme_entry = fetch_crs_file(readme_url, readme_path, skip_existing=skip_existing)
    readme_entry["source_kind"] = "pds_ppi"
    readme_entry["role"] = "volume_readme"
    entries.append(readme_entry)

    index_tab_entry = fetch_crs_file(index_url, index_tab_path, skip_existing=skip_existing)
    index_tab_entry["source_kind"] = "pds_ppi"
    index_tab_entry["role"] = "index_tab"
    entries.append(index_tab_entry)

    index_lbl_entry = fetch_crs_file(index_lbl_url, index_lbl_path, skip_existing=skip_existing)
    index_lbl_entry["source_kind"] = "pds_ppi"
    index_lbl_entry["role"] = "index_label"
    entries.append(index_lbl_entry)

    if not os.path.exists(index_tab_path):
        return entries

    with open(index_tab_path, encoding="utf-8") as fh:
        index_rows = parse_pds_index_tab(fh.read())

    bundle_abs = os.path.realpath(bundle_dir)

    for row in index_rows:
        label_rel = row.get("FILE_SPECIFICATION_NAME", "").strip()
        if not label_rel:
            continue
        label_rel = label_rel.lstrip("/")
        data_rel = label_rel[:-4] + ".TAB" if label_rel.endswith(".LBL") else label_rel
        label_url = f"{dataset_root}/{label_rel}"
        data_url = f"{dataset_root}/{data_rel}"
        label_dest = os.path.realpath(os.path.join(bundle_dir, *label_rel.split("/")))
        data_dest = os.path.realpath(os.path.join(bundle_dir, *data_rel.split("/")))

        # Reject paths that escape the bundle directory (defense against
        # compromised upstream INDEX.TAB with ".." components).
        if not label_dest.startswith(bundle_abs + os.sep):
            print(f"  [SKIP-TRAVERSAL] {label_rel}", file=sys.stderr)
            continue
        if not data_dest.startswith(bundle_abs + os.sep):
            print(f"  [SKIP-TRAVERSAL] {data_rel}", file=sys.stderr)
            continue

        label_entry = fetch_crs_file(label_url, label_dest, skip_existing=skip_existing)
        label_entry["source_kind"] = "pds_ppi"
        label_entry["role"] = "data_label"
        label_entry["product_id"] = row.get("PRODUCT_ID")
        label_entry["start_time"] = row.get("START_TIME")
        label_entry["stop_time"] = row.get("STOP_TIME")
        entries.append(label_entry)

        data_entry = fetch_crs_file(data_url, data_dest, skip_existing=skip_existing)
        data_entry["source_kind"] = "pds_ppi"
        data_entry["role"] = "data_table"
        data_entry["product_id"] = row.get("PRODUCT_ID")
        data_entry["start_time"] = row.get("START_TIME")
        data_entry["stop_time"] = row.get("STOP_TIME")
        entries.append(data_entry)

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spacecraft",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        help="Voyager spacecraft numbers (default: 1 2)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="1977-2024",
        help="Year range as START-END (default: 1977-2024)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/external/voyager",
        help="Output base directory (default: data/external/voyager)",
    )
    parser.add_argument(
        "--cadence",
        choices=["one_day", "six_hour", "fifteen_minute"],
        default="one_day",
        help="CRS cadence directory to fetch from (default: one_day)",
    )
    parser.add_argument(
        "--product",
        choices=["rate_ascii", "daily_flux", "jupiter_encounter"],
        default="rate_ascii",
        help="CRS product family to stage (default: rate_ascii)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "spdf", "cdas", "open_data", "catalog_data_gov", "pds_ppi", "inbox"],
        default="auto",
        help="Acquisition source (default: auto)",
    )
    parser.add_argument(
        "--inbox-dir",
        default="data/inbox/heliosphere",
        help="Manual-import inbox root used by the inbox fallback",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if file already exists",
    )
    args = parser.parse_args()

    start_year, end_year = (int(y) for y in args.years.split("-"))
    years = range(start_year, end_year + 1)

    total_fetched = 0
    total_failed = 0

    for sc in args.spacecraft:
        if args.product == "daily_flux":
            subdir = "flux_daily"
        elif args.product == "jupiter_encounter":
            subdir = "pds_ppi/jupiter_encounter"
        else:
            subdir = os.path.join("crs", args.cadence)
        out_dir = os.path.join(args.out_dir, f"voyager{sc}", subdir)
        print(
            f"Fetching Voyager {sc} CRS {args.product} via {args.source} "
            f"({start_year}-{end_year}) -> {out_dir}"
        )
        base = BASE_URLS[sc].rstrip("/") + f"/{args.cadence}/"
        files: list[dict[str, object]] = []
        selected_source = args.source

        if args.product == "rate_ascii" and args.source in {"auto", "spdf"}:
            try:
                discovered = discover_crs_files(sc, args.cadence, years)
            except (urllib.error.URLError, OSError) as exc:
                print(f"  [fail] SPDF index discovery failed for Voyager {sc}: {exc}")
                if args.source == "spdf":
                    discovered = {}
                else:
                    selected_source = "inbox"
                    discovered = {}

            if selected_source != "inbox":
                for year in years:
                    matches = discovered.get(year, [])
                    if not matches:
                        print(f"  [fail] no CRS file discovered for {year}")
                        total_failed += 1
                        files.append(
                            {
                                "url": base,
                                "path": os.path.join(out_dir, f"missing_{year}.asc"),
                                "status": "failed",
                                "year": year,
                                "reason": "missing_from_index",
                                "source_kind": "spdf",
                            }
                        )
                        continue

                    for fname in sorted(matches):
                        dest = os.path.join(out_dir, fname)
                        entry = fetch_crs_file(
                            base + fname, dest, skip_existing=args.skip_existing
                        )
                        entry["year"] = year
                        entry["source_kind"] = "spdf"
                        files.append(entry)
                        if entry["status"] == "fetched":
                            total_fetched += 1
                        elif entry["status"] == "failed":
                            total_failed += 1

        if selected_source == "inbox" or args.source == "inbox":
            inbox_dir = os.path.join(
                args.inbox_dir,
                f"voyager{sc}",
                "daily_flux" if args.product == "daily_flux" else (
                    "jupiter_encounter" if args.product == "jupiter_encounter" else args.cadence
                ),
            )
            inbox_entries = copy_inbox_files(
                inbox_dir,
                out_dir,
                skip_existing=args.skip_existing,
            )
            if inbox_entries:
                files.extend(inbox_entries)
                total_fetched += sum(entry["status"] == "copied" for entry in inbox_entries)
            else:
                files.append(
                    {
                        "url": NOTESV_URLS[sc] if args.product == "daily_flux" else base,
                        "path": out_dir,
                        "status": "failed",
                        "reason": "no_inbox_files_found",
                        "source_kind": "inbox",
                    }
                )
                total_failed += 1

        if args.product == "daily_flux" and args.source in {
            "auto",
            "cdas",
            "open_data",
            "catalog_data_gov",
        }:
            metadata = discover_daily_flux_package(
                sc,
                "catalog_data_gov" if args.source == "catalog_data_gov" else (
                    "open_data" if args.source == "open_data" else "auto"
                ),
            )
            cda_probe = probe_host("cdaweb.gsfc.nasa.gov")
            output_name = f"voyager{sc}_crs_daily_flux_{start_year}_{end_year}.csv"
            output_path = os.path.join(out_dir, output_name)
            metadata_entry = {
                "url": metadata.get("landing_page") or NOTESV_URLS[sc],
                "path": output_path,
                "status": "metadata_only",
                "source_kind": metadata.get("source_kind", "unresolved"),
                "notes_url": NOTESV_URLS[sc],
                "dataset_page": OPEN_DATA_PAGES[sc],
                "doi": DAILY_FLUX_METADATA[sc]["doi"],
                "spase_resource_id": DAILY_FLUX_METADATA[sc]["spase_resource_id"],
                "hapi_dataset_id": HAPI_DATASET_IDS[sc],
                "package_id": metadata.get("package_id"),
                "package_name": metadata.get("package_name"),
                "publisher": metadata.get("publisher"),
                "modified": metadata.get("modified"),
                "resources": metadata.get("resources", []),
                "cdaweb_probe": cda_probe,
            }

            if cda_probe.get("reachable"):
                entry = fetch_hapi_csv(
                    sc, start_year, end_year, output_path,
                    skip_existing=args.skip_existing,
                )
                entry.update(metadata_entry)
                entry["status"] = entry.get("status", "fetched")
                files.append(entry)
                if entry["status"] == "fetched":
                    total_fetched += 1
                elif entry["status"] == "failed":
                    total_failed += 1
            else:
                metadata_entry["reason"] = (
                    "mirror_metadata_resolved_but_gsfc_data_host_unreachable_from_shell"
                )
                files.append(metadata_entry)

        if args.product == "jupiter_encounter" and args.source in {"auto", "pds_ppi"}:
            dataset = PDS_PPI_DATASETS[sc]
            if end_year < dataset["start_year"] or start_year > dataset["end_year"]:
                files.append(
                    {
                        "url": f"{PDS_PPI_ROOT}/{dataset['dataset_id']}/",
                        "path": out_dir,
                        "status": "skipped",
                        "source_kind": "pds_ppi",
                        "dataset_id": dataset["dataset_id"],
                        "reason": "requested_year_range_outside_dataset_coverage",
                    }
                )
            else:
                pds_entries = fetch_pds_bundle(sc, out_dir, skip_existing=args.skip_existing)
                files.extend(pds_entries)
                total_fetched += sum(entry["status"] == "fetched" for entry in pds_entries)
                total_failed += sum(entry["status"] == "failed" for entry in pds_entries)

        manifest_path = os.path.join(
            args.out_dir,
            f"voyager{sc}",
            f"crs_{args.product}_{args.cadence}_{start_year}_{end_year}_manifest.json",
        )
        write_manifest(
            manifest_path,
            {
                "spacecraft": sc,
                "cadence": args.cadence,
                "product": args.product,
                "source": args.source,
                "start_year": start_year,
                "end_year": end_year,
                "notes_url": NOTESV_URLS[sc],
                "dataset_page": OPEN_DATA_PAGES[sc],
                "pds_ppi_dataset": PDS_PPI_DATASETS[sc]["dataset_id"],
                "files": files,
            },
        )
        print(f"  manifest: {manifest_path}")

    print(f"\nSummary: fetched={total_fetched}, failed={total_failed}")
    if total_failed > 0 and total_fetched == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
