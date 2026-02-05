from __future__ import annotations

import argparse
from pathlib import Path

from gemini_physics.materials_jarvis import (
    FIGSHARE_ARTICLE_ID_JARVIS_DFT,
    download,
    jarvis_subset_to_dataframe,
    list_figshare_files,
    load_json_records,
    select_figshare_file,
    unzip,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and cache a small JARVIS-DFT subset.")
    parser.add_argument(
        "--name-regex",
        default=r"^jdft_3d-4-26-2020\.zip$",
        help="Regex to select a Figshare file name (default: small 2020 zip).",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of records to sample.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling.")
    parser.add_argument(
        "--out-csv",
        default="data/csv/materials_jarvis_subset.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/external/jarvis_dft",
        help="Cache directory for downloaded/extracted files.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = list_figshare_files(FIGSHARE_ARTICLE_ID_JARVIS_DFT)
    chosen = select_figshare_file(files, args.name_regex)

    zip_path = cache_dir / chosen.name
    download(chosen.download_url, zip_path)

    extracted = unzip(zip_path, cache_dir / zip_path.stem)
    json_files = [p for p in extracted if p.suffix.lower() == ".json"]
    if not json_files:
        raise SystemExit(f"No JSON file found inside {zip_path}")
    # Prefer the largest JSON (often the main payload) if multiple are present.
    json_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    json_path = json_files[0]

    records = load_json_records(json_path)
    df = jarvis_subset_to_dataframe(records, n=args.n, seed=args.seed)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_provenance(
        cache_dir / "PROVENANCE.json",
        article_id=FIGSHARE_ARTICLE_ID_JARVIS_DFT,
        file=chosen,
        downloaded_path=zip_path,
    )

    print(f"Wrote {len(df)} records to {out_csv}")
    print(f"Provenance: {cache_dir / 'PROVENANCE.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
