#!/usr/bin/env python3
"""
Verify CSV -> TOML migration coverage and semantic parity.

Default checks:
- every data/csv/legacy/*.csv is represented in registry/legacy_csv_datasets.toml
- canonical per-dataset TOML files exist
- dataset metadata checksums and parsed rows match source CSV semantics

Manifest mode:
- pass --source-manifest to verify an explicit CSV subset instead of --source-glob.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path


DEFAULT_INDEX_PATH = "registry/legacy_csv_datasets.toml"
DEFAULT_SOURCE_GLOB = "data/csv/legacy/*.csv"
DEFAULT_CORPUS_LABEL = "legacy CSV"

csv.field_size_limit(sys.maxsize)


def _sha_text(values: object) -> str:
    blob = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def _sanitize_header_token(token: str, index: int) -> str:
    token = token.strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = token.strip("_")
    if not token:
        token = f"col_{index + 1}"
    if token[0].isdigit():
        token = f"col_{token}"
    return token


def _make_unique(tokens: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for token in tokens:
        count = seen.get(token, 0) + 1
        seen[token] = count
        if count == 1:
            out.append(token)
        else:
            out.append(f"{token}_{count}")
    return out


def _to_float(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _infer_type(values: list[str]) -> str:
    non_empty = [item.strip() for item in values if item.strip()]
    if not non_empty:
        return "empty"

    lowered = [item.lower() for item in non_empty]
    if all(item in {"true", "false", "yes", "no"} for item in lowered):
        return "bool"

    if all(re.fullmatch(r"[+-]?\d+", item) for item in non_empty):
        return "int"

    if all(_to_float(item) is not None for item in non_empty):
        return "float"

    return "string"


def _parse_source(
    path: Path,
    has_header: bool,
    delimiter: str,
    quotechar: str,
) -> tuple[list[str], list[str], list[list[str]], list[str], list[int], list[int]]:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:65536]
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    sniff_delimiter = dialect.delimiter if dialect.delimiter else ","
    sniff_quotechar = dialect.quotechar if dialect.quotechar else '"'
    if delimiter != sniff_delimiter:
        raise SystemExit(
            f"ERROR: {path}: delimiter mismatch in canonical TOML "
            f"(expected {sniff_delimiter!r}, found {delimiter!r})"
        )
    if quotechar != sniff_quotechar:
        raise SystemExit(
            f"ERROR: {path}: quotechar mismatch in canonical TOML "
            f"(expected {sniff_quotechar!r}, found {quotechar!r})"
        )

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle, dialect)
        parsed = [list(row) for row in reader]

    if parsed and parsed[0]:
        parsed[0][0] = parsed[0][0].lstrip("\ufeff")

    if has_header and parsed:
        original_header = parsed[0]
        data_rows = parsed[1:]
    else:
        original_header = []
        data_rows = parsed

    max_cols = max([len(original_header)] + [len(row) for row in data_rows] + [0])

    if max_cols == 0:
        header_tokens = []
    elif has_header:
        padded_header = original_header + [""] * (max_cols - len(original_header))
        header_tokens = [
            _sanitize_header_token(token, idx) for idx, token in enumerate(padded_header)
        ]
    else:
        header_tokens = [f"col_{idx + 1}" for idx in range(max_cols)]

    header = _make_unique(header_tokens)

    rows: list[list[str]] = []
    for row in data_rows:
        padded = row + [""] * (max_cols - len(row))
        rows.append(padded[:max_cols])

    types: list[str] = []
    non_empty_counts: list[int] = []
    empty_counts: list[int] = []
    for index in range(max_cols):
        values = [row[index] for row in rows]
        non_empty = sum(1 for value in values if value.strip())
        empty = len(values) - non_empty
        types.append(_infer_type(values))
        non_empty_counts.append(non_empty)
        empty_counts.append(empty)

    return header, original_header, rows, types, non_empty_counts, empty_counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root.",
    )
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help="Path to aggregate CSV->TOML index.",
    )
    parser.add_argument(
        "--source-glob",
        default=DEFAULT_SOURCE_GLOB,
        help="Glob of source CSV files expected to be covered by index.",
    )
    parser.add_argument(
        "--source-manifest",
        default=None,
        help="Optional path to a newline-delimited manifest of source CSV paths.",
    )
    parser.add_argument(
        "--corpus-label",
        default=DEFAULT_CORPUS_LABEL,
        help="Human-readable label for output messages.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    index_path = repo_root / args.index_path
    data = tomllib.loads(index_path.read_text(encoding="utf-8"))

    datasets = data.get("dataset", [])
    if args.source_manifest:
        manifest_path = repo_root / args.source_manifest
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        source_paths = {
            line.strip()
            for line in manifest_lines
            if line.strip() and not line.strip().startswith("#")
        }
    else:
        source_files = sorted((repo_root).glob(args.source_glob))
        source_paths = {path.relative_to(repo_root).as_posix() for path in source_files}

    failures: list[str] = []

    indexed_paths = {str(row.get("source_csv", "")) for row in datasets}
    missing_from_index = sorted(source_paths - indexed_paths)
    extra_in_index = sorted(indexed_paths - source_paths)

    if missing_from_index:
        failures.append(f"Missing {len(missing_from_index)} {args.corpus_label} entries in index.")
        failures.extend(f"- missing: {item}" for item in missing_from_index[:20])
    if extra_in_index:
        failures.append(f"Index has {len(extra_in_index)} non-existent {args.corpus_label} entries.")
        failures.extend(f"- extra: {item}" for item in extra_in_index[:20])

    if len(datasets) != len(source_paths):
        failures.append(
            f"Index dataset_count mismatch: index={len(datasets)} source={len(source_paths)}"
        )

    for row in datasets:
        source_csv = str(row.get("source_csv", ""))
        canonical_toml = str(row.get("canonical_toml", ""))
        source_path = repo_root / source_csv
        canon_path = repo_root / canonical_toml

        if not source_path.exists():
            failures.append(f"{source_csv}: source CSV missing")
            continue
        if not canon_path.exists():
            failures.append(f"{source_csv}: canonical TOML missing at {canonical_toml}")
            continue

        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if source_sha != str(row.get("source_sha256", "")):
            failures.append(f"{source_csv}: source_sha256 mismatch in index")

        canon = tomllib.loads(canon_path.read_text(encoding="utf-8"))
        dataset = canon.get("dataset", {})
        columns = canon.get("column", [])
        toml_header = list(dataset.get("header", []))
        toml_original_header = list(dataset.get("original_header", []))
        toml_rows = list(dataset.get("rows", []))

        has_header = bool(dataset.get("has_header", False))
        delimiter = str(dataset.get("delimiter", ","))
        quotechar = str(dataset.get("quotechar", '"'))

        header, original_header, parsed_rows, inferred_types, non_empty_counts, empty_counts = _parse_source(
            source_path,
            has_header=has_header,
            delimiter=delimiter,
            quotechar=quotechar,
        )

        expected_header_sha = _sha_text(header)
        expected_row_sha = _sha_text(parsed_rows)

        if header != toml_header:
            failures.append(f"{source_csv}: header mismatch")
        if original_header != toml_original_header:
            failures.append(f"{source_csv}: original_header mismatch")

        if parsed_rows != toml_rows:
            failures.append(f"{source_csv}: row payload mismatch")

        if expected_header_sha != str(dataset.get("header_value_sha256", "")):
            failures.append(f"{source_csv}: dataset header checksum mismatch")
        if expected_row_sha != str(dataset.get("row_value_sha256", "")):
            failures.append(f"{source_csv}: dataset row checksum mismatch")

        if expected_header_sha != str(row.get("header_value_sha256", "")):
            failures.append(f"{source_csv}: index header checksum mismatch")
        if expected_row_sha != str(row.get("row_value_sha256", "")):
            failures.append(f"{source_csv}: index row checksum mismatch")

        if int(dataset.get("row_count", -1)) != len(parsed_rows):
            failures.append(f"{source_csv}: row_count mismatch")
        if int(dataset.get("column_count", -1)) != len(header):
            failures.append(f"{source_csv}: column_count mismatch")

        if len(columns) != len(header):
            failures.append(f"{source_csv}: column profile count mismatch")
        else:
            for idx, column in enumerate(columns):
                if str(column.get("name", "")) != header[idx]:
                    failures.append(f"{source_csv}: column name mismatch at index {idx + 1}")
                if str(column.get("inferred_type", "")) != inferred_types[idx]:
                    failures.append(f"{source_csv}: inferred_type mismatch at index {idx + 1}")
                if int(column.get("non_empty_count", -1)) != non_empty_counts[idx]:
                    failures.append(f"{source_csv}: non_empty_count mismatch at index {idx + 1}")
                if int(column.get("empty_count", -1)) != empty_counts[idx]:
                    failures.append(f"{source_csv}: empty_count mismatch at index {idx + 1}")

    if failures:
        print(f"ERROR: {args.corpus_label} TOML parity verification failed.")
        for item in failures[:300]:
            print(f"- {item}")
        if len(failures) > 300:
            print(f"- ... and {len(failures) - 300} more failures")
        return 1

    print(
        f"OK: {args.corpus_label} corpus is fully represented in canonical TOML with semantic parity. "
        f"datasets={len(datasets)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
