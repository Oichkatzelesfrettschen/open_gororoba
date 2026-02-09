#!/usr/bin/env python3
"""
Migrate legacy CSV corpora into canonical TOML datasets.

Source:
- data/csv/legacy/*.csv

Canonical outputs:
- registry/legacy_csv_datasets.toml
- registry/data/legacy_csv/*.toml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


LEGACY_GLOB = "data/csv/legacy/*.csv"
CANON_DIR = "registry/data/legacy_csv"
INDEX_PATH = "registry/legacy_csv_datasets.toml"
UPDATED_STAMP = "2026-02-09"


def _assert_ascii(text: str, context: str) -> None:
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII output in {context}: {sample!r}")


def _esc(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _slugify(name: str) -> str:
    stem = name.rsplit(".", 1)[0].lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "dataset"
    return stem


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


def _sniff(sample: str) -> tuple[csv.Dialect, bool]:
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    try:
        has_header = sniffer.has_header(sample)
    except csv.Error:
        has_header = True
    return dialect, has_header


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


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    slug: str
    source_csv: str
    source_sha256: str
    source_size_bytes: int
    has_header: bool
    delimiter: str
    quotechar: str
    row_count: int
    column_count: int
    header: list[str]
    original_header: list[str]
    rows: list[list[str]]
    header_value_sha256: str
    row_value_sha256: str
    canonical_toml: str
    column_types: list[str]
    non_empty_counts: list[int]
    empty_counts: list[int]


def _parse_csv(path: Path) -> tuple[bool, str, str, list[str], list[str], list[list[str]]]:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:65536]
    dialect, has_header = _sniff(sample)

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

    max_cols = max(
        [len(original_header)] + [len(row) for row in data_rows] + [0]
    )

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

    normalized_rows: list[list[str]] = []
    for row in data_rows:
        padded = row + [""] * (max_cols - len(row))
        normalized_rows.append(padded[:max_cols])

    quotechar = dialect.quotechar if dialect.quotechar else '"'
    delimiter = dialect.delimiter if dialect.delimiter else ","

    return has_header, delimiter, quotechar, header, original_header, normalized_rows


def _profile_columns(rows: list[list[str]], column_count: int) -> tuple[list[str], list[int], list[int]]:
    types: list[str] = []
    non_empty_counts: list[int] = []
    empty_counts: list[int] = []

    for index in range(column_count):
        values = [row[index] for row in rows]
        non_empty = sum(1 for value in values if value.strip())
        empty = len(values) - non_empty
        col_type = _infer_type(values)
        types.append(col_type)
        non_empty_counts.append(non_empty)
        empty_counts.append(empty)

    return types, non_empty_counts, empty_counts


def _render_dataset(dataset: Dataset) -> str:
    lines: list[str] = []
    lines.append("# Canonical TOML dataset migrated from legacy CSV.")
    lines.append("# Generated by src/scripts/analysis/migrate_legacy_csv_to_toml.py")
    lines.append("")
    lines.append("[dataset]")
    lines.append(f"id = {_esc(dataset.dataset_id)}")
    lines.append(f"slug = {_esc(dataset.slug)}")
    lines.append(f"source_csv = {_esc(dataset.source_csv)}")
    lines.append(f"source_sha256 = {_esc(dataset.source_sha256)}")
    lines.append(f"source_size_bytes = {dataset.source_size_bytes}")
    lines.append(f"has_header = {str(dataset.has_header).lower()}")
    lines.append(f"delimiter = {_esc(dataset.delimiter)}")
    lines.append(f"quotechar = {_esc(dataset.quotechar)}")
    lines.append(f"row_count = {dataset.row_count}")
    lines.append(f"column_count = {dataset.column_count}")
    lines.append(f"header_value_sha256 = {_esc(dataset.header_value_sha256)}")
    lines.append(f"row_value_sha256 = {_esc(dataset.row_value_sha256)}")
    lines.append('migrated_on = "2026-02-09"')
    lines.append('migrated_by = "src/scripts/analysis/migrate_legacy_csv_to_toml.py"')
    lines.append("")

    lines.append("header = [")
    for token in dataset.header:
        lines.append(f"  {_esc(token)},")
    lines.append("]")
    lines.append("")

    lines.append("original_header = [")
    for token in dataset.original_header:
        lines.append(f"  {_esc(token)},")
    lines.append("]")
    lines.append("")

    lines.append("rows = [")
    for row in dataset.rows:
        encoded = ", ".join(_esc(value) for value in row)
        lines.append(f"  [{encoded}],")
    lines.append("]")
    lines.append("")

    for idx, name in enumerate(dataset.header):
        lines.append("[[column]]")
        lines.append(f"index = {idx + 1}")
        lines.append(f"name = {_esc(name)}")
        original_name = dataset.original_header[idx] if idx < len(dataset.original_header) else ""
        lines.append(f"original_name = {_esc(original_name)}")
        lines.append(f"inferred_type = {_esc(dataset.column_types[idx])}")
        lines.append(f"non_empty_count = {dataset.non_empty_counts[idx]}")
        lines.append(f"empty_count = {dataset.empty_counts[idx]}")
        lines.append("")

    return "\n".join(lines)


def _render_index(datasets: list[Dataset]) -> str:
    lines: list[str] = []
    lines.append("# Canonical index for legacy CSV datasets migrated to TOML.")
    lines.append("# Generated by src/scripts/analysis/migrate_legacy_csv_to_toml.py")
    lines.append("")
    lines.append("[legacy_csv_datasets]")
    lines.append('updated = "2026-02-09"')
    lines.append("authoritative = true")
    lines.append(f"source_glob = {_esc(LEGACY_GLOB)}")
    lines.append(f"canonical_dir = {_esc(CANON_DIR)}")
    lines.append(f"dataset_count = {len(datasets)}")
    lines.append("")

    for dataset in datasets:
        lines.append("[[dataset]]")
        lines.append(f"id = {_esc(dataset.dataset_id)}")
        lines.append(f"slug = {_esc(dataset.slug)}")
        lines.append(f"source_csv = {_esc(dataset.source_csv)}")
        lines.append(f"source_sha256 = {_esc(dataset.source_sha256)}")
        lines.append(f"source_size_bytes = {dataset.source_size_bytes}")
        lines.append(f"canonical_toml = {_esc(dataset.canonical_toml)}")
        lines.append(f"row_count = {dataset.row_count}")
        lines.append(f"column_count = {dataset.column_count}")
        lines.append(f"has_header = {str(dataset.has_header).lower()}")
        lines.append(f"delimiter = {_esc(dataset.delimiter)}")
        lines.append(f"quotechar = {_esc(dataset.quotechar)}")
        lines.append(f"header_value_sha256 = {_esc(dataset.header_value_sha256)}")
        lines.append(f"row_value_sha256 = {_esc(dataset.row_value_sha256)}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Repository root.",
    )
    parser.add_argument(
        "--legacy-glob",
        default=LEGACY_GLOB,
        help="Glob for legacy CSV sources.",
    )
    parser.add_argument(
        "--out-index",
        default=INDEX_PATH,
        help="Output TOML index path.",
    )
    parser.add_argument(
        "--out-dir",
        default=CANON_DIR,
        help="Output directory for per-dataset TOML files.",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted((root).glob(args.legacy_glob))
    if not sources:
        raise SystemExit(f"ERROR: no source CSV files found for glob {args.legacy_glob!r}")

    datasets: list[Dataset] = []

    for idx, source in enumerate(sources, start=1):
        rel = source.relative_to(root).as_posix()
        raw = source.read_bytes()
        source_sha = hashlib.sha256(raw).hexdigest()

        has_header, delimiter, quotechar, header, original_header, rows = _parse_csv(source)
        row_count = len(rows)
        column_count = len(header)
        column_types, non_empty_counts, empty_counts = _profile_columns(rows, column_count)

        dataset_id = f"LC-{idx:04d}"
        slug = _slugify(source.name)
        canon_name = f"{dataset_id}_{slug}.toml"
        canon_rel = Path(args.out_dir).joinpath(canon_name).as_posix()

        dataset = Dataset(
            dataset_id=dataset_id,
            slug=slug,
            source_csv=rel,
            source_sha256=source_sha,
            source_size_bytes=len(raw),
            has_header=has_header,
            delimiter=delimiter,
            quotechar=quotechar,
            row_count=row_count,
            column_count=column_count,
            header=header,
            original_header=original_header,
            rows=rows,
            header_value_sha256=_sha_text(header),
            row_value_sha256=_sha_text(rows),
            canonical_toml=canon_rel,
            column_types=column_types,
            non_empty_counts=non_empty_counts,
            empty_counts=empty_counts,
        )
        datasets.append(dataset)

    rendered_by_path: dict[Path, str] = {}
    for dataset in datasets:
        out_path = root / dataset.canonical_toml
        text = _render_dataset(dataset)
        _assert_ascii(text, str(out_path))
        rendered_by_path[out_path] = text

    # Remove stale generated TOML dataset files.
    expected = {path.resolve() for path in rendered_by_path}
    for existing in out_dir.glob("*.toml"):
        if existing.resolve() not in expected:
            existing.unlink()

    for out_path, text in rendered_by_path.items():
        out_path.write_text(text, encoding="utf-8")

    index_text = _render_index(datasets)
    index_path = root / args.out_index
    _assert_ascii(index_text, str(index_path))
    index_path.write_text(index_text, encoding="utf-8")

    print(f"Wrote {index_path} with {len(datasets)} datasets.")
    print(f"Wrote {len(datasets)} canonical TOML dataset files under {out_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
