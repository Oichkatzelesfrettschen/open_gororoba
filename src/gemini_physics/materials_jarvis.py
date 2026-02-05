from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

FIGSHARE_ARTICLE_ID_JARVIS_DFT = 6815699


@dataclass(frozen=True)
class FigshareFile:
    id: int
    name: str
    size: int
    download_url: str
    md5: str | None


def list_figshare_files(article_id: int) -> list[FigshareFile]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    files: list[FigshareFile] = []
    for entry in payload.get("files", []):
        files.append(
            FigshareFile(
                id=int(entry["id"]),
                name=str(entry["name"]),
                size=int(entry["size"]),
                download_url=str(entry["download_url"]),
                md5=entry.get("computed_md5") or entry.get("supplied_md5"),
            )
        )
    return files


def select_figshare_file(files: list[FigshareFile], name_regex: str) -> FigshareFile:
    rx = re.compile(name_regex)
    matches = [f for f in files if rx.search(f.name)]
    if not matches:
        raise ValueError(f"No figshare file matched regex: {name_regex!r}")
    # Prefer smaller downloads by default (good for reproducible CI).
    matches.sort(key=lambda f: f.size)
    return matches[0]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def unzip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            dest = out_dir / member.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, dest.open("wb") as dst:
                dst.write(src.read())
            extracted.append(dest)
    return extracted


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list of records in {path}, got {type(data)}")
    return data


def jarvis_subset_to_dataframe(
    records: list[dict[str, Any]],
    *,
    n: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n <= 0:
        raise ValueError("n must be positive")
    if len(records) == 0:
        raise ValueError("No records to sample from")

    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    rows = []
    for i in idx:
        rec = records[int(i)]
        atoms = rec.get("atoms") or {}
        atom_elements = atoms.get("elements") or []
        if not isinstance(atom_elements, list):
            atom_elements = []

        # Derive basic chemistry fields if missing in the record.
        formula = rec.get("formula")
        if formula is None and atom_elements:
            counts: dict[str, int] = {}
            for el in atom_elements:
                counts[str(el)] = counts.get(str(el), 0) + 1
            formula = "".join(
                f"{el}{counts[el] if counts[el] != 1 else ''}"
                for el in sorted(counts.keys())
            )

        elements = rec.get("elements")
        if elements is None and atom_elements:
            elements = sorted({str(e) for e in atom_elements})

        nelements = rec.get("nelements")
        if nelements is None and elements is not None:
            nelements = len(elements)

        # Lattice volume (Angstrom^3) if present.
        volume = rec.get("volume")
        lattice_mat = atoms.get("lattice_mat")
        if volume is None and isinstance(lattice_mat, list) and len(lattice_mat) == 3:
            try:
                volume = float(abs(np.linalg.det(np.array(lattice_mat, dtype=float))))
            except Exception:
                volume = None

        rows.append(
            {
                "jid": rec.get("jid"),
                "formula": formula,
                "elements": ",".join(elements or []),
                "nelements": nelements,
                "energy_per_atom": rec.get("energy_per_atom"),
                "formation_energy_peratom": rec.get("formation_energy_peratom"),
                "optb88vdw_bandgap": rec.get("optb88vdw_bandgap"),
                "ehull": rec.get("ehull"),
                "spg_symbol": rec.get("spg_symbol"),
                "spg_number": rec.get("spg_number"),
                "density": rec.get("density"),
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


def write_provenance(
    dest: Path,
    *,
    article_id: int,
    file: FigshareFile,
    downloaded_path: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "figshare",
        "article_id": article_id,
        "file_id": file.id,
        "file_name": file.name,
        "file_size": file.size,
        "download_url": file.download_url,
        "md5": file.md5,
        "downloaded_path": os.fspath(downloaded_path),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
