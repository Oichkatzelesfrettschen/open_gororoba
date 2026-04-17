#!/usr/bin/env python3
"""DOI audit for refs_heliosphere.bib.

Resolves each @article entry against the CrossRef REST API and flags any entry
where >= 2 metadata fields (author-family, title, journal, year) do not match
the bib file, or where the DOI 404s.

Usage:
    python scripts/check_refs.py [--strict] [bib_file]

--strict: exit 1 if any mismatch or 404 is found (for CI use).
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib import request, error as urllib_error


BIB_DEFAULT = Path(__file__).parent.parent / "docs/latex/heliosphere/refs_heliosphere.bib"
CROSSREF_API = "https://api.crossref.org/works/{doi}"
REQUEST_DELAY = 0.5  # seconds between CrossRef requests (polite pool)


def normalize(s: str) -> str:
    """Fold to ASCII lowercase for fuzzy comparison."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"'", "", s)  # collapse possessives: "humanity's" -> "humanitys"
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())


def extract_entries(bib_text: str) -> list[dict]:
    """Parse @article entries from bib file."""
    entries = []
    # Find all @article blocks
    for m in re.finditer(r"@article\{([^,]+),(.*?)\n\}", bib_text, re.DOTALL):
        key = m.group(1).strip()
        body = m.group(2)
        entry = {"key": key, "raw": m.group(0)}
        # Extract fields
        for field in ("author", "title", "journal", "year", "doi", "volume", "pages"):
            fm = re.search(rf"\b{field}\s*=\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}", body, re.DOTALL)
            if fm:
                val = re.sub(r"\s+", " ", fm.group(1)).strip()
                # Strip LaTeX braces and commands for comparison
                val = re.sub(r"\{([^{}]*)\}", r"\1", val)
                val = re.sub(r"\\[a-zA-Z]+\s*", "", val)
                entry[field] = val.strip()
        entries.append(entry)
    return entries


def fetch_crossref(doi: str) -> dict | None:
    """Fetch CrossRef metadata for a DOI. Returns None on 404 or error."""
    url = CROSSREF_API.format(doi=doi.strip())
    try:
        req = request.Request(url, headers={"User-Agent": "check_refs/1.0 (open_gororoba CI)"})
        with request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("message", {})
    except urllib_error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


def compare_entry(entry: dict, cr: dict) -> list[str]:
    """Return list of mismatch descriptions."""
    mismatches = []

    # Year
    bib_year = entry.get("year", "")
    cr_year = str(cr.get("published", {}).get("date-parts", [[]])[0][0] or "")
    if bib_year and cr_year and bib_year != cr_year:
        mismatches.append(f"year: bib={bib_year!r} cr={cr_year!r}")

    # Journal
    bib_journal = normalize(entry.get("journal", ""))
    cr_journals = cr.get("container-title", [])
    cr_journal = normalize(cr_journals[0]) if cr_journals else ""
    if bib_journal and cr_journal:
        # Check if either is substring of the other (handles abbreviations)
        if bib_journal not in cr_journal and cr_journal not in bib_journal:
            mismatches.append(f"journal: bib={entry.get('journal','')!r} cr={cr_journals[0] if cr_journals else ''!r}")

    # Title (fuzzy: check first 6 words match)
    bib_title_words = normalize(entry.get("title", "")).split()[:6]
    cr_title = normalize(cr.get("title", [""])[0])
    cr_title_words = cr_title.split()[:6]
    if bib_title_words and cr_title_words and bib_title_words != cr_title_words:
        mismatches.append(f"title prefix: bib={' '.join(bib_title_words)!r} cr={' '.join(cr_title_words)!r}")

    # First author family name
    bib_authors = entry.get("author", "")
    bib_first_family = normalize(bib_authors.split(" and ")[0].split(",")[0].strip())
    cr_authors = cr.get("author", [])
    cr_first_family = normalize(cr_authors[0].get("family", "")) if cr_authors else ""
    if bib_first_family and cr_first_family and bib_first_family not in cr_first_family and cr_first_family not in bib_first_family:
        mismatches.append(f"first-author: bib={bib_first_family!r} cr={cr_first_family!r}")

    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="DOI audit for bib file")
    parser.add_argument("bib_file", nargs="?", default=str(BIB_DEFAULT))
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any mismatch")
    args = parser.parse_args()

    bib_text = Path(args.bib_file).read_text(encoding="utf-8")
    entries = extract_entries(bib_text)

    print(f"Auditing {len(entries)} @article entries in {args.bib_file}\n")

    failures = []
    warnings = []

    for entry in entries:
        key = entry["key"]
        doi = entry.get("doi", "")

        if not doi:
            warnings.append(f"  WARN  {key}: no DOI field -- cannot verify")
            print(f"  WARN  {key}: no DOI")
            continue

        time.sleep(REQUEST_DELAY)
        cr = fetch_crossref(doi)

        if cr is None:
            failures.append(f"  FAIL  {key}: DOI {doi!r} returned 404 or network error")
            print(f"  FAIL  {key}: DOI 404 ({doi})")
            continue

        mismatches = compare_entry(entry, cr)
        if len(mismatches) >= 2:
            msg = f"  FAIL  {key}: {len(mismatches)} field mismatches: {'; '.join(mismatches)}"
            failures.append(msg)
            print(msg)
        elif mismatches:
            msg = f"  WARN  {key}: 1 minor mismatch: {mismatches[0]}"
            warnings.append(msg)
            print(msg)
        else:
            print(f"  OK    {key} ({doi})")

    print(f"\n--- Summary ---")
    print(f"  OK:       {len(entries) - len(failures) - len(warnings)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Failures: {len(failures)}")

    if failures:
        print("\nFailed entries:")
        for f in failures:
            print(f)

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
