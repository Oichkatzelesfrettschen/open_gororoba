#!/bin/sh
# Fetch NASA OMNI2 hourly solar wind + IMF data.
#
# Downloads yearly ASCII files from NASA/GSFC SPDF.
# Each file is ~2.8 MB (8760 rows, 57 columns per row).
#
# Source: King & Papitashvili (2005), J. Geophys. Res., 110, A02104.
# URL: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
#
# Usage:
#   bin/fetch_omni.sh              # fetch 2020-2025 (default)
#   bin/fetch_omni.sh 1990 2024    # fetch custom range
#   bin/fetch_omni.sh --skip       # skip if files exist

set -eu

BASE_URL="https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_"
OUT_DIR="data/external/omni2"
YEAR_START="${1:-2020}"
YEAR_END="${2:-2025}"
SKIP_EXISTING=false

if [ "${1:-}" = "--skip" ]; then
    SKIP_EXISTING=true
    YEAR_START="${2:-2020}"
    YEAR_END="${3:-2025}"
fi

mkdir -p "$OUT_DIR"

fetched=0
skipped=0
failed=0

for year in $(seq "$YEAR_START" "$YEAR_END"); do
    fname="omni2_${year}.dat"
    out="$OUT_DIR/$fname"

    if [ "$SKIP_EXISTING" = true ] && [ -f "$out" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    url="${BASE_URL}${year}.dat"
    printf "Fetching %s ... " "$fname"

    if curl -sS --fail -o "$out" "$url" 2>/dev/null; then
        lines=$(wc -l < "$out" | tr -d ' ')
        bytes=$(wc -c < "$out" | tr -d ' ')
        printf "OK (%s lines, %s bytes)\n" "$lines" "$bytes"
        fetched=$((fetched + 1))
    elif command -v wget >/dev/null 2>&1; then
        printf "(curl failed, trying wget) ... "
        if wget -q -O "$out" "$url" 2>/dev/null; then
            lines=$(wc -l < "$out" | tr -d ' ')
            printf "OK (%s lines)\n" "$lines"
            fetched=$((fetched + 1))
        else
            printf "FAILED\n"
            rm -f "$out"
            failed=$((failed + 1))
        fi
    else
        printf "FAILED\n"
        rm -f "$out"
        failed=$((failed + 1))
    fi
done

echo ""
echo "OMNI2 fetch: $fetched fetched, $skipped skipped, $failed failed"
echo "Data directory: $OUT_DIR"
