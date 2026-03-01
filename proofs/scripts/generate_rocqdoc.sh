#!/bin/sh
# generate_rocqdoc.sh -- produce browsable HTML documentation via rocq doc.
#
# WHY:  Replaces Alectryon (blocked on SerAPI) with built-in rocq doc --html.
# WHAT: Generates HTML for all theory and verified .v files.
# HOW:  rocq doc --html -R ... each file, output to proofs/html/.

set -eu

PROOFS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROOFS_DIR}/html"

if ! command -v rocq >/dev/null 2>&1; then
    echo "SKIP: rocq not found"
    exit 0
fi

mkdir -p "${OUT_DIR}"

echo "Generating rocq doc HTML..."

for vfile in "${PROOFS_DIR}"/theories/*.v "${PROOFS_DIR}"/verified/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}" .v)"
    echo "  rocq doc: ${base}"
    rocq doc --html \
        -R "${PROOFS_DIR}/theories" OpenGororoba \
        -R "${PROOFS_DIR}/verified" OpenGororobaVerified \
        "${vfile}" -o "${OUT_DIR}/${base}.html" 2>/dev/null || \
        echo "  (skipped: rocq doc failed for ${base})"
done

# Generate a simple index page
cat > "${OUT_DIR}/index.html" << 'INDEX_HEAD'
<!DOCTYPE html>
<html><head><title>open_gororoba Proof Documentation</title>
<style>body{font-family:monospace;max-width:800px;margin:2em auto}
a{color:#2200cc}h2{border-bottom:1px solid #ccc;padding-bottom:0.3em}</style>
</head><body>
<h1>open_gororoba: Formal Proofs</h1>
<h2>Theories</h2><ul>
INDEX_HEAD

for vfile in "${PROOFS_DIR}"/theories/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}" .v)"
    echo "<li><a href=\"${base}.html\">${base}</a></li>" >> "${OUT_DIR}/index.html"
done

cat >> "${OUT_DIR}/index.html" << 'INDEX_MID'
</ul>
<h2>Verified Claims</h2><ul>
INDEX_MID

for vfile in "${PROOFS_DIR}"/verified/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}" .v)"
    echo "<li><a href=\"${base}.html\">${base}</a></li>" >> "${OUT_DIR}/index.html"
done

echo '</ul></body></html>' >> "${OUT_DIR}/index.html"

echo ""
echo "HTML documentation in: ${OUT_DIR}/"
echo "Open: ${OUT_DIR}/index.html"
