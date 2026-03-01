#!/bin/sh
# dep_graph.sh -- generate proof dependency graph in DOT format.
#
# Output: proofs/metrics/dep_graph.dot (and .pdf if dot is available)
#
# WHY:  Visualizes which theories depend on which, for the paper.
# WHAT: Parses _CoqProject and Require Import lines to build a DAG.
# HOW:  grep imports from .v files, emit DOT, render with graphviz.

set -eu

PROOFS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROOFS_DIR}/metrics"
DOT_FILE="${OUT_DIR}/dep_graph.dot"

mkdir -p "${OUT_DIR}"

cat > "${DOT_FILE}" << 'HEADER'
digraph ProofDependencies {
  rankdir=BT;
  node [shape=box, fontsize=10, fontname="monospace"];
  edge [color="#666666"];

  // Theory nodes (blue)
  node [style=filled, fillcolor="#d4e6f1"];
HEADER

# Add theory nodes
for vfile in "${PROOFS_DIR}"/theories/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}" .v)"
    echo "  \"${base}\";" >> "${DOT_FILE}"
done

# Add verified nodes (green)
echo "" >> "${DOT_FILE}"
echo "  // Verified claim nodes (green)" >> "${DOT_FILE}"
echo "  node [style=filled, fillcolor=\"#d5f5e3\"];" >> "${DOT_FILE}"

for vfile in "${PROOFS_DIR}"/verified/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}" .v)"
    echo "  \"${base}\";" >> "${DOT_FILE}"
done

# Add edges based on Require Import
echo "" >> "${DOT_FILE}"
echo "  // Dependencies" >> "${DOT_FILE}"

for vfile in "${PROOFS_DIR}"/theories/*.v "${PROOFS_DIR}"/verified/*.v; do
    [ -f "${vfile}" ] || continue
    src="$(basename "${vfile}" .v)"
    # Parse "From OpenGororoba Require Import X Y Z." lines
    grep -oP 'From OpenGororoba Require Import \K[^.]+' "${vfile}" 2>/dev/null | tr ' ' '\n' | while read -r dep; do
        [ -n "${dep}" ] && echo "  \"${src}\" -> \"${dep}\";" >> "${DOT_FILE}"
    done
    grep -oP 'From OpenGororobaVerified Require Import \K[^.]+' "${vfile}" 2>/dev/null | tr ' ' '\n' | while read -r dep; do
        [ -n "${dep}" ] && echo "  \"${src}\" -> \"${dep}\";" >> "${DOT_FILE}"
    done
done

echo "}" >> "${DOT_FILE}"

# Render if graphviz is available
if command -v dot >/dev/null 2>&1; then
    dot -Tpdf "${DOT_FILE}" -o "${OUT_DIR}/dep_graph.pdf" 2>/dev/null && \
        echo "Generated: ${OUT_DIR}/dep_graph.pdf" || \
        echo "PDF generation failed (non-fatal)"
    dot -Tsvg "${DOT_FILE}" -o "${OUT_DIR}/dep_graph.svg" 2>/dev/null && \
        echo "Generated: ${OUT_DIR}/dep_graph.svg" || \
        echo "SVG generation failed (non-fatal)"
fi

# Convert DOT to TikZ for paper inclusion (requires dot2tex)
if command -v dot2tex >/dev/null 2>&1; then
    dot2tex -f tikz --figonly "${DOT_FILE}" > "${OUT_DIR}/dep_graph.tikz" 2>/dev/null && \
        echo "Generated: ${OUT_DIR}/dep_graph.tikz" || \
        echo "TikZ generation failed (non-fatal)"
fi

echo "Generated: ${DOT_FILE}"
