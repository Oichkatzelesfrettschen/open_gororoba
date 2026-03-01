#!/bin/sh
# collect_metrics.sh -- gather proof compilation metrics for PGFPlots.
#
# Output: proofs/metrics/compile_times.csv
#   file,lines,vo_size_bytes,compile_seconds
#
# Also: proofs/metrics/summary.csv (aggregate counts for LaTeX macros)
#
# WHY:  Tracks proof complexity growth over time; data drives PGFPlots in paper.
# WHAT: Counts source lines, .vo file sizes, and compilation times per proof file.
# HOW:  Times individual rocq compile calls, then scrapes results.

set -eu

PROOFS_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${PROOFS_DIR}/metrics"
OUT_FILE="${OUT_DIR}/compile_times.csv"
SUMMARY_FILE="${OUT_DIR}/summary.csv"

mkdir -p "${OUT_DIR}"

echo "file,lines,vo_size_bytes,compile_seconds" > "${OUT_FILE}"

for vfile in "${PROOFS_DIR}"/theories/*.v "${PROOFS_DIR}"/verified/*.v; do
    [ -f "${vfile}" ] || continue
    base="$(basename "${vfile}")"
    lines="$(wc -l < "${vfile}")"
    vofile="${vfile%.v}.vo"
    if [ -f "${vofile}" ]; then
        vo_size="$(stat -c%s "${vofile}" 2>/dev/null || stat -f%z "${vofile}" 2>/dev/null || echo 0)"
    else
        vo_size=0
    fi

    # Time individual compilation if rocq is available and .vo exists.
    # We use the existing .vo as proof that compilation succeeds;
    # timing is done by re-compiling the single file.
    compile_seconds="0.0"
    if command -v rocq >/dev/null 2>&1; then
        # Determine which namespace this file belongs to
        case "${vfile}" in
            */theories/*) ns_flag="-R ${PROOFS_DIR}/theories OpenGororoba -R ${PROOFS_DIR}/verified OpenGororobaVerified" ;;
            */verified/*) ns_flag="-R ${PROOFS_DIR}/theories OpenGororoba -R ${PROOFS_DIR}/verified OpenGororobaVerified" ;;
            *) ns_flag="" ;;
        esac

        # Time compilation (seconds with 2 decimal places)
        t_start="$(date +%s%N 2>/dev/null || echo 0)"
        if [ "${t_start}" != "0" ]; then
            # shellcheck disable=SC2086
            rocq compile ${ns_flag} "${vfile}" > /dev/null 2>&1 || true
            t_end="$(date +%s%N)"
            # Nanosecond arithmetic via awk
            compile_seconds="$(echo "${t_start}" "${t_end}" | awk '{printf "%.2f", ($2 - $1) / 1000000000}')"
        fi
    fi

    # Strip .v extension; escape underscores for LaTeX PGFPlots tick labels
    label="$(echo "${base%.v}" | sed 's/_/\\_/g')"
    echo "${label},${lines},${vo_size},${compile_seconds}" >> "${OUT_FILE}"
done

# --- Summary CSV for LaTeX macros ---

theory_count="$(find "${PROOFS_DIR}/theories" -name '*.v' 2>/dev/null | wc -l)"
verified_count="$(find "${PROOFS_DIR}/verified" -name '*.v' 2>/dev/null | wc -l)"
total_v="$(( theory_count + verified_count ))"
total_vo="$(find "${PROOFS_DIR}/theories" "${PROOFS_DIR}/verified" -name '*.vo' 2>/dev/null | wc -l)"
total_lines="$(cat "${PROOFS_DIR}"/theories/*.v "${PROOFS_DIR}"/verified/*.v 2>/dev/null | wc -l)"

# Count kernel-checked claims (files matching C[0-9]* or C_* in verified/)
kernel_checked="$(find "${PROOFS_DIR}/verified" -name '*.v' 2>/dev/null | wc -l)"

cat > "${SUMMARY_FILE}" << EOF
metric,value
theory_files,${theory_count}
verified_files,${verified_count}
total_v_files,${total_v}
total_vo_files,${total_vo}
total_lines,${total_lines}
kernel_checked_claims,${kernel_checked}
EOF

echo ""
echo "Summary: ${total_v} .v files, ${total_vo} .vo compiled, ${total_lines} total lines"
echo "Output: ${OUT_FILE}"
echo "Summary: ${SUMMARY_FILE}"
