#!/bin/sh

set -eu

emit_if_positive() {
    value="$1"
    case "$value" in
        ''|*[!0-9]*)
            return 1
            ;;
        0)
            return 1
            ;;
        *)
            printf '%s\n' "$value"
            return 0
            ;;
    esac
}

if emit_if_positive "${WORKER_BUDGET:-}" >/dev/null 2>&1; then
    emit_if_positive "${WORKER_BUDGET:-}"
    exit 0
fi

detect_threads() {
    if command -v nproc >/dev/null 2>&1; then
        threads="$(nproc 2>/dev/null || true)"
        if emit_if_positive "${threads:-}" >/dev/null 2>&1; then
            printf '%s\n' "$threads"
            return 0
        fi
    fi

    if command -v getconf >/dev/null 2>&1; then
        threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
        if emit_if_positive "${threads:-}" >/dev/null 2>&1; then
            printf '%s\n' "$threads"
            return 0
        fi
    fi

    if command -v sysctl >/dev/null 2>&1; then
        threads="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
        if emit_if_positive "${threads:-}" >/dev/null 2>&1; then
            printf '%s\n' "$threads"
            return 0
        fi
    fi

    if command -v lscpu >/dev/null 2>&1; then
        threads="$(
            lscpu -p=cpu 2>/dev/null \
                | awk '!/^#/ { print $1 }' \
                | wc -l \
                | tr -d ' '
        )"
        if emit_if_positive "${threads:-}" >/dev/null 2>&1; then
            printf '%s\n' "$threads"
            return 0
        fi
    fi

    if [ -r /proc/cpuinfo ]; then
        threads="$(
            awk '
                /^processor[[:space:]]*:/ { count += 1 }
                END { print count }
            ' /proc/cpuinfo \
                | tr -d ' '
        )"
        if emit_if_positive "${threads:-}" >/dev/null 2>&1; then
            printf '%s\n' "$threads"
            return 0
        fi
    fi

    printf '1\n'
}

threads="$(detect_threads)"
budget="$(expr "$threads" / 2)"

if [ "$budget" -lt 1 ]; then
    budget=1
fi

printf '%s\n' "$budget"
