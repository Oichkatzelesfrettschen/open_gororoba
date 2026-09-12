#!/bin/sh
# Bound Cargo, nextest, Rust test, and Rayon concurrency before compilation.
# Local mode reserves workstation capacity and caps the shared budget at two.
# CI mode may use half of the runner CPUs. Both modes reserve memory per worker.
set -eu

mode="${1:-local}"
case "$mode" in
    local|ci) ;;
    *)
        printf 'usage: %s [local|ci]\n' "$0" >&2
        exit 2
        ;;
esac

threads="${GOROROBA_WORKER_TEST_CPUS:-}"
if [ -z "$threads" ]; then
    threads="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2\n')"
fi
case "$threads" in
    ''|*[!0-9]*|0)
        printf 'worker CPU count must be a positive integer, got %s\n' "$threads" >&2
        exit 2
        ;;
esac

available_memory_mb="${GOROROBA_WORKER_TEST_MEMORY_MB:-}"
if [ -z "$available_memory_mb" ] && [ -r /proc/meminfo ]; then
    available_memory_kb="$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo)"
    case "$available_memory_kb" in
        ''|*[!0-9]*) ;;
        *) available_memory_mb="$((available_memory_kb / 1024))" ;;
    esac
fi

# A container can expose host MemAvailable while enforcing a smaller cgroup
# limit. Use the remaining cgroup allowance when it is the tighter boundary.
if [ -z "${GOROROBA_WORKER_TEST_MEMORY_MB:-}" ] && \
   [ -r /sys/fs/cgroup/memory.max ] && [ -r /sys/fs/cgroup/memory.current ]; then
    cgroup_max="$(cat /sys/fs/cgroup/memory.max)"
    cgroup_current="$(cat /sys/fs/cgroup/memory.current)"
    case "$cgroup_max:$cgroup_current" in
        *[!0-9:]*|:*) ;;
        *)
            if [ "$cgroup_max" -gt "$cgroup_current" ]; then
                cgroup_available_mb="$(((cgroup_max - cgroup_current) / 1048576))"
                if [ -z "$available_memory_mb" ] || \
                   [ "$cgroup_available_mb" -lt "$available_memory_mb" ]; then
                    available_memory_mb="$cgroup_available_mb"
                fi
            fi
            ;;
    esac
fi
case "$available_memory_mb" in
    ''|*[!0-9]*) available_memory_mb=2048 ;;
esac

cpu_budget="$((threads / 2))"
if [ "$cpu_budget" -lt 1 ]; then cpu_budget=1; fi

# Rust compilation and linking can use substantially more memory than the
# process count suggests. Reserve 2 GiB per concurrent worker.
memory_budget="$((available_memory_mb / 2048))"
if [ "$memory_budget" -lt 1 ]; then memory_budget=1; fi

budget="$cpu_budget"
if [ "$memory_budget" -lt "$budget" ]; then budget="$memory_budget"; fi
if [ "$mode" = local ] && [ "$budget" -gt 2 ]; then budget=2; fi

printf '%s\n' "$budget"
