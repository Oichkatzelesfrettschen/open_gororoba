#!/bin/sh

set -eu

trim_spaces() {
    tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//'
}

print_pkg_status() {
    pkg="$1"
    if ! command -v pkg-config >/dev/null 2>&1; then
        printf -- "- pkg-config %s: skipped (pkg-config not installed)\n" "$pkg"
        return 0
    fi

    if pkg-config --exists "$pkg" 2>/dev/null; then
        version="$(pkg-config --modversion "$pkg" 2>/dev/null | trim_spaces)"
        libs="$(pkg-config --libs "$pkg" 2>/dev/null | trim_spaces)"
        if [ -n "$libs" ]; then
            printf -- "- pkg-config %s: present (version %s; libs %s)\n" "$pkg" "$version" "$libs"
        else
            printf -- "- pkg-config %s: present (version %s)\n" "$pkg" "$version"
        fi
    else
        printf -- "- pkg-config %s: missing\n" "$pkg"
    fi
}

printf 'Native BLAS and LAPACK detection\n'
printf '%s\n' '- Repo-exposed Cargo feature: openblas-system'
printf '%s\n' '- Opt-in mechanism: the `Cargo.toml` `[features]` table, enabled with `cargo ... --features <name>`'
printf '%s\n' '- Default behavior: no native BLAS feature is enabled unless a build command opts in explicitly'
printf '%s\n' '- Probe method: pkg-config for installed native libraries plus platform notes for non-pkg-config backends'

print_pkg_status openblas
print_pkg_status blis
print_pkg_status blas
print_pkg_status lapack
print_pkg_status lapacke

case "$(uname -s 2>/dev/null || printf 'unknown')" in
    Darwin)
        printf '%s\n' '- Host note: macOS may provide Accelerate as a system framework, but this repo does not currently expose an `accelerate` feature.'
        ;;
esac

printf '%s\n' '- Repo mapping: `openblas-system` expects a system OpenBLAS install and is the only repo-exposed native BLAS feature.'
printf '%s\n' '- Upstream Burn also forwards `openblas`, `accelerate`, and `blas-netlib`, but this repo intentionally does not expose them because the source-backed path is not reproducible in offline gates.'
printf '%s\n' '- Upstream `blas-src` supports `blis`, `intel-mkl`, and `r`, but Burn does not forward those selectors for this crate, so they are not repo opt-ins today.'
