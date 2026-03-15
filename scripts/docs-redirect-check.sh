#!/usr/bin/env bash
set -euo pipefail

DOCS_SITE_DIR="${1:-target/site-docs}"

need_file() {
  local path="$1"
  if [ ! -f "$path" ]; then
    echo "ERROR: required docs artifact missing: $path" >&2
    exit 1
  fi
}

check_text() {
  local file="$1"
  local needle="$2"
  if ! grep -Fq "$needle" "$file"; then
    echo "ERROR: expected marker missing in $file: $needle" >&2
    exit 1
  fi
}

simulate_redirect() {
  local path="$1"
  local first=""
  local root="/"
  local suffix=""

  first="${path#/}"
  first="${first%%/*}"
  if [ -n "$first" ] && [ "$first" != "book" ] && [ "$first" != "rustdoc" ]; then
    root="/$first/"
  fi

  local legacy_prefixes=(
    "/.cache/cargo-default-target/doc"
    "/cache/cargo-default-target/doc"
    "/.cache/gate-target/doc"
    "/cache/gate-target/doc"
    "/target/docs-target/doc"
    "/target/doc"
  )

  for prefix in "${legacy_prefixes[@]}"; do
    if [[ "$path" == *"$prefix"* ]]; then
      suffix="${path#*${prefix}}"
      printf '%s\n' "${root}rustdoc${suffix}"
      return 0
    fi
  done

  if [ "$path" = "$root" ] || [ "$path" = "${root}book" ] || [ "$path" = "${root}book/" ]; then
    printf '%s\n' "${root}book/"
    return 0
  fi

  if [ "$path" = "${root}rustdoc" ] || [ "$path" = "${root}rustdoc/" ]; then
    printf '%s\n' "${root}rustdoc/"
    return 0
  fi

  printf '%s\n' "$root"
}

need_file "${DOCS_SITE_DIR}/404.html"
need_file "${DOCS_SITE_DIR}/book.html"
need_file "${DOCS_SITE_DIR}/rustdoc.html"
need_file "${DOCS_SITE_DIR}/index.html"
need_file "${DOCS_SITE_DIR}/.nojekyll"

check_text "${DOCS_SITE_DIR}/404.html" "/.cache/cargo-default-target/doc"
check_text "${DOCS_SITE_DIR}/404.html" "/cache/gate-target/doc"
check_text "${DOCS_SITE_DIR}/404.html" "window.location.replace"
check_text "${DOCS_SITE_DIR}/book.html" "./book/"
check_text "${DOCS_SITE_DIR}/rustdoc.html" "./rustdoc/"

status=0
for case in \
  "/.cache/cargo-default-target/doc/pkg/struct.SomeType.html|/rustdoc/pkg/struct.SomeType.html" \
  "/cache/gate-target/doc/index.html|/rustdoc/index.html" \
  "/repo/.cache/gate-target/doc/pkg/index.html|/repo/rustdoc/pkg/index.html" \
  "/book|/book/" \
  "/rustdoc|/rustdoc/" \
  "/repo/book|/repo/book/" \
  "/repo/rustdoc|/repo/rustdoc/" \
  "/repo/other|/repo/"
  do
  input="${case%%|*}"
  expected="${case#*|}"
  expected="${expected#|}"
  output="$(simulate_redirect "$input")"
  if [ "$output" != "$expected" ]; then
    echo "ERROR: redirect mismatch for '$input': expected '$expected', got '$output'" >&2
    status=1
  else
    echo "OK: $input -> $output"
  fi
done

if [ "$status" -ne 0 ]; then
  exit 1
fi

echo "OK: docs redirect checks passed."
