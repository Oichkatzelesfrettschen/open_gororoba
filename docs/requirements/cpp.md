# Requirements: C++ Kernels (Conan + CMake)

This repo includes optional C++ kernels under `cpp/` (C++20) with Makefile targets:

- `make cpp-deps`
- `make cpp-build`
- `make cpp-test`
- `make cpp-bench`
- `make cpp-clean`

## Toolchain

Typical requirements:
- CMake >= 3.25
- Conan 2.x
- A C++20 compiler (GCC or Clang)

Notes:
- Conan profiles are machine-specific. Prefer keeping profiles local and avoid
  committing machine-specific paths.
- This is an optional acceleration layer; core correctness should remain covered
  by Python tests and verifiers.

## Build (example)

```bash
make cpp-build
make cpp-test
```

If Conan/CMake are missing, install them via your system package manager or via
pipx, then retry.

