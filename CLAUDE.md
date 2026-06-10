# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pycorrfunc is a pythonic rewrite of the Corrfunc software infrastructure for computing correlation functions. The core goal is to provide a modern Python interface with wheels that support true runtime SIMD dispatch. It wraps optimized C/C++ kernels using nanobind.

**Key characteristics:**
- Pure Python + C/C++ hybrid (no Cython)
- Meson build system for compilation
- Dual precision modules: `_pycorrfunc` (double) and `_pycorrfuncf` (float)
- SIMD dispatch at runtime via `isa_t` enum (SSE4.2, AVX, AVX512)
- Uses OpenMP for parallelization

## Relevant Science Papers
Look in `papers/*.tex` for background on any science-y questions.

## Development Setup

### Install for Development

```bash
uv venv
. .venv/bin/activate
uv export --only-group=dev | uv pip install -r -
uv pip install -v .
```

Or for development builds (editable):
```bash
uv pip install -v -e .
```

### Build Configuration

Key meson options in `meson.options`:
- `openmp` (default: enabled) — Enable OpenMP parallelization
- `double_accum` (default: enabled) — Use double precision for accumulators in float module

Enable custom options at build time:
```bash
uv pip install -v -Csetup-args="-Dopenmp=disabled" .
```

## Running Tests

```bash
pytest -v                    # Run all tests
pytest -v -k 'not avx512'   # Run without AVX512 (CI default for wide compatibility)
pytest -v tests/test_theory.py::test_name  # Run specific test
```

Tests use a brute-force reference implementation for validation. See `tests/test_theory.py`.

## Code Structure

### Python Layer
- **[src/pycorrfunc/__init__.py](src/pycorrfunc/__init__.py)** — Entry point, imports C extensions and orchestrates API
- **[src/pycorrfunc/theory.py](src/pycorrfunc/theory.py)** — High-level correlation function APIs
- **[src/pycorrfunc/isa.py](src/pycorrfunc/isa.py)** — ISA enum helper and dispatch logic

### C/C++ Core ([lib/](lib/) and [include/](include/))

**Main entry:** [lib/main.cpp](lib/main.cpp) — nanobind module definition for both `_pycorrfunc` and `_pycorrfuncf`

**SIMD Kernels:** All in [lib/theory/DD/](lib/theory/DD/)
- `kernel_sse42.c` — SSE4.2 kernels
- `kernel_avx.c` — AVX kernels
- `kernel_avx512.c` — AVX512 kernels
- `kernel_fallback.c` — Fallback (scalar) implementation

The meson build compiles SIMD kernels into separate static libraries, and the linker dispatches to them at runtime via the `isa_t` enum.

**Common utilities:** [lib/common/](lib/common/)
- `cellarray.c`, `gridlink.c`, `gridlink_utils.c` — Spatial data structures
- `options.c`, `weights.c` — Configuration and weight handling
- `cpp_utils.cpp` — C++ helper functions (OpenMP, error handling)

### Build Configuration
- **[meson.build](meson.build)** — Main build script; orchestrates SIMD library creation and Python extension modules
- **[pyproject.toml](pyproject.toml)** — PEP 517 metadata; specifies meson-python backend and cibuildwheel options

## Key Architecture Details

### Dual-Module Design

Two extension modules are built from the same sources:
- `_pycorrfunc` — Double precision (64-bit floats)
- `_pycorrfuncf` — Single precision (32-bit floats)

Controlled by `-DPYCORRFUNC_USE_DOUBLE` and `-DNB_DOMAIN` preprocessor flags at compile time.

### Runtime ISA Dispatch

The `isa_t` enum is defined in both modules. Python code selects an ISA by name and passes the enum value to the C++ bindings, which then call the appropriate SIMD kernel. This allows wheels to run on systems with varying CPU capabilities without relying on CPU feature detection at import time.

### Precision Accumulation Option

The `double_accum` meson option lets the float module (`_pycorrfuncf`) use 64-bit accumulators for `ravg` and `wavg` even though outputs are 32-bit. This improves numerical accuracy without full double-precision computation.

## Build System Notes

- **Meson** is preferred over setuptools; [meson.build](meson.build) is the single source of truth.
- **nanobind** (not pybind11) is used for Python bindings; faster compile times and smaller binaries.
- **setuptools_scm** is used for versioning from git tags.
- **Limited API** support (Python 3.12+): opt-in via `-Dpython.allow_limited_api=true` for forward compatibility wheels.

The `lib/cpp_utils.cpp` file centralizes C++ logic (OpenMP pragmas, error handling) to keep the main module clean.

## Testing and CI

- **Local tests** via pytest (see "Running Tests" above).
- **GitHub Actions** ([.github/workflows/tests.yaml](.github/workflows/tests.yaml)): Runs on Python 3.12–3.14 with both GCC and Clang, tests the no-AVX512 codepath.
- **Jenkins** (internal): Runs on Flatiron infrastructure for additional OS/compiler coverage.
- **cibuildwheel** ([pyproject.toml](pyproject.toml)): Builds wheels for multiple platforms; enforces ABI3 audit on limited-API wheels.

## Common Tasks

### Add a new ISA kernel
1. Create `lib/theory/DD/kernel_newisa.c` with the kernel function.
2. Update [meson.build](meson.build) to compile it (add to `simd_check_kwargs` or create a static library like AVX512).
3. Update `isa_t` enum in [lib/main.cpp](lib/main.cpp) and export from Python in [src/pycorrfunc/isa.py](src/pycorrfunc/isa.py).

### Debug build issues
- Check that Meson >= 1.3.0 is installed.
- If Clang is used, ensure `libomp-dev` is installed (OpenMP support).
- Build with verbose output: `uv pip install -v -Csetup-args="-Dcompile_args=['-v']" .`
- Run `meson --version` and ensure the system compiler supports C++17.

### Verify SIMD dispatch
The `theory` module accepts an `isa` parameter. Test dispatch manually:
```python
import pycorrfunc.theory as theory
result = theory.some_function(..., isa='avx')  # Force specific ISA
```

## Release Process

1. Tag the version: `git tag -a v2.0.0 -m 'Releasing version 2.0.0'`
2. Push the tag: `git push origin v2.0.0`
3. GitHub Actions triggers wheel builds and PyPI upload via cibuildwheel.
4. Verify build and upload succeeded in Actions logs.
5. Create a GitHub release from the tag.

See [RELEASING.md](RELEASING.md) for details.

## Code Style

- **Python:** Enforced by ruff (linter + formatter) in pre-commit.
- **C/C++:** Enforced by clang-format (config in [.clang-format](.clang-format)).
- **Quote style:** Single quotes in Python (configured in [pyproject.toml](pyproject.toml)).

Run pre-commit checks: `pre-commit run --all-files`

## Known Limitations and TODOs

- This repo is a **proof-of-concept** and not intended for public use (per README).
- AVX512 support is checked at compile time; not natively supported by Meson's SIMD module yet.
- Compile commands symlink install is not yet working (see [meson.build](meson.build) line 154).

## References

- **Corrfunc:** https://github.com/manodeep/Corrfunc (original C library)
- **nanobind:** https://nanobind.readthedocs.io
- **Meson:** https://mesonbuild.com
- **astropy:** Used for cosmological unit handling
