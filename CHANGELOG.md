# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased][Unreleased]

### Added

- 33 gradient-based UDF classes for smooth custom losses (Cauchy, Log-Cosh, Wing,
  Smooth Quantile, Gamma Regression, Poisson, Logistic, Student-t, and 25 more)
- `grad()`-based UDF path: implement `eval()` + `grad()` as an alternative to
  `eval()` + `argmin()` for smooth user-defined functions
- 6 grad-UDF documentation examples with step-by-step tutorials
- Documentation for accessing variable values via `.X` attribute in User Guide
- Error handling examples and troubleshooting guide in solve.rst
- HTML-only build instructions in README (avoiding LaTeX dependency)
- Standalone example scripts in `examples/` directory (now 40 total)

### Changed

- Clarified installation instructions with two options (one-step vs step-by-step)
- Improved documentation for L1 norm usage (`admm.norm(x, ord=1)`)
- UDF library expanded from 15 to 48 classes (15 proximal + 33 gradient-based)

### Fixed

- RST markup warnings in API documentation files

## [1.0.0][1.0.0] - 2026-04-02

### Added

- Initial release of ADMM (Automatic Decomposition Method by MindOpt)
- Support for structured optimization problems:
  - Linear Programming (LP)
  - Quadratic Programming (QP)
  - Second-Order Cone Programming (SOCP)
  - Semidefinite Programming (SDP)
- NumPy-friendly Python API with symbolic expressions
- User-defined proximal functions (UDF) for custom nonconvex modeling
- Example applications covering:
  - Portfolio optimization
  - Least squares and ridge regression
  - LASSO and sparse logistic regression
  - Support vector machines
  - Robust PCA
  - Entropy maximization
  - Image deblurring and signal processing
- Comprehensive documentation:
  - User Guide with step-by-step modeling workflow
  - API Reference with detailed function signatures
  - Example gallery with runnable code
- Automated testing suite with 430+ tests
- CI/CD pipeline with GitHub Actions

### Technical Details

- C++ backend with Python bindings (via Cython)
- Cross-platform support: Linux (x86_64), macOS (ARM64), Windows (x86_64)
- Automatic canonicalization and decomposition
- ADMM-based solver with warm-start capability
- Pre-built `admmlib` dependency for easy installation

### Known Limitations

- Variable dimensions limited to at most 2 (no higher-dimensional arrays)
- PDF documentation requires LaTeX installation (xelatex/latexmk)
- Some advanced solver parameters require manual tuning for large-scale problems

[Unreleased]: https://github.com/alibaba-damo-academy/admm/compare/1.0.0...HEAD
[1.0.0]: https://github.com/alibaba-damo-academy/admm/releases/tag/1.0.0
