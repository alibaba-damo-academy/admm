# ADMM: Automatic Decomposition Method by MindOpt

[![PyPI version](https://badge.fury.io/py/admm.svg)](https://badge.fury.io/py/admm)
[![Documentation Status](https://readthedocs.org/projects/admm/badge/?version=latest)](https://admm.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ADMM (Automatic Decomposition Method by MindOpt) is a Python library for building and solving structured optimization models. You describe objectives and constraints in a natural mathematical style, and ADMM turns that model into an efficient numerical solve through automatic canonicalization, decomposition, and a high-performance C++ backend.

The core idea is simple: model the problem, not the solver internals. ADMM is designed for formulations that combine linear or quadratic terms, smooth fitting losses, nonsmooth regularization, affine constraints, matrix structure, and custom proximal terms.

For convex models, ADMM targets the global optimum. Through user-defined proximal extensions, it also supports selected nonconvex formulations such as exact sparsity, rank constraints, and manifold-style projections, where the solver acts as a practical local method.

## Features

- **Model-first optimization workflow**: Write the mathematics directly and let ADMM handle canonicalization, decomposition, and solver orchestration automatically.
- **Rich structured problem support**: Combine linear and quadratic objectives, smooth losses, nonsmooth regularizers, affine constraints, and matrix-valued structure such as symmetry and PSD constraints.
- **Beyond standard convex modeling**: Extend the library with custom proximal operators through `UDFBase` to handle selected nonconvex penalties and constraints, including L0, rank, and manifold-style projections.
- **Built for real applications**: Use the same interface for portfolio optimization, sparse and regularized learning, covariance estimation, semidefinite modeling, compressed sensing, and signal or image processing.
- **NumPy-friendly Python API**: Work naturally with scalars, vectors, and matrices in a concise Python interface instead of hand-coding low-level updates.
- **Fast backend, practical deployment**: Run on a C++ backend with Python bindings across Linux, macOS, and Windows.

## When to Use ADMM

ADMM is a strong fit when your model combines several structured ingredients in one formulation, such as:

- linear or quadratic objectives
- smooth fitting terms such as least squares, logistic regression, or Huber loss
- nonsmooth regularization such as L1 or nuclear norm
- affine equality or inequality constraints
- structural constraints on variables such as nonnegativity, symmetry, or PSD
- matrix-valued objectives such as trace, log-determinant, or Frobenius norm
- custom proximal terms for advanced nonconvex modeling

## Installation

### From PyPI

```bash
pip install admm
```

This will automatically install all dependencies including `admmlib` (the pre-built admm C++ core dependency library).

### From Source

```bash
git clone https://github.com/alibaba-damo-academy/admm.git
cd admm
pip install . -r requirements.txt
```

## Quick Start

The following example shows a mean-variance portfolio optimization problem:

```
min    -mu^T w + gamma * w^T Sigma w
s.t.   sum(w) = 1,   w >= 0
```

The corresponding ADMM code:

```python
import admm
import numpy as np

n = 20
mu = np.abs(np.random.randn(n))
F = np.random.randn(n + 3, n)              # random factor matrix
Sigma = F.T @ F + 0.1 * np.eye(n)          # covariance matrix (PSD)
gamma = 0.5                                # risk-aversion parameter

model = admm.Model()
w = admm.Var("w", n)
model.setObjective(-mu.T @ w + gamma * (w.T @ Sigma @ w))
model.addConstr(admm.sum(w) == 1)
model.addConstr(w >= 0)
model.optimize()

print(f"status: {model.StatusString}")
print(f"obj:    {model.ObjVal:.6f}")
```

## Examples

The [`examples/`](examples/) folder contains 34 standalone scripts covering every documented use case — from basic LP/QP to UDF-based nonconvex models. Each script runs independently:

```bash
python examples/portfolio_optimization.py
python examples/sparse_logistic_regression.py
python examples/udf_l0_norm.py
```

See [`examples/README.md`](examples/README.md) for the full list.

## User-Defined Proximal Functions

The [`udf/`](udf/) folder ships 15 ready-to-use proximal operators for nonconvex and convex penalties (L0, rank, manifold projections, etc.) that go beyond standard convex modeling tools. Example usage:

```python
from udf.L0Norm import L0Norm

model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * L0Norm(x))
```

See [`udf/README.md`](udf/README.md) for the full class list, how to write your own, and how to contribute.

## Documentation

- **Online**: [admm.readthedocs.io](https://admm.readthedocs.io)
- **PDF**: [ADMM User Manual (PDF)](https://admm.readthedocs.io/_/downloads/en/latest/pdf/)
- [User Guide](https://admm.readthedocs.io/en/latest/3_User_guide/)
- [Examples](https://admm.readthedocs.io/en/latest/4_Examples/)
- [API Reference](https://admm.readthedocs.io/en/latest/5_API_Document/)

## Building from Source

### Prerequisites

- Python >= 3.9
- C++ compiler (GCC, Clang, or MSVC)
- Cython >= 0.29.0, setuptools >= 61, NumPy >= 1.20.0, SciPy >= 1.7.0
- admmlib >= 2026.4.4 (pre-built C++ core dependency)
- xelatex / latexmk (only needed for PDF documentation)

### Supported Platforms

- **Linux**: x86_64 (GCC)
- **macOS**: ARM64 (Apple Silicon, Clang)
- **Windows**: x86_64 (MSVC)

### Build and Install

```bash
pip install . -r requirements.txt
```

### Run Tests

```bash
pytest tests/                # all tests
pytest tests/test_ut.py      # unit tests
pytest tests/test_admm.py    # application tests
pytest tests/test_udf.py     # user-defined function tests
pytest tests/test_doc.py     # documentation example tests
```

### Build Documentation Locally

```bash
cd docs
bash build.sh
```

Open `docs/_build/html/index.html` in your browser. To build HTML only (without LaTeX/PDF):

```bash
cd docs
python -c "import genrst; genrst.writeRst()"
python -m sphinx -b dirhtml ./ ./_build/html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, including how to contribute new user-defined proximal function classes under `udf/`.

## Citing ADMM

If you use ADMM in your research or work, please cite:

```bibtex
@software{admm2026,
  title  = {{ADMM}: {A}utomatic {D}ecomposition {M}ethod by {MindOpt}},
  author = {{MindOpt Team, Alibaba DAMO Academy}},
  year   = {2026},
  url    = {https://github.com/alibaba-damo-academy/admm},
  note   = {Open-source Python library for structured optimization}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is maintained by the MindOpt Team at Alibaba DAMO Academy.

## Support

- **Issues**: [GitHub Issues](https://github.com/alibaba-damo-academy/admm/issues)
- **Email**: solver.damo@list.alibaba-inc.com
- **Documentation**: [https://admm.readthedocs.io](https://admm.readthedocs.io)
