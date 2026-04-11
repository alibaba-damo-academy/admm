# Contributing to ADMM

Thank you for your interest in contributing to ADMM! This document provides guidelines and instructions for contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please:

1. Check if the issue already exists in the [issue tracker](https://github.com/alibaba-damo-academy/admm/issues)
2. Update to the latest version to see if the bug is already fixed

When submitting a bug report, please include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **Environment information**:
  - Python version
  - Operating system
  - ADMM version
  - NumPy/SciPy versions
- **Code samples** or test cases that reproduce the issue

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Clear use case** - why is this enhancement needed?
- **Detailed description** of the proposed feature
- **Possible implementation** approach (if you have ideas)
- **Examples** of how the feature would be used

### Contributing a UDF Class

Community-contributed user-defined proximal functions live in the top-level `udf/` directory. The package is auto-discovered, so in most cases adding a new UDF only requires adding one new Python file whose class name matches the file name.

For example, `udf/L0Norm.py` contains a single class `L0Norm`:

```python
import admm
import math
import numpy as np


class L0Norm(admm.UDFBase):
    """L0 norm: count of nonzero entries.

    Function:

        f(x) = ||x||_0 = #{i : x_i != 0}

    Proximal operator (coordinatewise hard threshold):

        prox_{lam * f}(v)_i = v_i    if |v_i| > sqrt(2 * lam)
                            = 0      otherwise

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        return float(np.count_nonzero(np.abs(vector) > 1e-12))

    def argmin(self, lamb, tensorlist):
        v = np.asarray(tensorlist[0], dtype=float)
        threshold = math.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]
```

ADMM supports two UDF paths. Choose whichever fits your function:

**Path 1: `eval` + `argmin`** — you supply the closed-form proximal operator. Best for nonsmooth or indicator functions (L0, rank, projections).

**Path 2: `eval` + `grad`** — you supply the gradient. Best for smooth functions where the proximal operator has no simple formula (log-cosh, Cauchy loss, quantile regression). The C++ backend solves the proximal subproblem via gradient descent with backtracking line search.

Example of a `grad`-based UDF:

```python
import admm
import numpy as np

class LogCoshLoss(admm.UDFBase):
    """Log-cosh loss: f(x) = sum(log(cosh(x_i))).

    Gradient: grad_i = tanh(x_i)
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(np.cosh(x))))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [np.tanh(x)]
```

When contributing a new UDF:

1. Create a new file in `udf/`, such as `udf/MyPenalty.py`
2. Define one class with the same name as the file, such as `class MyPenalty(admm.UDFBase)`
3. Add a docstring with the function definition and proximal operator or gradient in plain-text math notation
4. Implement `arguments()`, `eval()`, and **either** `argmin()` or `grad()`
5. Keep dependencies local to the file so the class is easy to review and maintain
6. Add or update tests in `tests/test_udf.py` (for `argmin`-based) or `tests/test_udf_grad.py` (for `grad`-based) to show how the new class is used

After adding the file, it will be available through `import udf` as `udf.MyPenalty`.

### Pull Requests

1. Fork the repository
2. Create a new branch from `main` for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add or update tests as needed
5. Update documentation as needed
6. Ensure all tests pass (`pytest tests/`)
7. Commit your changes (`git commit -am 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Development Setup

### Prerequisites

- Git
- C++ compiler
- Python >= 3.10
- Cython >= 0.29.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- admmlib >= 2026.4.9 (pre-built C++ core dependency)

### Setup Development Environment

```bash
# Clone your fork (replace YOUR_USERNAME with your GitHub username)
git clone https://github.com/YOUR_USERNAME/admm.git
cd admm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Build the package (uses current environment)
python -m build -n
```

### Running Tests

From the repository root:

```bash
# run all tests
pytest tests/

# run all unit tests
pytest tests/test_ut.py

# run all tests with user-defined proximal functions
pytest tests/test_udf.py

# run all tests with grad-based UDFs
pytest tests/test_udf_grad.py

# run all tests with common ADMM applications
pytest tests/test_admm.py

# run all documentation example tests
pytest tests/test_doc.py
```

### Documentation

Documentation is built with Sphinx. To build locally:

```bash
cd docs
bash build.sh
```

Open `_build/html/index.html` in your browser to view the documentation.

## Pull Request Checklist

Before opening a pull request, make sure:

- the change is scoped to one purpose
- relevant tests pass locally
- docs are updated if behavior or APIs changed
- generated or temporary files are not included accidentally

## Version Management

The package version is defined in `docs/descinfo.py`. Edit `PACKAGE_VERSION` field to update the version:

```python
PACKAGE_VERSION = "1.0.0" # Update this
```

After updating the version, rebuild the package:

```bash
python -m build -n
```

## Questions

If you are unsure about a change:

- open an issue:
  [GitHub Issues](https://github.com/alibaba-damo-academy/admm/issues)
- or contact:
  `solver.damo@list.alibaba-inc.com`

Thank you for contributing.
