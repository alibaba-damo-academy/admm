# User-Defined Proximal Functions (UDF)

Ready-to-use proximal operators for penalties and constraints beyond the built-in ADMM atoms.

## Quick Start

```python
import admm
import numpy as np
from udf.L0Norm import L0Norm

y = np.array([0.2, 2.0, 0.6, 2.2])
model = admm.Model()
x = admm.Var("x", len(y))
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + 1.0 * L0Norm(x))
model.optimize()
print(x.X)  # [0, 2, 0, 2.2] — small entries thresholded to zero
```

## Available Classes

| Class | Type | Function |
|-------|------|----------|
| `L0Norm` | Penalty | Exact sparsity count: f(x) = \\|x\\|\_0 |
| `L0BallIndicator` | Indicator | Cardinality constraint: \\|x\\|\_0 ≤ k |
| `LHalfNorm` | Penalty | L1/2 quasi-norm: Σ √\|x\_i\| |
| `L21Norm` | Penalty | Group sparsity: Σ\_j \\|x\_{:,j}\\|\_2 |
| `MCPPenalty` | Penalty | Minimax Concave Penalty |
| `SCADPenalty` | Penalty | Smoothly Clipped Absolute Deviation |
| `QuarticPenalty` | Penalty | Quartic: Σ x\_i⁴ |
| `RankPenalty` | Penalty | Matrix rank: # nonzero singular values |
| `RankRIndicator` | Indicator | Rank constraint: rank(X) ≤ r |
| `OrthogonalMatrixIndicator` | Indicator | Orthogonal matrix: X^T X = I |
| `StiefelIndicator` | Indicator | Stiefel manifold: X^T X = I\_p (tall X) |
| `UnitSphereIndicator` | Indicator | Unit sphere: \\|x\\|\_2 = 1 |
| `SimplexIndicator` | Indicator | Simplex: x ≥ 0, Σx\_i = r |
| `BinaryIndicator` | Indicator | Binary cube: x ∈ {0,1}^n |
| `GroupSparsityPenalty` | Penalty | Column-group L0 count |

## Writing Your Own UDF

A UDF is a Python class that inherits from `admm.UDFBase` and implements three methods:

```python
import admm
import numpy as np

class MyPenalty(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        """Return the list of ADMM variables this function depends on."""
        return [self.arg]

    def eval(self, arglist):
        """Evaluate f(x) at a concrete numeric point.

        arglist[i] corresponds to arguments()[i].
        Must return a scalar float.
        """
        x = np.asarray(arglist[0], dtype=float)
        return float(...)  # your function value

    def argmin(self, lamb, arglist):
        """Compute the proximal operator: argmin_x { lamb * f(x) + (1/2)||x - v||^2 }.

        arglist[0] is the current point v.
        Must return a list of arrays matching arguments().
        """
        v = np.asarray(arglist[0], dtype=float)
        prox = ...  # your proximal computation
        return [prox.tolist()]
```

**Key rules:**
- `arguments()` returns a list of `admm.Var` objects — the solver passes their current values into `eval` and `argmin` as `arglist`
- `eval` returns a **scalar float** (the function value)
- `argmin` returns a **list of arrays** (one per variable in `arguments()`)
- The proximal operator solves: minimize `lamb * f(x) + (1/2) * ||x - v||^2`

## Usage in a Model

```python
from udf.MyPenalty import MyPenalty

model = admm.Model()
x = admm.Var("x", n)
model.setObjective(admm.sum(admm.square(A @ x - b)) + lam * MyPenalty(x))
model.optimize()
```

UDFs compose naturally with built-in atoms — use standard ADMM for the data-fit term and constraints, and only implement the custom proximal block.

## Contributing a UDF

1. Create `udf/MyPenalty.py` with a single class `MyPenalty(admm.UDFBase)`
2. Add a docstring with the mathematical definition and proximal operator formula
3. Implement `arguments()`, `eval()`, and `argmin()`
4. Add a test in `tests/test_udf.py`
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.
