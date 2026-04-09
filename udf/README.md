# User-Defined Proximal Functions (UDF)

Ready-to-use proximal operators and gradient functions for penalties, losses, and constraints beyond the built-in ADMM atoms.

ADMM supports two UDF paths:

- **`eval` + `argmin`** — you supply the closed-form proximal operator. Best for nonsmooth or indicator functions (L0, rank, projections).
- **`eval` + `grad`** — you supply the gradient. Best for smooth functions where the proximal operator has no simple formula (log-cosh, Cauchy, quantile loss). The C++ backend solves the proximal subproblem via gradient descent with Armijo backtracking line search.

## Quick Start

### Path 1: argmin — nonconvex penalty

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

### Path 2: grad — smooth robust loss

```python
import admm
import numpy as np
from udf.LogCoshLoss import LogCoshLoss

A = np.random.randn(50, 5)
b = np.random.randn(50)
model = admm.Model()
x = admm.Var("x", 5)
model.setObjective(LogCoshLoss(A @ x - b))
model.optimize()
print(x.X)  # robust regression coefficients
```

## Available Classes

### Proximal operator path (`eval` + `argmin`)

| Class | Type | Function |
|-------|------|----------|
| `L0Norm` | Penalty | Exact sparsity count ‖x‖₀ |
| `L0BallIndicator` | Indicator | Cardinality constraint ‖x‖₀ ≤ k |
| `LHalfNorm` | Penalty | L1/2 quasi-norm Σ √‖xᵢ‖ |
| `L21Norm` | Penalty | Group sparsity Σⱼ ‖x₍:,ⱼ₎‖₂ |
| `MCPPenalty` | Penalty | Minimax Concave Penalty |
| `SCADPenalty` | Penalty | Smoothly Clipped Absolute Deviation |
| `QuarticPenalty` | Penalty | Quartic Σ xᵢ⁴ |
| `RankPenalty` | Penalty | Matrix rank: number of nonzero singular values |
| `RankRIndicator` | Indicator | Rank constraint rank(X) ≤ r |
| `OrthogonalMatrixIndicator` | Indicator | Orthogonal matrix XᵀX = I |
| `StiefelIndicator` | Indicator | Stiefel manifold XᵀX = Iₚ (tall X) |
| `UnitSphereIndicator` | Indicator | Unit sphere ‖x‖₂ = 1 |
| `SimplexIndicator` | Indicator | Simplex x ≥ 0, Σxᵢ = r |
| `BinaryIndicator` | Indicator | Binary cube x ∈ {0,1}ⁿ |
| `GroupSparsityPenalty` | Penalty | Column-group L0 count |

### Gradient path (`eval` + `grad`)

#### Robust M-estimator losses

| Class | Type | Function |
|-------|------|----------|
| `LogCoshLoss` | Loss | Smooth L1 approx: f(r) = Σ log(cosh(rᵢ)), grad = tanh(r) |
| `CauchyLoss` | Loss | Heavy-tailed robust: f(r) = Σ log(1+(rᵢ/c)²), redescending gradient |
| `PseudoHuberLoss` | Loss | Smooth Huber: f(r) = δ²·Σ(√(1+(r/δ)²)−1), grad = r/√(1+(r/δ)²) |
| `WelschLoss` | Loss | Gaussian-shaped: f(r) = (c²/2)·Σ(1−exp(−r²/c²)), bounded, redescending |
| `GemanMcClureLoss` | Loss | Bounded estimator: f(r) = Σ r²/(2(1+r²)), saturates at n/2 |
| `FairLoss` | Loss | Between L2 and L1: f(r) = c²·Σ(‖r‖/c − log(1+‖r‖/c)) |
| `TukeyBisquareLoss` | Loss | Complete outlier rejection: grad = 0 for ‖r‖ > c |
| `StudentTLoss` | Loss | Heavy-tailed: f(r) = Σ log(1+r²/ν), ν=1 gives Cauchy |
| `BerhuLoss` | Loss | Reverse Huber: L1 for small ‖r‖, L2 for large ‖r‖ |
| `CorrentropyLoss` | Loss | Max correntropy: f(r) = Σ(1−exp(−r²/(2σ²))), kernel-based robust |
| `BoundedRatioLoss` | Loss | Smooth L0 approx: f(x) = Σ x²/(1+x²), bounded in [0, n) |
| `MorsePotential` | Loss | Molecular physics: f(x) = D·Σ(1−exp(−a(x−r₀)))², bounded |

#### Quantile, asymmetric and structured losses

| Class | Type | Function |
|-------|------|----------|
| `SmoothQuantileLoss` | Loss | Smooth pinball: f(u) = Σ[τu + (1/β)log(1+exp(−βu))] |
| `QuantileHuberLoss` | Loss | Asymmetric smooth quantile via softplus, parameterized by (τ, ε) |
| `AsymmetricLoss` | Loss | Different penalties for +/−: f(r) = Σ(w₊·max(r,0)² + w₋·max(−r,0)²) |
| `WingLoss` | Loss | Face alignment: f(r) = Σ w·ln(1 + √(rᵢ²+δ²)/ε), steep grad near zero |

#### Classification and GLM losses

| Class | Type | Function |
|-------|------|----------|
| `LogisticLoss` | Loss | Logistic regression: f(w) = Σ log(1+exp(−yᵢ·aᵢᵀw)) |
| `SmoothHingeLoss` | Loss | Smooth SVM hinge (C¹): quadratic for 0<z<1, linear for z≤0 |
| `SquaredHingeLoss` | Loss | L2-SVM: f(w) = Σ max(0, 1−yᵢ·aᵢᵀw)² |
| `BinaryCrossEntropyLoss` | Loss | Cross-entropy: f(p) = −Σ[t·log(p)+(1−t)·log(1−p)] |
| `SmoothEpsilonLoss` | Loss | SVR ε-insensitive (smooth): zero penalty inside ε-tube |
| `GammaRegressionLoss` | Loss | GLM Gamma deviance: f(μ) = Σ(y·exp(−μ) + μ), for positive data |
| `PoissonLoss` | Loss | Poisson NLL: f(w) = Σ(exp(aᵢᵀw) − bᵢ·aᵢᵀw) |

#### Divergences and information-theoretic

| Class | Type | Function |
|-------|------|----------|
| `KLDivergence` | Divergence | KL divergence: f(p) = Σ pᵢ·log(pᵢ/qᵢ) |
| `ItakuraSaitoDivergence` | Divergence | Audio/spectral: f(x) = Σ(y/x − log(y/x) − 1), scale-invariant |
| `ChiSquaredDivergence` | Divergence | χ² divergence: f(x) = Σ(xᵢ−yᵢ)²/yᵢ |
| `NegEntropy` | Penalty | Negative entropy: f(x) = Σ xᵢ·log(xᵢ), convex on x > 0 |
| `LogBarrier` | Barrier | Interior point: f(x) = −Σ log(xᵢ), self-concordant |

#### Structural and signal processing

| Class | Type | Function |
|-------|------|----------|
| `SmoothTV` | Penalty | Total variation: f(x) = Σ √((xᵢ₊₁−xᵢ)²+ε), structural gradient |
| `GraphLaplacianSmoothing` | Penalty | Graph smoothing: f(x) = Σ_{(i,j)∈E} (xᵢ−xⱼ)², arbitrary topology |
| `SmoothLpNorm` | Penalty | Smooth Lp: f(x) = Σ(xᵢ²+ε)^(p/2), bridges L1 and L2 |

#### Multi-argument and special

| Class | Type | Function |
|-------|------|----------|
| `SmoothHingePairLoss` | Loss | Pairwise ranking: f(x,y) = Σ softplus(xᵢ−yᵢ+margin), 2-arg UDF |
| `DoubleWellPotential` | Potential | Non-convex: f(x) = Σ(xᵢ²−1)², minima at ±1 |

## Writing Your Own UDF

A UDF is a Python class that inherits from `admm.UDFBase` and implements `arguments()`, `eval()`, plus **either** `argmin()` or `grad()`:

### Path 1: `eval` + `argmin` (proximal operator)

Best for nonsmooth or indicator functions where you know the closed-form proximal operator.

```python
import admm
import numpy as np

class MyPenalty(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(...)  # your function value

    def argmin(self, lamb, arglist):
        """Solve: argmin_x { lamb * f(x) + (1/2)||x - v||^2 }."""
        v = np.asarray(arglist[0], dtype=float)
        prox = ...  # your proximal computation
        return [prox.tolist()]
```

### Path 2: `eval` + `grad` (gradient)

Best for smooth functions where the proximal operator has no simple formula. The C++ backend solves the proximal subproblem via gradient descent with backtracking line search.

```python
import admm
import numpy as np

class MyLoss(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(...)  # your function value

    def grad(self, arglist):
        """Return the gradient as a list of arrays (one per variable)."""
        x = np.asarray(arglist[0], dtype=float)
        return [...]  # gradient array, same shape as x
```

### Key rules

- `arguments()` returns a list of `admm.Var` objects — the solver passes their current values into `eval` and `argmin`/`grad` as `arglist`
- `eval` returns a **scalar float** (the function value)
- Implement **either** `argmin` or `grad`, not both
- `argmin` returns a **list of arrays** — solves: minimize `lamb * f(x) + (1/2) * ||x - v||^2`
- `grad` returns a **list of arrays** — each array has the same shape as the corresponding input
- Fixed data (targets, weights, hyperparameters) should be stored in `__init__` — only optimization variables go in `arguments()`

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
2. Add a docstring with the mathematical definition and proximal operator or gradient formula
3. Implement `arguments()`, `eval()`, and **either** `argmin()` or `grad()`
4. Add a test in `tests/test_udf.py` (for `argmin`) or `tests/test_udf_grad.py` (for `grad`)
5. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full guidelines.
