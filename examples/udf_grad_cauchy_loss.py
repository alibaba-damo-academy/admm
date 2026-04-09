"""
User-Defined Smooth Function (grad UDF): Cauchy Loss Robust Regression

RUN COMMAND: python examples/udf_grad_cauchy_loss.py

This example demonstrates robust regression using the Cauchy (Lorentzian)
loss, a heavy-tailed loss function from robust statistics whose influence
function is bounded and redescending — making it even more outlier-resistant
than Huber or log-cosh.

Problem: Robust regression with Cauchy loss and L2 regularization

Mathematical formulation:
    min_x   sum(log(1 + ((Ax - b)_i / c)^2)) + (lam/2) ||x||_2^2

Where:
    f(r) = log(1 + (r/c)^2)         is the Cauchy loss
    grad f(r) = 2r / (c^2 + r^2)    is bounded and → 0 for large |r|
    c > 0 is a scale parameter controlling the transition from quadratic to flat

Key property: the Cauchy loss has a REDESCENDING influence function — as
residuals grow larger, their influence on the fit actually DECREASES.
This makes it exceptionally robust to gross outliers.
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the Cauchy Loss as a grad-based UDF
# ============================================================================
class CauchyLoss(admm.UDFBase):
    """Cauchy/Lorentzian loss: f(r) = sum(log(1 + (r_i/c)^2)).

    Properties:
        - Behaves like r^2/c^2 near zero (quadratic)
        - Grows logarithmically for large |r| (heavy-tailed)
        - Influence function 2r/(c^2 + r^2) → 0 as r → ∞ (redescending)

    Parameters
    ----------
    arg : admm.Var or expression
        The residual vector.
    c : float
        Scale parameter. Smaller c = more aggressive outlier rejection.
    """

    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(1 + (r / self.c) ** 2)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [2.0 * r / (self.c ** 2 + r ** 2)]


# ============================================================================
# Step 2: Generate data with gross outliers
# ============================================================================
np.random.seed(7)
n, p = 80, 5
A = np.random.randn(n, p)
x_true = np.array([3.0, -1.0, 2.0, 0.5, -0.5])
b = A @ x_true + 0.2 * np.random.randn(n)

# Inject 25% gross outliers (very large)
n_outliers = 20
outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
b[outlier_idx] += np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(20, 50, size=n_outliers)

lam = 0.01
c = 2.0  # Cauchy scale

print("=" * 70)
print("Grad UDF: Cauchy Loss Robust Regression")
print("=" * 70)
print(f"Data: {n} samples, {p} features, {n_outliers} gross outliers (25%)")
print(f"Outlier magnitude: 20-50x signal level")
print(f"Cauchy scale c = {c}")
print()

# ============================================================================
# Step 3: Solve with ADMM
# ============================================================================
model = admm.Model()
x = admm.Var("x", p)
residual = A @ x - b

model.setObjective(CauchyLoss(residual, c=c) + (lam / 2) * admm.sum(admm.square(x)))
model.optimize()

print("Results:")
print("-" * 70)
print(f"  Status: {model.StatusString}")
print(f"  Objective: {model.ObjVal:.6f}")
print()

x_val = np.asarray(x.X)
print("  Coefficient comparison:")
print(f"  {'Index':<8} {'True':<12} {'Cauchy':<12} {'Error':<12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for i in range(p):
    print(f"  {i:<8} {x_true[i]:<12.4f} {x_val[i]:<12.4f} {abs(x_val[i] - x_true[i]):<12.4f}")

# ============================================================================
# Step 4: Compare with OLS
# ============================================================================
x_ols = np.linalg.lstsq(A, b, rcond=None)[0]
print(f"\n  OLS error (no robustness):    {np.linalg.norm(x_ols - x_true):.6f}")
print(f"  Cauchy loss error (robust):   {np.linalg.norm(x_val - x_true):.6f}")
print(f"\n  Cauchy loss reduces error by {np.linalg.norm(x_ols - x_true) / max(np.linalg.norm(x_val - x_true), 1e-10):.1f}x vs OLS")

print("=" * 70)
