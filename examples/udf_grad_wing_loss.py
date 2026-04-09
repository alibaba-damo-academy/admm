"""
User-Defined Smooth Function (grad UDF): Wing Loss for Precise Regression

RUN COMMAND: python examples/udf_grad_wing_loss.py

This example demonstrates the Wing loss, originally proposed for face
landmark localization in computer vision. Wing loss amplifies attention
to small-to-medium range errors compared to L2, while still handling
large errors gracefully.

Problem: Regression with Wing loss — emphasizes small errors

Mathematical formulation:
    min_beta   sum( w * ln(1 + sqrt((A @ beta - b)_i^2 + delta^2) / eps) )

Where:
    w     controls the overall loss magnitude
    eps   sets the transition between the linear and logarithmic regimes
    delta smooths the loss near zero (avoids non-differentiability)

Key insight: Unlike L2 (which has vanishing gradient near zero), Wing loss
has a steeper gradient for small errors. This forces the optimizer to
"pay more attention" to reducing small residuals — useful when precision
at all data points matters more than handling outliers.

This example shows how a domain-specific loss from deep learning can be
plugged into ADMM via the grad path.
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the Wing Loss as a grad-based UDF
# ============================================================================
class WingLoss(admm.UDFBase):
    """Wing loss for high-precision regression.

    f(r) = sum( w * ln(1 + sqrt(r_i^2 + delta^2) / eps) )

    Properties:
        - Near zero: gradient ≈ w / (eps + delta), steeper than L2's zero gradient
        - Large |r|: grows as w * ln(|r|/eps), slower than L2
        - Everywhere smooth (delta > 0)

    Parameters
    ----------
    arg : admm.Var or expression
        The residual vector.
    w : float
        Loss magnitude.
    eps : float
        Transition scale.
    delta : float
        Smoothing at zero.
    """

    def __init__(self, arg, w=10.0, eps=2.0, delta=0.01):
        self.arg = arg
        self.w = w
        self.eps = eps
        self.delta = delta

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        return float(np.sum(self.w * np.log(1 + s / self.eps)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        s = np.sqrt(r ** 2 + self.delta ** 2)
        ds = r / s  # d(sqrt(r^2+delta^2))/dr
        return [self.w * ds / (self.eps + s)]


# ============================================================================
# Step 2: Generate data — clean observations with small noise
# ============================================================================
np.random.seed(2026)
n, p = 60, 5
A = np.random.randn(n, p)
beta_true = np.array([1.5, -0.8, 2.0, 0.3, -1.2])
b = A @ beta_true + 0.3 * np.random.randn(n)

# Also inject a few moderate outliers
outlier_idx = np.random.choice(n, size=6, replace=False)
b[outlier_idx] += np.random.choice([-1, 1], size=6) * np.random.uniform(3, 6, size=6)

print("=" * 70)
print("Grad UDF: Wing Loss for Precise Regression")
print("=" * 70)
print(f"Data: {n} samples, {p} features, {len(outlier_idx)} moderate outliers")
print(f"True coefficients: {beta_true}")
print()

# ============================================================================
# Step 3: Solve with ADMM using Wing loss
# ============================================================================
model = admm.Model()
beta = admm.Var("beta", p)
residual = A @ beta - b

model.setObjective(WingLoss(residual, w=10.0, eps=2.0))
model.setOption(admm.Options.admm_max_iteration, 5000)
model.optimize()

print(f"Status: {model.StatusString}")
print()

beta_val = np.asarray(beta.X)

# ============================================================================
# Step 4: Compare with OLS and Huber
# ============================================================================
beta_ols = np.linalg.lstsq(A, b, rcond=None)[0]

print("Coefficient comparison:")
print(f"  {'Index':<8} {'True':<12} {'Wing':<12} {'OLS':<12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
for i in range(p):
    print(f"  {i:<8} {beta_true[i]:<12.4f} {beta_val[i]:<12.4f} {beta_ols[i]:<12.4f}")

err_wing = np.linalg.norm(beta_val - beta_true)
err_ols = np.linalg.norm(beta_ols - beta_true)
print(f"\n  ||beta_wing - true||_2 = {err_wing:.6f}")
print(f"  ||beta_ols  - true||_2 = {err_ols:.6f}")

# Residual analysis
res_wing = A @ beta_val - b
res_ols = A @ beta_ols - b
print(f"\n  Median |residual| (Wing): {np.median(np.abs(res_wing)):.4f}")
print(f"  Median |residual| (OLS):  {np.median(np.abs(res_ols)):.4f}")

print("=" * 70)
