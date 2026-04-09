"""
User-Defined Smooth Function (grad UDF): Smooth Quantile Regression

RUN COMMAND: python examples/udf_grad_smooth_quantile.py

This example demonstrates quantile regression using a smooth approximation
to the pinball (check) loss. Unlike the non-differentiable pinball loss,
this UDF is everywhere smooth, making it a perfect fit for the grad path.

Problem: Estimate the tau-th conditional quantile of y given A

Mathematical formulation:
    min_x   sum( tau * u_i + (1/beta) * log(1 + exp(-beta * u_i)) )
    where   u = A @ x - b

The smooth pinball loss approximates the classical quantile loss:
    rho_tau(u) = u * (tau - I(u < 0))

As beta → ∞, the smooth version converges to the exact pinball loss.
With finite beta (e.g. 20), it remains differentiable while staying
very close to the true quantile loss.

This is useful for:
    - Prediction intervals (fit tau=0.1 and tau=0.9)
    - Asymmetric risk modeling
    - Understanding distributional effects beyond the mean
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the Smooth Quantile Loss as a grad-based UDF
# ============================================================================
class SmoothQuantileLoss(admm.UDFBase):
    """Smooth quantile/pinball loss.

    f(u) = sum( tau * u_i + (1/beta) * log(1 + exp(-beta * u_i)) )

    Parameters
    ----------
    arg : admm.Var or expression
        The residual vector u = prediction - target.
    tau : float
        Quantile level in (0, 1). tau=0.5 gives the median.
    beta : float
        Smoothing parameter. Larger beta = closer to exact pinball.
    """

    def __init__(self, arg, tau=0.5, beta=20.0):
        self.arg = arg
        self.tau = tau
        self.beta = beta

    def arguments(self):
        return [self.arg]

    def _softplus(self, z):
        """Numerically stable log(1 + exp(z))."""
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def eval(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        return float(np.sum(self.tau * u + (1.0 / self.beta) * self._softplus(-self.beta * u)))

    def grad(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        sig = self._sigmoid(self.beta * u)
        return [self.tau - (1.0 - sig)]


# ============================================================================
# Step 2: Generate heteroscedastic data
# ============================================================================
np.random.seed(123)
n, p = 100, 3
A = np.random.randn(n, p)
x_true = np.array([2.0, -1.0, 0.5])
# Noise variance increases with A[:,0] — heteroscedastic
noise_scale = 0.5 + 2.0 * np.abs(A[:, 0])
b = A @ x_true + noise_scale * np.random.randn(n)

print("=" * 70)
print("Grad UDF: Smooth Quantile Regression")
print("=" * 70)
print(f"Data: {n} samples, {p} features, heteroscedastic noise")
print(f"True coefficients: {x_true}")
print()

# ============================================================================
# Step 3: Fit three quantiles (0.1, 0.5, 0.9) to get prediction interval
# ============================================================================
results = {}
for tau in [0.1, 0.5, 0.9]:
    model = admm.Model()
    x = admm.Var("x", p)
    residual = A @ x - b

    model.setObjective(SmoothQuantileLoss(residual, tau=tau, beta=20.0))
    model.optimize()

    x_val = np.asarray(x.X)
    results[tau] = x_val

    print(f"  tau = {tau}:  x = [{', '.join(f'{v:.4f}' for v in x_val)}]"
          f"  (status: {model.StatusString})")

print()
print("Interpretation:")
print(f"  Median regression (tau=0.5):  [{', '.join(f'{v:.4f}' for v in results[0.5])}]")
print(f"  True coefficients:            [{', '.join(f'{v:.4f}' for v in x_true)}]")
print(f"  Median error: {np.linalg.norm(results[0.5] - x_true):.6f}")
print()

# Show coefficient spread across quantiles
print("  Coefficient spread across quantiles (wider = more heteroscedastic effect):")
for i in range(p):
    spread = results[0.9][i] - results[0.1][i]
    print(f"    x[{i}]: [{results[0.1][i]:+.4f}, {results[0.9][i]:+.4f}]  spread = {spread:.4f}")

print("=" * 70)
