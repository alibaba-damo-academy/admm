"""
User-Defined Smooth Function (grad UDF): Gamma Regression (GLM)

RUN COMMAND: python examples/udf_grad_gamma_regression.py

This example demonstrates Gamma regression with a log link, a common
generalized linear model (GLM) for positive-valued response variables.
The canonical loss has no closed-form proximal operator, but its gradient
is straightforward — making it a natural fit for the grad UDF path.

Problem: Fit a GLM with Gamma-distributed responses

Mathematical formulation:
    min_mu   sum( y_i * exp(-mu_i) + mu_i )  +  (lam/2) ||mu||_2^2
    s.t.     mu >= -5   (prevent numerical underflow of exp(-mu))

Where:
    mu_i = log(mean_i)   is the log-scale parameter (log link)
    y_i > 0              is the observed positive response
    f(mu) = y*exp(-mu) + mu  is the Gamma deviance (up to constants)
    grad f(mu) = -y*exp(-mu) + 1

Applications:
    - Insurance claims modeling (claim sizes are positive, right-skewed)
    - Survival analysis (waiting times, durations)
    - Environmental modeling (rainfall, pollutant concentrations)
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the Gamma Regression Loss as a grad-based UDF
# ============================================================================
class GammaRegressionLoss(admm.UDFBase):
    """Gamma regression loss with log link: f(mu) = sum(y * exp(-mu) + mu).

    This is the negative log-likelihood of the Gamma distribution (up to
    constants) under the log link, where mu = log(E[y]).

    The minimum is at mu_i = log(y_i) (sample log-mean).

    Parameters
    ----------
    arg : admm.Var
        The log-scale parameter mu (same length as y).
    y : array
        Observed positive responses.
    """

    def __init__(self, arg, y):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        return float(np.sum(self.y * np.exp(-mu) + mu))

    def grad(self, arglist):
        mu = np.asarray(arglist[0], dtype=float).ravel()
        return [(-self.y * np.exp(-mu) + 1.0).reshape(arglist[0].shape)]


# ============================================================================
# Step 2: Generate Gamma-distributed data
# ============================================================================
np.random.seed(314)
n = 30

# True log-means
mu_true = np.random.uniform(0.5, 3.0, size=n)

# Generate Gamma samples: shape k, scale theta = exp(mu)/k
k = 5.0  # shape parameter
y = np.random.gamma(shape=k, scale=np.exp(mu_true) / k, size=n)

lam = 0.05

print("=" * 70)
print("Grad UDF: Gamma Regression (GLM with Log Link)")
print("=" * 70)
print(f"Samples: {n}")
print(f"Gamma shape parameter: {k}")
print(f"Regularization (lam): {lam}")
print(f"Response range: [{y.min():.2f}, {y.max():.2f}]")
print()

# ============================================================================
# Step 3: Solve with ADMM
# ============================================================================
model = admm.Model()
mu = admm.Var("mu", n)

# Gamma loss + L2 regularization
model.setObjective(GammaRegressionLoss(mu, y) + (lam / 2) * admm.sum(admm.square(mu)))

# Prevent extreme values
model.addConstr(mu >= -5)
model.addConstr(mu <= 10)

model.optimize()

print(f"Status: {model.StatusString}")
print(f"Objective: {model.ObjVal:.4f}")
print()

mu_val = np.asarray(mu.X).ravel()

# ============================================================================
# Step 4: Evaluate the fit
# ============================================================================
# The analytical solution without regularization is mu_i = log(y_i)
mu_mle = np.log(y)

print("Log-mean recovery (first 10 samples):")
print(f"  {'Index':<8} {'True mu':<12} {'Fitted mu':<12} {'MLE mu':<12} {'y_i':<12}")
print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
for i in range(min(10, n)):
    print(f"  {i:<8} {mu_true[i]:<12.4f} {mu_val[i]:<12.4f} {mu_mle[i]:<12.4f} {y[i]:<12.4f}")

print()
print(f"  RMSE (fitted vs true):  {np.sqrt(np.mean((mu_val - mu_true) ** 2)):.6f}")
print(f"  RMSE (MLE vs true):     {np.sqrt(np.mean((mu_mle - mu_true) ** 2)):.6f}")
print()

# Show predicted means vs observed
mean_pred = np.exp(mu_val)
mean_true = np.exp(mu_true)
print(f"  Mean prediction correlation: {np.corrcoef(mean_pred, mean_true)[0, 1]:.6f}")

print("=" * 70)
