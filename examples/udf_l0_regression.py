"""
UDF L0-Regularized Regression Example

RUN COMMAND: python examples/udf_l0_regression.py

This example demonstrates L0-regularized regression via a User-Defined
Function (UDF). The L0 norm counts the number of nonzero entries (sparsity)
and its proximal operator hard-thresholds entries below sqrt(2*lam).

Problem: Recover a sparse nonnegative signal from noisy linear measurements
using L0 regularization.

Mathematical formulation:
    min_x   (1/2) ||A x - b||_2^2  +  lam * ||x||_0
    s.t.    x >= 0

Where:
    ||x||_0 = number of nonzero entries (nonconvex)
    A: sensing matrix (30x20)
    b: noisy observations
    lam: regularization weight (0.5)
    k: true sparsity level (3 nonzero entries)
    seed: 42
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class L0Norm(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.count_nonzero(np.abs(x) > 1e-12))

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        threshold = np.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate synthetic data
# ============================================================================
np.random.seed(42)
n, m, k = 20, 30, 3

# True sparse signal
x_true = np.zeros(n)
x_true[np.random.choice(n, k, replace=False)] = np.random.rand(k) * 2 + 0.5

# Sensing matrix and noisy observations
A = np.random.randn(m, n)
b = A @ x_true + 0.01 * np.random.randn(m)
lam = 0.5

print("=" * 70)
print("UDF L0-Regularized Regression Example")
print("=" * 70)
print(f"Problem size: m={m} measurements, n={n} variables")
print(f"True sparsity: k={k} nonzero entries")
print(f"Regularization weight (lam): {lam}")
print(f"Hard threshold sqrt(2*lam) = {np.sqrt(2.0 * lam):.4f}")
print(f"True nonzero indices: {np.nonzero(x_true)[0].tolist()}")
print(f"True nonzero values: {x_true[x_true > 0].tolist()}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 4: Set objective and constraints
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(A @ x - b)) + lam * L0Norm(x))
model.addConstr(x >= 0)

# ============================================================================
# Step 5: Solve the model
# ============================================================================
model.optimize()

# ============================================================================
# Step 6: Check status and print results
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"\u2713 Status: {model.StatusString}")
    print(f"\u2713 Optimal objective value: {model.ObjVal:.6f}")
    print(f"\u2713 Solver time: {model.SolverTime:.4f} seconds")
    print()

    x_sol = np.asarray(x.X)
    nnz = np.count_nonzero(np.abs(x_sol) > 1e-6)
    print(f"Number of nonzero entries: {nnz} (true: {k})")
    print()

    # Show recovered nonzero entries
    nz_idx = np.nonzero(np.abs(x_sol) > 1e-6)[0]
    print("Recovered nonzero entries:")
    for idx in nz_idx:
        true_val = x_true[idx] if x_true[idx] > 0 else 0.0
        print(f"  x[{idx}] = {x_sol[idx]:.6f} (true: {true_val:.6f})")

    # Recovery quality
    print(f"\nRecovery error ||x - x_true||_2 = {np.linalg.norm(x_sol - x_true):.6f}")
    print(f"Residual ||A x - b||_2 = {np.linalg.norm(A @ x_sol - b):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
