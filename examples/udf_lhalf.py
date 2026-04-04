"""
User-Defined Function (UDF): L1/2 Quasi-Norm Example

RUN COMMAND: python examples/udf_lhalf.py

This example demonstrates the L_{1/2} quasi-norm as a sparsity-inducing
penalty using a User-Defined Function (UDF) in ADMM.

Problem: Sparse signal recovery using the L_{1/2} quasi-norm, which provides
stronger sparsity promotion than the L1 norm (nonconvex).

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  lam * sum_i sqrt(|x_i|)

Where:
    x: decision variable
    y: observed signal
    lam: regularization parameter
    sum_i sqrt(|x_i|): L_{1/2} quasi-norm (nonconvex)

The proximal operator uses the closed-form half-thresholding rule:
    - If |v_i| <= threshold: prox = 0
    - Otherwise: prox = sign(v) * (2/3)|v| * (1 + cos(2pi/3 - 2phi/3))
      where phi = arccos((3*sqrt(3)*lam) / (4*|v|^{3/2}))
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define the L1/2 Quasi-Norm UDF
# ============================================================================
class LHalfNorm(admm.UDFBase):
    """L_{1/2} quasi-norm: sum of sqrt(|x_i|)."""

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.sqrt(np.abs(x))))

    def argmin(self, lamb, arglist):
        v = np.asarray(arglist[0], dtype=float)
        abs_v = np.abs(v)
        threshold = 1.5 * (lamb ** (2.0 / 3.0))
        prox = np.zeros_like(v)
        active = abs_v > threshold
        if np.any(active):
            phi = np.arccos(
                np.clip(
                    (3.0 * np.sqrt(3.0) * lamb) / (4.0 * np.power(abs_v[active], 1.5)),
                    -1.0,
                    1.0,
                )
            )
            prox_abs = (2.0 * abs_v[active] / 3.0) * (
                1.0 + np.cos((2.0 * np.pi / 3.0) - (2.0 * phi / 3.0))
            )
            prox[active] = np.sign(v[active]) * prox_abs
        return [prox.tolist()]


# ============================================================================
# Step 2: Define problem data
# ============================================================================
y = np.array([0.2, 1.0, 2.0])
lam = 0.5

print("=" * 70)
print("User-Defined Function (UDF): L1/2 Quasi-Norm Example")
print("=" * 70)
print(f"Observed signal y: {y}")
print(f"Regularization (lam): {lam}")
print(f"Half-threshold: 1.5 * lam^(2/3) = {1.5 * (lam ** (2.0 / 3.0)):.6f}")
print()
print("Expected: small entries shrunk toward zero more aggressively")
print("than L1 regularization. Entry y[0]=0.2 should be zeroed.")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", len(y))

# ============================================================================
# Step 4: Set objective with UDF
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * LHalfNorm(x))

# ============================================================================
# Step 5: Add constraints
# ============================================================================
# Unconstrained problem

# ============================================================================
# Step 6: Solve and print results
# ============================================================================
model.optimize()

print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Optimal objective value: {model.ObjVal:.6f}")
    print(f"  Solver time: {model.SolverTime:.4f} seconds")
    print()

    x_val = np.asarray(x.X)

    print("Solution analysis:")
    print(f"  {'Index':<8} {'y_i':<10} {'x_i':<12} {'Shrinkage'}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10}")
    for i in range(len(y)):
        shrinkage = y[i] - x_val[i] if abs(x_val[i]) > 1e-6 else y[i]
        print(f"  {i:<8} {y[i]:<10.4f} {x_val[i]:<12.6f} {shrinkage:<10.6f}")

    print()
    print(f"  Nonzero entries: {np.sum(np.abs(x_val) > 1e-4)}/{len(y)}")

    # L1/2 quasi-norm value
    lhalf_val = np.sum(np.sqrt(np.abs(x_val)))
    print(f"  L1/2 quasi-norm value: {lhalf_val:.6f}")

    # Decomposition of objective
    data_fidelity = 0.5 * np.sum((x_val - y) ** 2)
    penalty = lam * lhalf_val
    print(f"\n  Data fidelity: {data_fidelity:.6f}")
    print(f"  Penalty term:  {penalty:.6f}")
    print(f"  Total:         {data_fidelity + penalty:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
