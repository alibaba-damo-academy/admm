"""
UDF Group Sparsity Example

RUN COMMAND: python examples/udf_group_sparsity.py

This example demonstrates group sparsity regularization via a User-Defined
Function (UDF). The penalty counts the number of non-zero column groups and
its proximal operator hard-thresholds entire columns.

Problem: Find a matrix X close to a given Y while encouraging entire columns
to be zero (group sparsity).

Mathematical formulation:
    min_X   (1/2) ||X - Y||_F^2  +  lam * g(X)

Where:
    g(X) = number of columns with nonzero norm (group L0)
    Y: target matrix (2x3)
    lam: regularization weight
    Prox: zero column j if ||Z_j||^2 <= 2*lam, else keep Z_j
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class GroupSparsityPenalty(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        column_norms = np.linalg.norm(X, axis=0)
        return float(np.count_nonzero(column_norms > 1e-12))

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        column_norm_sq = np.sum(Z * Z, axis=0)
        keep_mask = column_norm_sq > 2.0 * lamb
        prox = Z * keep_mask[np.newaxis, :]
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
Y = np.array([[0.2, 2.0, 0.3], [0.1, 1.0, 0.4]])
lam = 1.0

print("=" * 70)
print("UDF Group Sparsity Example")
print("=" * 70)
print(f"Target matrix Y shape: {Y.shape}")
print(f"Regularization weight (lam): {lam}")
print(f"Y =\n{Y}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
X = admm.Var("X", 2, 3)

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * GroupSparsityPenalty(X))

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

    X_sol = np.asarray(X.X)
    print(f"Solution X =\n{X_sol}")
    print()

    # Column analysis
    column_norms = np.linalg.norm(X_sol, axis=0)
    print("Column norms:")
    for j in range(X_sol.shape[1]):
        status = "nonzero" if column_norms[j] > 1e-6 else "zeroed"
        print(f"  Column {j}: ||X[:,{j}]|| = {column_norms[j]:.6f} ({status})")

    num_active = np.sum(column_norms > 1e-6)
    print(f"\nActive column groups: {num_active} / {X_sol.shape[1]}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
