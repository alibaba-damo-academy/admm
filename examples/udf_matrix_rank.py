"""
UDF Matrix Rank Example

RUN COMMAND: python examples/udf_matrix_rank.py

This example demonstrates matrix rank minimization via a User-Defined
Function (UDF). The penalty counts the number of nonzero singular values
(matrix rank) and its proximal operator hard-thresholds singular values.

Problem: Find a matrix X close to a given Y while penalizing its rank.

Mathematical formulation:
    min_X   (1/2) ||X - Y||_F^2  +  lam * rank(X)

Where:
    rank(X) = number of nonzero singular values (nonconvex)
    Y: target matrix (2x2)
    lam: regularization weight
    Prox: hard threshold singular values at sqrt(2*lam)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class RankPenalty(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        singular_v = np.linalg.svd(X, compute_uv=False)
        return float(np.sum(singular_v > 1e-10))

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
        threshold = np.sqrt(2.0 * lamb)
        singular_v = np.where(singular_v <= threshold, 0.0, singular_v)
        prox = (u * singular_v) @ vt
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
Y = np.array([[2.0, 0.0], [0.0, 0.5]])
lam = 0.5

print("=" * 70)
print("UDF Matrix Rank Example")
print("=" * 70)
print(f"Target matrix Y shape: {Y.shape}")
print(f"Regularization weight (lam): {lam}")
print(f"Y =\n{Y}")
print(f"Singular values of Y: {np.linalg.svd(Y, compute_uv=False)}")
print(f"Threshold sqrt(2*lam) = {np.sqrt(2.0 * lam):.4f}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
X = admm.Var("X", 2, 2)

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + lam * RankPenalty(X))

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

    sv_sol = np.linalg.svd(X_sol, compute_uv=False)
    print(f"Singular values of X: {sv_sol}")
    print(f"Rank of solution: {np.sum(sv_sol > 1e-6)}")
    print(f"Frobenius error ||X - Y||_F = {np.linalg.norm(X_sol - Y, 'fro'):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
