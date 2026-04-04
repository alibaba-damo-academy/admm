"""
UDF Rank-r Indicator Example

RUN COMMAND: python examples/udf_rank_r.py

This example demonstrates rank-constrained matrix approximation via a
User-Defined Function (UDF). The indicator enforces rank(X) <= r and its
proximal operator is the truncated SVD (keep top r singular values).

Problem: Find the closest matrix X to a given Y such that rank(X) <= r.

Mathematical formulation:
    min_X   (1/2) ||X - Y||_F^2  +  delta_{rank(X) <= r}(X)

Where:
    delta is the indicator function (0 if feasible, +inf otherwise)
    Y: target matrix (2x2)
    r: rank bound (1)
    Prox: truncated SVD keeping top r singular values
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class RankRIndicator(admm.UDFBase):
    def __init__(self, arg, rank_bound=1):
        self.arg = arg
        self.rank_bound = rank_bound

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        singular_v = np.linalg.svd(X, compute_uv=False)
        return 0.0 if np.sum(singular_v > 1e-10) <= self.rank_bound else float("inf")

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, singular_v, vt = np.linalg.svd(Z, full_matrices=False)
        singular_v[min(self.rank_bound, len(singular_v)):] = 0.0
        prox = (u * singular_v) @ vt
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
Y = np.array([[3.0, 0.0], [0.0, 1.0]])
rank_bound = 1

print("=" * 70)
print("UDF Rank-r Indicator Example")
print("=" * 70)
print(f"Target matrix Y shape: {Y.shape}")
print(f"Rank bound r: {rank_bound}")
print(f"Y =\n{Y}")
print(f"Singular values of Y: {np.linalg.svd(Y, compute_uv=False)}")
print(f"Rank of Y: {np.linalg.matrix_rank(Y)}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
X = admm.Var("X", 2, 2)

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + RankRIndicator(X, rank_bound))

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
