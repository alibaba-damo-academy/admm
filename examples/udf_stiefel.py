"""
UDF Stiefel-Manifold Indicator Example

RUN COMMAND: python examples/udf_stiefel.py

This example demonstrates projection onto the Stiefel manifold via a
User-Defined Function (UDF). The indicator enforces X^T X = I_n
(orthonormal columns) and its proximal operator uses the polar factor
from the SVD.

Problem: Find the closest matrix X with orthonormal columns to a given Y.

Mathematical formulation:
    min_X   (1/2) ||X - Y||_F^2  +  delta_{St(m,n)}(X)

Where:
    St(m,n) = {X in R^{m x n} : X^T X = I_n}  (Stiefel manifold)
    Y: target matrix (3x2)
    Prox (polar factor): if Z = U Sigma V^T then prox = U V^T
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define the UDF class
# ============================================================================


class StiefelIndicator(admm.UDFBase):
    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        X = np.asarray(arglist[0], dtype=float)
        identity = np.eye(X.shape[1])
        return 0.0 if np.linalg.norm(X.T @ X - identity) <= 1e-9 else float("inf")

    def argmin(self, lamb, arglist):
        Z = np.asarray(arglist[0], dtype=float)
        u, _, vt = np.linalg.svd(Z, full_matrices=False)
        prox = u @ vt
        return [prox.tolist()]


# ============================================================================
# Step 2: Generate data
# ============================================================================
Y = np.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])

print("=" * 70)
print("UDF Stiefel-Manifold Indicator Example")
print("=" * 70)
print(f"Target matrix Y shape: {Y.shape}")
print(f"Y =\n{Y}")
print(f"Y^T Y =\n{Y.T @ Y}")
print()

# ============================================================================
# Step 3: Create model and decision variables
# ============================================================================
model = admm.Model()
X = admm.Var("X", 3, 2)

# ============================================================================
# Step 4: Set objective
# ============================================================================
model.setObjective(0.5 * admm.sum(admm.square(X - Y)) + StiefelIndicator(X))

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

    # Verify orthonormality
    XtX = X_sol.T @ X_sol
    print(f"X^T X =\n{np.round(XtX, 6)}")
    print(f"||X^T X - I||_F = {np.linalg.norm(XtX - np.eye(2)):.6e} (should be ~0)")
    print(f"Frobenius error ||X - Y||_F = {np.linalg.norm(X_sol - Y, 'fro'):.6f}")

else:
    print(f"\u2717 Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
