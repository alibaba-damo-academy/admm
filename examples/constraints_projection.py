"""
Constraints Projection Example

RUN COMMAND: python examples/constraints_projection.py

This example demonstrates various constraint types available in ADMM, including
norm constraints, non-negativity, and positive semidefinite (PSD) constraints.

Problem: Project target points onto the feasible set defined by multiple
norm constraints and a PSD constraint.

Mathematical formulation:
    min_{x, X}  ||x - x_target||_2^2 + ||X - X_target||_F^2
    s.t.        x >= 0
                ||x||_1 <= 1.2
                ||x||_2 <= 1.0
                ||X||_* <= 1.1        (nuclear norm)
                ||X||_F <= 1.0        (Frobenius norm)
                X >> 0                (positive semidefinite)

Where:
    x: 2-dimensional vector variable
    X: 2x2 symmetric matrix variable (PSD)
    x_target, X_target: target values to project from
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define target values
# ============================================================================
x_target = np.array([2.0, 1.0])
X_target = np.array([[2.0, 0.0], [0.0, 1.0]])

print("=" * 70)
print("Constraints Projection Example")
print("=" * 70)
print(f"Vector target x_target: {x_target}")
print("Matrix target X_target:")
print(f"  {X_target[0]}")
print(f"  {X_target[1]}")
print()
print("Constraints:")
print("  x >= 0")
print("  ||x||_1 <= 1.2")
print("  ||x||_2 <= 1.0")
print("  ||X||_* <= 1.1  (nuclear norm)")
print("  ||X||_F <= 1.0  (Frobenius norm)")
print("  X >> 0          (positive semidefinite)")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", 2)
X = admm.Var("X", 2, 2, symmetric=True)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Minimize distance to target values
model.setObjective(
    admm.sum(admm.square(x - x_target))
    + admm.sum(admm.square(X - X_target))
)

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# Vector constraints
model.addConstr(x >= 0)                         # non-negativity
model.addConstr(admm.norm(x, ord=1) <= 1.2)     # L1 norm ball
model.addConstr(admm.norm(x, ord=2) <= 1.0)     # L2 norm ball

# Matrix constraints
model.addConstr(admm.norm(X, ord='nuc') <= 1.1)  # nuclear norm ball
model.addConstr(admm.norm(X, ord='fro') <= 1.0)  # Frobenius norm ball
model.addConstr(X >> 0)                           # PSD constraint

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
    print(f"  Status: {model.StatusString}")
    print(f"  Optimal objective value: {model.ObjVal:.6f}")
    print(f"  Solver time: {model.SolverTime:.4f} seconds")
    print()

    x_val = np.asarray(x.X)
    X_val = np.asarray(X.X)

    print("Projected vector x:")
    print(f"  x = {np.round(x_val, 6)}")
    print(f"  ||x||_1 = {np.linalg.norm(x_val, 1):.6f} (limit: 1.2)")
    print(f"  ||x||_2 = {np.linalg.norm(x_val, 2):.6f} (limit: 1.0)")
    print(f"  min(x)  = {x_val.min():.6f} (should be >= 0)")
    print()

    print("Projected matrix X:")
    print(f"  X = {np.round(X_val, 6)}")
    nuc_norm = np.linalg.norm(np.linalg.svd(X_val, compute_uv=False), 1)
    fro_norm = np.linalg.norm(X_val, 'fro')
    eigenvalues = np.linalg.eigvalsh(X_val)
    print(f"  ||X||_* = {nuc_norm:.6f} (limit: 1.1)")
    print(f"  ||X||_F = {fro_norm:.6f} (limit: 1.0)")
    print(f"  eigenvalues = {np.round(eigenvalues, 6)} (should be >= 0)")
    print()

    # Distance from targets
    x_dist = np.linalg.norm(x_val - x_target)
    X_dist = np.linalg.norm(X_val - X_target, 'fro')
    print("Projection distances:")
    print(f"  ||x - x_target||_2 = {x_dist:.6f}")
    print(f"  ||X - X_target||_F = {X_dist:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
