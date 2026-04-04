"""
Robust PCA Example

RUN COMMAND: python examples/robust_pca.py

This example demonstrates robust PCA: decompose an observed matrix M into
a low-rank component L and a sparse component S.

Mathematical formulation:
    min_{L, S}   ||L||_*  +  lam * sum(|S_ij|)
    s.t.         L + S = M

Where:
    ||L||_* = nuclear norm (promotes low rank)
    sum(|S_ij|) = entrywise L1 norm (promotes element-wise sparsity)
    M = low-rank (m=50, r=10, n=40) + small noise, seed = 1.
    lam = 1 / sqrt(max(m, n)).
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
m = 50
r = 10
n = 40
M = np.random.randn(m, r) @ np.random.randn(r, n)
M = M + 0.1 * np.random.randn(m, n)
lam = 1.0 / np.sqrt(max(m, n))

print("=" * 70)
print("Robust PCA Example")
print("=" * 70)
print(f"Matrix size: {m} x {n}, true rank: {r}")
print(f"Regularization (lam): {lam:.4f}")
print()

# ============================================================================
# Step 2: Create model and variables
# ============================================================================
model = admm.Model()
L = admm.Var("L", m, n)
S = admm.Var("S", m, n)

# ============================================================================
# Step 3: Set objective — nuclear norm + entrywise L1
# ============================================================================
model.setObjective(admm.norm(L, ord="nuc") + lam * admm.sum(admm.abs(S)))

# ============================================================================
# Step 4: Add constraint
# ============================================================================
model.addConstr(L + S == M)

# ============================================================================
# Step 5: Solve
# ============================================================================
model.optimize()

# ============================================================================
# Step 6: Results
# ============================================================================
print()
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"  Status: {model.StatusString}")
    print(f"  Objective value: {model.ObjVal:.6f}")
    print(f"  Reconstruction error: {np.linalg.norm(np.asarray(L.X) + np.asarray(S.X) - M):.6f}")
else:
    print(f"  Optimization failed: {model.StatusString}")

print("=" * 70)
