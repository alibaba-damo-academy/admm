"""
Semidefinite Programming Example

RUN COMMAND: python examples/semidefinite_program.py

Semidefinite program with linear trace objective and trace equality
constraints over a PSD matrix variable.

Mathematical formulation:
    min_X  tr(C X)
    s.t.   tr(A_i X) = b_i,  i = 1, ..., p
           X >> 0            (positive semidefinite)

Where:
    n = 4  matrix dimension
    p = 3  trace equality constraints
    C = R^T R  (PSD cost matrix)
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Generate problem data
# ============================================================================
np.random.seed(1)

n = 4  # matrix dimension
p = 3  # number of trace equality constraints

R = np.random.randn(n, n)
C = R.T @ R                       # PSD: tr(C X) >= 0 for all X >> 0

A = []
b = []
for _ in range(p):
    Ai = np.random.randn(n, n)
    Ai = 0.5 * (Ai + Ai.T)        # symmetric constraint matrices
    A.append(Ai)
    b.append(np.random.randn())
A = np.array(A)
b = np.array(b)

print("=" * 70)
print("Semidefinite Programming Example")
print("=" * 70)
print(f"Matrix dimension (n): {n}")
print(f"Trace constraints (p): {p}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
X = admm.Var("X", n, n, PSD=True)  # PSD matrix variable

# ============================================================================
# Step 3: Set objective
# ============================================================================
model.setObjective(admm.trace(C @ X))

# ============================================================================
# Step 4: Add trace equality constraints
# ============================================================================
for i in range(p):
    model.addConstr(admm.trace(A[i] @ X) == b[i])

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
    print(f"  Status:    {model.StatusString}")
    print(f"  Objective: {model.ObjVal:.6f}")
    print(f"  X =\n{np.round(np.asarray(X.X), 6)}")
    eigvals = np.linalg.eigvalsh(np.asarray(X.X))
    print(f"  Eigenvalues of X: {np.round(eigvals, 6)}")
    print(f"  All eigenvalues >= 0 (PSD): {np.all(eigvals >= -1e-4)}")
else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
