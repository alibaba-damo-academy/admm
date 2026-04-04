"""
Modeling Workflow Example

RUN COMMAND: python examples/modeling_workflow.py

This example demonstrates the complete ADMM modeling workflow from start to finish.
It includes variables, a parameter, constraints, solver options, and post-solve inspection.

Mathematical formulation:
    min   0.5 * ||Ax - b||_2^2 + alpha * ||x||_1
    s.t.  x >= 0

Where:
    x: coefficient vector (optimization variable)
    A: data matrix
    b: observation vector
    alpha: regularization parameter (bound at solve time)
"""

import admm
import numpy as np

print("=" * 70)
print("Modeling Workflow Example")
print("=" * 70)
print("Mathematical formulation:")
print("    min   0.5 * ||Ax - b||_2^2 + alpha * ||x||_1")
print("    s.t.  x >= 0")
print()
print("Where:")
print("    x: coefficient vector (optimization variable)")
print("    A: data matrix")
print("    b: observation vector")
print("    alpha: regularization parameter (bound at solve time)")
print()

# ============================================================================
# Step 1: Create decision variables and parameters
# ============================================================================
print("Step 1: Create decision variables and parameters")
np.random.seed(1)
m = 30  # number of observations
n = 10  # number of variables
A = np.random.randn(m, n)  # Data matrix
b = np.random.randn(m)     # Observation vector

model = admm.Model()  # Create the model
x = admm.Var("x", n)  # Optimization variable
alpha = admm.Param("alpha")  # Regularization parameter

print("  - Model created")
print(f"  - Created variable: {x.name} with shape {x.shape}")
print(f"  - Created parameter: {alpha.name}")
print(f"  - Data matrix A: {A.shape}")
print(f"  - Observation vector b: {b.shape}")
print()

# ============================================================================
# Step 2: Build expressions from operators and atoms
# ============================================================================
print("Step 2: Build expressions")
residual = A @ x - b
loss = 0.5 * admm.sum(admm.square(residual))
regularizer = alpha * admm.norm(x, ord=1)

print("  - Built residual expression: A @ x - b")
print("  - Built loss expression: 0.5 * sum(square(residual))")
print("  - Built regularizer expression: alpha * norm(x, ord=1)")
print()

# ============================================================================
# Step 3: Set one scalar objective
# ============================================================================
print("Step 3: Set scalar objective")
model.setObjective(loss + regularizer)
print("  - Objective set: loss + regularizer")
print()

# ============================================================================
# Step 4: Add feasibility conditions (constraints)
# ============================================================================
print("Step 4: Add constraints")
model.addConstr(x >= 0)  # Structural feasibility
print("  - Added constraint: x >= 0")
print()

# ============================================================================
# Step 5: Tune options (only after model structure is correct)
# ============================================================================
print("Step 5: Tune solver options")
model.setOption(admm.Options.admm_max_iteration, 5000)  # Iteration budget
print("  - Set max iterations: 5000")
print()

# ============================================================================
# Step 6: Solve with concrete parameter data
# ============================================================================
print("Step 6: Solve with parameter value alpha = 0.1")
model.optimize({"alpha": 0.1})  # Bind parameter data and solve
print()

# ============================================================================
# Step 7: Read the results in order
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"✓ Status: {model.StatusString}")
    print(f"✓ Optimal objective value: {model.ObjVal:.6f}")
    print(f"✓ Solver time: {model.SolverTime:.4f} seconds")
    print()

    print("Solution vector x:")
    print(f"  - Shape: {x.X.shape}")
    print(f"  - Values: {np.round(np.asarray(x.X), 6)}")
    print(f"  - Min value: {x.X.min():.6f}")
    print(f"  - Max value: {x.X.max():.6f}")
    print(f"  - Number of non-zeros: {np.sum(np.abs(x.X) > 1e-6)}/{n}")

    # Verify constraint satisfaction
    print(f"  - Min value: {x.X.min():.6f}")

else:
    print(f"✗ Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print()
print("Explanation:")
print("  The objective combines a least squares loss (0.5 * ||Ax - b||_2^2)")
print("  with an L1 regularization term (alpha * ||x||_1) to promote sparsity.")
print("  The constraint x >= 0 enforces nonnegativity on all coefficients.")
print("  The L1 penalty encourages many coefficients to be exactly zero.")
print("=" * 70)
