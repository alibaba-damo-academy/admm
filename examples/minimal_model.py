"""
Minimal Model Example

RUN COMMAND: python examples/minimal_model.py

This example demonstrates the smallest complete ADMM modeling workflow from start to finish.
The model is intentionally tiny, so you can focus on the pattern: define variables,
define parameters, build the model, solve it, and then read the solution.

Mathematical formulation:
    min   x1 + x2
    s.t.  x1 >= p,
          x2 >= 0,

Where:
    x1, x2: scalar decision variables
    p: scalar parameter (bound at solve time)
"""

import admm

print("=" * 60)
print("Minimal Model Example")
print("=" * 60)
print("Mathematical formulation:")
print("    min   x1 + x2")
print("    s.t.  x1 >= p,")
print("          x2 >= 0,")
print()
print("Where:")
print("    x1, x2: scalar decision variables")
print("    p: scalar parameter (bound at solve time)")
print()

# ============================================================================
# Step 1: Create the variables and parameter
# ============================================================================
print("Step 1: Create variables and parameter")
x1 = admm.Var("x1")
x2 = admm.Var("x2")
p = admm.Param("p")

print(f"  - Created variable: {x1.name}")
print(f"  - Created variable: {x2.name}")
print(f"  - Created parameter: {p.name}")
print()

# ============================================================================
# Step 2: Create the model
# ============================================================================
print("Step 2: Create the model")
model = admm.Model()
print("  - Model created")
print()

# ============================================================================
# Step 3: Add the objective
# ============================================================================
print("Step 3: Add the objective (minimize x1 + x2)")
model.setObjective(x1 + x2)
print("  - Objective set: minimize x1 + x2")
print()

# ============================================================================
# Step 4: Add the constraints
# ============================================================================
print("Step 4: Add constraints")
model.addConstr(x1 >= p)
model.addConstr(x2 >= 0)
print("  - Added constraint: x1 >= p")
print("  - Added constraint: x2 >= 0")
print()

# ============================================================================
# Step 5: Solve with a concrete parameter value
# ============================================================================
print("Step 5: Solve with parameter value p = 2")
model.setOption(admm.Options.admm_max_iteration, 1000)
model.optimize({"p": 2})

print()
print("Results:")
print("-" * 60)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"✓ Status: {model.StatusString}")
    print(f"✓ Optimal objective value: {model.ObjVal:.6f} (expected: 2.0)")
    print(f"✓ x1 value: {x1.X:.6f} (expected: 2.0)")
    print(f"✓ x2 value: {x2.X:.6f} (expected: 0.0)")

    print()
    print("Explanation:")
    print("  The constraint x1 >= 2 forces the first variable to be at least 2,")
    print("  and x2 >= 0 forces the second variable to be at least 0.")
    print("  Since the objective tries to minimize their sum, the best")
    print("  choice is to sit exactly on those lower bounds:")
    print("  x1* = 2.0, x2* = 0.0")

else:
    print(f"✗ Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 60)
