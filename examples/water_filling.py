"""
Water Filling Example

RUN COMMAND: python examples/water_filling.py

This example demonstrates the classic water-filling power allocation problem
using ADMM.

Problem: Allocate total power P across n channels to maximize the sum of
log-capacities, subject to a total power constraint and non-negativity.

Mathematical formulation:
    max_x   sum_i log(alpha_i + x_i)
    s.t.    sum(x) = P
            x >= 0

Equivalently (minimization):
    min_x   sum_i -log(alpha_i + x_i)
    s.t.    sum(x) = P
            x >= 0

Where:
    x: power allocation per channel (decision variable)
    alpha: channel gain for each channel
    P: total power budget
"""

import admm
import numpy as np

# ============================================================================
# Step 1: Define problem data
# ============================================================================
alpha = np.array([0.5, 0.8, 1.0, 1.3, 1.6])   # channel gains
total_power = 2.0                                # total power budget
n = len(alpha)

print("=" * 70)
print("Water Filling Example")
print("=" * 70)
print(f"Number of channels (n): {n}")
print(f"Total power budget (P): {total_power}")
print(f"Channel gains (alpha): {alpha}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Maximize sum(log(alpha + x)) = minimize sum(-log(alpha + x))
model.setObjective(admm.sum(-admm.log(alpha + x)))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
model.addConstr(admm.sum(x) == total_power)     # total power constraint
model.addConstr(x >= 0)                          # non-negativity

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

    # Water level (alpha_i + x_i should be constant for active channels)
    water_levels = alpha + x_val
    active_channels = x_val > 1e-4

    print("Power allocation:")
    print(f"  {'Channel':<10} {'Alpha':<10} {'Power':<12} {'Level':<12} {'Active'}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*6}")
    for i in range(n):
        mark = "yes" if active_channels[i] else "no"
        print(f"  {i:<10} {alpha[i]:<10.3f} {x_val[i]:<12.6f} {water_levels[i]:<12.6f} {mark}")

    print()
    print(f"  Total power allocated: {np.sum(x_val):.6f} (budget: {total_power})")
    print(f"  Active channels: {np.sum(active_channels)}/{n}")

    # Total capacity
    total_capacity = np.sum(np.log(water_levels))
    print(f"  Total capacity (sum log): {total_capacity:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
