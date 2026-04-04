"""
Least Squares Example

RUN COMMAND: python examples/least_squares.py

This example demonstrates linear least squares fitting using ADMM.

Problem: Fit a linear model to observed data by minimizing the sum of
squared residuals.

Mathematical formulation:
    min_x ||Ax - b||_2^2

Where:
    A: design matrix (m x n)
    b: observation vector (m)
    x: parameter vector to estimate (n)
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
m = 40  # number of observations
n = 12  # number of parameters

# Design matrix
A = np.random.randn(m, n)

# True parameter vector (unknown in practice)
x_true = np.random.randn(n)

# Observations with noise
noise_level = 0.1
b = A @ x_true + noise_level * np.random.randn(m)

print("=" * 70)
print("Least Squares Example")
print("=" * 70)
print(f"Number of observations (m): {m}")
print(f"Number of parameters (n): {n}")
print(f"Noise level: {noise_level}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)  # parameter vector to estimate

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Minimize sum of squared residuals: ||Ax - b||_2^2
residual = A @ x - b
model.setObjective(admm.sum(admm.square(residual)))

# ============================================================================
# Step 4: Solve the model
# ============================================================================
model.optimize()

# ============================================================================
# Step 5: Check status and print results
# ============================================================================
print("Results:")
print("-" * 70)

if model.StatusString == "SOLVE_OPT_SUCCESS":
    print(f"✓ Status: {model.StatusString}")
    print(f"✓ Optimal objective value: {model.ObjVal:.6f}")
    print(f"  (Expected: ~{noise_level**2 * m:.3f} for noisy data)")
    print(f"✓ Solver time: {model.SolverTime:.4f} seconds")
    print()

    # Solution quality
    solution_error = np.linalg.norm(x.X - x_true)
    print("Solution quality:")
    print(f"  Estimation error ||x - x_true||: {solution_error:.6f}")
    print(f"  Relative error: {solution_error / np.linalg.norm(x_true) * 100:.2f}%")
    print()

    # Show first few parameters
    print("First 5 parameter estimates:")
    for i in range(min(5, n)):
        print(f"  x[{i}] = {x.X[i]:.6f} (true: {x_true[i]:.6f})")

    # Residual analysis
    residuals = A @ x.X - b
    print()
    print("Residual statistics:")
    print(f"  Mean: {residuals.mean():.6f}")
    print(f"  Std dev: {residuals.std():.6f} (should be close to {noise_level})")
    print(f"  Max abs: {np.abs(residuals).max():.6f}")

else:
    print(f"✗ Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
