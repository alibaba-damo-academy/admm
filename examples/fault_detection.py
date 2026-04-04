"""
Fault Detection Example

RUN COMMAND: python examples/fault_detection.py

This example demonstrates fault detection via box-constrained sparse regression
using ADMM.

Problem: Identify faulty sensors in a network by solving a box-constrained
quadratic program with an L1-like penalty that promotes sparse fault indicators.

Mathematical formulation:
    min_x   ||A x - y||_2^2  +  tau * 1^T x
    s.t.    0 <= x <= 1

Where:
    x: fault indicator vector (0 = normal, 1 = faulty)
    A: measurement matrix (m x n)
    y: observed measurements
    tau: penalty parameter (controls sparsity of faults)
    n: number of sensors
    m: number of measurements
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 200         # number of sensors
m = 40          # number of measurements
p_fault = 0.03  # probability of a sensor being faulty
snr = 5.0       # signal-to-noise ratio

sigma = np.sqrt(p_fault * n / (snr ** 2))
A = np.random.randn(m, n)

# Generate ground truth: sparse fault indicators
x_true = (np.random.rand(n) <= p_fault).astype(float)
y = A @ x_true + sigma * np.random.randn(m)

# Penalty parameter (derived from Bayesian analysis)
tau = 2 * np.log(1 / p_fault - 1) * sigma ** 2

num_true_faults = int(np.sum(x_true))
print("=" * 70)
print("Fault Detection Example")
print("=" * 70)
print(f"Number of sensors (n): {n}")
print(f"Number of measurements (m): {m}")
print(f"Fault probability (p_fault): {p_fault}")
print(f"True number of faults: {num_true_faults}")
print(f"SNR: {snr}")
print(f"Penalty parameter (tau): {tau:.6f}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
model.setOption(admm.Options.admm_max_iteration, 5000)
x = admm.Var("x", n)

# ============================================================================
# Step 3: Set objective
# ============================================================================
model.setObjective(admm.sum(admm.square(A @ x - y)) + tau * admm.sum(x))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
model.addConstr(x >= 0)     # fault indicator lower bound
model.addConstr(x <= 1)     # fault indicator upper bound

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

    # Detection statistics
    threshold = 0.5
    detected_faults = np.sum(x_val > threshold)
    true_positives = np.sum((x_val > threshold) & (x_true > 0.5))
    false_positives = np.sum((x_val > threshold) & (x_true < 0.5))
    false_negatives = np.sum((x_val <= threshold) & (x_true > 0.5))

    print(f"Detection results (threshold = {threshold}):")
    print(f"  True faults: {num_true_faults}")
    print(f"  Detected faults: {int(detected_faults)}")
    print(f"  True positives: {int(true_positives)}")
    print(f"  False positives: {int(false_positives)}")
    print(f"  False negatives: {int(false_negatives)}")
    print()

    # Solution statistics
    print("Solution statistics:")
    print(f"  Min fault indicator: {x_val.min():.6f}")
    print(f"  Max fault indicator: {x_val.max():.6f}")
    print(f"  Mean fault indicator: {x_val.mean():.6f}")
    print(f"  Indicators > 0.1: {np.sum(x_val > 0.1)}")

    # Show detected fault locations
    fault_indices = np.where(x_val > threshold)[0]
    if len(fault_indices) > 0:
        print(f"\n  Detected fault sensor indices: {fault_indices.tolist()}")
        true_fault_indices = np.where(x_true > 0.5)[0]
        print(f"  True fault sensor indices:     {true_fault_indices.tolist()}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
