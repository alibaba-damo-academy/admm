"""
Sparse Inverse Covariance Estimation Example

RUN COMMAND: python examples/sparse_inverse_covariance.py

This example demonstrates sparse inverse covariance (precision matrix) estimation
using the graphical lasso formulation with ADMM.

Problem: Estimate a sparse precision matrix from observed data, which reveals
the conditional independence structure of a multivariate Gaussian distribution.

Mathematical formulation:
    min_{Theta >> 0}   -log det(Theta) + tr(S Theta) + lam * ||vec(Theta)||_1

Where:
    Theta: precision matrix (n x n, positive definite, decision variable)
    S: sample covariance matrix
    lam: L1 regularization parameter (promotes sparsity)
    log det: log-determinant barrier for positive definiteness
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
n = 30              # dimension of the precision matrix
sample_num = 60     # number of samples

# Generate a random true precision matrix (PSD)
A = np.random.randn(n, n)
true_precision = A.T @ A + 0.5 * np.eye(n)

# Draw samples from the corresponding Gaussian distribution
samples = np.random.multivariate_normal(
    mean=np.zeros(n),
    cov=np.linalg.inv(true_precision),
    size=sample_num,
)

# Compute sample covariance
S = np.cov(samples, rowvar=False)

# Regularization parameter
lam = 0.05

print("=" * 70)
print("Sparse Inverse Covariance Estimation Example")
print("=" * 70)
print(f"Matrix dimension (n): {n}")
print(f"Number of samples: {sample_num}")
print(f"L1 regularization (lam): {lam}")
print(f"True precision matrix density: {np.mean(np.abs(true_precision) > 0.01):.2%}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
Theta = admm.Var("Theta", n, n, PSD=True)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Graphical lasso: -log det(Theta) + tr(S @ Theta) + lam * ||Theta||_1
model.setObjective(
    -admm.log_det(Theta) + admm.trace(S @ Theta) + lam * admm.sum(admm.abs(Theta))
)

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# PSD constraint is enforced through the variable declaration (PSD=True)

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

    # Sparsity analysis
    Theta_val = np.asarray(Theta.X)
    total_entries = n * n
    nonzero_entries = np.sum(np.abs(Theta_val) > 1e-4)
    sparsity = 1.0 - nonzero_entries / total_entries

    print("Precision matrix statistics:")
    print(f"  Total entries: {total_entries}")
    print(f"  Nonzero entries (|Theta_ij| > 1e-4): {nonzero_entries}")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Matrix norm (Frobenius): {np.linalg.norm(Theta_val, 'fro'):.6f}")
    print()

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(Theta_val)
    print(f"  Minimum eigenvalue: {eigenvalues.min():.6f} (should be > 0)")
    print(f"  Maximum eigenvalue: {eigenvalues.max():.6f}")

    # Recovery quality
    recovery_error = np.linalg.norm(Theta_val - true_precision, 'fro')
    print(f"\n  Recovery error (||Theta - Theta_true||_F): {recovery_error:.6f}")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
