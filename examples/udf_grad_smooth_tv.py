"""
User-Defined Smooth Function (grad UDF): Smooth Total Variation Denoising

RUN COMMAND: python examples/udf_grad_smooth_tv.py

This example demonstrates signal denoising using a smooth total variation
(TV) penalty. Standard TV (sum of |x_{i+1} - x_i|) is nonsmooth and
requires a proximal operator. The smooth variant replaces the absolute
value with sqrt(d^2 + eps), making it differentiable everywhere — a
perfect fit for the grad UDF path.

Problem: Piecewise-smooth signal recovery

Mathematical formulation:
    min_x   (1/2) ||x - y||_2^2  +  lam * sum(sqrt((x_{i+1} - x_i)^2 + eps))

Where:
    - The first term is data fidelity (stay close to noisy observation y)
    - The second term is smooth TV (encourage piecewise smoothness)
    - eps > 0 ensures differentiability everywhere

The gradient of smooth TV involves finite differences:
    grad_i = -(d_i / sqrt(d_i^2 + eps)) + (d_{i-1} / sqrt(d_{i-1}^2 + eps))

This shows how grad-based UDFs can encode STRUCTURAL gradient computations
(involving neighboring elements), not just elementwise operations.
"""

import admm
import numpy as np


# ============================================================================
# Step 1: Define Smooth Total Variation as a grad-based UDF
# ============================================================================
class SmoothTV(admm.UDFBase):
    """Smooth total variation: f(x) = sum(sqrt((x_{i+1} - x_i)^2 + eps)).

    Approximates the standard TV penalty sum(|x_{i+1} - x_i|) but is
    differentiable everywhere, enabling the grad UDF path.

    Parameters
    ----------
    arg : admm.Var
        Signal variable (1-D vector).
    eps : float
        Smoothing parameter. Smaller eps = closer to true TV.
    """

    def __init__(self, arg, eps=1e-4):
        self.arg = arg
        self.eps = eps

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        return float(np.sum(np.sqrt(d ** 2 + self.eps)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float).ravel()
        d = np.diff(x)
        dd = d / np.sqrt(d ** 2 + self.eps)
        g = np.zeros_like(x)
        g[:-1] -= dd   # contribution from d_i to x_i
        g[1:] += dd    # contribution from d_i to x_{i+1}
        return [g.reshape(arglist[0].shape)]


# ============================================================================
# Step 2: Generate a piecewise-constant signal with noise
# ============================================================================
np.random.seed(99)
n = 100

# True signal: piecewise constant with 4 levels
signal_true = np.zeros(n)
signal_true[0:25] = 1.0
signal_true[25:50] = 3.0
signal_true[50:75] = 0.5
signal_true[75:100] = 2.0

# Noisy observation
sigma = 0.5
y = signal_true + sigma * np.random.randn(n)

lam = 2.0

print("=" * 70)
print("Grad UDF: Smooth Total Variation Denoising")
print("=" * 70)
print(f"Signal length: {n}")
print(f"Noise level (sigma): {sigma}")
print(f"TV regularization (lam): {lam}")
print(f"True signal levels: [1.0, 3.0, 0.5, 2.0]")
print()

# ============================================================================
# Step 3: Denoise with ADMM
# ============================================================================
model = admm.Model()
x = admm.Var("x", n)

# Data fidelity + smooth TV regularization
model.setObjective(0.5 * admm.sum(admm.square(x - y)) + lam * SmoothTV(x))
model.optimize()

print(f"Status: {model.StatusString}")
print(f"Objective: {model.ObjVal:.4f}")
print()

x_val = np.asarray(x.X).ravel()

# ============================================================================
# Step 4: Evaluate recovery quality
# ============================================================================
mse_noisy = np.mean((y - signal_true) ** 2)
mse_denoised = np.mean((x_val - signal_true) ** 2)

print("Recovery quality:")
print(f"  MSE (noisy input):   {mse_noisy:.6f}")
print(f"  MSE (TV denoised):   {mse_denoised:.6f}")
print(f"  MSE reduction:       {mse_noisy / mse_denoised:.1f}x")
print()

# Show recovered levels at segment centers
segments = [(12, "segment 1"), (37, "segment 2"), (62, "segment 3"), (87, "segment 4")]
print("  Level recovery at segment centers:")
print(f"  {'Segment':<14} {'True':<10} {'Noisy':<10} {'Denoised':<10}")
print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
for idx, name in segments:
    print(f"  {name:<14} {signal_true[idx]:<10.4f} {y[idx]:<10.4f} {x_val[idx]:<10.4f}")

print("=" * 70)
