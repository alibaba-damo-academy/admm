"""
Convolutional Image Deblurring Example

RUN COMMAND: python examples/image_deblurring.py

This example demonstrates image deblurring using total variation regularization
and 2D convolution with ADMM.

Problem: Recover a sharp image U from a blurred observation B by minimizing
a combination of total variation (edge-preserving regularizer) and data
fidelity (squared error with the blurred image).

Mathematical formulation:
    min_U   lam * TV(U)  +  (1/2) ||K * U - B||_F^2

Where:
    U: recovered image (decision variable)
    B: observed blurred image
    K: 3x3 Gaussian blur kernel
    TV(U): total variation penalty (promotes piecewise smoothness)
    lam: regularization parameter
    K * U: 2D convolution with mode "same"
"""

import admm
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# ============================================================================
# Step 1: Generate synthetic data
# ============================================================================
height = 40
width = 50

# Piecewise-constant synthetic image (blocks) — TV regularization is effective here
image = np.zeros((height, width))
image[:20, :25] = 0.8
image[20:, 25:] = 0.6
image[10:30, 10:40] = 1.0
image += 0.02 * np.random.randn(height, width)  # slight noise

# Gaussian blur kernel (3x3)
kernel = np.array([
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
])

# Create blurred image by convolving with the kernel
image_blur = admm.conv2d(image, kernel, "same")

lam = 0.1   # regularization parameter

print("=" * 70)
print("Convolutional Image Deblurring Example")
print("=" * 70)
print(f"Image size: {height} x {width}")
print(f"Kernel size: {kernel.shape[0]} x {kernel.shape[1]} (Gaussian)")
print(f"Regularization (lam): {lam}")
print(f"Original image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Blurred image type: {type(image_blur).__name__}")
print()

# ============================================================================
# Step 2: Create model and decision variables
# ============================================================================
model = admm.Model()
U = admm.Var("U", image.shape)

# ============================================================================
# Step 3: Set objective
# ============================================================================
# Total variation + data fidelity
tv = admm.tv2d(U, p=1)
residual = admm.conv2d(U, kernel, "same") - image_blur
model.setObjective(lam * tv + 0.5 * admm.sum(admm.square(residual)))

# ============================================================================
# Step 4: Add constraints
# ============================================================================
# Unconstrained problem (regularization handles smoothness)

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

    U_val = np.asarray(U.X)

    # Recovery quality
    recovery_error = np.linalg.norm(U_val - image, 'fro')
    psnr = 10 * np.log10(1.0 / np.mean((U_val - image) ** 2))

    print("Recovery statistics:")
    print(f"  Recovery error (||U - I_orig||_F): {recovery_error:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Recovered image range: [{U_val.min():.3f}, {U_val.max():.3f}]")
    print()

    blur_error = np.linalg.norm(np.asarray(image_blur) - image, 'fro')
    reduction = (1 - recovery_error / blur_error) * 100
    print(f"  Blurred error:  {blur_error:.4f}")
    print(f"  Recovery error: {recovery_error:.4f}")
    print(f"  Error reduction: {reduction:.0f}%")

else:
    print(f"  Optimization failed: {model.StatusString}")
    print(f"  Primal gap: {model.PrimalGap}")
    print(f"  Dual gap: {model.DualGap}")

print("=" * 70)
