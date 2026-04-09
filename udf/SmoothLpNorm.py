import admm
import numpy as np


class SmoothLpNorm(admm.UDFBase):
    r"""Smooth Lp pseudo-norm: differentiable approximation to ‖x‖ₚᵖ.

    Mathematical definition:

        f(x) = Σᵢ (xᵢ² + ε)^(p/2)

    where p ∈ (0, 2] and ε > 0 is a smoothing parameter.

    Behavior:
        ε → 0:  Converges to ‖x‖ₚᵖ = Σ|xᵢ|ᵖ
        p = 2:  f(x) = Σ(xᵢ² + ε) = ‖x‖₂² + nε  (≈ squared L2)
        p = 1:  f(x) = Σ√(xᵢ² + ε)                (≈ L1 norm)
        p < 1:  Promotes stronger sparsity than L1   (non-convex)

    Gradient:

        ∇f(x)ᵢ = p · xᵢ · (xᵢ² + ε)^(p/2 − 1)

    Properties:
    - Smooth everywhere (C∞) due to the ε perturbation
    - Convex for p ≥ 1
    - Non-convex but still useful for p < 1 (SCAD-like sparsity)
    - Reduces to smooth absolute power for general p

    The smooth Lp norm bridges L1 and L2 regularization and can
    approximate non-convex penalties (Lp, p < 1) in a differentiable
    manner, enabling gradient-based optimization for sparse estimation.

    Parameters
    ----------
    arg : admm.Var or expression
        The input vector.
    p : float
        The norm exponent, typically in (0, 2].
    eps : float
        Smoothing parameter (ε > 0). Smaller ε gives tighter
        approximation but larger Lipschitz constant near zero.
    """

    def __init__(self, arg, p=1.5, eps=1e-4):
        self.arg = arg
        self.p = p
        self.eps = eps

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum((x ** 2 + self.eps) ** (self.p / 2)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [self.p * x * (x ** 2 + self.eps) ** (self.p / 2 - 1)]
