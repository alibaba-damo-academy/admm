import admm
import numpy as np


class BoundedRatioLoss(admm.UDFBase):
    r"""Bounded ratio penalty: smooth approximation to L0 norm.

    Mathematical definition:

        f(x) = Σᵢ xᵢ² / (1 + xᵢ²)

    Behavior:
        xᵢ = 0:    f = 0                 (minimum)
        |xᵢ| → ∞:  f → 1                 (saturates at 1 per element)
        |xᵢ| ≪ 1:  f ≈ xᵢ²              (like L2 near zero)
        |xᵢ| ≫ 1:  f ≈ 1 − 1/xᵢ²        (nearly constant)

    The total loss is bounded in [0, n) where n is the dimension.

    Gradient:

        ∇f(x)ᵢ = 2xᵢ / (1 + xᵢ²)²

    The gradient is redescending: peaks at |xᵢ| = 1/√3 and decays
    as O(1/xᵢ³) for large residuals.

    Properties:
    - Smooth (C∞) everywhere
    - Non-convex, but bounded
    - Approximates ‖x‖₀ (counts nonzeros) as elements grow
    - Each element contributes at most 1 to the total loss
    - Same influence function shape as Geman-McClure (without the 1/2 factor)

    Used in:
    - Sparse signal recovery (smooth L0 relaxation)
    - Robust estimation (bounded loss)
    - Feature selection (approximate cardinality penalty)

    Parameters
    ----------
    arg : admm.Var or expression
        The input vector.
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum(x ** 2 / (1 + x ** 2)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [2.0 * x / (1 + x ** 2) ** 2]
