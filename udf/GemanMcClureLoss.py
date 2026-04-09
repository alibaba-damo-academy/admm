import admm
import numpy as np


class GemanMcClureLoss(admm.UDFBase):
    r"""Geman-McClure loss: a bounded, smooth robust estimator.

    Mathematical definition:

        f(r) = Σᵢ rᵢ² / (2·(1 + rᵢ²))

    Behavior:
        |r| ≪ 1:  f(r) ≈ r²/2                      (like L2)
        |r| ≫ 1:  f(r) → 1/2                        (bounded, saturates)

    The loss is bounded above by n/2. Each residual contributes at
    most 1/2 to the total cost, regardless of magnitude.

    Gradient:

        ∇f(r)ᵢ = rᵢ / (1 + rᵢ²)²

    The gradient is redescending: it peaks at |r| = 1/√3 and decays
    as O(1/r³) for large residuals.

    Used in robust estimation, image processing, and computer vision
    where gross outliers must be completely suppressed.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(r ** 2 / (2.0 * (1 + r ** 2))))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [r / (1 + r ** 2) ** 2]
