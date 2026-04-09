import admm
import numpy as np


class BerhuLoss(admm.UDFBase):
    r"""Reverse Huber (Berhu) loss: L1 for small, L2 for large residuals.

    Mathematical definition:

        f(r) = Σᵢ ρ(rᵢ)

        ρ(r) = |r|                       if |r| ≤ c
             = (r² + c²) / (2c)         if |r| > c

    where c > 0 is the transition threshold.

    This is the OPPOSITE of the standard Huber loss:
        - Huber: L2 for small |r|, L1 for large |r|
        - Berhu: L1 for small |r|, L2 for large |r|

    Gradient:

        ∇f(r)ᵢ = sign(rᵢ)    if |rᵢ| ≤ c
                = rᵢ / c      if |rᵢ| > c

    The Berhu loss is used in monocular depth estimation (Laina et al.,
    2016) where small residuals should be treated as L1 (sparsity-like)
    but large residuals should be penalized quadratically to ensure
    reconstruction accuracy.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    c : float
        Transition threshold between L1 and L2 regimes.
    """

    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        ar = np.abs(r)
        return float(np.sum(np.where(
            ar <= self.c, ar,
            (r ** 2 + self.c ** 2) / (2 * self.c)
        )))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        ar = np.abs(r)
        return [np.where(ar <= self.c, np.sign(r), r / self.c)]
