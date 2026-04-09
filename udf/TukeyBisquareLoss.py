import admm
import numpy as np


class TukeyBisquareLoss(admm.UDFBase):
    r"""Tukey's bisquare (biweight) loss: complete outlier rejection.

    Mathematical definition:

        f(r) = Σᵢ ρ(rᵢ)

        ρ(r) = (c²/6) · (1 − (1 − (r/c)²)³)   if |r| ≤ c
             = c²/6                               if |r| > c

    where c > 0 is the rejection threshold (default 4.685 for 95%
    efficiency at Gaussian data).

    Behavior:
        |r| ≪ c:  f(r) ≈ r²/2                      (like L2)
        |r| > c:   f(r) = c²/6                      (constant — outlier fully rejected)

    This is the only common M-estimator that completely ignores
    residuals beyond the threshold c — the gradient is exactly zero
    for |r| > c.

    Gradient:

        ∇f(r)ᵢ = rᵢ · (1 − (rᵢ/c)²)²   if |rᵢ| ≤ c
                = 0                        if |rᵢ| > c

    Non-convex globally, but widely used in robust statistics (IRLS,
    M-estimation) for its breakdown point properties.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    c : float
        Rejection threshold. Default 4.685 (95% Gaussian efficiency).
    """

    def __init__(self, arg, c=4.685):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        u = r / self.c
        mask = np.abs(u) <= 1
        val = np.where(mask,
                       self.c ** 2 / 6.0 * (1 - (1 - u ** 2) ** 3),
                       self.c ** 2 / 6.0)
        return float(np.sum(val))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        u = r / self.c
        mask = np.abs(u) <= 1
        return [np.where(mask, r * (1 - u ** 2) ** 2, 0.0)]
