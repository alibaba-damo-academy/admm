import admm
import numpy as np


class WelschLoss(admm.UDFBase):
    r"""Welsch (Leclerc) loss: a bounded robust loss with Gaussian-shaped influence.

    Mathematical definition:

        f(r) = (c²/2) · Σᵢ (1 − exp(−rᵢ²/c²))

    where c > 0 is the scale parameter.

    Behavior:
        |r| ≪ c:  f(r) ≈ r²/2                     (like L2)
        |r| ≫ c:  f(r) → c²/2                      (bounded, saturates)

    The loss is bounded above by n·c²/2, meaning outliers have
    strictly limited impact on the total cost.

    Gradient:

        ∇f(r)ᵢ = rᵢ · exp(−rᵢ²/c²)

    The gradient has a Gaussian profile: it rises linearly near zero,
    peaks at |r| = c/√2, then decays to zero — making it strongly
    redescending. Outliers beyond ~2c have essentially zero influence.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    c : float
        Scale parameter. Controls the width of the Gaussian influence.
    """

    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(0.5 * self.c ** 2 * np.sum(1 - np.exp(-(r / self.c) ** 2)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [r * np.exp(-(r / self.c) ** 2)]
