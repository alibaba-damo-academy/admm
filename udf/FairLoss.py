import admm
import numpy as np


class FairLoss(admm.UDFBase):
    r"""Fair loss: a smooth robust loss between L2 and L1.

    Mathematical definition:

        f(r) = c² · Σᵢ (|rᵢ|/c − log(1 + |rᵢ|/c))

    where c > 0 is the scale parameter.

    Behavior:
        |r| ≪ c:  f(r) ≈ r²/(2c) · c = r²/2       (like L2)
        |r| ≫ c:  f(r) ≈ c·|r| − c²·log(|r|/c)    (sub-linear, between L1 and log)

    The Fair loss grows slower than L1 for large residuals but faster
    than Cauchy or Welsch. It provides moderate outlier resistance while
    maintaining a stronger pull toward fitting inliers than the fully
    bounded losses.

    Gradient:

        ∇f(r)ᵢ = rᵢ / (1 + |rᵢ|/c)

    The gradient is bounded by c and monotonically increasing, but
    sub-linear — providing the "fair" compromise between efficiency
    and robustness.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    c : float
        Scale parameter.
    """

    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        ar = np.abs(r)
        return float(self.c ** 2 * np.sum(ar / self.c - np.log(1 + ar / self.c)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [r / (1 + np.abs(r) / self.c)]
