import admm
import numpy as np


class CauchyLoss(admm.UDFBase):
    """Cauchy (Lorentzian) loss: heavy-tailed robust loss.

    Function:

        f(r) = sum(log(1 + (r_i / c)^2))

    Behavior:
        - Quadratic near zero: ≈ r^2 / c^2
        - Logarithmic for large |r|: ≈ 2 * log(|r| / c)

    Gradient:

        grad_i = 2 * r_i / (c^2 + r_i^2)

    The gradient is redescending — it tends to zero as |r| → ∞.
    This means very large outliers are automatically downweighted.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    c : float
        Scale parameter. Smaller c = more aggressive outlier rejection.
    """

    def __init__(self, arg, c=1.0):
        self.arg = arg
        self.c = c

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(np.log(1 + (r / self.c) ** 2)))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [2.0 * r / (self.c ** 2 + r ** 2)]
