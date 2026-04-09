import admm
import numpy as np


class LogCoshLoss(admm.UDFBase):
    """Log-cosh loss: a smooth approximation to L1.

    Function:

        f(r) = sum(log(cosh(r_i)))

    Behavior:
        log(cosh(r)) ≈ r^2/2      for small |r|  (like L2)
        log(cosh(r)) ≈ |r| - log2 for large |r|  (like L1)

    Gradient:

        grad_i = tanh(r_i)

    The gradient is bounded in [-1, 1], so large residuals (outliers) have
    limited influence on the fit.

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
        # Numerically stable: log(cosh(x)) = |x| - log(2) + log1p(exp(-2|x|))
        ar = np.abs(r)
        return float(np.sum(ar - np.log(2) + np.log1p(np.exp(-2 * ar))))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [np.tanh(r)]
