import admm
import numpy as np


class BinaryIndicator(admm.UDFBase):
    """Indicator of the binary cube {0, 1}^n.

    Function:

        f(x) = delta_{{0,1}^n}(x)

    Proximal operator (coordinatewise rounding):

        prox_{lam * f}(v)_i = 1   if v_i >= 0.5
                            = 0   if v_i <  0.5

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        is_binary = np.logical_or(np.abs(vector) <= 1e-9, np.abs(vector - 1.0) <= 1e-9)
        return 0.0 if np.all(is_binary) else float("inf")

    def argmin(self, lamb, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        prox = np.where(vector >= 0.5, 1.0, 0.0)
        return [prox.tolist()]
