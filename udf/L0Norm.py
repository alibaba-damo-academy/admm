import admm
import math
import numpy as np


class L0Norm(admm.UDFBase):
    """L0 norm: count of nonzero entries.

    Function:

        f(x) = ||x||_0 = #{i : x_i != 0}

    Proximal operator (coordinatewise hard threshold):

        prox_{lam * f}(v)_i = v_i    if |v_i| > sqrt(2 * lam)
                            = 0      otherwise

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
        return float(np.count_nonzero(np.abs(vector) > 1e-12))

    def argmin(self, lamb, tensorlist):
        v = np.asarray(tensorlist[0], dtype=float)
        threshold = math.sqrt(2.0 * lamb)
        prox = np.where(np.abs(v) <= threshold, 0.0, v)
        return [prox.tolist()]
