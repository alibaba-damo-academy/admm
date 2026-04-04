import admm
import numpy as np


class QuarticPenalty(admm.UDFBase):
    """Quartic penalty: sum of fourth powers.

    Function:

        f(x) = ||x||_4^4 = sum_i x_i^4

    Proximal operator (coordinatewise, via Cardano's formula):

    Each component p_i = prox_{lam * f}(v)_i satisfies p_i + 4*lam*p_i^3 = v_i.
    Setting xi_i = sqrt(v_i^2 + 1 / (27 * lam)), the closed-form solution is:

        p_i = cbrt((xi_i + v_i) / (8 * lam)) - cbrt((xi_i - v_i) / (8 * lam))

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
        x = np.asarray(tensorlist[0], dtype=float)
        return float(np.sum(np.abs(x) ** 4))

    def argmin(self, lamb, tensorlist):
        v = np.asarray(tensorlist[0], dtype=float)
        if lamb <= 0:
            return [v.tolist()]
        xi = np.sqrt(v * v + 1.0 / (27.0 * lamb))
        prox = np.cbrt((xi + v) / (8.0 * lamb)) - np.cbrt((xi - v) / (8.0 * lamb))
        return [prox.tolist()]
