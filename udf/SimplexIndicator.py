import admm
import numpy as np


class SimplexIndicator(admm.UDFBase):
    """Indicator of the r-simplex.

    Function:

        f(x) = delta_{Delta_r}(x),    Delta_r = {x in R^n : x_i >= 0, sum_i x_i = r}

    Proximal operator (Euclidean projection onto the simplex):

        prox_{lam * f}(v)_i = max(v_i - theta, 0)

    where theta is chosen so that sum_i max(v_i - theta, 0) = r.

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    radius : float, optional
        Simplex radius r (default: 1.0).
    """

    def __init__(self, arg, radius=1.0):
        self.arg = arg
        self.radius = radius

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        if np.min(vector) >= -1e-9 and abs(np.sum(vector) - self.radius) <= 1e-9:
            return 0.0
        return float("inf")

    def argmin(self, lamb, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        sorted_vector = np.sort(vector)[::-1]
        cumulative = np.cumsum(sorted_vector) - self.radius
        indices = np.arange(1, len(vector) + 1)
        rho = np.nonzero(sorted_vector - cumulative / indices > 0)[0][-1]
        theta = cumulative[rho] / (rho + 1)
        prox = np.maximum(vector - theta, 0.0)
        return [prox.tolist()]
