import admm
import numpy as np


class UnitSphereIndicator(admm.UDFBase):
    """Indicator of the unit sphere in R^n.

    Function:

        f(x) = delta_{S^{n-1}}(x),    S^{n-1} = {x in R^n : ||x||_2 = 1}

    Proximal operator (projection onto the unit sphere):

        prox_{lam * f}(v) = v / ||v||_2    if v != 0
                          = e_1            if v = 0

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
        norm = np.linalg.norm(vector)
        return 0.0 if abs(norm - 1.0) <= 1e-9 else float("inf")

    def argmin(self, lamb, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            prox = np.zeros_like(vector)
            prox[0] = 1.0
            return [prox.tolist()]
        return [(vector / norm).tolist()]
