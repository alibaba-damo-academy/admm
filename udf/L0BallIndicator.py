import admm
import numpy as np


class L0BallIndicator(admm.UDFBase):
    """Indicator of the L0 ball: at most k nonzero entries.

    Function:

        f(x) = delta_{B0_k}(x),    B0_k = {x in R^n : ||x||_0 <= k}

    Proximal operator (keep the k largest-magnitude entries):

        prox_{lam * f}(v)_i = v_i   if i in S_k(v)
                            = 0      otherwise

    where S_k(v) is the index set of the k entries of v with the
    largest absolute values.

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    k : int, optional
        Maximum number of nonzero entries (default: 1).
    """

    def __init__(self, arg, k=1):
        self.arg = arg
        self.k = k

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        return 0.0 if np.count_nonzero(np.abs(vector) > 1e-12) <= self.k else float("inf")

    def argmin(self, lamb, tensorlist):
        vector = np.asarray(tensorlist[0], dtype=float)
        prox = np.zeros_like(vector)
        keep_count = min(max(self.k, 0), vector.size)
        if keep_count > 0:
            keep_idx = np.argpartition(np.abs(vector), -keep_count)[-keep_count:]
            prox[keep_idx] = vector[keep_idx]
        return [prox.tolist()]
