import admm
import math
import numpy as np


class RankPenalty(admm.UDFBase):
    """Matrix rank penalty.

    Function:

        f(X) = rank(X)

    Proximal operator (hard singular-value thresholding):

    Let X = U * diag(sigma) * V^T be the compact SVD. Then:

        prox_{lam * f}(X) = U * diag(sigma_hat) * V^T

        sigma_hat_i = sigma_i   if sigma_i > sqrt(2 * lam)
                    = 0         otherwise

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (matrix).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        return float(np.sum(singular_values > 1e-10))

    def argmin(self, lamb, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        threshold = math.sqrt(2.0 * lamb)
        singular_values = np.where(singular_values <= threshold, 0.0, singular_values)
        prox = (u * singular_values) @ vt
        return [prox.tolist()]
