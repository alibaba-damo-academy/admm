import admm
import numpy as np


class RankRIndicator(admm.UDFBase):
    """Indicator of the set of matrices with rank at most r.

    Function:

        f(X) = delta_{rank(X) <= r}(X)

    Proximal operator (truncated SVD projection):

    Let X = U * diag(sigma) * V^T be the compact SVD. Then:

        prox_{lam * f}(X) = U_r * diag(sigma_r) * V_r^T

    where U_r, sigma_r, V_r retain only the r largest singular values
    and their corresponding singular vectors.

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (matrix).
    rank_bound : int, optional
        Maximum allowed rank r (default: 1).
    """

    def __init__(self, arg, rank_bound=1):
        self.arg = arg
        self.rank_bound = rank_bound

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        return 0.0 if np.sum(singular_values > 1e-10) <= self.rank_bound else float("inf")

    def argmin(self, lamb, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
        singular_values[min(self.rank_bound, len(singular_values)):] = 0.0
        prox = (u * singular_values) @ vt
        return [prox.tolist()]
