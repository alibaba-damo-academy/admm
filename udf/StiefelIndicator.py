import admm
import numpy as np


class StiefelIndicator(admm.UDFBase):
    """Indicator of the Stiefel manifold St(m, n).

    Function:

        f(X) = delta_{St(m,n)}(X),    St(m,n) = {X in R^{m x n} : X^T X = I_n},  m >= n

    The Stiefel manifold is the set of matrices with orthonormal columns.
    When m = n, it reduces to the orthogonal group O(n).

    Proximal operator (projection via polar factor):

    Let Z = U * Sigma * V^T be the compact SVD. Then:

        prox_{lam * f}(Z) = U * V^T

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (matrix), shape (m, n) with m >= n.
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
            return float("inf")
        identity = np.eye(matrix.shape[1])
        return 0.0 if np.linalg.norm(matrix.T @ matrix - identity) <= 1e-9 else float("inf")

    def argmin(self, lamb, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        prox = u @ vt
        return [prox.tolist()]
