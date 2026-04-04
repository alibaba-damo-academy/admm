import admm
import numpy as np


class OrthogonalMatrixIndicator(admm.UDFBase):
    """Indicator of the orthogonal group O(n).

    Function:

        f(X) = delta_{O(n)}(X),    O(n) = {X in R^{n x n} : X^T X = I_n}

    Proximal operator (projection via polar factor):

    Let X = U * Sigma * V^T be the compact SVD. Then:

        prox_{lam * f}(X) = U * V^T

    This is the square-matrix special case of StiefelIndicator.

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (square matrix), shape (n, n).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return float("inf")
        identity = np.eye(matrix.shape[0])
        return 0.0 if np.linalg.norm(matrix.T @ matrix - identity) <= 1e-9 else float("inf")

    def argmin(self, lamb, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        prox = u @ vt
        return [prox.tolist()]
