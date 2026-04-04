import admm
import numpy as np


class L21Norm(admm.UDFBase):
    """L_{2,1} norm: sum of column-wise L2 norms.

    Function:

        f(X) = ||X||_{2,1} = sum_j ||X[:,j]||_2

    The L_{2,1} norm promotes column (group) sparsity in a matrix
    X in R^{m x n}.

    Proximal operator (block soft-thresholding, applied columnwise):

        prox_{lam * f}(V)[:,j] = max(1 - lam / ||V[:,j]||_2, 0) * V[:,j]

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
        matrix = np.atleast_2d(np.asarray(tensorlist[0], dtype=float))
        return float(np.sum(np.linalg.norm(matrix, axis=0)))

    def argmin(self, lamb, tensorlist):
        matrix = np.atleast_2d(np.asarray(tensorlist[0], dtype=float))
        column_norms = np.linalg.norm(matrix, axis=0)
        scale = np.where(
            column_norms > 1e-12,
            np.maximum(1.0 - lamb / column_norms, 0.0),
            0.0,
        )
        prox = matrix * scale[np.newaxis, :]
        return [prox.tolist()]
