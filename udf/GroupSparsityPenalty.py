import admm
import numpy as np


class GroupSparsityPenalty(admm.UDFBase):
    """Group sparsity penalty: number of nonzero columns.

    Function:

        G(X) = #{j : ||X[:,j]||_2 > 0}

    for a matrix X in R^{m x n}. This is the column-group analogue of
    the L0 norm and promotes exact block (column) sparsity.

    Proximal operator (columnwise hard threshold):

        prox_{lam * G}(V)[:,j] = V[:,j]    if ||V[:,j]||_2^2 > 2 * lam
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
        column_norms = np.linalg.norm(matrix, axis=0)
        return float(np.count_nonzero(column_norms > 1e-12))

    def argmin(self, lamb, tensorlist):
        matrix = np.asarray(tensorlist[0], dtype=float)
        column_norm_sq = np.sum(matrix * matrix, axis=0)
        keep_mask = column_norm_sq > 2.0 * lamb
        prox = matrix * keep_mask[np.newaxis, :]
        return [prox.tolist()]
