import admm
import numpy as np


class SmoothHingeLoss(admm.UDFBase):
    r"""Smooth hinge loss for SVM classification.

    Mathematical definition:

        f(w) = Σᵢ h(yᵢ · aᵢᵀw)

        h(z) = 0                if z ≥ 1
             = ½(1 − z)²       if 0 < z < 1
             = ½ − z            if z ≤ 0

    where A is the data matrix (m × n), y ∈ {−1, +1}ᵐ are labels,
    and w ∈ ℝⁿ is the weight vector.

    Behavior:
        z ≥ 1 (correct classification with margin):  zero loss
        0 < z < 1 (correct but insufficient margin):  quadratic penalty
        z ≤ 0 (misclassification):  linear penalty (like hinge)

    This is a C¹-smooth version of the standard hinge loss
    max(0, 1−z), which is non-differentiable at z = 1.

    Gradient:

        ∇f(w) = Aᵀ · (y ⊙ h'(y ⊙ Aw))

        h'(z) = 0          if z ≥ 1
              = −(1 − z)   if 0 < z < 1
              = −1          if z ≤ 0

    The smooth hinge loss is convex and has Lipschitz continuous
    gradient, making it well-suited for gradient-based optimization.

    Parameters
    ----------
    arg : admm.Var or expression
        The weight vector w.
    A : array_like, shape (m, n)
        Data matrix (each row is a data point).
    y : array_like, shape (m,)
        Labels in {−1, +1}.
    """

    def __init__(self, arg, A, y):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        loss = np.where(z >= 1, 0.0,
               np.where(z >= 0, 0.5 * (1 - z) ** 2,
                        0.5 - z))
        return float(np.sum(loss))

    def grad(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.y * (self.A @ w)
        dloss_dz = np.where(z >= 1, 0.0,
                   np.where(z >= 0, -(1 - z),
                            -1.0))
        g = self.A.T @ (self.y * dloss_dz)
        return [g.reshape(arglist[0].shape)]
