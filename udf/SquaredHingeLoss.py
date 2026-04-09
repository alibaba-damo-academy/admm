import admm
import numpy as np


class SquaredHingeLoss(admm.UDFBase):
    r"""Squared hinge loss for SVM classification.

    Mathematical definition:

        f(w) = Σᵢ max(0, 1 − yᵢ · aᵢᵀw)²

    where A is the data matrix (m × n), y ∈ {−1, +1}ᵐ are labels,
    and w ∈ ℝⁿ is the weight vector.

    Behavior:
        yᵢ·aᵢᵀw ≥ 1:   loss = 0          (correct with sufficient margin)
        yᵢ·aᵢᵀw < 1:    loss = (1 − margin)²  (quadratic penalty)

    The squared hinge loss is a smooth (C¹) alternative to the
    standard hinge loss max(0, 1−z), which has a kink at z = 1.

    Gradient:

        ∇f(w)ⱼ = −2 · Σ_{i: marginᵢ<1} (1 − yᵢ·aᵢᵀw) · yᵢ · Aᵢⱼ

    Properties:
    - Convex (composition of convex increasing with convex)
    - C¹ smooth (gradient is continuous, unlike hinge)
    - Not C² (second derivative has a discontinuity at margin = 1)
    - Penalizes margin violations more aggressively than hinge (quadratic vs linear)

    Used in:
    - SVM variants (L2-SVM)
    - Structured prediction
    - Multi-class classification (Crammer-Singer formulation)

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
        margins = self.y * (self.A @ w)
        losses = np.maximum(0, 1 - margins) ** 2
        return float(np.sum(losses))

    def grad(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        margins = self.y * (self.A @ w)
        violations = np.maximum(0, 1 - margins)
        g = -2.0 * self.A.T @ (violations * self.y)
        return [g.reshape(arglist[0].shape)]
