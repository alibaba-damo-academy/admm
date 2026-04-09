import admm
import numpy as np


class KLDivergence(admm.UDFBase):
    r"""Kullback-Leibler divergence from a fixed reference distribution.

    Mathematical definition:

        f(p) = Σᵢ pᵢ · log(pᵢ / qᵢ)
             = Σᵢ [pᵢ · log(pᵢ) − pᵢ · log(qᵢ)]

    where q is a fixed reference distribution (q > 0) and p > 0
    is the variable distribution.

    Behavior:
        p = q:    f(p) = 0                 (minimum)
        p ≠ q:    f(p) > 0                 (Gibbs' inequality)
        pᵢ → 0:  pᵢ·log(pᵢ/qᵢ) → 0      (by convention)

    Gradient:

        ∇f(p)ᵢ = log(pᵢ/qᵢ) + 1

    The KL divergence is a fundamental measure of dissimilarity
    between probability distributions. It is used in:
    - Variational inference (ELBO objective)
    - Information geometry
    - Maximum entropy estimation
    - Bayesian model comparison

    Note: The KL divergence is NOT symmetric: KL(p‖q) ≠ KL(q‖p).
    The variable p should be constrained to be positive.

    Parameters
    ----------
    arg : admm.Var or expression
        The variable distribution p (must be positive).
    q : array_like
        The fixed reference distribution (must be positive).
    """

    def __init__(self, arg, q):
        self.arg = arg
        self.q = np.asarray(q, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        p = np.maximum(np.asarray(arglist[0], dtype=float), 1e-30)
        return float(np.sum(p * np.log(p / self.q)))

    def grad(self, arglist):
        p = np.maximum(np.asarray(arglist[0], dtype=float), 1e-30)
        return [np.log(p / self.q) + 1.0]
