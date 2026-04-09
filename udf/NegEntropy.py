import admm
import numpy as np


class NegEntropy(admm.UDFBase):
    r"""Negative entropy: the convex conjugate of log-sum-exp.

    Mathematical definition:

        f(x) = Σᵢ xᵢ · log(xᵢ)

    Domain: x > 0 (componentwise). By convention, 0·log(0) = 0.

    Behavior:
        xᵢ → 0⁺:   xᵢ·log(xᵢ) → 0     (from above)
        xᵢ = 1/e:   minimum of each component (= −1/e)
        xᵢ → +∞:    f → +∞              (super-linear growth)

    Gradient:

        ∇f(x)ᵢ = log(xᵢ) + 1

    Hessian:

        ∇²f(x)ᵢᵢ = 1/xᵢ    (diagonal, positive definite on x > 0)

    Properties:
    - Strictly convex on x > 0
    - The negative of Shannon entropy: H(x) = −Σ xᵢ·log(xᵢ)
    - Minimizing f(x) subject to Σxᵢ = 1 gives the uniform distribution
      (maximum entropy principle)
    - The Bregman divergence of f is the KL divergence

    Used in:
    - Maximum entropy estimation (regularizer)
    - Information-theoretic optimization
    - Mirror descent with entropic regularization
    - Optimal transport (entropy-regularized Wasserstein)

    Parameters
    ----------
    arg : admm.Var or expression
        The input (must be positive).
    """

    def __init__(self, arg):
        self.arg = arg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return float(np.sum(x * np.log(x)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        x = np.maximum(x, 1e-30)
        return [np.log(x) + 1.0]
