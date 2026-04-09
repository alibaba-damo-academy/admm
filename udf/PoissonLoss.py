import admm
import numpy as np


class PoissonLoss(admm.UDFBase):
    r"""Poisson negative log-likelihood loss.

    Mathematical definition:

        f(w) = Σᵢ [exp(aᵢᵀw) − bᵢ · aᵢᵀw]

    where A is the design matrix (m × n), b ∈ ℝ₊ᵐ are observed
    counts, and w ∈ ℝⁿ is the parameter vector.

    This corresponds to the negative log-likelihood of a Poisson GLM
    with log link: E[bᵢ] = exp(aᵢᵀw), i.e., the canonical link.

    Behavior:
        aᵢᵀw = log(bᵢ):  optimal (mean = observation)
        aᵢᵀw too large:   exp penalty dominates (overshoot)
        aᵢᵀw too small:   −b·z term dominates (undershoot)

    Gradient:

        ∇f(w) = Aᵀ · (exp(Aw) − b)

    This is the standard score function for Poisson regression.

    Properties:
    - Convex (exp is convex, composition with affine is convex)
    - Smooth (C∞)
    - The natural loss for count data modeling
    - MLE under the Poisson assumption

    Used in:
    - Count data regression (epidemiology, ecology, insurance)
    - Photon-limited imaging (Poisson noise model)
    - Network traffic modeling
    - Topic models (Poisson factorization)

    Parameters
    ----------
    arg : admm.Var or expression
        The parameter vector w.
    A : array_like, shape (m, n)
        Design matrix.
    b : array_like, shape (m,)
        Observed counts (non-negative).
    """

    def __init__(self, arg, A, b):
        self.arg = arg
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.A @ w
        return float(np.sum(np.exp(z) - self.b * z))

    def grad(self, arglist):
        w = np.asarray(arglist[0], dtype=float).ravel()
        z = self.A @ w
        g = self.A.T @ (np.exp(z) - self.b)
        return [g.reshape(arglist[0].shape)]
