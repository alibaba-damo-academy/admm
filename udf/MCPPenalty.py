import admm
import math
import numpy as np


class MCPPenalty(admm.UDFBase):
    """Minimax Concave Penalty (MCP; Zhang, 2010).

    The MCP interpolates between the L1 norm and the L0 norm by gradually
    relaxing the L1 bias as |x_i| grows. It is nonconvex.

    Function (coordinatewise, t = |x_i|):

        m(x_i) = lam * t - t^2 / (2 * beta)    if t <= beta * lam
               = beta * lam^2 / 2              if t > beta * lam

    Proximal operator (coordinatewise):

    When beta > lam_s (ADMM step size):

        prox_{lam_s * m}(v)_i = sgn(v_i) * min(beta * max(|v_i| - lam*lam_s, 0) / (beta - lam_s),  |v_i|)

    When beta <= lam_s, hard-threshold at sqrt(beta * lam_s) * lam.

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    lam : float, optional
        Penalty scale lambda (default: 0.4).
    beta : float, optional
        Concavity parameter beta > 0 (default: 2.0).
    """

    def __init__(self, arg, lam=0.4, beta=2.0):
        if beta <= 0:
            raise ValueError("beta must be positive, got {}".format(beta))
        if lam < 0:
            raise ValueError("lam must be non-negative, got {}".format(lam))
        self.arg = arg
        self.lam = lam
        self.beta = beta

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        result = np.empty_like(ax)
        mask = ax <= self.beta * self.lam
        result[mask] = self.lam * ax[mask] - (ax[mask] * ax[mask]) / (2.0 * self.beta)
        result[~mask] = 0.5 * self.beta * self.lam * self.lam
        return float(np.sum(result))

    def argmin(self, lamb, tensorlist):
        # lamb is the ADMM step size (called lam_s in the docstring formulas,
        # distinct from self.lam which is the MCP penalty scale).
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        s = np.sign(x)

        if self.beta <= lamb:
            threshold = math.sqrt(self.beta * lamb) * self.lam
            result = np.where(ax <= threshold, 0.0, x)
            return [result.tolist()]

        result = s * np.minimum(
            self.beta * np.maximum(ax - self.lam * lamb, 0.0) / (self.beta - lamb),
            ax,
        )
        return [result.tolist()]
