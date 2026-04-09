import admm
import numpy as np


class SmoothQuantileLoss(admm.UDFBase):
    """Smooth quantile (pinball) loss for quantile regression.

    Function:

        f(u) = sum(tau * u_i + (1/beta) * log(1 + exp(-beta * u_i)))

    This is a smooth approximation to the classical pinball loss:
        rho_tau(u) = u * (tau - I(u < 0))

    As beta → ∞, the smooth version converges to the exact pinball loss.

    Gradient:

        grad_i = tau - (1 - sigmoid(beta * u_i))

    Parameters
    ----------
    arg : admm.Var or expression
        The residual u = prediction - target.
    tau : float
        Quantile level in (0, 1). tau=0.5 gives the median.
    beta : float
        Smoothing parameter. Larger beta = closer to exact pinball.
    """

    def __init__(self, arg, tau=0.5, beta=20.0):
        self.arg = arg
        self.tau = tau
        self.beta = beta

    def arguments(self):
        return [self.arg]

    def _softplus(self, z):
        """Numerically stable log(1 + exp(z))."""
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def eval(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        return float(np.sum(self.tau * u + (1.0 / self.beta) * self._softplus(-self.beta * u)))

    def grad(self, arglist):
        u = np.asarray(arglist[0], dtype=float)
        return [self.tau - (1.0 - self._sigmoid(self.beta * u))]
