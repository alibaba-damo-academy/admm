import admm
import numpy as np


class QuantileHuberLoss(admm.UDFBase):
    r"""Quantile Huber loss: asymmetric smooth loss for quantile regression.

    Mathematical definition:

        f(r) = ε · Σᵢ [τ · softplus(rᵢ/ε) + (1−τ) · softplus(−rᵢ/ε)]

    where softplus(z) = log(1 + exp(z)), τ ∈ (0,1) is the quantile level,
    and ε > 0 controls the smoothing.

    Behavior:
        ε → 0:  Converges to the exact pinball (quantile) loss
                 ρ_τ(r) = r·(τ − I(r < 0))
        r → +∞: f(r) ≈ τ·r                  (linear with slope τ)
        r → −∞: f(r) ≈ (1−τ)·|r|            (linear with slope 1−τ)

    Gradient:

        ∇f(r)ᵢ = τ · σ(rᵢ/ε) − (1−τ) · (1 − σ(rᵢ/ε))
               = σ(rᵢ/ε) − (1−τ)

    where σ(z) = 1/(1+exp(−z)) is the sigmoid function.

    The Quantile Huber loss provides a smooth, differentiable
    approximation to the check function used in quantile regression.
    Unlike least squares which estimates the conditional mean, quantile
    regression estimates conditional quantiles (median, percentiles).

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically residual r = x − y).
    tau : float
        Quantile level in (0, 1). τ=0.5 gives the smoothed median.
    eps : float
        Smoothing parameter. Smaller ε gives a tighter approximation
        to the exact pinball loss.
    """

    def __init__(self, arg, tau=0.5, eps=0.1):
        self.arg = arg
        self.tau = tau
        self.eps = eps

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        z = r / self.eps
        return float(self.eps * np.sum(
            self.tau * np.logaddexp(0, z)
            + (1 - self.tau) * np.logaddexp(0, -z)
        ))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        sig = 1.0 / (1.0 + np.exp(-r / self.eps))
        return [self.tau * sig - (1 - self.tau) * (1 - sig)]
