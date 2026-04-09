import admm
import numpy as np


class CorrentropyLoss(admm.UDFBase):
    r"""Correntropy-induced loss: maximum correntropy criterion (MCC).

    Mathematical definition:

        f(r) = Σᵢ [1 − exp(−rᵢ² / (2σ²))]

    where σ > 0 is the kernel bandwidth.

    Behavior:
        rᵢ = 0:     f = 0                 (minimum)
        |rᵢ| → ∞:   f → 1 per element     (bounded, saturates at n)
        |rᵢ| ≪ σ:   f ≈ rᵢ²/(2σ²)        (like scaled L2)
        |rᵢ| ≫ σ:   f ≈ 1                 (constant, outlier rejection)

    Gradient:

        ∇f(r)ᵢ = rᵢ/σ² · exp(−rᵢ² / (2σ²))

    The gradient is redescending: peaks at |rᵢ| = σ and decays
    as a Gaussian envelope for large residuals.

    Properties:
    - Smooth (C∞) and bounded in [0, n)
    - Non-convex globally
    - Closely related to Welsch loss (same shape, different parameterization)
    - Based on the correntropy kernel: V(e) = E[κ(X, X−e)]
      where κ is a Gaussian kernel

    The correntropy loss originates from information-theoretic learning
    and is used in:
    - Robust adaptive filtering (MCC filters)
    - Signal processing in impulsive noise
    - Robust regression and classification
    - Brain-computer interfaces

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    sigma : float
        Kernel bandwidth. Controls the transition from quadratic
        to constant behavior.
    """

    def __init__(self, arg, sigma=1.0):
        self.arg = arg
        self.sigma = sigma

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(1 - np.exp(-r ** 2 / (2 * self.sigma ** 2))))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [r / self.sigma ** 2 * np.exp(-r ** 2 / (2 * self.sigma ** 2))]
