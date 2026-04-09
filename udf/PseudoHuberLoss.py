import admm
import numpy as np


class PseudoHuberLoss(admm.UDFBase):
    r"""Pseudo-Huber loss: a smooth approximation to the Huber loss.

    Mathematical definition:

        f(r) = δ² · Σᵢ (√(1 + (rᵢ/δ)²) − 1)

    where δ > 0 is the transition parameter.

    Behavior:
        |r| ≪ δ:  f(r) ≈ r²/(2δ²) · δ² = r²/2    (like L2)
        |r| ≫ δ:  f(r) ≈ δ · |r| − δ²              (like L1)

    Unlike the standard Huber loss which has a kink at |r| = δ,
    the Pseudo-Huber loss is smooth (C∞) everywhere.

    Gradient:

        ∇f(r)ᵢ = rᵢ / √(1 + (rᵢ/δ)²)

    The gradient is bounded in (−δ, δ), providing outlier robustness.

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    delta : float
        Transition parameter. Controls the L2-to-L1 transition scale.
    """

    def __init__(self, arg, delta=1.0):
        self.arg = arg
        self.delta = delta

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        d = self.delta
        return float(d ** 2 * np.sum(np.sqrt(1 + (r / d) ** 2) - 1))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        d = self.delta
        return [r / np.sqrt(1 + (r / d) ** 2)]
