import admm
import numpy as np


class SmoothEpsilonLoss(admm.UDFBase):
    r"""Smooth ε-insensitive loss for Support Vector Regression (SVR).

    Mathematical definition:

        f(x) = Σᵢ [sp(xᵢ − yᵢ − ε) + sp(yᵢ − xᵢ − ε)]

    where sp(z) = log(1 + exp(z)) is the softplus function,
    y is the target vector, and ε ≥ 0 is the insensitivity tube width.

    Behavior:
        |xᵢ − yᵢ| < ε:    ≈ 0           (inside the tube, no penalty)
        |xᵢ − yᵢ| > ε:    ≈ |xᵢ − yᵢ| − ε  (linear penalty outside tube)

    This is a smooth approximation to the standard ε-insensitive loss:
        ρ_ε(r) = max(0, |r| − ε)

    which is non-differentiable at r = ±ε.

    Gradient:

        ∇f(x)ᵢ = σ(xᵢ − yᵢ − ε) − σ(yᵢ − xᵢ − ε)

    where σ(z) = 1/(1+exp(−z)) is the sigmoid function.

    The smooth ε-insensitive loss is used in SVR where predictions
    within ε of the target are considered exact (zero loss), providing
    built-in noise tolerance.

    Parameters
    ----------
    arg : admm.Var or expression
        The prediction vector x.
    y : array_like
        Target values.
    eps : float
        Half-width of the insensitivity tube (ε ≥ 0).
    """

    def __init__(self, arg, y, eps=0.5):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)
        self.eps = eps

    def arguments(self):
        return [self.arg]

    def _softplus(self, z):
        """Numerically stable softplus: log(1 + exp(z))."""
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        r = x - self.y
        return float(np.sum(
            self._softplus(r - self.eps) + self._softplus(-r - self.eps)
        ))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        r = x - self.y
        return [self._sigmoid(r - self.eps) - self._sigmoid(-r - self.eps)]
