import admm
import numpy as np


class SmoothHingePairLoss(admm.UDFBase):
    r"""Smooth pairwise ranking loss: encourages y > x + margin.

    Mathematical definition:

        f(x, y) = Σᵢ softplus(xᵢ − yᵢ + margin)

    where softplus(z) = log(1 + exp(z)).

    Behavior:
        yᵢ ≫ xᵢ + margin:   f ≈ 0        (margin satisfied, no loss)
        yᵢ ≈ xᵢ + margin:   f ≈ log(2)   (at the boundary)
        yᵢ ≪ xᵢ + margin:   f ≈ xᵢ − yᵢ + margin  (linear penalty)

    This is a smooth approximation to the hinge-based ranking loss:
        max(0, xᵢ − yᵢ + margin)

    Gradient (two-argument):

        ∂f/∂xᵢ = σ(xᵢ − yᵢ + margin)
        ∂f/∂yᵢ = −σ(xᵢ − yᵢ + margin)

    where σ(z) = 1/(1+exp(−z)) is the sigmoid function.

    Used in:
    - Learning to rank (pairwise approach)
    - Metric learning (embedding distances)
    - Recommendation systems (BPR — Bayesian Personalized Ranking)
    - Contrastive learning objectives

    Parameters
    ----------
    x_arg : admm.Var or expression
        First variable (the "negative" or lower-ranked items).
    y_arg : admm.Var or expression
        Second variable (the "positive" or higher-ranked items).
    margin : float
        Desired minimum gap y − x. Default 1.0.
    """

    def __init__(self, x_arg, y_arg, margin=1.0):
        self.x_arg = x_arg
        self.y_arg = y_arg
        self.margin = margin

    def arguments(self):
        return [self.x_arg, self.y_arg]

    def _softplus(self, z):
        """Numerically stable softplus."""
        return np.where(z > 20, z, np.log(1 + np.exp(np.clip(z, -500, 20))))

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        y = np.asarray(arglist[1], dtype=float)
        return float(np.sum(self._softplus(x - y + self.margin)))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        y = np.asarray(arglist[1], dtype=float)
        s = self._sigmoid(x - y + self.margin)
        return [s, -s]
