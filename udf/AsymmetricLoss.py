import admm
import numpy as np


class AsymmetricLoss(admm.UDFBase):
    r"""Asymmetric quadratic loss: different penalties for positive/negative errors.

    Mathematical definition:

        f(r) = Σᵢ [w₊ · max(rᵢ, 0)² + w₋ · max(−rᵢ, 0)²]

    where w₊ > 0 and w₋ > 0 are the weights for positive and negative
    errors respectively.

    Behavior:
        rᵢ > 0:  contribution = w₊ · rᵢ²   (positive error penalty)
        rᵢ < 0:  contribution = w₋ · rᵢ²   (negative error penalty)
        rᵢ = 0:  contribution = 0           (no penalty)

    Gradient:

        ∇f(r)ᵢ = 2w₊ · rᵢ    if rᵢ ≥ 0
                = 2w₋ · rᵢ    if rᵢ < 0

    Properties:
    - Convex (piecewise quadratic with matching value at 0)
    - C¹ smooth (continuous gradient, kink in second derivative at 0)
    - Reduces to standard L2 when w₊ = w₋

    Used in:
    - Newsvendor problem (inventory optimization: overage ≠ underage cost)
    - Asymmetric risk assessment (downside risk ≠ upside risk)
    - Quantile-like regression (ratio w₋/w₊ determines the quantile)
    - Forecast evaluation where over/under-prediction have different costs

    Parameters
    ----------
    arg : admm.Var or expression
        The input (typically a residual vector).
    w_pos : float
        Weight for positive errors (r > 0).
    w_neg : float
        Weight for negative errors (r < 0).
    """

    def __init__(self, arg, w_pos=1.0, w_neg=2.0):
        self.arg = arg
        self.w_pos = w_pos
        self.w_neg = w_neg

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return float(np.sum(
            self.w_pos * np.maximum(r, 0) ** 2
            + self.w_neg * np.maximum(-r, 0) ** 2
        ))

    def grad(self, arglist):
        r = np.asarray(arglist[0], dtype=float)
        return [np.where(r >= 0, 2 * self.w_pos * r, 2 * self.w_neg * r)]
