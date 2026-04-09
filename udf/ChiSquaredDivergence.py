import admm
import numpy as np


class ChiSquaredDivergence(admm.UDFBase):
    r"""Chi-squared (χ²) divergence: a symmetric-looking f-divergence.

    Mathematical definition:

        f(x) = Σᵢ (xᵢ − yᵢ)² / yᵢ

    where y > 0 is the fixed reference distribution and x is the variable.

    Behavior:
        x = y:    f(x) = 0                 (minimum)
        x ≠ y:    f(x) > 0                 (always non-negative)

    The χ² divergence is a weighted sum of squared differences, where
    elements with small yᵢ contribute more to the divergence. This makes
    it sensitive to deviations in low-probability regions.

    Gradient:

        ∇f(x)ᵢ = 2(xᵢ − yᵢ) / yᵢ

    Properties:
    - Convex in x (weighted quadratic)
    - An f-divergence with generator φ(t) = (t − 1)²
    - Related to Pearson's chi-squared test statistic
    - Upper bounds the KL divergence: KL(x‖y) ≤ χ²(x‖y) / (2·min(y))

    Used in:
    - Goodness-of-fit testing
    - Distribution matching (GANs with χ² penalty)
    - Importance sampling diagnostics

    Parameters
    ----------
    arg : admm.Var or expression
        The variable distribution x.
    y : array_like
        The fixed reference distribution (must be positive).
    """

    def __init__(self, arg, y):
        self.arg = arg
        self.y = np.asarray(y, dtype=float)

    def arguments(self):
        return [self.arg]

    def eval(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return float(np.sum((x - self.y) ** 2 / self.y))

    def grad(self, arglist):
        x = np.asarray(arglist[0], dtype=float)
        return [2.0 * (x - self.y) / self.y]
