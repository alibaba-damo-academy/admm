import admm
import numpy as np


class SCADPenalty(admm.UDFBase):
    """Smoothly Clipped Absolute Deviation (SCAD) penalty (Fan & Li, 2001).

    The SCAD penalty is nonconvex and piecewise smooth. It combines the
    unbiasedness of hard thresholding with the continuity of soft thresholding.

    Function (coordinatewise, t = |x_i|):

        p(x_i) = alpha * t                                      if t <= alpha
               = (-t^2 + 2*a*alpha*t - alpha^2) / (2*(a - 1))   if alpha < t <= a*alpha
               = (a + 1) * alpha^2 / 2                          if t > a*alpha

    Proximal operator (coordinatewise; requires a > 1):

    When a > 1 + lam:

        prox_{lam * p}(v)_i
            = sgn(v_i) * max(|v_i| - alpha*lam, 0)               if |v_i| <= (1 + lam)*alpha
            = ((a-1)*v_i - sgn(v_i)*a*alpha*lam) / (a-1-lam)     if (1+lam)*alpha < |v_i| <= a*alpha
            = v_i                                                if |v_i| > a*alpha

    When 1 < a <= 1 + lam (edge case):

        prox_{lam * p}(v)_i
            = sgn(v_i) * max(|v_i| - alpha*lam, 0)               if |v_i| <= (a+1+lam)*alpha/2
            = v_i                                                otherwise

    Parameters
    ----------
    arg : admm.Var
        The optimization variable (vector).
    alpha : float, optional
        Regularization scale (default: 0.4).
    a : float, optional
        Concavity parameter, must satisfy a > 1 (default: 3.7).
    """

    def __init__(self, arg, alpha=0.4, a=3.7):
        if a <= 1:
            raise ValueError("a must be > 1, got {}".format(a))
        if alpha < 0:
            raise ValueError("alpha must be non-negative, got {}".format(alpha))
        self.arg = arg
        self.alpha = alpha
        self.a = a

    def arguments(self):
        return [self.arg]

    def eval(self, tensorlist):
        x = np.asarray(tensorlist[0], dtype=float)
        ax = np.abs(x)
        result = np.empty_like(ax)

        mask1 = ax <= self.alpha
        mask2 = (~mask1) & (ax <= self.a * self.alpha)
        mask3 = ~(mask1 | mask2)

        result[mask1] = self.alpha * ax[mask1]
        result[mask2] = (
            -ax[mask2] * ax[mask2]
            + 2.0 * self.a * self.alpha * ax[mask2]
            - self.alpha * self.alpha
        ) / (2.0 * (self.a - 1.0))
        result[mask3] = 0.5 * (self.a + 1.0) * self.alpha * self.alpha
        return float(np.sum(result))

    def argmin(self, lamb, tensorlist):
        v = np.asarray(tensorlist[0], dtype=float)
        av = np.abs(v)
        s = np.sign(v)
        alpha = self.alpha
        a = self.a

        result = np.empty_like(v)
        if a > 1.0 + lamb:
            mask1 = av <= (1.0 + lamb) * alpha
            mask2 = (~mask1) & (av <= a * alpha)
            mask3 = ~(mask1 | mask2)

            result[mask1] = s[mask1] * np.maximum(av[mask1] - alpha * lamb, 0.0)
            result[mask2] = ((a - 1.0) * v[mask2] - s[mask2] * a * alpha * lamb) / (
                a - 1.0 - lamb
            )
            result[mask3] = v[mask3]
            return [result.tolist()]

        cutoff = 0.5 * (a + 1.0 + lamb) * alpha
        mask1 = av <= cutoff
        mask2 = ~mask1
        result[mask1] = s[mask1] * np.maximum(av[mask1] - alpha * lamb, 0.0)
        result[mask2] = v[mask2]
        return [result.tolist()]
